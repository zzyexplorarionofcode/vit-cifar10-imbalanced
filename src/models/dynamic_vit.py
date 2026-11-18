"""DynamicViT 风格的渐进式 token 剪枝（progressive token pruning）。

这个文件解决什么问题
--------------------
普通 ViT 对每个 patch token 都做同样多的注意力计算，图像里大量背景 patch 其实
对分类帮助不大。DynamicViT 在若干 Transformer Block 之后插入一个很小的打分
网络（``TokenPruner``），给每个 patch token 算“重要性”，只保留得分最高的固定
比例 token，再让后续 Block 在更短的序列上计算。因为 token 是真被删掉而不是被
mask，后续 FLOPs 会实际下降。

在“数据→模型→训练→评估”流程中的位置
------------------------------------
1. ``src/data.py`` 产生 ``[B,3,32,32]`` 图像 batch；
2. **本文件**继承 ``vit.py`` 的 ``VisionTransformer``，在前向中途裁剪 token，
   最终仍输出 ``[B,10]`` logits；
3. ``src/engine.py`` 用交叉熵/蒸馏损失训练；剪枝器没有单独的 ratio loss，
   保留数量由固定 schedule 决定；
4. 评估时 ``token_ratio`` 可以报告实际保留比例，``scripts/dynamic_tokens.py``
   会把它记录到结果文件。

阅读前需要知道的概念
--------------------
- **token 显著性**：用一个两层 MLP 对每个 token 输出一个标量分数，分数越高越
  可能保留。这个分数不是分类 logits，而是“排序依据”。
- **Gumbel 噪声**：训练时给分数加独立 Gumbel 噪声再做 top-k，相当于从一个
  随机排序中采样，能让剪枝器探索不同 token 组合；推理时去掉噪声，结果确定。
- **top-k 与 gather**：``topk`` 返回索引，``gather`` 按索引从原张量取 token。
  索引操作本身不可导，所以需要下面的 straight-through estimator。
- **Straight-Through Estimator（STE，直通估计器）**：前向用硬 0/1 选择（真的
  丢 token），反向却把梯度送给软门控值。常见写法
  ``hard_value * (soft - soft.detach() + 1)``：前向因子恰好是 1，反向则把
  ``hard_value`` 的梯度按链式法则传回 ``soft``，从而绕过不可导的 top-k。
- **detach**：返回数值相同但脱离计算图的张量；它在前向不改变结果，在反向
  截断梯度。STE 里用它构造“只保留 soft 梯度”的减法。
- **cls token 永远保留**：分类只依赖第 0 个 token，剪掉它模型就没有输出了。

主要类/函数
-----------
- ``_gumbel_noise``：采样标准 Gumbel 噪声。
- ``TokenPruner``：给 patch token 打分并执行 top-k 保留 + STE。
- ``DynamicViT``：在 ViT Block 之间插入 pruner 的模型。
- ``dynamic_vit_tiny``：把扁平关键字参数拆成“ViT 配置 + 剪枝配置”的便捷函数。
"""

from __future__ import annotations  # 延迟类型注解求值，允许在注解里使用较新的写法而不影响运行时

from typing import List, Sequence  # List 仅用于 token_counts 的类型提示；Sequence 表示只读序列

import torch
import torch.nn as nn

from .vit import ViTConfig, VisionTransformer  # 复用基线的 patch embedding、Block、norm 和 head


def _gumbel_noise(shape, device, dtype) -> torch.Tensor:
    """采样标准 Gumbel(0, 1) 噪声。

    Gumbel 噪声的经典构造是 ``-log(-log(U))``，其中 ``U ~ Uniform(0,1]``。
    ``torch.rand`` 可能产生 0，直接取 log 会得到 ``-inf``，所以先 ``clamp_min``
    到一个极小正数。噪声本身不需要梯度，它只用来扰动排序。

    参数
    ----
    shape : tuple
        输出形状，例如 ``(B, N)``。
    device : torch.device
        噪声所在设备，必须与分数张量一致。
    dtype : torch.dtype
        噪声数据类型，通常与分数一致（如 float32）。

    返回
    ----
    torch.Tensor
        形状为 ``shape`` 的 Gumbel 噪声，数值可正可负。
    """
    u = torch.rand(shape, device=device, dtype=dtype).clamp_min(1e-20)  # U ~ [0,1)，夹到正数避免 log(0)
    return -torch.log(-torch.log(u))  # 标准 Gumbel 变换；输出与 u 同形状


class TokenPruner(nn.Module):
    """轻量 token 打分器：保留 cls token 和得分最高的 k 个 patch token。

    结构是 ``LayerNorm -> Linear(C, hidden) -> GELU -> Linear(hidden, 1)``，
    对每个 patch token 独立输出一个分数。训练时分数加 Gumbel 噪声后做 top-k，
    并用 STE 让梯度通过被选中 token 的软门控回传；推理时直接取 top-k，不做
    随机扰动。被丢弃的 token 不参与后续计算，所以后续 Block 的序列长度变短。
    """

    def __init__(self, dim: int, hidden_dim: int | None = None, temperature: float = 1.0):
        """初始化打分 MLP。

        参数
        ----
        dim : int
            输入 token 维度 C。
        hidden_dim : int or None
            隐藏层宽度；为 ``None`` 时取 ``max(dim // 2, 16)``，保证小模型也有
            至少 16 个隐藏单元。隐藏层刻意比 C 窄，使额外参数和 FLOPs 很小。
        temperature : float
            软门控 ``sigmoid(score / temperature)`` 的温度。温度越小，门控越
            接近 0/1；默认 1.0。它只影响训练时的梯度，不影响前向硬选择。
        """
        super().__init__()
        hidden_dim = hidden_dim or max(dim // 2, 16)  # 注意 `or`：传入 0 也会退回默认值，这是有意的容错
        self.norm = nn.LayerNorm(dim)  # 先归一化，避免不同 token 的尺度差异主导打分
        self.fc1 = nn.Linear(dim, hidden_dim)  # [B,N,C] -> [B,N,hidden]
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, 1)  # [B,N,hidden] -> [B,N,1]，每个 token 一个分数
        self.temperature = temperature
        self.last_keep = 0  # 诊断信息：最近一次前向保留的 patch 数（不含 cls），不是 buffer
        self.last_ratio = 1.0  # 诊断信息：最近一次实际保留比例 k/n

    def forward(self, x: torch.Tensor, keep_ratio: float) -> torch.Tensor:
        """对一个序列执行 token 选择。

        参数
        ----
        x : torch.Tensor
            形状 ``[B, 1+N, C]``；第 0 个是 cls token，后 N 个是 patch token。
        keep_ratio : float
            要保留的 patch 比例，例如 0.5 表示保留一半。

        返回
        ----
        torch.Tensor
            形状 ``[B, 1+k, C]``，其中
            ``k = max(1, min(N, int(round(N * keep_ratio))))``；cls token 固定在
            第 0 位，后面是选中的 patch token。
        """
        cls, tokens = x[:, :1], x[:, 1:]  # [B,1,C] 与 [B,N,C]；切片是视图，不复制数据
        b, n, c = tokens.shape  # n 是当前 patch 数，会随剪枝逐级减少
        k = max(1, min(n, int(round(n * keep_ratio))))  # 至少留 1 个、至多 n 个；round 后转 int 作为 topk 的 k
        score = self.fc2(self.act(self.fc1(self.norm(tokens)))).squeeze(-1)  # [B,N,C] -> [B,N,1] -> [B,N]

        if self.training:  # 训练时加 Gumbel 噪声：每次前向的 top-k 排序会随机变化，起到探索作用
            ranking = score + _gumbel_noise(score.shape, score.device, score.dtype)  # [B,N] + [B,N]
        else:  # 推理时使用原始分数，保证输出确定、可复现
            ranking = score
        idx = ranking.topk(k, dim=1, largest=True, sorted=False).indices  # [B,k] 的 token 下标；sorted=False 省一次排序
        gather_idx = idx.unsqueeze(-1).expand(-1, -1, c)  # [B,k] -> [B,k,1] -> [B,k,C]；expand 是视图，最后一维 stride=0
        kept = torch.gather(tokens, 1, gather_idx)  # 沿 token 维按索引取特征：[B,N,C] -> [B,k,C]

        if self.training:
            # 前向值：kept_soft - kept_soft.detach() + 1 恰好等于 1，所以 kept 不变；
            # 反向：detach 的那份被当作常数，门控因子的梯度只通过 kept_soft 回传，形成 STE。
            soft = torch.sigmoid(score / self.temperature)  # [B,N]，每个 token 的软保留概率
            kept_soft = torch.gather(soft, 1, idx).unsqueeze(-1)  # [B,k] -> [B,k,1]，只取被选中 token 的门控值
            kept = kept * (kept_soft - kept_soft.detach() + 1.0)  # 前向乘 1；反向把梯度送到 score

        self.last_keep = k
        self.last_ratio = k / n  # 记录实际比例，便于脚本统计；不参与损失
        return torch.cat((cls, kept), dim=1)  # [B,1,C] 与 [B,k,C] 拼接 -> [B,1+k,C]


class DynamicViT(VisionTransformer):
    """在 ViT 的指定 Block 之后插入 token 剪枝器的分类模型。

    它继承 ``VisionTransformer`` 的 patch embedding、cls token、位置嵌入、Block、
    最终 LayerNorm 和分类头，只重写 ``forward_features``：每经过一个剪枝点，就把
    当前 patch token 数按 schedule 减少。继承来的 ``forward`` 会动态调用子类
    重写后的 ``forward_features``，因此无需再复制分类头逻辑。

    默认 ``prune_after=(1,2,3)``、``keep_ratios=(0.75,0.5,0.25)`` 表示在 Block 1、
    2、3 之后分别保留 75%、50%、25% 的 patch token。注意这个默认值要求
    ``cfg.depth >= 5``（因为最后一个下标不能等于 ``depth-1``）；本项目默认的
    ``ViTConfig.depth=4`` 下应使用工厂的 ``(0,1,2)``，具体以配置为准。
    """

    def __init__(
        self,
        cfg: ViTConfig,
        prune_after: Sequence[int] = (1, 2, 3),
        keep_ratios: Sequence[float] = (0.75, 0.5, 0.25),
        predictor_hidden: int | None = None,
        temperature: float = 1.0,
    ):
        """初始化 DynamicViT。

        参数
        ----
        cfg : ViTConfig
            主干 ViT 配置；要求 ``max(prune_after) <= cfg.depth - 2``，也就是
            必须留出至少一个 Block 在剪枝之后继续计算。
        prune_after : Sequence[int]
            在哪些 Block 之后剪枝，使用 0-based 下标。例如 ``(0,1,2)`` 表示
            Block 0、1、2 的输出各剪一次；下标必须落在 ``[0, depth-2]``。
        keep_ratios : Sequence[float]
            与 ``prune_after`` 等长的保留比例，每个元素对应一个剪枝点。
        predictor_hidden : int or None
            打分 MLP 的隐藏宽度；``None`` 时由 ``TokenPruner`` 选默认值。
        temperature : float
            软门控温度，仅影响训练时 STE 的梯度。

        关键设计：保留数量由固定比例 schedule 决定，而不是让网络自己学一个
        可微的保留率，因此不需要额外的 ratio loss，训练也更稳定。
        """
        super().__init__(cfg)  # 先搭好完整的 ViT 主干（包括 blocks、norm、head）
        if len(prune_after) != len(keep_ratios):  # 每个剪枝点都必须有对应的保留比例
            raise ValueError("prune_after and keep_ratios must have the same length")
        if any(i < 0 or i >= cfg.depth - 1 for i in prune_after):  # 只允许在最后一个 Block 之前剪
            raise ValueError("pruning must happen before the final block")
        self.prune_after = tuple(int(i) for i in prune_after)  # 转成 tuple，避免外部 list 被修改后行为漂移
        self.keep_ratios = tuple(float(r) for r in keep_ratios)  # 转成 float，兼容配置里的整数字面量
        self.pruners = nn.ModuleList(
            [TokenPruner(cfg.embed_dim, predictor_hidden, temperature) for _ in self.prune_after]
        )  # 每个剪枝点一个独立 pruner；ModuleList 保证参数被注册
        self._prune_map = {layer: j for j, layer in enumerate(self.prune_after)}  # 层下标 -> pruner 下标，前向时 O(1) 查找
        self.token_counts: List[int] = []  # 记录每个阶段的 token 总数（含 cls），供 token_ratio 使用

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """带渐进剪枝的特征提取。

        参数
        ----
        x : torch.Tensor
            图像 batch，形状 ``[B,3,32,32]``。

        返回
        ----
        torch.Tensor
            形状 ``[B,1+k_last,C]``，已过最终 LayerNorm；``k_last`` 是最后一次
            剪枝后保留的 patch 数。
        """
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # [B,C,8,8] -> [B,C,64] -> [B,64,C]
        cls = self.cls_token.expand(x.shape[0], -1, -1)  # [1,1,C] -> [B,1,C]
        x = torch.cat((cls, x), dim=1) + self.pos_embed  # [B,1+64,C]，加可学习位置嵌入
        x = self.pos_drop(x)
        self.token_counts = [x.shape[1]]  # 初始 token 数 = 1 + 64；每次前向都重置，避免累积旧值
        for i, block in enumerate(self.blocks):
            x = block(x)  # 先完整跑完当前 Block
            if i in self._prune_map:  # 当前层之后有剪枝点
                j = self._prune_map[i]
                x = self.pruners[j](x, self.keep_ratios[j])  # [B,1+N,C] -> [B,1+k,C]
                self.token_counts.append(x.shape[1])  # 记录剪枝后的 token 数，含 cls
        return self.norm(x)  # Pre-LN 出口归一化；形状不变

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """完整前向：图像 batch → 分类 logits。

        参数
        ----
        x : torch.Tensor
            形状 ``[B,3,32,32]``。

        返回
        ----
        torch.Tensor
            形状 ``[B,num_classes]`` 的 logits。即使 patch token 被剪到很少，
            cls token 始终保留，所以分类头仍能工作。
        """
        return self.head(self.forward_features(x)[:, 0])  # 取 cls token [B,C] -> [B,10]

    @torch.no_grad()  # 只做统计，不需要构建计算图
    def token_ratio(self, x: torch.Tensor) -> float:
        """测量一次前向后最终保留的 token 比例。

        参数
        ----
        x : torch.Tensor
            形状 ``[B,3,32,32]`` 的一批图像；比例与 batch size 无关。

        返回
        ----
        float
            ``token_counts[-1] / token_counts[0]``，即最终 token 数（含 cls）与
            初始 token 数（含 cls）之比。例如 64 个 patch 剪到 16 个时约为
            ``(1+16)/(1+64) ≈ 0.262``，而不是 0.25。

        注意：函数会临时切到 eval 模式，再用 ``self.train(was_training)`` 恢复；
        如果前向中途抛异常，恢复语句不会执行，调用方需要自行处理。
        """
        was_training = self.training  # 记录调用前的全局训练/评估状态
        self.eval()  # 关闭 dropout、Gumbel 噪声等随机行为
        self.forward_features(x)  # 只跑特征提取，填充 token_counts；不关心返回值
        ratio = self.token_counts[-1] / self.token_counts[0]  # Python float
        self.train(was_training)  # 递归恢复原来的模式
        return ratio


def dynamic_vit_tiny(**kwargs) -> DynamicViT:
    """便捷构造函数：把扁平关键字参数拆给 ViTConfig 和 DynamicViT。

    ``**kwargs`` 可能同时包含两类参数：``img_size``、``embed_dim`` 等属于
    ``ViTConfig``；``prune_after``、``keep_ratios``、``predictor_hidden`` 等属于
    ``DynamicViT``。这里利用 dataclass 的 ``__dataclass_fields__`` 判断键的归属，
    分别打包，避免调用方手动嵌套配置对象。

    参数
    ----
    **kwargs
        任意 ViTConfig 字段，或 DynamicViT 的构造参数。

    返回
    ----
    DynamicViT
        按给定配置构建、尚未训练的模型。

    注意：如果传入既不是 dataclass 字段、也不是 DynamicViT 参数的键，会在
    ``DynamicViT(...)`` 处触发 ``TypeError``，而不是被静默忽略。
    """
    cfg_keys = ViTConfig.__dataclass_fields__.keys()  # 所有 dataclass 字段名（不含 num_patches 这个 property）
    cfg_kwargs = {k: v for k, v in kwargs.items() if k in cfg_keys}  # 字典推导：挑出 ViT 配置
    model_kwargs = {k: v for k, v in kwargs.items() if k not in cfg_keys}  # 其余键留给剪枝参数
    return DynamicViT(ViTConfig(**cfg_kwargs), **model_kwargs)  # 两次 ** 展开，分别构造配置和模型
