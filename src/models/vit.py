"""ViT 基线模型：在 32x32 CIFAR 图像上从零实现 Vision Transformer。

这个文件解决什么问题
--------------------
CIFAR-10 的一张图是 ``[3, 32, 32]`` 的 RGB 像素张量，而 Transformer 习惯处理
一串向量 ``[N, C]``。本文件用“把图像切成不重叠的小块（patch），每个 patch 线性
投影成一个 token”的方式完成这种转换，再堆叠若干 Pre-LN Transformer Block，最后
用分类 token（cls token）输出 10 类 logits。这里不依赖 timm 等模型库，目的是让
初学者能完整看到 ViT 的每一个形状变换和每一项计算。

在“数据→模型→训练→评估”流程中的位置
------------------------------------
1. ``src/data.py`` 把 npy 图像读成 ``[B, 3, 32, 32]`` 的 float batch；
2. **本文件**把 batch 变成 ``[B, 10]`` 的 logits（未归一化分数）；
3. ``src/losses.py`` 用 logits 和标签算交叉熵 / Class-Balanced Loss；
4. ``src/engine.py`` 负责反向传播、验证和测试指标；
5. ``src/models/factory.py`` 根据配置字符串选择本文件的 ``VisionTransformer``。

阅读前需要知道的概念
--------------------
- **张量形状记号**：``B`` 是 batch size，``N`` 是 token 数，``C`` 是每个 token 的
  通道数（embedding 维度），``H/W`` 是图像高宽。注释里常用
  ``# [B, N, C] -> ...`` 标出形状变化。
- **卷积作 patch embedding**：``kernel_size = stride = patch_size`` 的卷积，等价于
  把图像切成 patch 后逐 patch 做线性层，但写法更短、更快。
- **多头自注意力**：每个 token 产生 Q/K/V，用 Q 与所有 K 的点积算相关性，softmax
  归一化后对 V 加权求和。多个 head 在不同子空间并行做这件事。
- **Pre-LN 残差块**：先 LayerNorm 再进子层，最后加回输入；比 Post-LN 更容易训练。
- **随机深度 DropPath**：训练时按样本随机丢掉整条残差分支，推理时关闭。
- **logits**：模型最后不接 softmax；PyTorch 的交叉熵损失内部会做 log-softmax。

主要类/函数
-----------
- ``ViTConfig``：集中存放超参数的 dataclass。
- ``DropPath``：逐样本随机丢弃残差分支的正则化模块。
- ``Attention``：显式矩阵乘实现的多头自注意力。
- ``MLP``：Transformer 里的两层前馈网络。
- ``Block``：Pre-LN 的“注意力 + MLP”残差块。
- ``trunc_normal_``：截断正态初始化工具。
- ``VisionTransformer``：完整分类模型。
- ``vit_tiny``：用关键字参数快速构造模型的便捷函数。
"""

from __future__ import annotations  # 延迟求值类型注解：允许写 list[int] / int | None，且不会在导入时真正解析前向引用

from dataclasses import dataclass  # 自动生成 __init__/__repr__/__eq__，减少配置类样板代码
from typing import Optional  # Optional[T] 表示“T 或 None”，用于 Attention.last_attn

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass  # 装饰器：根据下面的字段自动生成 __init__、__repr__、__eq__，减少样板代码
class ViTConfig:
    """ViT 的超参数容器（纯配置对象，不是 ``nn.Module``）。

    ``@dataclass`` 会自动生成 ``__init__(img_size=32, patch_size=4, ...)``：
    下面每个带默认值的类属性都是一个可传入参数。这里只保存配置，不参与前向计算，
    因此不会出现在 ``model.parameters()`` 里，也不会被 ``.to(device)`` 移动。

    参数
    ----
    img_size : int
        输入图像边长，默认 32。当前实现假设图像是正方形。
    patch_size : int
        每个 patch 的边长，默认 4；32/4 得到每边 8 个 patch，共 64 个 token。
    in_chans : int
        输入通道数，RGB 为 3。
    num_classes : int
        CIFAR-10 为 10，决定分类头输出维度。
    embed_dim : int
        每个 token 的向量维度 C，也是所有 Block 内部的特征宽度。
    depth : int
        Transformer Block 的层数。
    num_heads : int
        多头注意力的 head 数；必须能整除 ``embed_dim``。
    mlp_ratio : float
        MLP 隐藏层宽度相对 ``embed_dim`` 的倍数。
    dropout : float
        token 丢弃率（位置嵌入后）和 MLP 输出丢弃率。
    attn_dropout : float
        注意力 softmax 权重上的 dropout 概率。
    drop_path : float
        DropPath 的最大丢弃率；实际值会按层从 0 线性增加到该值。
    init_scale : float
        线性层、卷积层、cls/位置嵌入的截断正态初始化标准差。
    """

    img_size: int = 32  # 输入边长；改了它要同时保证能被 patch_size 整除
    patch_size: int = 4  # 每个 patch 的边长；32/4=8，所以每边 8 个 patch
    in_chans: int = 3  # RGB 三通道
    num_classes: int = 10  # CIFAR-10 类别数
    embed_dim: int = 96  # token 维度 C；注意力里会拆成 num_heads 个 head
    depth: int = 4  # Block 堆叠层数
    num_heads: int = 4  # 每个 head 的维度 = embed_dim // num_heads = 24
    mlp_ratio: float = 2.0  # MLP 隐藏宽度 = 96 * 2 = 192
    dropout: float = 0.0  # 这个小模型默认不做普通 dropout，主要靠 DropPath 正则
    attn_dropout: float = 0.0  # 注意力权重 dropout，默认关闭
    drop_path: float = 0.05  # 随机深度上限；小数据上常用的轻量正则
    init_scale: float = 0.02  # ViT 对初始化敏感，使用较小的标准差

    @property  # 把方法伪装成只读属性：用 cfg.num_patches，而不是 cfg.num_patches()
    def num_patches(self) -> int:
        """由图像尺寸和 patch 尺寸推导出的 patch 数（不含 cls token）。

        返回
        ----
        int
            ``(img_size // patch_size) ** 2``。默认配置下为 ``(32 // 4) ** 2 = 64``。
            这是派生量，不单独存字段，避免修改 ``img_size`` 后忘记同步。
        """
        return (self.img_size // self.patch_size) ** 2


class DropPath(nn.Module):
    """随机深度（Stochastic Depth）：训练时按样本丢弃整条残差分支。

    对输入 ``x`` 的形状没有限制，通常是 ``[B, N, C]`` 或 ``[B, C, H, W]``。
    训练时每个样本独立采样一个 0/1 mask：保留时把输出除以 ``keep``，使训练期
    望与推理时一致；被丢弃的样本这一整条分支输出 0，相当于只保留残差直连。
    推理时（``self.training == False``）直接返回输入，不引入随机性。
    """

    def __init__(self, p: float = 0.0):
        """初始化 DropPath。

        参数
        ----
        p : float
            丢弃概率，取值通常在 ``[0, 1)``。``p=0`` 时 forward 直接走恒等路径。
        """
        super().__init__()  # 必须先初始化 nn.Module，否则子模块/参数注册和 train()/eval() 都不会正常工作
        self.p = p  # 普通 Python 属性，不是 nn.Parameter；它不需要学习，也会随模型保存到 state_dict 之外的配置里

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对输入做逐样本的随机深度。

        参数
        ----
        x : torch.Tensor
            任意形状的特征，首维必须是 batch，例如 ``[B, N, C]``。

        返回
        ----
        torch.Tensor
            与 ``x`` 同形状；训练时部分样本的整条分支被置零并缩放，推理时原样返回。
        """
        if self.p == 0.0 or not self.training:  # 推理阶段必须关闭随机性，保证同一输入得到同一输出
            return x
        keep = 1.0 - self.p  # 保留概率；下面除以它做重标定，保持期望不变
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # 例：[B,N,C] -> (B,1,1)，mask 可广播到每个 token/channel
        mask = x.new_empty(shape).bernoulli_(keep)  # 在 x 的 device/dtype 上采样 Bernoulli(keep)，得到 0/1 mask
        return x * mask / keep  # 被保留的样本乘 1/keep，被丢弃的样本整条分支变 0


class Attention(nn.Module):
    """用显式矩阵乘实现的多头自注意力（Multi-Head Self-Attention）。

    输入是一串 token ``[B, N, C]``。模块先用一个线性层一次性产生 Q/K/V，再把
    最后一维拆成 ``num_heads`` 个头，每个头在 ``head_dim = C / num_heads`` 的子空间
    里独立计算“缩放点积注意力”。显式写 matmul 而不是调用
    ``F.scaled_dot_product_attention``，是为了让中间 attention map 可见，方便
    初学者检查形状、做可视化和统计 FLOPs。

    关键形状：``[B, N, C] -> [B, H, N, Dh] -> [B, H, N, N] -> [B, N, C]``。
    """

    def __init__(self, dim: int, num_heads: int, attn_dropout: float = 0.0):
        """初始化 QKV 投影、输出投影和注意力 dropout。

        参数
        ----
        dim : int
            输入/输出 token 维度 C。
        num_heads : int
            head 数 H；要求 ``dim % num_heads == 0``，否则无法把 C 平均分给各头。
        attn_dropout : float
            对 softmax 后的注意力权重做 dropout 的概率；训练时随机屏蔽部分
            key 的贡献，推理时关闭。
        """
        super().__init__()
        if dim % num_heads != 0:  # 例如 dim=96、heads=5 就无法整除，直接报错比后面 reshape 失败更清楚
            raise ValueError("embed_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # 每个 head 的通道数 Dh
        self.scale = self.head_dim**-0.5  # 1/sqrt(Dh)，防止点积随维度增大而方差爆炸
        self.qkv = nn.Linear(dim, dim * 3, bias=True)  # 一次算完 Q/K/V，比三个独立 Linear 少一次调度
        self.proj = nn.Linear(dim, dim)  # 把各 head 拼接后的结果混合回 C 维
        self.attn_dropout = attn_dropout
        self.last_attn: Optional[torch.Tensor] = None  # 仅用于推理时保存注意力图，不参与梯度，也不是可学习参数

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """计算多头自注意力。

        参数
        ----
        x : torch.Tensor
            形状 ``[B, N, C]`` 的 token 序列；N 包含 cls token 和所有 patch token。

        返回
        ----
        torch.Tensor
            形状 ``[B, N, C]``，已经过输出投影，可与残差输入相加。
        """
        b, n, c = x.shape  # 解包出 batch、token 数、通道数；后续 reshape 都依赖它们
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)  # [B,N,C] -> [B,N,3,H,Dh]
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # [B,N,3,H,Dh] -> [3,B,H,N,Dh] -> 三个 [B,H,N,Dh]
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B,H,N,Dh] @ [B,H,Dh,N] -> [B,H,N,N]
        attn = attn.softmax(dim=-1)  # 对最后一维（每个 query 对应的所有 key）归一化，使每行和为 1
        attn = F.dropout(attn, p=self.attn_dropout, training=self.training)  # 训练时随机丢弃部分注意力连接
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)  # [B,H,N,N]@[B,H,N,Dh] -> [B,H,N,Dh] -> [B,N,C]
        if not self.training:  # 只在评估/推理时缓存注意力图，避免训练时持有计算图造成显存泄漏
            self.last_attn = attn.detach()  # detach 后不参与反向传播；形状 [B,H,N,N]
        return self.proj(out)  # 把 H 个头拼接后的信息再线性混合一次


class MLP(nn.Module):
    """Transformer Block 中的两层前馈网络（FFN）。

    结构是 ``Linear(C, hidden) -> GELU -> Linear(hidden, C) -> Dropout``，对每个
    token 独立作用。``hidden`` 通常比 ``C`` 宽（``mlp_ratio`` 倍），给模型更大的
    逐 token 非线性变换容量；因为不跨 token 交互，它不改变 N。
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        """初始化前馈网络。

        参数
        ----
        dim : int
            输入/输出通道数 C。
        hidden_dim : int
            中间隐藏层宽度，通常为 ``int(dim * mlp_ratio)``。
        dropout : float
            第二个线性层输出上的 dropout 概率。
        """
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)  # [B,N,C] -> [B,N,hidden_dim]
        self.act = nn.GELU()  # GELU 比 ReLU 平滑，是 ViT 的常用激活函数
        self.fc2 = nn.Linear(hidden_dim, dim)  # [B,N,hidden_dim] -> [B,N,C]
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """逐 token 做两层非线性变换。

        参数
        ----
        x : torch.Tensor
            形状 ``[B, N, C]``。

        返回
        ----
        torch.Tensor
            形状 ``[B, N, C]``，与输入可直接残差相加。
        """
        return self.drop(self.fc2(self.act(self.fc1(x))))  # 形状始终是 [B,N,C] -> [B,N,hidden] -> [B,N,C]


class Block(nn.Module):
    """Pre-LN 的 Transformer Block：注意力子层 + MLP 子层，各带残差。

    计算顺序是 ``x = x + DropPath(Attn(LN(x)))``，再
    ``x = x + DropPath(MLP(LN(x)))``。LayerNorm 放在子层之前（Pre-LN），使得
    残差主干上始终有一条“干净”的恒等路径，深层网络也更容易优化；作为代价，
    最后需要在整个模型出口再补一个 ``LayerNorm``（见 ``VisionTransformer.norm``）。
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        """初始化一个 Transformer Block。

        参数
        ----
        dim : int
            token 维度 C，也是残差主干的宽度。
        num_heads : int
            注意力头数。
        mlp_ratio : float
            MLP 隐藏宽度相对 ``dim`` 的倍数。
        dropout : float
            MLP 输出 dropout 概率。
        attn_dropout : float
            注意力权重 dropout 概率。
        drop_path : float
            本 Block 两条残差分支的随机深度概率；为 0 时用 ``nn.Identity``
            代替，既省一点开销，也避免无意义的随机数生成。
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)  # 注意力前的归一化，稳定 Q/K/V 的尺度
        self.attn = Attention(dim, num_heads, attn_dropout=attn_dropout)
        self.norm2 = nn.LayerNorm(dim)  # MLP 前的归一化
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout=dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()  # 0 时用恒等映射，语义相同但更快

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行一个 Pre-LN Block。

        参数
        ----
        x : torch.Tensor
            形状 ``[B, N, C]``。

        返回
        ----
        torch.Tensor
            形状 ``[B, N, C]``；残差相加要求两边形状完全一致。
        """
        x = x + self.drop_path(self.attn(self.norm1(x)))  # 注意力分支；DropPath 可能把整条分支置零
        x = x + self.drop_path(self.mlp(self.norm2(x)))  # MLP 分支；两条分支各自独立采样 mask
        return x


# 下面签名里的 `tensor: torch.Tensor` 和 `-> None` 是类型注解，只用于阅读和静态检查，不改变运行逻辑。
def trunc_normal_(tensor: torch.Tensor, std: float = 0.02) -> None:
    """用截断正态分布原地初始化张量。

    先按 ``N(0, std^2)`` 采样，再把数值截断到 ``[-2*std, 2*std]``。ViT 对初始化
    比较敏感：标准差太大时注意力 logits 容易饱和，太小时各 token 几乎相同；
    DeiT 系列常用这种小方差、去极值的初始化。注意这是简化版“截断”，不是严格
    的条件正态采样，但对教学和这个小模型足够。

    参数
    ----
    tensor : torch.Tensor
        要初始化的张量，函数会原地修改它（返回值为 ``None``）。
    std : float
        正态分布的标准差，默认 0.02。

    返回
    ----
    None
        原地修改，不返回新张量。
    """
    with torch.no_grad():  # 初始化不需要梯度，关闭 autograd 可避免无谓的计算图记录
        tensor.normal_(0.0, std)  # 原地采样 N(0, std^2)
        # 截断到两倍标准差，避免极端值破坏训练初期的稳定性（DeiT 的常见做法）。
        tensor.clamp_(-2 * std, 2 * std)  # 原地截断，形状不变


class VisionTransformer(nn.Module):
    """面向 32x32 CIFAR 图像的紧凑 Vision Transformer。

    前向流程：图像 ``[B,3,H,W]`` → 卷积切 patch → ``[B,C,H/p,W/p]`` →
    展平成 token ``[B,N,C]`` → 拼 cls token 并加位置嵌入 → 多个 Block →
    最终 LayerNorm → 取 cls token → 分类头 ``[B,num_classes]``。

    设计取舍：
    - 用 stride=patch_size 的卷积做 patch embedding，避免显式 unfold；
    - 使用可学习的 cls token 和绝对位置嵌入，结构最接近原始 ViT；
    - 分类头之前做最终 LayerNorm，因为 Block 是 Pre-LN；
    - 输出 logits 而不是概率，交给训练脚本的交叉熵损失处理。
    """

    def __init__(self, cfg: ViTConfig):
        """根据配置搭建完整 ViT。

        参数
        ----
        cfg : ViTConfig
            超参数容器；其中 ``embed_dim`` 必须能被 ``num_heads`` 整除，
            ``img_size`` 必须能被 ``patch_size`` 整除。
        """
        super().__init__()  # 注册子模块和参数；没有这一行，后面的层不会出现在 model.parameters() 中
        self.cfg = cfg  # 保存配置，便于初始化函数读取 init_scale，也方便调试
        self.patch_embed = nn.Conv2d(
            cfg.in_chans, cfg.embed_dim, kernel_size=cfg.patch_size, stride=cfg.patch_size
        )  # [B,3,32,32] -> [B,C,8,8]；kernel 与 stride 相等 => 不重叠的 patch 线性投影
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))  # 先用 zeros 占位，再 trunc_normal_ 覆盖；nn.Parameter 让这个张量成为可训练参数
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg.num_patches + 1, cfg.embed_dim))  # [1,1+N,C]，N=64；同样先占位后初始化，训练中会更新
        self.pos_drop = nn.Dropout(cfg.dropout)  # 位置嵌入后的 dropout，默认 0
        dpr = [x.item() for x in torch.linspace(0, cfg.drop_path, cfg.depth)]  # 每层一个 Python float，0 -> drop_path 线性增加
        self.blocks = nn.ModuleList(
            [
                Block(
                    cfg.embed_dim,
                    cfg.num_heads,
                    cfg.mlp_ratio,
                    dropout=cfg.dropout,
                    attn_dropout=cfg.attn_dropout,
                    drop_path=dpr[i],
                )
                for i in range(cfg.depth)
            ]
        )  # 必须用 nn.ModuleList：普通 list 不会注册子模块，.to(device)/parameters() 会漏掉这些层
        self.norm = nn.LayerNorm(cfg.embed_dim)  # Pre-LN 结构的最终归一化
        self.head = nn.Linear(cfg.embed_dim, cfg.num_classes)  # [B,C] -> [B,10]
        self.apply(self._init_weights)  # 递归地对所有子模块执行初始化
        trunc_normal_(self.cls_token, std=cfg.init_scale)  # nn.Parameter 不是 Module，apply 不会处理它，需单独初始化
        trunc_normal_(self.pos_embed, std=cfg.init_scale)  # 同上；位置嵌入也使用小方差截断正态

    def _init_weights(self, m: nn.Module) -> None:
        """``nn.Module.apply`` 的回调：按层类型初始化权重。

        参数
        ----
        m : nn.Module
            当前被遍历到的模块；``apply`` 会递归访问所有子模块。

        返回
        ----
        None
            原地修改 ``m`` 的参数，不返回任何值。

        关键设计：线性层和卷积层用截断正态、偏置置零；LayerNorm 的缩放置 1、
        偏置置 0（这也是 LayerNorm 的默认值，这里显式写出便于教学）。
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.cfg.init_scale)  # 权重小方差初始化
            if m.bias is not None:
                nn.init.zeros_(m.bias)  # 偏置从 0 开始
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)  # 缩放 gamma = 1
            nn.init.zeros_(m.bias)  # 偏移 beta = 0
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.cfg.init_scale)  # patch embedding 的卷积权重
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """把图像编码成经过最终 LayerNorm 的 token 序列。

        参数
        ----
        x : torch.Tensor
            图像 batch，形状 ``[B, in_chans, H, W]``，默认 ``[B,3,32,32]``。

        返回
        ----
        torch.Tensor
            形状 ``[B, 1+N, C]``；第 0 个 token 是 cls token，后 N 个是 patch token。
        """
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # [B,C,8,8] -> [B,C,64] -> [B,64,C]
        cls = self.cls_token.expand(x.shape[0], -1, -1)  # [1,1,C] -> [B,1,C]；expand 只创建视图，不复制数据
        x = torch.cat((cls, x), dim=1) + self.pos_embed  # [B,1+N,C] + [1,1+N,C]，batch 维自动广播
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x)  # 每层保持 [B,1+N,C]，Prune 版本会在这里改变 N
        return self.norm(x)  # Pre-LN 的出口归一化，形状不变

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """完整前向：图像 batch → 10 类 logits。

        参数
        ----
        x : torch.Tensor
            形状 ``[B,3,32,32]``。

        返回
        ----
        torch.Tensor
            形状 ``[B,num_classes]`` 的 logits；未做 softmax，供交叉熵损失使用。
        """
        return self.head(self.forward_features(x)[:, 0])  # 取 cls token [B,C]，再经分类头得到 [B,10]


def vit_tiny(**kwargs) -> VisionTransformer:
    """便捷构造函数：把关键字参数直接传给 ``ViTConfig``。

    ``**kwargs`` 会把调用处的所有关键字参数打包成一个 dict，例如
    ``vit_tiny(embed_dim=64, depth=3)`` 等价于
    ``VisionTransformer(ViTConfig(embed_dim=64, depth=3))``。名字里的 tiny 只是
    习惯叫法，实际大小仍由传入的参数决定。

    参数
    ----
    **kwargs
        任意 ``ViTConfig`` 字段，如 ``img_size``、``patch_size``、``embed_dim``、
        ``depth``、``num_heads`` 等。

    返回
    ----
    VisionTransformer
        按给定配置构建、尚未训练的模型。
    """
    return VisionTransformer(ViTConfig(**kwargs))  # ** 在这里把 dict 重新展开成关键字参数
