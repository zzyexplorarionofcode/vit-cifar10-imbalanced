"""MobileViT-Lite：面向 CIFAR-10 的轻量卷积—Transformer 混合模型。

这个文件解决什么问题
--------------------
普通 ViT 让所有 patch token 两两做全局注意力，计算量随 token 数平方增长；在
32x32 图像上还勉强可用，但在移动端/高分辨率场景代价很高。MobileViT 的思路是：
先用卷积提取局部特征并降采样，再把特征图切成不重叠的 patch，**只在每个 patch
内部**做自注意力，最后把结果折叠回特征图。这样既保留了注意力的内容自适应
混合能力，又把注意力矩阵限制在 patch 内，参数量和计算量都显著降低。

在“数据→模型→训练→评估”流程中的位置
------------------------------------
1. ``src/data.py`` 产生 ``[B,3,32,32]`` 图像 batch；
2. **本文件**用卷积 stem 把空间尺寸降到 8x8，经过若干 MobileViT Block，
   再可选地接一个全局 Transformer Block，最后全局平均池化并输出 ``[B,10]``；
3. ``src/engine.py`` 负责训练/评估；``src/models/factory.py`` 在
   ``--model mobile`` 时构建本文件模型；
4. ``scripts/measure_complexity.py`` 会统计它的参数量、FLOPs、延迟和吞吐。

阅读前需要知道的概念
--------------------
- **depthwise convolution（深度可分离卷积）**：``groups = in_channels`` 时，
  每个输入通道单独用一个卷积核，通道之间不混合；后面再接 1x1 卷积做通道混合。
  它比普通卷积便宜很多，是 MobileNet 系列的核心。
- **特征图形状 ``[B, C, H, W]``**：B 是 batch，C 是通道，H/W 是空间高宽。
  PyTorch 卷积默认使用这种 channel-first 布局。
- **F.unfold / F.fold**：``unfold`` 把每个滑动窗口（这里是互不重叠的 p×p
  patch）展平成一列，得到 ``[B, C*p*p, L]``；``fold`` 是它的逆操作，把列按
  原位置拼回 ``[B, C, H, W]``。两者都要求 kernel/stride 等参数匹配。
- **patch 内注意力**：每个 patch 有 ``p*p`` 个位置，把每个位置的特征当作一个
  token，只在这 ``p*p`` 个 token 之间做注意力。不同 patch 之间在这一层不直接
  交互，长距离信息主要靠前面的卷积和后面的全局 Block 传递。
- **残差与 DropPath**：Block 输出加回输入，保证信息直通；训练时整条分支可能
  被随机丢弃，推理时关闭。

主要类/函数
-----------
- ``ConvBNSiLU``：卷积 + BatchNorm + SiLU 的基础卷积块。
- ``MobileViTBlock``：局部卷积 + patch 内注意力 + 折叠 + 残差。
- ``MobileViTConfig``：模型超参数 dataclass。
- ``MobileViTLite``：完整分类模型。
- ``mobile_vit_lite``：用关键字参数快速构造模型的便捷函数。
"""

from __future__ import annotations  # 延迟类型注解求值，便于使用较新的类型写法且不影响运行时

from dataclasses import dataclass  # 自动生成配置类的 __init__/__repr__/__eq__

import torch
import torch.nn as nn
import torch.nn.functional as F  # 这里主要用 F.unfold / F.fold 做 patch 的展开与折叠

from .vit import Block, DropPath, trunc_normal_  # 复用 ViT 的 Pre-LN Block、随机深度和初始化工具


class ConvBNSiLU(nn.Module):
    """卷积 + BatchNorm + SiLU 的标准组合。

    顺序是 ``Conv2d -> BatchNorm2d -> SiLU``。卷积默认不设 bias，因为紧接的
    BatchNorm 会学习偏移（beta），再加 bias 是冗余的。``groups`` 参数控制
    分组卷积：当 ``groups == in_ch == out_ch`` 时就是 depthwise 卷积。
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1, groups: int = 1):
        """初始化卷积块。

        参数
        ----
        in_ch : int
            输入通道数。
        out_ch : int
            输出通道数。
        kernel : int
            卷积核边长；使用 ``kernel // 2`` 作为 padding，奇数核且 stride=1 时
            可以保持 H/W 不变。
        stride : int
            步长；2 表示空间尺寸减半，1 表示不变。
        groups : int
            分组数。``groups=in_ch=out_ch`` 表示 depthwise 卷积；默认 1 是普通卷积。
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel, stride, kernel // 2, groups=groups, bias=False
        )  # [B,in_ch,H,W] -> [B,out_ch,H_out,W_out]；bias 交给 BN
        self.bn = nn.BatchNorm2d(out_ch)  # 在通道维上做归一化，训练时维护 running stats
        self.act = nn.SiLU(inplace=True)  # SiLU(x)=x*sigmoid(x)，平滑且常用；inplace 省一点内存

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """卷积 → 归一化 → 激活。

        参数
        ----
        x : torch.Tensor
            形状 ``[B, in_ch, H, W]``。

        返回
        ----
        torch.Tensor
            形状 ``[B, out_ch, H_out, W_out]``；stride=2 时 H_out≈H/2。
        """
        return self.act(self.bn(self.conv(x)))  # 形状：卷积改变通道/空间，BN 和 SiLU 不改变形状


class MobileViTBlock(nn.Module):
    """MobileViT 的核心块：局部卷积 + patch 内自注意力 + 残差。

    前向分四步：
    1. ``self.local`` 用 depthwise 3x3 + 1x1 卷积得到局部增强后的特征图；
    2. ``F.unfold`` 把特征图切成互不重叠的 p×p patch，得到 ``[B, C*p*p, L]``；
    3. 重排成 ``[B*L, p*p, C]``，让每个 patch 成为一个独立的小序列，交给
       ``Block`` 做 patch 内注意力；
    4. ``F.fold`` 把每个 patch 的 ``p*p`` 个 token 拼回特征图，1x1 投影后加残差。

    这样注意力的复杂度从全局的 ``O((HW)^2)`` 降到约 ``O(L * p^4)``，其中
    ``L = HW / p^2``，在 p 较小时便宜得多。注意这里没有 cls token，也没有
    额外的位置嵌入；空间信息主要由卷积的局部感受野和 patch 的切分方式提供，
    注意力本身并不显式区分 patch 内 token 的二维位置。
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        patch_size: int = 4,
        mlp_ratio: float = 2.0,
        drop_path: float = 0.0,
    ):
        """初始化 MobileViT Block。

        参数
        ----
        dim : int
            特征通道数 C，也是 token 维度。
        num_heads : int
            patch 内注意力的 head 数，必须整除 ``dim``。
        patch_size : int
            每个 patch 的边长 p；p=4 时每个 patch 内有 16 个 token。
        mlp_ratio : float
            ``Block`` 内 MLP 隐藏宽度相对 ``dim`` 的倍数。
        drop_path : float
            随机深度概率；既传给内部 ``Block`` 的两条子分支，也用于外层融合
            分支的 ``DropPath``。
        """
        super().__init__()
        self.patch_size = patch_size
        self.local = nn.Sequential(
            ConvBNSiLU(dim, dim, 3, groups=dim),  # depthwise 3x3：每个通道独立卷积，扩大感受野
            nn.Conv2d(dim, dim, 1, bias=False),  # pointwise 1x1：混合通道信息
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
        )  # 输出仍是 [B,C,H,W]；这一步不改变空间尺寸
        self.transformer = Block(
            dim, num_heads, mlp_ratio, dropout=0.0, attn_dropout=0.0, drop_path=drop_path
        )  # 这里输入的是 [B*L, p*p, C]，注意力只在每个 patch 的 p*p 个 token 内进行
        self.project = nn.Conv2d(dim, dim, 1, bias=False)  # 1x1 投影，把注意力输出映射回特征图通道
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()  # 外层残差分支的随机深度

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行一次 MobileViT Block。

        参数
        ----
        x : torch.Tensor
            形状 ``[B, C, H, W]`` 的特征图；H、W 必须能被 ``patch_size`` 整除，
            否则无法无重叠地切成 patch。

        返回
        ----
        torch.Tensor
            形状 ``[B, C, H, W]``，与输入可直接残差相加。
        """
        b, c, h, w = x.shape  # 保存原始空间尺寸，最后 fold 和残差都要用
        p = self.patch_size
        if h % p or w % p:  # 例如 7x7 无法切成 4x4 的整 patch，提前报清晰错误
            raise ValueError(f"feature map {h}x{w} is not divisible by patch_size={p}")
        y = self.local(x)  # [B,C,H,W] -> [B,C,H,W]；depthwise + pointwise 做局部特征增强
        unfolded = F.unfold(y, kernel_size=p, stride=p)  # [B,C,H,W] -> [B, C*p*p, L]，L=(H/p)*(W/p)
        num_patches = unfolded.shape[-1]  # L
        tokens = unfolded.view(b, c, p * p, num_patches).permute(0, 3, 2, 1)  # [B,C,p*p,L] -> [B,L,p*p,C]
        tokens = tokens.reshape(b * num_patches, p * p, c)  # 把 batch 和 patch 合并：每个 patch 是一条长度为 p*p 的序列
        tokens = self.transformer(tokens)  # 注意力只在每个 patch 内部进行，形状 [B*L,p*p,C] 不变
        tokens = tokens.view(b, num_patches, p * p, c).permute(0, 3, 2, 1)  # [B,L,p*p,C] -> [B,C,p*p,L]
        folded = F.fold(
            tokens.reshape(b, c * p * p, num_patches),
            output_size=(h, w),
            kernel_size=p,
            stride=p,
        )  # [B,C*p*p,L] -> [B,C,H,W]；stride=kernel 时 patch 不重叠，不会相加
        return x + self.drop_path(self.project(folded))  # 残差：投影后的注意力特征加回输入，形状 [B,C,H,W]


@dataclass  # 自动生成 __init__：下面每个字段都是可传参数
class MobileViTConfig:
    """MobileViT-Lite 的超参数容器（纯配置，不是 ``nn.Module``）。

    参数
    ----
    img_size : int
        输入图像边长，默认 32；当前实现按正方形处理。
    in_chans : int
        输入通道数，RGB 为 3。
    num_classes : int
        分类类别数，CIFAR-10 为 10。
    dim : int
        模型主特征通道数 C，也是 token 维度。
    num_heads : int
        patch 内注意力的 head 数，必须整除 ``dim``。
    mobile_depth : int
        MobileViT Block 的个数。
    global_depth : int
        最后的全局 Transformer Block 个数；0 表示不用全局注意力。
    patch_size : int
        MobileViT Block 内每个 patch 的边长 p。
    mlp_ratio : float
        Block 内 MLP 隐藏宽度相对 ``dim`` 的倍数。
    drop_path : float
        随机深度概率；全局 Block 会按层线性递增。
    stem_dim : int
        卷积 stem 的中间通道数，最后再 1x1 投到 ``dim``。
    """

    img_size: int = 32
    in_chans: int = 3
    num_classes: int = 10
    dim: int = 64  # 主特征宽度；比基线 ViT 的 96 更窄，以降低参数量
    num_heads: int = 4  # 每个 head 维度 = 64/4 = 16
    mobile_depth: int = 2  # 两个 patch 内注意力块
    global_depth: int = 1  # 一个全局 Block，用来补足跨 patch 的信息混合
    patch_size: int = 4  # 8x8 特征图切成 2x2=4 个 patch，每个 patch 4x4=16 个 token
    mlp_ratio: float = 2.0
    drop_path: float = 0.05
    stem_dim: int = 32  # stem 先到 32 通道，再投到 dim，控制第一层卷积成本


class MobileViTLite(nn.Module):
    """轻量混合分类模型：卷积 stem + MobileViT Block + 可选全局 Block。

    整体流程：
    ``[B,3,32,32]`` → 两次 stride-2 卷积得到 ``[B,d,8,8]`` → 若干个
    ``MobileViTBlock``（空间尺寸不变）→ 展平成 ``[B,64,d]`` → 可选位置嵌入和
    全局 ``Block`` → LayerNorm → 对 token 做平均池化 → 分类头 ``[B,10]``。

    与基线 ViT 的差异：没有 cls token，使用全局平均池化；大部分注意力限制在
    patch 内；stem 用卷积而不是一次大 stride 的 patch embedding。这样参数和
    FLOPs 都更低，同时保留一个全局 Block 做跨 patch 融合。
    """

    def __init__(self, cfg: MobileViTConfig):
        """根据配置搭建 MobileViT-Lite。

        参数
        ----
        cfg : MobileViTConfig
            超参数容器。要求 ``img_size`` 能被 4 整除（两次 stride-2），并且
            下采样后的特征边长能被 ``patch_size`` 整除，否则 MobileViT Block
            会抛出维度错误。
        """
        super().__init__()
        self.cfg = cfg
        d = cfg.dim
        self.stem = nn.Sequential(
            ConvBNSiLU(cfg.in_chans, cfg.stem_dim, 3, stride=2),  # [B,3,32,32] -> [B,32,16,16]
            ConvBNSiLU(cfg.stem_dim, cfg.stem_dim, 3, stride=2, groups=cfg.stem_dim),  # depthwise 下采样 -> [B,32,8,8]
            ConvBNSiLU(cfg.stem_dim, d, 1),  # 1x1 投到主通道 d，空间仍为 8x8
        )
        feat = cfg.img_size // 4  # 两次 stride=2 后的特征边长；32 -> 8
        self.feat_size = feat
        self.mobile_blocks = nn.ModuleList(
            [
                MobileViTBlock(
                    d, cfg.num_heads, cfg.patch_size, cfg.mlp_ratio, cfg.drop_path
                )
                for _ in range(cfg.mobile_depth)
            ]
        )  # ModuleList 会注册所有 Block；普通 list 不会
        dpr = [x.item() for x in torch.linspace(0, cfg.drop_path, max(cfg.global_depth, 1))]  # 每层 drop_path，从 0 线性增加
        self.global_blocks = nn.ModuleList(
            [
                Block(d, cfg.num_heads, cfg.mlp_ratio, drop_path=dpr[i])
                for i in range(cfg.global_depth)
            ]
        )  # global_depth=0 时是空 ModuleList，前向会自然跳过
        self.pos_embed = (
            nn.Parameter(torch.zeros(1, feat * feat, d)) if cfg.global_depth > 0 else None
        )  # 只有全局 Block 需要位置嵌入；[1,L,d]，L=8*8=64
        self.norm = nn.LayerNorm(d)  # Pre-LN Block 的出口归一化
        self.head = nn.Linear(d, cfg.num_classes)  # [B,d] -> [B,10]
        self.apply(self._init_weights)  # 递归初始化所有子模块
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)  # nn.Parameter 不是 Module，apply 不会初始化它，需单独处理

    def _init_weights(self, m: nn.Module) -> None:
        """``nn.Module.apply`` 的回调：按层类型初始化参数。

        参数
        ----
        m : nn.Module
            当前遍历到的模块。

        返回
        ----
        None
            原地修改参数，不返回值。

        线性层/卷积层用截断正态，偏置置零；LayerNorm 和 BatchNorm 的缩放
        置 1、偏移置 0。BatchNorm 的 ``running_mean/running_var`` 是 buffer，
        不在这里手动改，保持默认初始化。
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取经过最终归一化的 token 序列。

        参数
        ----
        x : torch.Tensor
            图像 batch，形状 ``[B, in_chans, H, W]``，默认 ``[B,3,32,32]``。

        返回
        ----
        torch.Tensor
            形状 ``[B, L, d]``，其中 ``L=(H/4)*(W/4)``；32x32 输入对应 L=64。
        """
        x = self.stem(x)  # [B,3,32,32] -> [B,d,8,8]
        for block in self.mobile_blocks:
            x = block(x)  # 每个 MobileViT Block 都保持 [B,d,8,8]
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # [B,d,8,8] -> [B,d,64] -> [B,64,d]
        if self.pos_embed is not None:
            tokens = tokens + self.pos_embed  # [B,L,d] + [1,L,d]，batch 维广播
        for block in self.global_blocks:
            tokens = block(tokens)  # 全局注意力允许不同 patch 的 token 交互
        return self.norm(tokens)  # 形状 [B,L,d]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """完整前向：图像 batch → 分类 logits。

        参数
        ----
        x : torch.Tensor
            形状 ``[B,3,32,32]``。

        返回
        ----
        torch.Tensor
            形状 ``[B,num_classes]`` 的 logits；没有 cls token，使用 token 平均池化。
        """
        tokens = self.forward_features(x)  # [B,L,d]
        return self.head(tokens.mean(dim=1))  # [B,L,d] -> 对 L 求平均 -> [B,d] -> [B,10]


def mobile_vit_lite(**kwargs) -> MobileViTLite:
    """便捷构造函数：把关键字参数直接传给 ``MobileViTConfig``。

    ``**kwargs`` 会把调用处的关键字参数收集成 dict，再展开给 dataclass。
    例如 ``mobile_vit_lite(dim=72, mobile_depth=2)`` 等价于
    ``MobileViTLite(MobileViTConfig(dim=72, mobile_depth=2))``。

    参数
    ----
    **kwargs
        任意 ``MobileViTConfig`` 字段，如 ``img_size``、``dim``、``num_heads``、
        ``mobile_depth``、``global_depth``、``patch_size`` 等。

    返回
    ----
    MobileViTLite
        按给定配置构建、尚未训练的模型。
    """
    return MobileViTLite(MobileViTConfig(**kwargs))  # ** 把 dict 展开成关键字参数
