"""冒烟测试：用几秒钟检查模型、增广、损失函数是否“还能正常工作”。

===============================================================================
一、这个脚本做什么
===============================================================================
这是本项目唯一的测试文件，包含 5 个“冒烟测试（smoke test）”。它们不追求覆盖
所有分支，只回答一个很朴素的问题：改完代码后，最关键的东西有没有被改坏？

* ``test_vit_shape_and_params``       —— ViT 前向输出形状 (B, 10)、参数量 311,338
* ``test_dynamic_token_schedule``    —— DynamicViT 剪枝后 token 数 [65, 49, 25, 7]
* ``test_mobile_shape``              —— MobileViT-Lite 前向输出形状 (B, 10)
* ``test_augmentation_shape_and_range`` —— 增广后形状为 NCHW、数值全部有限
* ``test_cb_weights``                —— CB 权重“稀有类更大、归一化后均值为 1”

二、为什么这些测试不需要数据集 / GPU（本文件的核心设计）
-------------------------------------------------------------------------------
* 输入全是程序现场合成的随机张量（``torch.randn`` / ``torch.rand``），
  不读 ``data/*.npy``，所以不需要先跑 prepare_data.py，也不依赖课程附件；
* 模型用的是默认的 tiny 配置（ViT 只有约 0.31M 参数），一次前向在 CPU 上
  是毫秒级，因此不需要 GPU，也不会因为显存不足而失败；
* 不训练、不反传（不必要的部分用 ``torch.no_grad()`` 包起来），
  所以没有优化器、学习率、epoch 这些概念，测试结果是确定的；
* 唯一的“数值”断言的容差是固定的（``places=5``），不会因机器性能变化而波动。

换句话说，这是一组“结构/契约测试”：检查形状、数量、取值范围、统计性质，
而不是检查模型准不准——准确率要靠训练实验去验证。

三、输入 / 输出
-------------------------------------------------------------------------------
输入：无外部文件。测试内部构造的随机张量就是全部输入。
输出：终端打印 ``OK`` / ``FAILED`` 与统计信息；失败时抛出 AssertionError 并
以非零退出码结束。本文件不生成任何图片或表格。

四、在实验流程中的位置
-------------------------------------------------------------------------------
位于最前面：改完 ``src/`` 下的模型/损失/数据代码后，先跑它（几秒），确认没有
低级错误，再去跑十几分钟的 ``scripts/train.py`` 训练。它也可以放在 CI 里，
作为每次提交的快速门禁。

五、怎么运行
-------------------------------------------------------------------------------
在项目根目录 ``vit_cifar10_lab`` 下执行（``-t tests`` 把 tests/ 当作顶层目录，
避免本机 site-packages 里同名的 ``tests`` 包遮蔽它；本目录没有 ``__init__.py``）::

    python -m unittest discover -s tests -t tests -v

如果只想单独跑本文件，需要先让项目根进入模块搜索路径，否则 ``from src...``
会报 ModuleNotFoundError（因为直接执行脚本时 sys.path[0] 是 tests/ 而不是项目根）::

    # PowerShell
    $env:PYTHONPATH="."; python tests/test_smoke.py -v
"""

from __future__ import annotations

import unittest

import torch

# 只导入“被测对象”，不导入训练脚本或数据加载器：这提醒读者，本文件的依赖面
# 很小，所以跑得快、也不需要数据目录存在。
from src.data import augment_batch
from src.losses import effective_number_weights
from src.models import (
    DynamicViT,
    MobileViTConfig,
    MobileViTLite,
    ViTConfig,
    VisionTransformer,
)


class ModelSmokeTest(unittest.TestCase):
    """模型/数据/损失的冒烟测试集合。

    继承 ``unittest.TestCase`` 后，所有以 ``test_`` 开头的方法都会被测试运行器
    自动发现并执行；每个方法都用 ``self.assert*`` 系列函数做断言，失败即抛
    AssertionError。unittest 会在每个测试方法前后自动调用 setUp / tearDown，
    因此测试之间互不共享状态。
    """

    def setUp(self):
        """每个测试方法运行【之前】自动调用一次，用来准备公共输入。

        这里做两件事：固定随机种子，以及造一个假 batch。因为每个测试方法都会
        重新执行 setUp，所以某个测试改了全局随机状态也不会影响下一个测试。

        参数:
            无。由 unittest 框架调用，不接受额外参数。

        返回:
            None。副作用是给 ``self`` 挂上 ``self.x`` 这个属性。
        """
        # 固定 PyTorch 的随机种子，让 dropout、随机裁剪等操作在每次运行时一致，
        # 测试结果才可复现（否则偶发失败很难排查）。
        torch.manual_seed(0)
        # 一个假的 CIFAR batch：形状 (N, C, H, W) = (2, 3, 32, 32)。
        # 数值是标准正态噪声，模型只关心形状和计算是否正确，不关心像不像真图。
        # 用 batch=2 是为了让 batch 维度的 bug（比如误把 B 和通道混了）更容易暴露。
        self.x = torch.randn(2, 3, 32, 32)

    def test_vit_shape_and_params(self):
        """检查 ViT 前向输出形状，以及默认配置下的参数量是否仍是 311,338。

        这是最基础的“契约测试”：输入 (2, 3, 32, 32)，输出必须是 (2, 10)，
        即每个样本 10 个类别的 logits；参数量写死，是因为报告里的 0.311M 就来自
        这个默认配置，一旦有人改了默认维度/深度，这里会立刻失败，提醒同步更新
        报告数字（这是有意的“回归护栏”，不是脆弱的测试）。

        参数:
            无（unittest 自动调用）。

        返回:
            None；断言失败时抛 AssertionError。
        """
        # 不传任何参数 = 使用 ViTConfig 的全部默认值（dim=96、depth=4、patch=4 ...）。
        model = VisionTransformer(ViTConfig())
        # assertEqual 对 tuple 是逐元素的值比较，等价于同时检查 B=2、类别数=10。
        # 注意这里让模型处于默认的 train 模式，drop_path 有随机性，但不影响形状。
        self.assertEqual(model(self.x).shape, (2, 10))
        # p.numel() 返回单个参数张量的元素个数；生成器求和得到总参数量。
        # 311,338 与 README / 报告里的 0.311M 对应，改动模型结构时此数会变。
        self.assertEqual(sum(p.numel() for p in model.parameters()), 311338)

    def test_dynamic_token_schedule(self):
        """检查 DynamicViT 在每个剪枝阶段之后保留的 token 数。

        默认 64 个 patch token + 1 个 CLS = 65；按 75%/50%/25% 逐级保留后依次是
        49（48+1）、25（24+1）、7（6+1）。这个序列是“FLOPs 真的下降”的直接证据，
        所以用测试锁住，防止改剪枝逻辑时悄悄退化。

        参数:
            无（unittest 自动调用）。

        返回:
            None；断言失败时抛 AssertionError。
        """
        # prune_after=(0, 1, 2) 表示在第 0、1、2 个 Block 之后剪枝（索引从 0 开始），
        # 与训练脚本使用的 --prune-after 0,1,2 一致。
        model = DynamicViT(ViTConfig(), prune_after=(0, 1, 2), keep_ratios=(0.75, 0.5, 0.25))
        # eval() 关闭 dropout 和 Gumbel 噪声，让前向行为贴近推理；同时避免测试
        # 受随机状态影响（token 数量本身由保留率决定，但评估模式更符合真实用法）。
        model.eval()
        # no_grad：测试不反传，关掉计算图可以省内存、跑得更快。
        with torch.no_grad():
            model(self.x)
        # token_counts 由 forward_features 在每个阶段后记录，是“输入 + 三次剪枝”
        # 共 4 个长度值；最后一个 7 说明最终只保留 1 个 CLS + 6 个 patch token。
        self.assertEqual(model.token_counts, [65, 49, 25, 7])

    def test_mobile_shape(self):
        """检查 MobileViT-Lite（dim=72）的前向输出形状。

        轻量模型最容易在卷积 stem、Unfold/Fold 或池化处把空间尺寸/通道数算错，
        形状断言能第一时间发现这类问题；这里不关心精度，所以只检查输出 (2, 10)。

        参数:
            无（unittest 自动调用）。

        返回:
            None；断言失败时抛 AssertionError。
        """
        # dim=72 是训练脚本里使用的宽度；MobileViTConfig 的其它参数用默认值
        # （2 个 MobileViT Block、1 个全局 Block、patch=4）。
        model = MobileViTLite(MobileViTConfig(dim=72))
        self.assertEqual(model(self.x).shape, (2, 10))

    def test_augmentation_shape_and_range(self):
        """检查 batch 增广后的形状（NHWC uint8 -> NCHW float）与数值有限性。

        ``augment_batch`` 接收 uint8 的 NHWC 图片，内部依次做：转置成 NCHW、
        除以 255、随机裁剪、随机水平翻转、（可选）cutout、按 CIFAR 均值方差
        归一化。这里传入 ``erase_p=0.0`` 关掉随机擦除，让测试专注于“形状正确、
        没有 NaN/Inf”这两件事，减少随机分支带来的干扰。

        参数:
            无（unittest 自动调用）。

        返回:
            None；断言失败时抛 AssertionError。
        """
        # torch.rand 产生 [0,1) 的浮点数，乘 255 再转 uint8，模拟真实的像素范围；
        # 形状 (4, 32, 32, 3) 是 NHWC（样本、高、宽、通道），与 npy 缓存一致。
        images = (torch.rand(4, 32, 32, 3) * 255).to(torch.uint8)
        out = augment_batch(images, erase_p=0.0)
        # 形状从 NHWC 变成 NCHW：这是 PyTorch 卷积层要求的布局。
        self.assertEqual(out.shape, (4, 3, 32, 32))
        # isfinite 逐元素判断是否为有限值（排除 NaN 和 ±Inf），.all() 要求全为 True。
        # 归一化里若除零或出现非法运算，这里会立刻抓到。
        self.assertTrue(torch.isfinite(out).all())

    def test_cb_weights(self):
        """检查 Class-Balanced 权重的两个关键性质：稀有类权重更大、均值为 1。

        有效样本数公式 E_n = (1 - beta^n) / (1 - beta)，权重正比于 1/E_n：
        n 越小（类越稀有），E_n 越小，权重越大。``normalize=True`` 时再除以均值，
        使权重均值为 1，从而只改变类间相对大小，不改变 loss 的整体量级。

        参数:
            无（unittest 自动调用）。

        返回:
            None；断言失败时抛 AssertionError。
        """
        # 只有这个测试用到 numpy，局部导入可以避免在模块顶层引入额外依赖，
        # 也提醒读者：被测函数接受的是 numpy 数组而不是 torch 张量。
        import numpy as np

        # 模拟不均衡训练集的类别计数：0 类 4513 张（最多），1 类 2023 张（最少）。
        counts = np.array([4513, 2023, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000])
        w = effective_number_weights(counts)
        # assertGreater(a, b) 断言 a > b：1 类比 0 类稀有，权重必须更大。
        self.assertGreater(w[1], w[0])
        # assertAlmostEqual 用 7 位有效数字的舍入比较浮点；places=5 表示允许
        # 1e-5 级别的误差，避免浮点求和顺序带来的尾数差异导致误报。
        self.assertAlmostEqual(float(w.mean()), 1.0, places=5)


if __name__ == "__main__":
    # 直接执行本文件时，unittest.main() 会自动发现当前模块里所有 test_* 方法；
    # 被 unittest discover 导入时此块不执行，避免重复跑测试。
    unittest.main()
