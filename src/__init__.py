"""ViT CIFAR-10 实验的核心包。

包含四个子模块：
- ``data``：数据读取、npy 缓存、批量数据增广；
- ``models``：ViT 基线、DynamicViT、MobileViT-Lite；
- ``losses``：交叉熵、Class-Balanced Loss、蒸馏损失；
- ``engine``：训练循环、评估指标、参数量统计。

脚本位于项目根目录的 ``scripts/``，它们负责把这里的能力串成一次完整实验。
"""
