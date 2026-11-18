"""面向长尾 / 类别不平衡分类的损失函数 —— 教学注释版。

=====================================================================
一、这个模块解决什么问题
=====================================================================
本课程的数据集是「长尾」的：训练集里有些类有几千张图，有些类只有几百张。
如果直接用普通交叉熵，模型会倾向于「多预测大类」——因为把样本判成大类，
平均来看更容易蒙对，loss 下降得更快。结果就是：整体 accuracy 看着不错，
但小类的召回率惨不忍睹，而这往往才是我们真正关心的。

本模块提供两个武器：
1. **类别平衡损失（Class-Balanced Loss）**：给稀有类更高的损失权重，
   让模型不敢忽视它们。
2. **知识蒸馏损失（Knowledge Distillation）**：让一个「学生」模型的输出分布
   去逼近「教师」模型的输出分布。教师在大类上学到的「类间相似性」这种暗知识
   （比如猫和狗的分布比猫和飞机更接近），会以软标签的形式传给学生，
   通常能提升小类的泛化。

=====================================================================
二、在「数据 → 模型 → 训练 → 评估」流程中的位置
=====================================================================
   data.py                 models/                losses.py（本文件）        engine.py
 ┌───────────┐        ┌──────────────┐        ┌────────────────────┐      ┌──────────────┐
 │ 统计每类   │        │ 输出 logits  │        │ CE / CB-CE         │      │ train_one_   │
 │ 样本数     │───────▶│  (B, C)      │───────▶│ distillation_loss  │─────▶│ epoch 里     │
 │ counts[]  │        │              │        │ 组合成总损失        │      │ loss.backward│
 └───────────┘        └──────────────┘        └────────────────────┘      └──────────────┘

具体来说：
- `scripts/train.py` 先用 `np.bincount` 数出每类样本数 `counts`；
- 再用 `effective_number_weights(counts)` 把它变成每类的损失权重；
- `build_loss("cb", weights, smoothing)` 组装出 `ClassBalancedCrossEntropy`；
- 训练循环里 `loss = criterion(logits, labels)`；若开了蒸馏，再叠加
  `distill_alpha * distillation_loss(student, teacher, T)`；
- 最后 `loss.backward()` 把梯度传回模型。

=====================================================================
三、阅读前需要知道的概念
=====================================================================
1. logits 与 softmax：模型最后一层输出的是**未归一化的打分** logits (B,C)。
   softmax 把它变成概率分布；交叉熵内部会自动做 log_softmax，
   所以传给 `F.cross_entropy` 的必须是 logits，不能自己先 softmax。
2. 损失函数是「按样本平均」的标量：反向传播需要一个能求导的标量，
   所以所有损失最后都归约成一个数（reduction="mean"/"batchmean"）。
3. 类别权重（class weight）：给每个类的损失乘一个系数。权重越大，
   模型越「怕」在这个类上犯错，于是被迫关注它。
4. 有效样本数（effective number）：随着样本数增加，新增样本带来的信息增益
   是**递减**的（大量重复样本几乎不提供新信息）。所以「5000 张图的类」
   并不该只比「500 张图的类」重 10 倍，而应重得少一些。
5. label smoothing（标签平滑）：不再要求模型输出 one-hot（100% 把握），
   而是 (1-ε) 给正确类、ε/C 均分给其他类。这能抑制过度自信、改善校准。
6. 知识蒸馏的温度 T：把 logits 除以 T 再 softmax，T>1 会让分布更平滑，
   类间的相对大小（暗知识）更容易被学生学到。梯度会随 T 变小，所以要乘 T² 补偿。

=====================================================================
四、本文件的主要函数 / 类
=====================================================================
- `effective_number_weights(counts, beta, normalize)`：由每类样本数算出 CB 权重。
- `ClassBalancedCrossEntropy`：带类别权重 + label smoothing 的交叉熵模块。
- `distillation_loss(student_logits, teacher_logits, temperature)`：KL 蒸馏损失。
- `build_loss(name, class_weights, label_smoothing)`：按字符串名字造损失函数
  （工厂函数，让训练脚本只靠一个 `--loss ce|cb` 参数就能切换）。
"""

from __future__ import annotations

# `from __future__ import annotations`：让类型注解延迟成字符串求值。
# 这样 `Optional[np.ndarray]`、`np.ndarray` 这类注解在定义时不会真的去解析，
# 既避免前向引用问题，也让模块导入更快。它必须是文档字符串后的第一条语句。

from typing import Optional  # 仅用于类型注解：表示「可以是 np.ndarray，也可以是 None」

import numpy as np  # 计算有效样本数和权重
import torch
import torch.nn as nn  # 用于继承 nn.Module 定义损失类
import torch.nn.functional as F  # cross_entropy / log_softmax / softmax / kl_div


def effective_number_weights(
    counts: np.ndarray, beta: float = 0.999, normalize: bool = True
) -> np.ndarray:
    """用「有效样本数」公式，把每类样本数换算成类别平衡（CB）损失权重。

    参数
    ----
    counts : np.ndarray
        shape=(C,), dtype 可为整型；第 c 个元素是第 c 类的训练样本数。
        通常来自 `np.bincount(train_labels, minlength=C)`。
    beta : float
        控制「重加权强度」的超参，默认 0.999，取值应在 [0, 1)。
        - beta = 0   → 所有类的有效样本数都是 1，权重全部相等（等于不加权）；
        - beta → 1   → E_n ≈ n，权重退化成「逆频率」（1/样本数），加权最猛；
        - 中间值（如 0.999 / 0.9999）→ 介于两者之间，是论文推荐的甜点区。
    normalize : bool
        是否把权重除以它们的均值，默认 True。
        归一化后权重的平均值是 1，这样「总损失的量级」不会因为加权而整体放大，
        学习率不用跟着重调。设为 False 则得到原始权重（论文里 w_c ∝ 1/E_n）。

    返回
    ----
    np.ndarray
        shape=(C,), dtype=float32 的权重数组。稀有类的值大，常见类的值小。

    公式与直觉（Cui et al., CVPR 2019）
    -----------------------------------
        E_n = (1 - beta**n) / (1 - beta)      # 有效样本数
        w_c ∝ 1 / E_n                          # 权重与有效样本数成反比

    为什么不是简单用 1/n（逆频率）？因为「样本数翻倍 = 信息翻倍」这个假设太乐观：
    同一类里第 1000 张图和第 1001 张图高度相似，新增的信息很少。
    有效样本数用一个几何级数去建模这种**边际递减**：
    beta=0.999 时，n=500 的 E_n 约 393，n=5000 的 E_n 约 993——
    样本数差 10 倍，有效样本数只差 2.5 倍，权重比也就温和得多。
    这样既能抬升小类，又不会因为权重过于极端而在小类上过拟合。

    关键设计原因
    ------------
    - 用 float64 计算：`beta**n` 在 beta 接近 1、n 上千时会出现
      「大数相减」的精度问题，float32 容易把 (1-beta^n) 算成 0。
    - 输出转成 float32：它最终要变成 torch 张量参与前向计算，
      float32 与模型参数精度一致，没必要用 double 拖慢训练。
    - 公式里分母 (1-beta) 是常数，对所有权重是同一个缩放因子，
      归一化之后会被消掉，所以保留它只是为了公式完整、便于对照论文。
    """
    # asarray(float64) 而不是 astype：如果传入的已经是 float64 数组，asarray 不复制，
    # 同时统一了类型，保证下面的 power 运算有足够精度。
    counts = np.asarray(counts, dtype=np.float64)
    # 有效样本数：beta**counts 是「每类一个」的逐元素幂运算，numpy 会广播到整个数组。
    # 当 counts=0 时，beta**0 = 1，E_n = 0 —— 这会让 1/E_n 变成 inf，
    # 实际使用中不会出现「样本数为 0 的类」（数据划分保证每类至少 1 张）。
    effective = (1.0 - np.power(beta, counts)) / (1.0 - beta)
    # 权重与有效样本数成反比：样本越少 → E_n 越小 → 权重越大。
    weights = 1.0 / effective
    if normalize:
        # 除以均值让权重「平均为 1」：这是训练稳定性的关键一步。
        # 不归一化的话，C 个类加起来的总权重可能远大于 C，等于悄悄放大了学习率。
        weights = weights / weights.mean()
    # 转 float32：与 torch 模型参数精度一致，后面 torch.tensor(weights) 也不会再转一次。
    return weights.astype(np.float32)


class ClassBalancedCrossEntropy(nn.Module):
    """带类别权重和 label smoothing 的交叉熵损失。

    继承 `nn.Module` 而不是写成普通函数，是为了让权重作为一个**模块状态**存在：
    它可以跟着模型 `.to(device)` 一起搬家、也能被 `state_dict()` 保存下来。

    参数
    ----
    weights : np.ndarray
        shape=(C,)，每个类的损失权重，通常来自 `effective_number_weights`。
    label_smoothing : float
        标签平滑系数 ε，默认 0.1，取值 [0,1)。0 表示不平滑（标准 one-hot）。

    forward 的输入 / 输出
    ----
    logits : torch.Tensor, shape=(B,C)
    target : torch.Tensor, shape=(B,), dtype=int64，取值 0~C-1
    返回：0 维标量 tensor（对 batch 取平均后的损失），可直接 `.backward()`。

    关键设计原因
    ------------
    为什么不直接用 `F.cross_entropy(weight=...)`？
    其实内部就是它——这个类只是把「权重缓冲区」和「平滑系数」打包在一起，
    让调用方（build_loss）不用每次手动传参，也方便以后扩展（比如改成 focal loss）。

    `register_buffer` 与 `nn.Parameter` 的区别（初学者最容易混淆的点）：
      - `nn.Parameter` 是**可训练**参数：会出现在 `model.parameters()` 里，
        优化器会更新它，`state_dict` 里也会保存。用于权重矩阵、bias 等。
      - `register_buffer` 是**不可训练**的张量：会出现在 `state_dict` 里、
        会跟着 `.to(device)` / `.half()` 一起搬家和变换，
        但**不会**被 `parameters()` 返回、优化器不会碰它、它也没有梯度。
        用于「需要随模型保存和搬运、但不用学的常量」。
    这里的类别权重显然不该被梯度更新（它是从数据统计出来的常数），
    所以必须用 buffer，绝不能用 Parameter——用错了会导致训练时权重自己被优化，
    越训越离谱。
    """

    def __init__(self, weights: np.ndarray, label_smoothing: float = 0.1):
        """把类别权重注册成 buffer，并记下平滑系数。

        参数
        ----
        weights : np.ndarray
            shape=(C,), 每类的损失权重。内部会 `torch.tensor(...)` 转成 float32 张量。
            注意必须在构造时就转成 tensor 再 register_buffer，
            不能直接 register 一个 numpy 数组（buffer 必须是 Tensor）。
        label_smoothing : float
            标签平滑系数 ε，默认 0.1。它不是张量，只是普通 Python 属性，
            不需要注册进 state_dict（从 config 里能重新构造出来）。

        关键设计原因
        ------------
        `torch.tensor(weights)` 会**复制**一份数据，因此之后外部再修改
        传入的 numpy 数组也不会影响到损失函数内部的权重——避免隐蔽的耦合。
        """
        # 必须先调 super().__init__()：nn.Module 在这里初始化内部字典
        # （_parameters、_buffers、_modules 等）。忘了它，register_buffer 会直接报错。
        super().__init__()
        # register_buffer("名字", 张量)：注册一个「随模型搬家但不训练」的张量。
        # 之后用 self.weights 就能访问；调用 loss_fn.to(device) 时它会自动跟着走。
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))
        # 普通属性：不需要梯度，也不需要进 state_dict。
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算加权 + 平滑后的交叉熵。

        参数
        ----
        logits : torch.Tensor
            shape=(B,C) 的**未归一化**打分。注意不要提前做 softmax，
            `F.cross_entropy` 内部会做 log_softmax（自己做会重复归一化，结果错）。
        target : torch.Tensor
            shape=(B,), dtype=int64 的类别下标。

        返回
        ----
        torch.Tensor
            0 维标量（reduction 默认 "mean"）：batch 内按样本平均的损失。

        关键设计原因
        ------------
        - `weight=` 让每个样本的损失乘以它真实类别的权重，稀有类的错误更「贵」。
        - `label_smoothing=ε` 的目标分布是：
              正确类 (1 - ε) + ε/C，其余类 ε/C
          这样模型不再被鼓励把正确类的概率推向 1.0。
          好处：抑制过度自信、降低 ECE（见 engine.py 的校准指标）、
          缓解「模型对错误标注过于自信」的问题。
        - `self.weights.to(logits.device)`：buffer 虽然会跟着模块搬家，但损失模块
          常常**没有**被 `.to(device)`（我们只对 model 调用了 .to）。这里防御性地
          再 .to() 一次，保证权重和 logits 在同一设备上，避免跨设备运算报错。
          `.to()` 对已在目标设备上的张量是空操作，没有额外开销。
        """
        return F.cross_entropy(
            logits,
            target,
            # 权重张量 (C,) 会按 target 逐样本取出对应类的权重，再乘到每个样本的 loss 上。
            weight=self.weights.to(logits.device),
            label_smoothing=self.label_smoothing,
        )


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """计算学生与教师预测分布之间的 KL 散度（知识蒸馏的「软标签」损失）。

    参数
    ----
    student_logits : torch.Tensor
        shape=(B,C)，学生模型的原始打分（未 softmax）。
    teacher_logits : torch.Tensor
        shape=(B,C)，教师模型对**同一批样本**的原始打分。
        训练时要用 `teacher_logits[indices]` 按原始下标取，保证样本对齐。
    temperature : float
        蒸馏温度 T，默认 2.0。T=1 就是普通 KL；T>1 时分布更平滑。

    返回
    ----
    torch.Tensor
        0 维标量。已乘上 T² 做梯度补偿，所以可以直接以 alpha 权重加进总损失。

    关键设计原因
    ------------
    1) 为什么要除以温度 T？
       教师模型对「猫」的 logits 可能是 [9.0(猫), 4.0(狗), 0.1(飞机)]。
       直接 softmax 后几乎是 [0.99, 0.01, ~0]，狗和飞机的差异被压没了。
       除以 T=4 后变成 [2.25, 1.0, 0.025]，softmax 得到 [0.80, 0.20, 0.006]——
       「猫和狗有点像、和飞机几乎不像」这种**类间结构**才显现出来。
       这正是蒸馏比 one-hot 标签多出来的信息（暗知识）。

    2) 为什么最后要乘 T²？
       logits 除以 T 之后，梯度幅度大约按 1/T 缩小（softmax 对 logits 的导数
       与概率差成正比，而概率差又被平滑了）。若不加补偿，T 越大蒸馏项越弱，
       就退化成了「几乎没蒸馏」。乘 T² 后，不同 T 下的梯度量级大致相当，
       调 T 时不用同步重调 distill_alpha。这是 Hinton 2015 原文的做法。

    3) KL 的方向：`F.kl_div(log_student, teacher)` 计算的是
           KL(teacher ‖ student) = Σ teacher · (log teacher - log student)
       即「用教师的分布去衡量学生的偏差」，教师概率为 0 的地方不产生惩罚。
       我们只对 student 求梯度（teacher 是常数、requires_grad=False），
       所以最小化它就是让学生的分布逼近教师。

    4) `reduction="batchmean"` 与 "mean" 的区别（易错点）：
       kl_div 默认的 "mean" 是除以**所有元素个数**（B×C），
       而 "batchmean" 是除以 **batch 大小 B**（先对每个样本的 C 个类求和，再对 B 平均）。
       蒸馏里的标准做法是 batchmean，因为「每个样本的 KL」才是我们要平均的量。
       用错 reduction，损失会平白小 C 倍，alpha 的最优值也跟着错。
    """
    # 学生用 log_softmax：kl_div 要求第一个参数是**对数概率**（它内部会做 exp）。
    # 若误传 softmax 后的概率，数值会完全不对（log 概率才是 kl_div 的输入约定）。
    student = F.log_softmax(student_logits / temperature, dim=-1)
    # 教师用 softmax 得到概率分布（它作为「目标分布」不需要 log）。
    # 注意 teacher 应当是不可导的常数（通常来自预先算好并保存的 logits），
    # 所以这里不会产生「教师也被更新」的问题。
    teacher = F.softmax(teacher_logits / temperature, dim=-1)
    # batchmean：先对类别维求和、再对 batch 求平均；
    # 乘 temperature**2 补偿温度带来的梯度缩放（见 docstring 第 2 点）。
    return F.kl_div(student, teacher, reduction="batchmean") * (temperature**2)


def build_loss(
    name: str,
    class_weights: Optional[np.ndarray] = None,
    label_smoothing: float = 0.1,
) -> nn.Module:
    """按名字构造损失函数（简单工厂模式）。

    参数
    ----
    name : str
        损失名，大小写不敏感，内部会先 `.lower()`：
        - "ce"：普通交叉熵（只带 label smoothing）；
        - "cb" / "class_balanced" / "class-balanced"：类别平衡交叉熵。
    class_weights : Optional[np.ndarray]
        shape=(C,) 的类别权重。只有 CB 损失需要；CE 会忽略它。
        传 None 而选了 CB 会直接报错（见下），而不是悄悄退化成不加权。
    label_smoothing : float
        标签平滑系数，默认 0.1，两种损失都会用上。

    返回
    ----
    nn.Module
        一个可调用的损失模块，签名统一为 `loss_fn(logits, target) -> 标量`。

    关键设计原因
    ------------
    工厂函数的好处是「把选择逻辑收在一处」：训练脚本只需要 `--loss ce|cb`
    这样一个字符串参数，不用写 if/else 去 new 不同的类；
    以后加 focal loss 之类的新损失，也只改这一个函数。

    参数校验很严格：选了 "cb" 却没给权重时**主动抛 ValueError**，
    而不是默默按 1.0 处理。因为「以为加了权、其实没加」是最难排查的静默错误——
    实验结论会直接错掉，但程序一点不报错。

    `name in {...}` 用集合做多别名匹配：比写 `name == "cb" or name == "class_balanced" or ...`
    更短，且集合查找是 O(1)（虽然这里只有三个元素，性能无所谓，主要是可读性）。
    """
    # 统一小写，这样 "CE"、"CE " 之外的大小写差异都能容错（空格仍需调用方自己清理）。
    name = name.lower()
    if name == "ce":
        # 普通交叉熵：weight 默认为 None，即所有类等权。
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if name in {"cb", "class_balanced", "class-balanced"}:
        if class_weights is None:
            # 快速失败：宁可让脚本在启动时就崩，也别让它「安静地」跑完 15 个 epoch
            # 才发现加权根本没生效。
            raise ValueError("class_weights are required for class-balanced loss")
        return ClassBalancedCrossEntropy(class_weights, label_smoothing)
    # 未知名字：把调用方传进来的值回显在错误里，方便定位拼写错误。
    raise ValueError(f"unknown loss: {name}")
