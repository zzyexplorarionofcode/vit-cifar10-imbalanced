"""训练 / 评估工具与分类指标 —— 教学注释版。

=====================================================================
一、这个模块解决什么问题
=====================================================================
`scripts/train.py` 里最不该出现的，是一大坨「前向、反向、更新参数、算指标」的
细节代码。那些细节写一次就容易出错，写十次（每个模型一份）更是灾难。
本模块把它们封装成几个职责单一的函数，让训练脚本只负责「读配置、循环 epoch、
记录结果」：

- 训练一步怎么做：`train_one_epoch`
- 评估一次怎么做：`evaluate` / `predict_logits`
- 优化器怎么分组、学习率怎么变：`build_optimizer` / `cosine_lambda`
- 指标怎么算：混淆矩阵、macro-F1、balanced accuracy、ECE
- 模型有多大、结果可不可复现：`count_parameters` / `set_seed`

换句话说：**data.py 决定「喂什么」，models/ 决定「算什么」，engine.py 决定
「怎么学、怎么考」。**

=====================================================================
二、在「数据 → 模型 → 训练 → 评估」流程中的位置
=====================================================================
    data.py                 models/                  engine.py（本文件）
  ┌──────────────┐      ┌───────────────┐      ┌────────────────────────────┐
  │ DataLoader   │      │ ViT / MobileViT│      │ train_one_epoch：           │
  │ augment_batch│─────▶│ forward       │─────▶│   zero_grad → forward →     │
  │ (B,3,32,32)  │      │ → logits(B,C) │      │   loss → backward → clip →  │
  └──────────────┘      └───────────────┘      │   step → scheduler.step     │
                                               │ evaluate：no_grad 前向 + 指标│
                                               │ predict_logits：导出 logits  │
                                               └────────────────────────────┘

一次完整实验的调用顺序（见 scripts/train.py）：
  1. set_seed → 2. build_optimizer → 3. cosine_lambda + LambdaLR →
  4. 每个 epoch：train_one_epoch（更新参数）→ evaluate（算验证指标）→
  5. 训练结束：evaluate 在测试集上给出最终指标；predict_logits 导出教师 logits。

=====================================================================
三、阅读前需要知道的概念
=====================================================================
1. 前向与反向：`logits = model(images)` 是前向；`loss.backward()` 沿计算图反传，
   把每个参数的梯度写进 `param.grad`；`optimizer.step()` 才真正更新参数。
2. 梯度是「累加」的：PyTorch 默认把新梯度加到 `.grad` 上，所以每个 batch 开头
   必须 `optimizer.zero_grad()` 清零，否则梯度会跨 batch 越积越大。
3. `model.train()` / `model.eval()` 切换的是**层的行为**：Dropout 是否随机丢弃、
   BatchNorm 用不用当前 batch 的统计量、是否更新 running stats。
4. `@torch.no_grad()` 关闭的是**自动求导**：不再记录计算图，省显存也更快。
   它和 `model.eval()` 是两件不同的事，评估时两个都要（详见 evaluate 的注释）。
5. 学习率调度（scheduler）：本项目用 `LambdaLR` + 一个返回倍率的函数，
   **按「step」（每个 batch）而不是按 epoch** 调整学习率的绝对值。
6. 长尾分布下的指标：整体 accuracy 会被大类主导，所以要同时看
   macro-F1（每类 F1 的算术平均）和 balanced accuracy（每类召回率的平均）。
7. 校准（calibration）：模型说「我有 80% 把握」时，是否真的对了 80%？
   ECE（Expected Calibration Error）就是衡量这个差距的指标。
8. 知识蒸馏：让学生模型的输出分布去逼近教师模型的输出分布（KL 散度）。

=====================================================================
四、本文件的主要函数
=====================================================================
- `set_seed(seed)`：固定 random / numpy / torch 的随机种子，保证可复现。
- `build_optimizer(model, lr, weight_decay)`：AdamW，且对 bias/归一化/位置编码
  这类参数**不做** weight decay。
- `cosine_lambda(warmup_steps, total_steps, min_lr_ratio)`：线性 warmup + 余弦衰减
  的倍率函数，喂给 `LambdaLR`。
- `evaluate(model, loader, device, num_classes, max_batches)`：在完整数据集上推理，
  返回 accuracy / macro-F1 / 混淆矩阵 / ECE 等一整套指标。
- `_expected_calibration_error(confidence, correct, n_bins)`：ECE 的分桶计算（内部工具）。
- `train_one_epoch(...)`：跑完一个 epoch，返回平均 loss / 准确率 / 耗时。
- `predict_logits(model, images, device, batch_size, max_batches)`：批量导出 logits，
  供知识蒸馏当教师信号用。
- `count_parameters(model, trainable_only)`：统计参数量。
"""

from __future__ import annotations

# `from __future__ import annotations`：把类型注解变成「延迟求值的字符串」。
# 好处是可以在注解里用 `Dict[str, float]`、`Optional[torch.Tensor]`、`str | Path`
# 这类写法而不必担心运行时求值顺序（比如引用还没定义的类）。
# 它只影响注解，不影响任何运行逻辑，且必须是文档字符串后的第一条语句。

import math  # 余弦调度里的 cos / pi
import random  # Python 内置随机数（如数据划分、shuffle 的某些路径）
import time  # 用 perf_counter 测每步耗时（比 time.time 更精确、不受系统时钟调整影响）
from typing import Dict, Iterable, Optional, Tuple  # 仅类型注解

import numpy as np
import torch
import torch.nn as nn

# 从同包内导入：数据增强/标准化来自 data.py，蒸馏损失来自 losses.py。
# 用相对导入（.）意味着本模块必须作为 `src.engine` 被导入，不能直接运行本文件。
from .data import augment_batch, normalize_batch
from .losses import distillation_loss


def set_seed(seed: int) -> None:
    """固定所有会影响实验结果的随机源，让「同一个 seed」跑出同一条曲线。

    参数
    ----
    seed : int
        随机种子。任何整数都行，习惯上用 42、0 这类。

    返回
    ----
    None
        纯粹是副作用（设置全局状态），不返回任何东西。

    关键设计原因
    ------------
    深度学习里随机性来源很多：参数初始化（torch）、数据打乱与增强（torch/numpy）、
    以及 Python 的 random 模块（某些第三方库会用它）。只固定 torch 一个是不够的，
    所以这里把三处都设一遍。

    `torch.cuda.manual_seed_all` 在纯 CPU 环境下是**空操作**，不会报错；
    保留它是为了让同一份代码在 GPU 机器上也能用，不用改。

    注意：固定种子只保证「同一台机器、同一版本、同一执行顺序」下可复现；
    换个 PyTorch 版本或改了算子实现，结果仍可能有微小差异。这不是 bug。
    """
    random.seed(seed)  # Python 标准库的随机数
    np.random.seed(seed)  # numpy 的全局随机状态（数据划分、shuffle 会用到）
    torch.manual_seed(seed)  # torch 的 CPU 随机状态：初始化、Dropout、增强都依赖它
    # 若没有 GPU，这一行是空操作；有 GPU 时它把所有设备的种子都设成同一个值。
    torch.cuda.manual_seed_all(seed)


def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """构造 AdamW 优化器，并把参数分成「要 weight decay」和「不要」两组。

    参数
    ----
    model : nn.Module
        待训练的模型。函数只读取它的 `named_parameters()`，不改变模型结构。
    lr : float
        基础学习率（AdamW 的初始 lr）。后续会被 LambdaLR 按倍率缩放。
    weight_decay : float
        作用在「权重矩阵」上的 L2 正则强度（AdamW 是解耦式衰减，与梯度更新分开）。

    返回
    ----
    torch.optim.Optimizer
        一个 `AdamW` 实例，内部含两个 param_group：
        group[0] 用 weight_decay，group[1] 用 0.0。

    关键设计原因
    ------------
    为什么不给所有参数都加 weight decay？因为衰减的直觉是「把权重往 0 拉，
    抑制过大的权重」，这对 Linear/Conv 的**权重矩阵**有意义；但下面几类参数
    一旦被拉向 0，伤害远大于收益：

      - `ndim == 1` 的参数：LayerNorm/BatchNorm 的 gamma/beta、各种 bias。
        它们本来就该自由地做「平移和缩放」，衰减会破坏归一化的表达能力。
      - 名字以 `.bias` 结尾的参数：同上，偏置不需要正则。
      - `cls_token` / `pos_embed`：ViT 的可学习「分类 token」和「位置编码」。
        它们是模型的结构性参数，不是拟合数据的权重；衰减会让位置信息整体变小，
        训练反而不稳。这是 ViT 微调里非常常见的实践。

    用 `named_parameters()` 而不是 `parameters()`，就是因为需要**名字**来判断
    哪些是 cls_token / pos_embed。

    betas=(0.9, 0.999) 是 Adam 的默认动量项；这里显式写出来是为了让学生看到
    「一阶矩、二阶矩的滑动平均系数」确实存在，而不是被默认值藏起来。
    """
    decay, no_decay = [], []
    # generator 逐个产出 (参数名, 参数张量)，顺序稳定（按注册顺序）。
    for name, param in model.named_parameters():
        # 冻结的参数（requires_grad=False）不该进优化器：
        # 放进去虽然不会更新，但会白白占用优化器状态内存。
        if not param.requires_grad:
            continue
        # 判断条件按「从宽到窄」排列：
        #   ndim == 1 → 所有向量型参数（bias、norm 的 gamma/beta）
        #   .bias     → 双保险（有些 bias 是 2 维的）
        #   cls_token / pos_embed → ViT 的结构性参数
        if param.ndim == 1 or name.endswith(".bias") or "cls_token" in name or "pos_embed" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    # param_group 是「一组共享超参的参数列表」。把两组分开，就能让同一个优化器
    # 对不同参数用不同的 weight_decay。
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.999))


def cosine_lambda(warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.02):
    """返回一个「学习率倍率函数」：先线性 warmup，再按余弦曲线衰减到 min_lr_ratio。

    参数
    ----
    warmup_steps : int
        预热步数（不是 epoch）。前 warmup_steps 步里，倍率从 ~0 线性升到 1。
        内部用 `max(1, ...)` 兜底，避免传 0 时除零。
    total_steps : int
        总训练步数（= steps_per_epoch × epochs），余弦曲线在它之后走到最低点。
    min_lr_ratio : float
        衰减终点相对基础学习率的比例，默认 0.02，即最终 lr = 0.02 × 初始 lr。
        不降到 0 是为了保留一点「微调能力」，也让训练末期不至于完全停住。

    返回
    ----
    Callable[[int], float]
        一个闭包 `fn(step) -> 倍率`，返回值在 [min_lr_ratio, 1] 区间内。
        它会被 `torch.optim.lr_scheduler.LambdaLR` 每步调用一次，
        实际 lr = base_lr × fn(step)。

    关键设计原因
    ------------
    1) 为什么需要 warmup？训练初期参数是随机的、梯度方向噪声很大，
       直接用大学习率容易把模型「炸」到坏的局部区域。先用小学习率走几步，
       等梯度方向稳定了再放大，是 Transformer 类模型的标准做法。
    2) 为什么要余弦衰减？相比「阶梯式下降」，余弦曲线前期降得慢、后期降得快，
       在接近最优解时步长足够小，收敛更平滑，且不需要手调「第几个 epoch 降一次」。
    3) 为什么返回函数而不是直接返回一个 list？因为 LambdaLR 要的就是
       `lambda step: 倍率` 这种可调用对象，它可以随时按当前 step 计算，
       不用预先存一张长度为 total_steps 的表。
    """
    # 兜底：warmup_steps 若为 0，下面 step/warmup_steps 会除零，所以至少取 1。
    warmup_steps = max(1, warmup_steps)

    def fn(step: int) -> float:
        """给定「第几步」，返回该步的学习率倍率（闭包，捕获上面的三个参数）。

        参数
        ----
        step : int
            当前步数（从 0 开始，LambdaLR 每调用一次 step() 就传进来一次）。
            注意它是**步数**，不是 epoch——本项目 steps_per_epoch 通常有几十到几百，
            所以一个 epoch 内学习率会连续变化很多次。

        返回
        ----
        float
            学习率倍率：
            - step < warmup_steps：step / warmup_steps（0 → 1 线性上升）；
            - 之后：从 1 沿余弦曲线降到 min_lr_ratio，且被夹在 [0,1] 的进度区间内。

        关键设计原因
        ------------
        返回值是「倍率」而不是「绝对学习率」，这样用户改 `--lr` 时不用改调度逻辑，
        两者天然解耦（LambdaLR 会做 base_lr × 倍率）。
        """
        # 第一阶段：线性 warmup。step=0 时倍率为 0（学习率为 0），
        # 相当于让第一步「空跑」一次；这是常见实现，影响可忽略。
        if step < warmup_steps:
            return step / warmup_steps
        # 第二阶段：把「预热之后的步数」归一化成 [0,1] 的进度。
        # max(1, ...) 防止 total_steps == warmup_steps 时除零。
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        # 夹紧到 [0,1]：训练步数可能超过 total_steps（例如 --resume 或改 epochs），
        # 越界后 cos 会开始回升，所以必须截断，保证倍率单调不增。
        progress = min(1.0, max(0.0, progress))
        # 余弦退火：progress=0 → cos(0)=1 → 倍率 1；
        # progress=1 → cos(pi)=-1 → 倍率 min_lr_ratio。
        # 中间是平滑的 (1+cos)/2 曲线。
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    # 返回闭包本身（不是调用结果），交给 LambdaLR 保存。
    return fn


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: Iterable,
    device: torch.device,
    num_classes: int = 10,
    max_batches: Optional[int] = None,
) -> Dict:
    """在给定数据集上做一次完整推理，并算出分类任务的一整套指标。

    参数
    ----
    model : nn.Module
        待评估模型，输入 (B,3,32,32) float，输出 (B,C) 的 logits。
    loader : Iterable
        DataLoader，每个 batch 至少含 (images, labels)；本函数只用前两个元素，
        所以传 IndexedArrayDataset（三元组）的 loader 也能正常工作。
    device : torch.device
        推理设备（本项目是 CPU）。
    num_classes : int
        类别数，默认 10。它决定了混淆矩阵是 C×C、以及 bincount 的长度。
    max_batches : Optional[int]
        最多跑多少个 batch，默认 None 表示跑完整个 loader。
        调试或做快速冒烟测试时设成 1~5 可以大幅省时间。

    返回
    ----
    Dict
        含以下键（注意 confusion_matrix 是「行=真实、列=预测」的 C×C 嵌套列表）：
          accuracy, macro_f1, balanced_accuracy,
          per_class_recall, per_class_f1, confusion_matrix, ece,
          num_samples, predictions, labels

    关键设计原因
    ------------
    1) `@torch.no_grad()` 与 `model.eval()` 为什么要一起用？
       - `@torch.no_grad()` 是**上下文级别的开关**：它告诉 autograd「接下来的运算
         不要记录计算图」。评估时不需要反向传播，不记录能省下大量内存和时间。
         但它**不改变任何层的行为**——Dropout 照样随机丢弃，BatchNorm 照样用
         当前 batch 的统计量。
       - `model.eval()` 是**模块级别的开关**：它把 Dropout 关掉、让 BatchNorm
         改用 running stats，从而让推理结果确定、可复现。
       两者作用在不同层面，缺一不可：只加 no_grad 会让指标带随机抖动，
       只加 eval() 会让显存/时间白白浪费。

    2) 为什么先把 logits 收集起来、最后统一算指标？
       因为 accuracy/F1/ECE 都是「整个数据集上的统计量」，必须看到全部样本才能算。
       如果边跑边算，分批的 softmax 和混淆矩阵拼接容易出错（尤其是 macro 平均）。

    3) `logits.float().cpu()`：统一成 float32 并搬回 CPU。
       评估结果最终要交给 numpy 做统计，而 numpy 不认识 GPU 张量；
       顺带把可能存在的 float16/bf16 统一成 fp32，避免 softmax 精度问题。
    """
    # 切到推理模式：Dropout 关闭、BatchNorm 用 running stats。
    # 这一步必须在循环之前，且和上面的 no_grad 是两件不同的事（见 docstring）。
    model.eval()
    all_logits, all_labels = [], []
    # enumerate 同时拿到「第几个 batch」和 batch 本身；
    # step 只在 max_batches 截断时用到，用 enumerate 比自己维护计数器更不容易出错。
    for step, batch in enumerate(loader):
        # batch 可能是 (images, labels) 或 (images, labels, indices)；
        # 这里只取前两个，所以两种 Dataset 都能兼容。
        images, labels = batch[0], batch[1]
        # 评估用 normalize_batch（确定性），绝不能用 augment_batch（随机增广）。
        # non_blocking=True 只对「锁页内存 + GPU」的异步拷贝有意义，CPU 上是空操作，
        # 写上是为了代码在 GPU 环境下不用改。
        images = normalize_batch(images).to(device, non_blocking=True)
        logits = model(images)
        # 先 .float() 统一精度再 .cpu() 搬到内存；积累成 list 后一次性 cat。
        all_logits.append(logits.float().cpu())
        all_labels.append(labels.cpu())
        if max_batches is not None and step + 1 >= max_batches:
            break
    # torch.cat 沿第 0 维拼接：(B1,C)+(B2,C)+... → (N,C)。
    # 顺序与 loader 的迭代顺序一致，所以能和 labels 严格对齐。
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels).numpy()
    # softmax(dim=-1)：沿类别维把 logits 变成概率（每行和为 1）。
    # 评估阶段不需要梯度，所以这里可以放心 .numpy()。
    probs = logits.softmax(dim=-1).numpy()
    # argmax(axis=1)：每行概率最大的那个类别的下标，就是预测类别。
    preds = probs.argmax(axis=1)

    # ---- 用 bincount 一步算出混淆矩阵（一个很值得学的技巧）----
    # 想把 (真实 i, 预测 j) 这一对计数加一，最直接的办法是二重循环，但那太慢。
    # 技巧：把二维坐标「编码」成一维下标 ——  flat = i * C + j。
    # 这样 i*C+j 正好落在 [0, C*C) 且互不重复（行优先展开），于是所有样本的
    # 一维下标可以用一次 np.bincount 统计频次，再 reshape 回 (C,C) 就得到混淆矩阵！
    #   cm[i, j] = 真实为 i 且被预测为 j 的样本数。
    # minlength=C*C 保证即使某些组合一次都没出现，数组长度也正好是 C*C。
    cm = np.bincount(
        labels.astype(np.int64) * num_classes + preds.astype(np.int64),
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)

    # 对角线上是「真实 i 且预测 i」，即真正例 TP（按类看）。
    tp = cm.diagonal().astype(np.float64)
    # 第 j 列求和 = 所有被预测成 j 的样本数；减去 TP 就是「假正例」FP
    # （别人被误判成了 j）。axis=0 表示沿「真实」方向求和。
    fp = cm.sum(axis=0) - tp
    # 第 i 行求和 = 所有真实为 i 的样本数；减去 TP 就是「假负例」FN
    # （i 类被漏判成别的类）。axis=1 表示沿「预测」方向求和。
    fn = cm.sum(axis=1) - tp
    # 召回率 recall = TP / (TP + FN)：真实为 i 的样本里，有多少被找出来了。
    # np.maximum(..., 1.0) 防止某类样本数为 0 时除零（长尾数据里可能真的出现）。
    per_class_recall = tp / np.maximum(tp + fn, 1.0)
    # F1 = 2TP / (2TP + FP + FN)，是 precision 与 recall 的调和平均。
    denom = 2 * tp + fp + fn
    # np.divide 的 where= 参数：只在分母 > 0 的位置做除法，其余位置保留 out 里的 0。
    # 这比「先判断再赋值」更向量化，也避免了 0/0 产生的 nan（nan 会污染后面的 mean）。
    per_class_f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    # macro-F1：对每个类的 F1 取**算术平均**，每类权重相同。
    # 这正是长尾场景需要的：大类再多也只算一票，小类的表现不会被淹没。
    macro_f1 = float(per_class_f1.mean())
    # 置信度 = 每行最大的那个概率值，代表模型对「自己的预测」有多大把握。
    confidence = probs.max(axis=1)
    # correct 是 0/1 的 float 数组，后面算 accuracy 和 ECE 都要用。
    correct = (preds == labels).astype(np.float64)
    # ECE 需要「置信度」和「是否预测正确」两个数组，按置信度分桶统计。
    ece = _expected_calibration_error(confidence, correct, n_bins=15)
    return {
        # accuracy：整体正确率。注意它会被大类主导，长尾时不能只看这个数。
        "accuracy": float(correct.mean()),
        "macro_f1": macro_f1,
        # balanced accuracy：各类召回率的平均，等价于「每类等权」的准确率。
        "balanced_accuracy": float(per_class_recall.mean()),
        # tolist() 把 numpy 数组变成 Python list，才能被 json.dumps 序列化。
        "per_class_recall": per_class_recall.tolist(),
        "per_class_f1": per_class_f1.tolist(),
        "confusion_matrix": cm.tolist(),
        # float(ece) 确保是 Python 原生 float 而不是 numpy 标量（后者不能 JSON 化）。
        "ece": float(ece),
        "num_samples": int(len(labels)),
        # 保存原始预测与标签，便于事后画图、做错误分析、写报告。
        "predictions": preds.tolist(),
        "labels": labels.tolist(),
    }


def _expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """计算 ECE（Expected Calibration Error）：预测置信度与真实准确率的加权差距。

    参数
    ----
    confidence : np.ndarray
        shape=(N,)，每个样本的「最大类别概率」，取值 [0,1]。
    correct : np.ndarray
        shape=(N,)，每个样本是否预测正确（0.0 / 1.0）。
    n_bins : int
        把 [0,1] 的置信度划成多少个等宽区间，默认 15。

    返回
    ----
    float
        ECE 值，越小说明「模型说几分把握就真有几分准」。0 表示完美校准。

    关键设计原因
    ------------
    交叉熵训练出来的模型通常**过度自信**：明明只对了 70%，却敢说 95%。
    ECE 的算法很直观：把预测按置信度分桶，在每个桶里比较
    「平均置信度」与「实际准确率」，差距越大越糟，最后按桶内样本占比加权求和。

    为什么要加权？因为高置信度的样本通常很少（比如 confidence>0.99 的桶可能只有
    几个样本），不加权的话一个只有 2 个样本的桶会和有几千个样本的桶同权，
    结果被噪声主导。

    分桶边界用 (lo, hi] 左开右闭：这样每个样本恰好落进一个桶，不会重复计数
    （若用闭区间，正好等于边界的样本会同时落进相邻两个桶）。
    """
    # linspace 生成 n_bins+1 个等距边界，例如 15 个桶就是 16 个点。
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    # zip(bins[:-1], bins[1:]) 把边界两两配对：(b0,b1), (b1,b2), ..., (b14,b15)。
    # 这是「相邻元素配对」的惯用写法，比用 index 循环更清晰。
    for lo, hi in zip(bins[:-1], bins[1:]):
        # 布尔掩码：选出置信度落在 (lo, hi] 区间的样本。
        # `&` 是 numpy 的逐元素逻辑与（注意不是 Python 的 `and`，后者会报错）。
        mask = (confidence > lo) & (confidence <= hi)
        # 空桶直接跳过，避免对空数组求 mean 得到 nan。
        if mask.sum() == 0:
            continue
        # 桶内实际准确率：correct 是 0/1，mean 就是「对的比例」。
        acc = correct[mask].mean()
        # 桶内平均置信度：模型自己觉得有多准。
        conf = confidence[mask].mean()
        # 加权：mask.mean() 就是「这个桶占总样本的比例」。
        # |acc - conf| 是这一桶的校准误差。
        ece += mask.mean() * abs(acc - conf)
    return ece


def train_one_epoch(
    model: nn.Module,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    criterion: nn.Module,
    device: torch.device,
    teacher_logits: Optional[torch.Tensor] = None,
    distill_alpha: float = 0.0,
    distill_temp: float = 2.0,
    max_grad_norm: float = 1.0,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """训练一个 epoch（跑完整个 loader），返回这一轮的平均 loss / 准确率 / 耗时。

    参数
    ----
    model : nn.Module
        待训练模型，输入 (B,3,32,32) float，输出 (B,C) logits。
    loader : Iterable
        DataLoader，必须产出**三元组** (images, labels, indices)，
        即需要搭配 `IndexedArrayDataset`（indices 用来查教师 logits）。
    optimizer : torch.optim.Optimizer
        优化器（通常是 build_optimizer 产出的 AdamW）。
    scheduler : torch.optim.lr_scheduler.LambdaLR
        按 step 调度的学习率调度器；本函数每个 batch 调一次 `scheduler.step()`。
    criterion : nn.Module
        主损失函数（CE 或 ClassBalancedCrossEntropy）。
    device : torch.device
        计算设备。
    teacher_logits : Optional[torch.Tensor]
        教师模型在**整个训练集**上的输出，shape=(N_total, C)，默认 None 表示不蒸馏。
        它按全集顺序存放，所以要用 batch 里的 indices 去索引。
    distill_alpha : float
        蒸馏损失的权重，默认 0.0（等于关闭蒸馏）。总损失 =
        criterion + alpha × distillation_loss。
    distill_temp : float
        蒸馏温度 T，默认 2.0。T 越大，概率分布越平滑，暗知识（类间相似性）越明显。
    max_grad_norm : float
        梯度裁剪的最大范数，默认 1.0；设为 0 或负数表示不裁剪。
    max_batches : Optional[int]
        最多跑多少个 batch，默认 None 表示跑完；调试时设小值可快速验证。

    返回
    ----
    Dict[str, float]
        {"loss": 样本加权平均损失, "accuracy": 训练准确率,
         "seconds": 本 epoch 耗时, "samples": 实际处理的样本数}

    关键设计原因
    ------------
    **每一步的操作顺序是刻意安排的，不能随意调换**：

        zero_grad → forward → loss → backward → clip_grad → step → scheduler.step

    - `zero_grad` 必须在 forward 之前：梯度是累加的，不清零就会串到上一个 batch。
    - `backward` 必须在 `step` 之前：`step` 读的是 `.grad`，没反传就更新不了。
    - `clip_grad` 必须在 `backward` 之后、`step` 之前：它需要梯度已经算好，
      又必须在参数被改动之前动手（否则裁的是旧梯度，白裁）。
    - `scheduler.step()` 放在 `optimizer.step()` **之后**：本项目每个 batch 更新一次
      学习率，放在参数更新之后，语义是「这一步用完当前 lr，再为下一步准备新 lr」。
      （PyTorch 2.x 起，在 optimizer.step 之前调用 scheduler 会告警。）
    """
    # 切到训练模式：Dropout 生效、BatchNorm 用当前 batch 统计并更新 running stats。
    # 如果忘了这一句（比如上一轮评估后模型还停在 eval 模式），Dropout 会失效，
    # 模型容易过拟合，而且这种 bug 非常难通过看 loss 曲线发现。
    model.train()
    # total_loss 累加的是「每个样本的损失之和」，最后除以样本数得到平均损失。
    # 为什么要乘 batch_size？因为最后一个 batch 可能不满，
    # 直接用「各 batch 平均 loss 再取平均」会让小 batch 被过度加权。
    total_loss = 0.0
    correct = 0
    total = 0
    # perf_counter 是高精度单调时钟，用来测「纯训练」耗时（不含评估）。
    start = time.perf_counter()
    # enumerate 提供 step 计数，仅用于 max_batches 截断。
    for step, batch in enumerate(loader):
        # 训练集用 IndexedArrayDataset，所以这里能取到第三个元素 indices。
        # indices 是这个 batch 里每个样本在**全集**中的原始下标。
        images, labels, indices = batch[0], batch[1], batch[2]
        # 训练用 augment_batch：随机裁剪 + 翻转 + Cutout + 标准化，输出 (B,3,32,32)。
        # 然后搬到 device（CPU 上 to("cpu") 是空操作）。
        images = augment_batch(images).to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        # set_to_none=True 比默认的「填 0」更好：
        #   1) 直接把 .grad 置为 None，省一次写内存（大模型上很可观）；
        #   2) 下一步 backward 会重新分配 .grad，而不是在旧值上累加，
        #      避免「某参数这一步没参与计算、却残留上一步梯度」的隐蔽 bug。
        optimizer.zero_grad(set_to_none=True)
        # 前向：得到 (B,C) 的 logits（还没经过 softmax 的原始打分）。
        logits = model(images)
        # 主损失：交叉熵内部会做 log_softmax + NLL，所以这里必须传 logits 而不是概率。
        loss = criterion(logits, labels)
        if teacher_logits is not None and distill_alpha > 0:
            # 用原始下标从教师 logits 里「按行取」本 batch 的教师输出：
            # 高级索引 teacher_logits[indices] → (B, C)。
            # 这一步是必须的：loader 有 shuffle，batch 内的样本顺序与全集顺序无关，
            # 不按 indices 取就会把 A 图的教师信号配给 B 图（标签错位，蒸馏失效）。
            # .to(dtype=torch.float32) 防止教师 logits 存成 fp16 时与 fp32 的
            # 学生 logits 做运算而报 dtype 不匹配。
            teacher = teacher_logits[indices].to(device=device, dtype=torch.float32)
            # 总损失 = 硬标签损失 + alpha × 软标签（蒸馏）损失。
            # 注意 loss = loss + ... 是「新建张量」而不是 += 原地操作，
            # 这样不会破坏 criterion 返回的计算图节点。
            loss = loss + distill_alpha * distillation_loss(logits, teacher, distill_temp)
        # 反向传播：从 loss 出发沿计算图求每个参数的梯度，写进 param.grad。
        # 注意它不会更新参数——真正的更新在 optimizer.step()。
        loss.backward()
        if max_grad_norm > 0:
            # 梯度裁剪：把所有参数的梯度拼成一个大向量，若它的 L2 范数超过
            # max_grad_norm，就整体等比缩小到这个阈值。
            # 作用：防止个别 batch（尤其是长尾数据里的小类或异常样本）产生巨大梯度，
            # 一步把参数推飞，导致 loss 变 NaN。这在 Transformer 上几乎是标配。
            # 它返回裁剪前的范数，这里不需要，所以不接收返回值。
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        # 用「已经裁剪过」的梯度更新参数（AdamW 内部还会做偏差修正和解耦衰减）。
        optimizer.step()
        # 每个 batch 更新一次学习率。因为本项目用的是 LambdaLR（按 step 计），
        # 所以这里必须在循环内调用，不能像 StepLR 那样放在 epoch 末尾。
        scheduler.step()

        # labels.shape[0] 是这个 batch 的真实样本数（最后一个 batch 可能更小）。
        batch_size = labels.shape[0]
        # loss.item() 把 0 维张量转成 Python float（会 detach，不再持有计算图）。
        # 乘 batch_size 后累加，等价于「对所有样本的损失求和」。
        total_loss += loss.item() * batch_size
        # argmax(dim=1) 取每行最大 logit 的下标作为预测类别；
        # == 得到 bool 张量，.sum().item() 数出正确的个数。
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size
        if max_batches is not None and step + 1 >= max_batches:
            break
    # 本 epoch 的墙钟耗时（秒）。
    elapsed = time.perf_counter() - start
    return {
        # max(total, 1) 防止空 loader（一个 batch 都没有）时除零。
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "seconds": elapsed,
        "samples": total,
    }


@torch.no_grad()
def predict_logits(
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
    max_batches: Optional[int] = None,
) -> torch.Tensor:
    """对整个图像张量做批量推理，返回拼接好的 logits（不做 softmax）。

    参数
    ----
    model : nn.Module
        推理用的模型，输出 (B,C) logits。
    images : torch.Tensor
        全部待推理的图像，shape=(N,32,32,3) uint8（NHWC），
        也接受已经是 float/NCHW 的输入（此时按「已经预处理过」处理）。
    device : torch.device
        推理设备。
    batch_size : int
        推理批大小，默认 256。它不影响结果，只影响内存峰值和速度。
    max_batches : Optional[int]
        最多推理多少个 batch，默认 None 表示全部。

    返回
    ----
    torch.Tensor
        shape=(N,C), dtype=float32，位于 **CPU** 上。
        因为它的典型用途是 `torch.save` 落盘当教师信号，放 CPU 才不会占显存。

    关键设计理由
    ------------
    为什么这里要手动切 batch，而不是直接用 DataLoader？
    因为输入只是一个「已经全在内存里的张量」，没有磁盘 I/O、也不需要 shuffle，
    用 DataLoader 反而多一层 Dataset 包装。手写 `range(0, N, batch_size)` 更直接，
    也方便控制 `max_batches`。

    为什么加 `@torch.no_grad()`？推理不需要梯度。若不关掉，PyTorch 会为每个
    batch 建一张计算图，几万个样本下来内存会持续增长（尤其是大模型），
    最后 OOM。注意它不会改变 Dropout 等层的行为，所以下面还要 `model.eval()`。
    """
    # 评估模式：关掉 Dropout、固定 BatchNorm 行为，保证结果确定。
    model.eval()
    outputs = []
    # range(start, stop, step)：i 依次为 0, batch_size, 2*batch_size, ...
    # 切片 images[i : i+batch_size] 自动处理最后一批不满的情况（Python 切片不越界）。
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        # 如果是 uint8，说明调用方传的是「原始像素」，需要走标准预处理；
        # 如果已经是 float，则假定调用方自己已经 permute + 标准化过了，
        # 不能重复标准化（否则等于做了两次 (x-mean)/std，分布完全错掉）。
        if batch.dtype == torch.uint8:
            batch = normalize_batch(batch)
        # .float() 统一精度、.cpu() 搬回内存，便于最后 cat 和 save。
        outputs.append(model(batch.to(device)).float().cpu())
        # i // batch_size 是「当前是第几个 batch」（从 0 开始），+1 表示已完成的个数。
        if max_batches is not None and i // batch_size + 1 >= max_batches:
            break
    # 沿第 0 维拼接所有 batch → (N', C)。
    return torch.cat(outputs)


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """统计模型的参数总量（按「标量个数」而不是「张量个数」）。

    参数
    ----
    model : nn.Module
        任意 PyTorch 模型。
    trainable_only : bool
        是否只统计「可训练」的参数（requires_grad=True），默认 False 表示全算。
        冻结骨干网络做微调时，对比这两个数能看出真正在学的参数量。

    返回
    ----
    int
        参数总个数。例如一个 (96, 96) 的 Linear 权重算 9216，而不是 1。

    关键设计原因
    ------------
    `p.numel()` 是 "number of elements" 的缩写，返回张量的元素总数，
    这正是我们要的（参数量通常按标量数报，单位是「个」而不是「层」）。

    用**生成器表达式** `sum(... for p in ...)` 而不是先建一个 list：
    生成器是惰性求值，不会为几十个张量额外分配一个列表，写法也更紧凑。

    `model.parameters()` 只返回叶子参数张量（已经去重，
    共享权重不会被算两次），所以直接遍历即可。
    """
    params = model.parameters()
    if trainable_only:
        # 生成器表达式 + if 过滤：只累加 requires_grad 为 True 的参数。
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)
