"""CIFAR-10 数据读取与 CPU 友好的「整批」数据增强 —— 教学注释版。

=====================================================================
一、这个模块解决什么问题
=====================================================================
课程数据集以「每个类别一个文件夹、里面若干 .jpg」的形式发放。如果每个 epoch
都用 PIL 逐张解码 4 万张小 JPEG，在 CPU 上会非常慢：JPEG 解码本身就贵，再加
上 Python 层的循环和字符串处理，数据读取会成为瓶颈，把 GPU/CPU 都饿着——
模型明明很小，训练却一小时才几个 epoch。

所以工程上分两步走：
1. 预处理（`scripts/prepare_data.py`）：一次性把文件夹读成两个 uint8 的 .npy
   大数组（图像 + 标签），之后每轮训练直接读数组，不再碰 JPEG 解码；
2. 本模块（data.py）：负责「读数组 → 包成 Dataset/DataLoader → 在整批
   Tensor 上做增强」，是训练循环唯一的数据来源。

这里的增强不是「一张一张地做」，而是「整批向量化」：随机裁剪用 gather 索引
一次算完整个 batch，水平翻转用布尔掩码一次翻完，只有 Cutout 因为每张图的
擦除框大小/位置都不同，才退化成少量 Python 循环。这样既保留了数据随机性，
又不会让 CPU 变成瓶颈。

=====================================================================
二、在「数据 → 模型 → 训练 → 评估」流程中的位置
=====================================================================
        data.py（本文件）              models/（ViT 等）        engine.py
   ┌────────────────────────┐   ┌────────────────────┐   ┌────────────────────┐
   │ .npy 数组               │   │ patch embedding    │   │ train_one_epoch    │
   │ → Dataset / DataLoader  │──▶│ Transformer 编码器  │──▶│ evaluate / 指标     │
   │ → augment_batch         │   │ 分类头 → logits     │   │ optimizer/scheduler│
   └────────────────────────┘   └────────────────────┘   └────────────────────┘

- 本文件是流水线的「入口」：它决定一个 batch 里的图长什么样——增广之后的
  NCHW、float32、且已经用 CIFAR-10 的均值方差标准化过；
- 训练循环（engine.py）拿到 batch 后只做前向/反向，不再关心数据细节；
- 评估阶段走 `normalize_batch`（只标准化、不做随机增强），保证同一张图每次
  评估得到的输入完全一致，指标才可复现、可比较。

=====================================================================
三、阅读前需要知道的概念
=====================================================================
1. 两种数组形状约定（全文最重要的一点）：
   - 磁盘上的 .npy 是 **NHWC + uint8**：shape = (N, 32, 32, 3)，像素值 0~255。
     N 是样本数，H/W 是高/宽，C 是 RGB 三通道。「NHWC」是 PIL / OpenCV /
     TensorFlow 的习惯，也是图像最直观的存储顺序（一行像素的 3 个通道挨着放）。
   - PyTorch 的卷积和模型要的是 **NCHW + float32**：shape = (N, 3, 32, 32)。
     所以本文件里到处能看到 `.permute(0, 3, 1, 2)`，就是把 (N,H,W,C) 重排成
     (N,C,H,W)，只换轴的顺序、不复制数据（得到的是原张量的视图）。
   - uint8 只是为了「省 4 倍内存地存图」；一旦要进模型，必须 `.float()` 再除以
     255 映射到 [0,1]。

2. Dataset 协议：`torch.utils.data.Dataset` 只要求实现两个方法——
   `__len__`（样本总数，DataLoader 靠它算有多少个 batch）和
   `__getitem__(i)`（取第 i 个样本，DataLoader 靠它拼出一个 batch）。
   Python 的 `len(ds)` 和 `ds[i]` 会自动调用这两个「魔法方法」（dunder）。

3. 标准化（normalization）：`(x/255 - mean) / std`，把每个通道拉到均值 0、
   方差 1。它让不同通道的数值尺度一致，优化更稳定、收敛更快。
   `CIFAR_MEAN` / `CIFAR_STD` 是 CIFAR-10 训练集统计出来的常数。

4. 数据增强（augmentation）：训练时随机裁剪 / 水平翻转 / 随机擦除（Cutout），
   制造「看起来不一样、但语义标签不变」的样本，等价于扩大了数据集，抑制过拟合。
   验证和测试时绝对不能用随机增强，否则每次评估结果都会抖动。

5. 随机种子：`torch.Generator().manual_seed(seed)` 让 DataLoader 的 shuffle
   顺序可复现——同一个种子跑两次，切出来的 batch 完全一样，实验才可对比。

=====================================================================
四、本文件的主要函数 / 类
=====================================================================
- `_find_split(root, kind)`：在解压目录里找出 train / test 的类别文件夹（内部工具）。
- `_read_image(path)`：读一张 JPEG → (32,32,3) 的 uint8 数组。
- `scan_dataset(root)`：返回 (train_dir, test_dir, 每个划分每类的图片数)。
- `build_arrays(root, out_dir, workers)`：文件夹 → train/test 的 .npy + meta.json。
- `ArrayDataset`：把 uint8 数组包成返回 (image, label) 的 TensorDataset。
- `IndexedArrayDataset`：额外返回「原始索引」的 Dataset（蒸馏时按索引查教师 logits）。
- `load_arrays(data_dir)`：读回两个 Dataset 和 meta 字典。
- `class_counts(labels)`：统计每个类别的样本数（处理长尾/不平衡分布用）。
- `make_loader(...)`：构造带固定随机种子的 DataLoader。
- `_normalize(x)` / `normalize_batch(images)`：NCHW 张量的标准化（后者含 NHWC→NCHW）。
- `augment_batch(images, erase_p)`：训练用整批增强（随机裁剪 + 翻转 + Cutout）。
- `denormalize(x)`：把标准化后的图还原到 [0,1]，用于可视化。
"""

from __future__ import annotations

# `from __future__ import annotations` 是「编译期开关」，作用是把所有类型注解
# 延迟成字符串保存，定义函数时并不真正求值。好处有二：
#   1) 可以放心写 `str | Path`、`Dict[str, int]` 这类新旧语法（Python 3.9 也能解析）；
#   2) 注解里引用还没定义的类（前向引用）不会抛 NameError。
# 注意：它必须是模块文档字符串之后的第一个语句（注释不算语句，但为了清晰别插东西）。

import json  # 读 meta.json、写数据集的元信息；JSON 是纯文本，便于人眼检查
from pathlib import Path  # 用「路径对象」代替字符串拼接：支持 / 运算符、跨平台
from typing import Dict, Tuple  # 仅用于类型注解，运行时零开销

import numpy as np  # .npy 的读写、bincount 统计都靠它
import torch
import torch.nn.functional as F  # 这里只用到 F.pad 一个函数式接口
from torch.utils.data import DataLoader, TensorDataset

# CIFAR-10 训练集的逐通道均值/标准差（RGB 三个数）。
# 注意顺序是 RGB，和图像的 C 维一致；标准化时必须用同一个顺序，不能拿 BGR 的常数。
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

# CIFAR-10 固定 10 类。代码里多处用它做「结构校验」和「数组长度」，
# 所以抽成常量，避免魔法数字散落各处。
NUM_CLASSES = 10

# 类别编号 0~9 对应的英文名。顺序必须与数据集文件夹名（0,1,...,9）一致，
# 否则画混淆矩阵、写报告时标签会整体错位。
CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def _find_split(root: Path, kind: str) -> Path:
    """在解压后的数据集目录里，定位 train 或 test 的「类别文件夹的父目录」。

    参数
    ----
    root : Path
        用户提供的解压根目录。实际压缩包可能一层套一层，例如
        `CIFAR10_balanced/CIFAR10_balance/0..9`，所以不能写死路径。
    kind : str
        只接受 "train" 或 "test"，决定返回哪一份划分。

    返回
    ----
    Path
        一个「直接子目录恰好是 0~9 十个纯数字目录」的目录。
        注意返回的是父目录，真正的图片在 `<返回值>/<类别号>/*.jpg`。

    关键设计原因
    ------------
    课程压缩包的层级并不统一（有的多套一层同名目录，balanced / unbalance 混在
    一起）。这里采用「先罗列所有可能的候选路径，再用目录结构去验证」的策略：
    只有当一个候选目录下恰有 10 个纯数字子目录时，才认为它是某个划分。
    这比写死一个路径更鲁棒，学生换一版数据也不至于直接报错。
    """
    candidates = []
    # 把常见的几种命名都试一遍：每个名字都考虑 root/name 与 root/name/name
    # 两种层级（后者对应「多套一层同名文件夹」的压缩包）。
    for name in ("CIFAR10_balance", "CIFAR10_balanced", "CIFAR10_unbalance", "CIFAR10_imbalanced"):
        candidates.extend([root / name, root / name / name])
    # The balanced archive nests as CIFAR10_balanced/CIFAR10_balance.
    # 上面这行是原作者的说明：balanced 压缩包会把同名目录再套一层。
    # `.glob("*")` 返回的是「生成器」，`list.extend` 会惰性消费它，
    # 所以这里等于把该目录下的所有一级子项都追加为候选（可能包含多余项，
    # 后面统一用结构校验过滤掉，不会误选）。
    candidates.extend((root / "CIFAR10_balanced" / "CIFAR10_balance").glob("*"))
    candidates.append(root / "CIFAR10_balanced" / "CIFAR10_balance")

    for cand in candidates:
        # 候选里可能有不存在/不是目录的项（例如 glob 出来的文件），先跳过。
        if not cand.is_dir():
            continue
        # 列表推导式：一行完成「遍历 + 过滤」。
        # `d.name.isdigit()` 只保留纯数字目录名（0~9），这样同级的 README、
        # __MACOSX 之类会被自动忽略，不用手写一堆排除条件。
        class_dirs = [d for d in cand.iterdir() if d.is_dir() and d.name.isdigit()]
        # 结构校验：不是「恰好 10 个类别目录」就不是我们要的划分，换下一个候选。
        # 用 != 而不是 <，是因为多出类别目录同样说明结构不对。
        if len(class_dirs) != NUM_CLASSES:
            continue
        # 统一转成小写、并换成 posix 风格（/）再匹配子串，
        # 避免 Windows 的反斜杠或大小写差异导致 "unbalanc" 匹配失败。
        path_lower = cand.as_posix().lower()
        name_lower = cand.name.lower()
        # train 划分来自 unbalance 的那份数据（课程设定：训练集长尾、测试集均衡）。
        if kind == "train" and "unbalanc" in path_lower:
            return cand
        # test 划分要的是 balanced 的那份；再排除 "unbalanc"，防止
        # "CIFAR10_unbalance" 里也含 "balanc" 子串而被误判成 balance。
        if kind == "test" and "balanc" in name_lower and "unbalanc" not in path_lower:
            return cand
    # 所有候选都不符合结构：明确报错，而不是返回 None 让调用方在后面莫名其妙崩掉。
    raise FileNotFoundError(f"Could not locate {kind} split under {root}")


def _read_image(path: Path) -> np.ndarray:
    """用 PIL 读一张 JPEG，并转成 RGB 的 (32,32,3) uint8 数组。

    参数
    ----
    path : Path
        单张图片的文件路径。

    返回
    ----
    np.ndarray
        shape=(32,32,3), dtype=uint8。注意是 HWC 顺序，与磁盘上的 NHWC 约定一致。

    关键设计原因
    ------------
    `from PIL import Image` 放在函数内部是「惰性导入」：只有真正去读图片
    （也就是跑预处理脚本）时才需要 PIL。训练/推理阶段直接读 .npy，就不必
    依赖 Pillow，启动也更快。

    `img.convert("RGB")` 是必需的：数据集里可能有灰度图或带 alpha 通道的 PNG，
    不统一成 3 通道，后面 `np.stack` 就会因为形状不一致而失败。
    `with ... as img` 保证文件句柄一定被关闭（Windows 上文件被占用会删不掉）。
    """
    from PIL import Image

    with Image.open(path) as img:
        # np.asarray 不复制像素数据（对 PIL 对象是零拷贝读取），
        # dtype=np.uint8 显式声明，避免不同版本 PIL 返回不同类型。
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def scan_dataset(root: str | Path) -> Tuple[Path, Path, Dict[str, Dict[str, int]]]:
    """扫描数据集目录，返回两个划分的路径以及每个类别的图片数量。

    参数
    ----
    root : str | Path
        数据集根目录。允许传字符串，函数内部第一件事就是 `Path(root)` 包一层，
        这样调用方不用关心类型（这也是 `str | Path` 注解的含义：两种都接受）。

    返回
    ----
    Tuple[Path, Path, Dict[str, Dict[str, int]]]
        (train_dir, test_dir, counts)，其中
        counts = {"train": {"0": 5000, ..., "9": 500}, "test": {...}}。
        注意类别的 key 是**字符串**，因为后面要写进 JSON——JSON 的键只能是字符串。

    关键设计原因
    ------------
    长尾实验需要先知道每个类有多少张图，才能算 class weights 和画分布图。
    在预处理阶段顺手统计一次，比训练时再扫一遍磁盘便宜得多。
    """
    # 把 str 统一成 Path，之后就能用 `/` 拼路径、用 `.iterdir()` 遍历。
    root = Path(root)
    train_dir = _find_split(root, "train")
    test_dir = _find_split(root, "test")
    counts: Dict[str, Dict[str, int]] = {}
    # 用 (名字, 路径) 元组来同时遍历两个划分，避免把同样的统计代码写两遍。
    for split, path in (("train", train_dir), ("test", test_dir)):
        counts[split] = {}
        for cls in range(NUM_CLASSES):
            # glob 返回生成器，必须 `list(...)` 包一层才能取长度；
            # 直接 len(glob(...)) 会报 TypeError（生成器没有长度）。
            counts[split][str(cls)] = len(list((path / str(cls)).glob("*.jpg")))
    return train_dir, test_dir, counts


def build_arrays(root: str | Path, out_dir: str | Path, workers: int = 16) -> None:
    """把「每类一个文件夹」的原始图片，转换成紧凑的 .npy 数组（NHWC uint8）。

    参数
    ----
    root : str | Path
        原始数据集根目录（含 CIFAR10_xxx 之类的子目录）。
    out_dir : str | Path
        输出目录；会写入 train_images.npy / train_labels.npy /
        test_images.npy / test_labels.npy / meta.json。
    workers : int
        线程池大小，默认 16。JPEG 解码在 C 层会释放 GIL，所以多线程确实能加速；
        但线程数超过 CPU 核数收益很小，反而增加内存峰值。

    返回
    ----
    None
        结果全部落盘（副作用型函数），不返回对象。

    关键设计原因
    ------------
    这一步是「用磁盘空间换训练速度」：把 4 万张 JPEG 一次性解码成一个大数组，
    以后每个 epoch 只做内存拷贝，不再有解码开销。uint8 而不是 float32，
    是因为图像原始精度就是 0~255，存成 float32 会白占 4 倍内存且损失不了信息。

    输出数组的形状是 **NHWC**（N,32,32,3），和 PIL 读出来的 HWC 保持一致，
    因此 `np.stack` 不需要任何转置；转成 NCHW 的事情留给 augment_batch。
    """
    # 线程池只在「预处理」这个场景需要，放函数内导入可以避免训练时无谓的依赖。
    from concurrent.futures import ThreadPoolExecutor

    train_dir, test_dir, counts = scan_dataset(root)
    out_dir = Path(out_dir)
    # parents=True 等价于 mkdir -p，逐级创建；exist_ok=True 表示目录已存在也不报错，
    # 这样脚本重复运行是安全的（幂等）。
    out_dir.mkdir(parents=True, exist_ok=True)

    for split, path in (("train", train_dir), ("test", test_dir)):
        # 这里刻意用「两个平行列表」而不是一个 [(file, label), ...] 列表：
        # 后面 pool.map 只接收文件列表，标签顺序靠「先按类循环、再按文件名排序」
        # 来保证与图像顺序严格一一对应。这个不变量一旦破坏，标签就全错位了。
        files = []
        labels = []
        for cls in range(NUM_CLASSES):
            # sorted 是为了让「同一份数据每次转换得到完全相同的顺序」，保证可复现。
            cls_files = sorted((path / str(cls)).glob("*.jpg"))
            files.extend(cls_files)
            # 列表乘法：[cls] * n 生成 n 个相同标签，再一次性接在 labels 后面。
            # 长度严格等于该类的图片数，因此和 files 完全对齐。
            labels.extend([cls] * len(cls_files))
        # ThreadPoolExecutor 用上下文管理器，退出时会 wait 所有任务并回收线程。
        # chunksize=64：每个线程一次领 64 个任务，减少任务分发（锁）的开销。
        # map 的结果顺序与输入顺序一致（不是完成顺序），所以图像和标签仍然对齐。
        with ThreadPoolExecutor(max_workers=workers) as pool:
            images = list(pool.map(_read_image, files, chunksize=64))
        # np.stack 沿着新的第 0 维堆叠：(32,32,3) × N → (N,32,32,3)。
        # 比 np.array 更明确，且能尽早发现形状不一致的图片。
        arr = np.stack(images, axis=0)
        # 形状校验：CIFAR-10 必须是 32x32 的彩色图。这里用 arr.shape[1:] 跳过 N，
        # 一旦数据集被换成别的东西，立刻报错而不是训练几十轮后才发现。
        if arr.shape[1:] != (32, 32, 3):
            raise ValueError(f"Unexpected image shape in {split}: {arr.shape}")
        # np.save 会写入 .npy 头（dtype、shape、C 顺序），文件本身是一整块数组；
        # 注意文件名若不以 .npy 结尾会自动补上，所以这里显式带上后缀。
        np.save(out_dir / f"{split}_images.npy", arr)
        # 标签用 int64：既是 PyTorch 交叉熵默认接受的整型，也避免 uint8 溢出。
        np.save(out_dir / f"{split}_labels.npy", np.asarray(labels, dtype=np.int64))
        print(f"{split}: {arr.shape[0]} images, counts={counts[split]}")

    # meta.json 记录「数据从哪来、每类多少张、类别名、图像形状」，
    # 让下游（训练脚本、报告生成）不必回头再猜目录结构。
    meta = {
        "train_dir": str(train_dir),  # Path 不能直接 JSON 序列化，先转成字符串
        "test_dir": str(test_dir),
        "counts": counts,
        "class_names": list(CLASS_NAMES),  # tuple 会被 json 当成数组，但显式转 list 更清楚
        "image_shape": [32, 32, 3],
    }
    # ensure_ascii=False：让中文/非 ASCII 路径原样写入，而不是变成 \uXXXX；
    # 必须显式指定 encoding="utf-8"，否则 Windows 默认用 cp936，可能写坏或读错。
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


class ArrayDataset(TensorDataset):
    """把 uint8 图像数组 + int64 标签数组包成最朴素的 (image, label) 数据集。

    继承自 `TensorDataset`：父类已经把「按下标同时索引多个 tensor」的逻辑写好了，
    所以这里只需要在 `__init__` 里把 numpy 数组转成 torch tensor 即可。

    参数
    ----
    images : np.ndarray
        shape=(N,32,32,3), dtype=uint8。注意保持 NHWC，标准化放到 batch 里做。
    labels : np.ndarray
        shape=(N,), dtype=int64。

    返回（通过 `__getitem__`）
    ----
    Tuple[torch.Tensor, torch.Tensor]
        (image, label)，image 是 (32,32,3) uint8，label 是 0 维 int64 tensor。
        这是「两元组」，训练循环里用 `batch[0], batch[1]` 取用。
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray):
        """用 numpy 数组构造数据集。

        参数
        ----
        images : np.ndarray
            NHWC 的 uint8 图像批，shape=(N,32,32,3)。
        labels : np.ndarray
            每个样本的类别编号，shape=(N,)，取值 0~9。

        关键设计原因
        ------------
        `torch.from_numpy` 与 numpy 数组**共享内存**（零拷贝），所以这里几乎不花时间、
        也不额外占内存。代价是：之后不要再去原地修改那个 numpy 数组，否则张量里的
        数据会跟着变——在本项目里数组读进来就只读，所以是安全的。

        为什么不用 float：uint8 存着省 4 倍内存，标准化/增广本来就要在 batch 上
        重新算一遍，提前转 float 没有收益。
        """
        # super().__init__ 把两个 tensor 注册进父类的 `tensors` 元组，
        # 父类的 __getitem__ 会对每个 tensor 取同一行，从而保证图文对齐。
        super().__init__(torch.from_numpy(images), torch.from_numpy(labels))


class IndexedArrayDataset(torch.utils.data.Dataset):
    """在一个 uint8 数组上的「视图」数据集，不复制图像数据，并额外返回原始索引。

    与 `ArrayDataset` 的区别有两点：
    1) 它持有 `indices`（例如训练集/验证集的下标切片），所以同一份 `images`
       可以被 train / val 两个 Dataset 共享，内存只有一份；
    2) `__getitem__` 返回的是 **三元组** (image, label, original_index)。

    参数
    ----
    images : np.ndarray
        完整图像数组，shape=(N,32,32,3), uint8。
    labels : np.ndarray
        完整标签数组，shape=(N,)。
    indices : np.ndarray
        本数据集实际用到的「原始下标」，shape=(M,)，M ≤ N。

    `__getitem__` 的返回
    ----
    Tuple[torch.Tensor, int, int]
        (image(32,32,3) uint8, label(int), original_index(int))。

    关键设计原因
    ------------
    为什么非要返回「原始索引」？因为知识蒸馏需要按样本查教师模型预先算好的 logits：
    教师 logits 是一个 (N, C) 的大张量，存的时候是按**全集**顺序存的；而训练时
    一个 batch 里的样本经过了划分（train/val 切分）和 shuffle，顺序完全打乱。
    只有把「这个样本在全集里的原始下标」一起带出来，才能写
    `teacher_logits[indices]` 精确取到对应的那一行。若只返回 image/label，
    蒸馏时就得靠猜或者额外维护映射，极易错位且难调试。
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray, indices: np.ndarray):
        """记录共享的数组和本视图使用的下标。

        参数
        ----
        images : np.ndarray
            全集图像，shape=(N,32,32,3), uint8；本类只读、不复制。
        labels : np.ndarray
            全集标签，shape=(N,)；与 images 的第 0 维一一对应。
        indices : np.ndarray
            本视图用到的原始下标，shape=(M,)。传 list 也可以，内部统一转 int64，
            因为后面要做 numpy 高级索引，索引数组必须是整型。

        关键设计原因
        ------------
        三个数组都是「引用」，构造 Dataset 几乎零开销。这样一来，train 和 val
        可以共享同一份 4 万张图的数组（只需要两个不同的 indices），内存占用不会翻倍。
        """
        self.images = images
        self.labels = labels
        # 统一成 int64：numpy 的高级索引要求整型；顺便把 list/tuple 也接受进来。
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        """返回本视图的样本数 M（而不是全集大小 N）。

        返回
        ----
        int
            有效样本数，即 `len(self.indices)`。DataLoader 靠它计算
            「一个 epoch 有多少个 batch」（ceil(M / batch_size)）。
        """
        # 注意：这里返回的是 indices 的长度，不是 images 的长度——
        # 这正是 train/val 能从同一份数组里切出不同子集的机制。
        return len(self.indices)

    def __getitem__(self, i: int):
        """按「视图内位置」i 取出样本，并附上它在全集中的原始下标。

        参数
        ----
        i : int
            视图内的位置，0 ≤ i < M。DataLoader 会用它（或一组它）来取样。

        返回
        ----
        Tuple[torch.Tensor, int, int]
            (image, label, original_index)：
            image  shape=(32,32,3), uint8（与 numpy 数组共享内存的视图）；
            label  是 Python int（不是 tensor，DataLoader 的默认 collate 会把它拼成
                   int64 tensor）；
            original_index 是全集下标，供蒸馏查 teacher_logits 用。

        关键设计原因
        ------------
        这里刻意返回 Python `int` 而不是 0 维 tensor：默认的 `collate_fn` 遇到
        int 会自动堆成 int64 tensor，且不会像 tensor 那样多一次包装；
        对 label 来说这样最省事。而索引也必须是 int，才能直接拿去做张量索引。
        """
        # 先把「视图内位置」翻译成「全集下标」，这是本类存在的核心意义。
        idx = int(self.indices[i])
        # images[idx] 是 numpy 的整数索引：取出 (32,32,3) 的子数组（视图，不拷贝）；
        # torch.from_numpy 再把它零拷贝地变成 tensor。
        # labels[idx] 取标量，转成 Python int 后交给 collate 拼 batch。
        return torch.from_numpy(self.images[idx]), int(self.labels[idx]), idx


def load_arrays(data_dir: str | Path) -> Tuple[ArrayDataset, ArrayDataset, Dict]:
    """从磁盘读回预处理好的数组，组装成 (训练集, 测试集, 元信息)。

    参数
    ----
    data_dir : str | Path
        存放 *_images.npy / *_labels.npy / meta.json 的目录。

    返回
    ----
    Tuple[ArrayDataset, ArrayDataset, Dict]
        (train, test, meta)：
        train/test 是返回 (image, label) 两元组的 Dataset；
        meta 是 meta.json 解析出来的普通字典（含 counts、class_names 等）。

    关键设计原因
    ------------
    `np.load` 默认会把整个文件读进内存（不解压、不惰性），对 CIFAR-10 这个体量
    （训练集约 150MB uint8）完全可以接受；换来的是之后访问极快。
    若数据再大几个数量级，可以改成 `mmap_mode="r"` 做内存映射，但本课程不需要。

    元信息用 `json.loads(Path.read_text(...))` 读取，等价于 `json.load(open(...))`，
    好处是不用自己管文件句柄；显式 `encoding="utf-8"` 是 Windows 上的必备习惯。
    """
    data_dir = Path(data_dir)
    # 每个 Dataset 各自加载一份数组（训练/测试本来就是两份不同的数据）。
    # 注意 load 出来的数组是 NHWC uint8，Dataset 内部不再做任何转换。
    train = ArrayDataset(np.load(data_dir / "train_images.npy"), np.load(data_dir / "train_labels.npy"))
    test = ArrayDataset(np.load(data_dir / "test_images.npy"), np.load(data_dir / "test_labels.npy"))
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    return train, test, meta


def class_counts(labels: np.ndarray | torch.Tensor, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """统计每个类别的样本数，用于长尾分析、class weights 和分布图。

    参数
    ----
    labels : np.ndarray | torch.Tensor
        shape=(N,) 的整数标签，取值应为 0~num_classes-1。两种类型都行——
        训练脚本手里可能是 tensor，直接用 numpy 版会更麻烦。
    num_classes : int
        类别总数，默认 10。它同时决定返回数组的长度。

    返回
    ----
    np.ndarray
        shape=(num_classes,), dtype=int64；第 c 个元素是「标签为 c 的样本数」。

    关键设计原因
    ------------
    用 `np.bincount` 而不是 `for c in range(C): (labels == c).sum()`：
    bincount 只扫一遍数组、用计数排序的思想直接累加，复杂度 O(N)；
    而循环写法是 O(N·C)，类别一多就慢一个量级。

    `minlength=num_classes` 很关键：如果某个类别一个样本都没有（长尾实验里很常见），
    bincount 默认只输出到「出现过的最大标签」为止，长度会不足 C，
    后面和 class weights、混淆矩阵对齐时就会错位。加上 minlength 保证长度恒为 C。
    """
    # Tensor 先搬到 CPU 再转 numpy：GPU 上的张量不能直接 .numpy()；
    # .cpu() 对本来就在 CPU 上的张量是空操作，所以这么写对两种设备都安全。
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    # astype(int64) 是必须的：bincount 要求非负整数输入，
    # 若 labels 是 float（例如从某处算出来的）会直接报错。
    return np.bincount(labels.astype(np.int64), minlength=num_classes)


def make_loader(
    dataset: TensorDataset,
    batch_size: int,
    shuffle: bool,
    seed: int = 0,
    drop_last: bool = False,
) -> DataLoader:
    """构造一个「随机性可控」的 DataLoader。

    参数
    ----
    dataset : TensorDataset
        任意符合 Dataset 协议的对象（这里类型标注写的是 TensorDataset，
        但 IndexedArrayDataset 也能传，因为它同样实现了 __len__/__getitem__）。
    batch_size : int
        每个 batch 的样本数。
    shuffle : bool
        是否在每个 epoch 打乱顺序。训练集应为 True，验证/测试集应为 False
        （不打乱才能让预测结果与样本顺序一一对应，方便对照分析）。
    seed : int
        shuffle 用的随机种子，默认 0。
    drop_last : bool
        是否丢掉最后那个不满 batch。默认 False；训练时若模型含 BatchNorm，
        丢掉不完整的尾批能避免「最后一批统计量异常」，本项目是 ViT（用 LayerNorm），
        所以保持 False 也没问题。

    返回
    ----
    DataLoader
        迭代一次产出 `batch_size` 个样本拼成的 batch；
        对 IndexedArrayDataset，每个 batch 是 3 个 tensor（图像、标签、索引）。

    关键设计原因
    ------------
    为什么要显式传入 `torch.Generator`？因为 DataLoader 的 `shuffle=True` 默认用
    **全局**随机数生成器，而全局 RNG 会被模型初始化、Dropout、数据增强等各处消耗，
    同一个 seed 跑两次，shuffle 的顺序可能不同——实验就不可复现了。

    给 DataLoader 一个「独立的生成器」并固定它的种子后，只要 seed 相同，
    每个 epoch 的 batch 划分就完全一致，与模型内部用了多少次随机数无关。

    `num_workers=0`：数据在**主进程**里加载。本项目的样本就是内存数组，
    开多进程反而要付进程间拷贝的代价；而且 num_workers>0 时每个 worker 有自己的
    随机状态，复现会更麻烦。`pin_memory=False` 同样是因为本项目只跑 CPU，
    锁页内存没有意义。
    """
    # 创建独立生成器并固定种子：它只服务于这个 DataLoader 的 shuffle，
    # 因此不会被训练循环里其他随机操作「抢走」随机数。
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=drop_last,
        generator=generator,
        pin_memory=False,
    )


def _normalize(x: torch.Tensor) -> torch.Tensor:
    """对已经是 NCHW、float 的图像张量做逐通道标准化。

    参数
    ----
    x : torch.Tensor
        shape=(N,3,H,W) 的 float 张量，取值范围应已在 [0,1]。
        函数假定调用方已经做过 permute 和 /255。

    返回
    ----
    torch.Tensor
        shape 与输入相同，逐通道执行 (x - mean) / std 后的新张量。

    关键设计原因
    ------------
    `x.new_tensor(...)` 会创建一个和 x **同 dtype、同 device** 的张量。
    这比 `torch.tensor(CIFAR_MEAN)` 更安全：后者默认是 CPU float32，
    一旦 x 在 GPU 上就会触发跨设备运算错误；用 new_tensor 就不用写 if 判断。

    `.view(1, 3, 1, 1)` 把长度为 3 的向量变成 (1,3,1,1)，这是为了**广播**：
    与 (N,3,H,W) 相减/相除时，N、H、W 三个维度自动扩展，只有通道维一一对应。
    广播不实际复制数据，所以既省内存又快。
    """
    # 形状 (1,3,1,1) 而非 (3,)：这样正好对齐 NCHW 的 C 维，无需再 unsqueeze。
    mean = x.new_tensor(CIFAR_MEAN).view(1, 3, 1, 1)
    std = x.new_tensor(CIFAR_STD).view(1, 3, 1, 1)
    # 普通除法生成新张量，不修改输入；调用方若还需要原始值也不会被破坏。
    return (x - mean) / std


def augment_batch(images: torch.Tensor, erase_p: float = 0.25) -> torch.Tensor:
    """对一个 uint8 NHWC 的图像 batch 做「向量化」随机裁剪 + 水平翻转 + Cutout。

    参数
    ----
    images : torch.Tensor
        shape=(B,32,32,3), dtype=uint8。B 是 batch 大小，NHWC 是磁盘上的存储顺序。
    erase_p : float
        每张图被 Cutout（随机擦除一块）的概率，默认 0.25。
        设为 0 可以完全关闭擦除，便于做消融实验。

    返回
    ----
    torch.Tensor
        shape=(B,3,32,32), dtype=float32，已标准化（均值 0、方差 1）。
        注意形状从 NHWC 变成了 NCHW，这是 PyTorch 模型要求的输入格式。

    关键设计原因
    ------------
    逐张调用 `torchvision.transforms` 意味着每个样本一次 Python 函数调用、一次
    随机数生成，一个 epoch 下来是几十万次，CPU 上非常可观。
    这里把整个 batch 当成一个张量来操作：
      - 随机裁剪：用「整数索引 + gather」一次性完成所有样本、所有通道的取样；
      - 水平翻转：先抽一个布尔掩码，再只对需要翻转的那些图做 flip；
      - Cutout：每张图的擦除框大小/位置都不同，只能循环，但通常只有 1/4 的样本
        命中，循环次数很少。

    另一个重点：所有随机数来自**全局 RNG**（`torch.randint` / `torch.rand`），
    所以增强的随机性由 `engine.set_seed` 统一控制；这也意味着只要 seed 相同、
    调用顺序相同，增强结果就可复现。
    """
    b = images.shape[0]
    # permute(0,3,1,2)：把 (B,H,W,C) 重排成 (B,C,H,W)——只换轴顺序，是视图操作。
    # .float() 把 uint8 变成 float32（这一步会真正复制一份，之后就可以安全地 in-place）。
    # .div_(255.0)：带下划线的版本是「原地」运算，把像素映射到 [0,1]。
    #   之所以能原地改，是因为 .float() 已经产生了一个新张量，
    #   不是在改调用方传进来的 uint8 原图（改原图会让下一个 epoch 的数据被污染）。
    x = images.permute(0, 3, 1, 2).float().div_(255.0)
    # F.pad 的 padding 元组顺序是「从最后一维往前数」：(left, right, top, bottom)。
    # 所以 (4,4,4,4) 表示左右各补 4 列、上下各补 4 行：32 → 40。
    # mode="reflect" 是镜像填充（边缘像素按镜面反射复制），比补 0 更接近自然图像，
    # 也让裁剪到边缘时不会突然出现一大块黑边。
    x = F.pad(x, (4, 4, 4, 4), mode="reflect")

    # 随机裁剪的起点：在 40x40 的填充图上取左上角 (y0, x0)。
    # torch.randint(0, 9, ...) 的上界是「开区间」，所以取值 0~8，
    # 加上 32 正好覆盖到 39，保证裁剪窗口不越界。
    y0 = torch.randint(0, 9, (b,))
    x0 = torch.randint(0, 9, (b,))

    # ---- gather 批量裁剪的索引原理（本文件最绕的一段，值得慢慢读）----
    # 目标：对每个样本、每个通道，都要从 40x40 的图里抠出 (y0..y0+31, x0..x0+31)。
    # 做法是把 40x40 拉平成长度 1600 的一维，再按「一维下标」去 gather。
    #   一维下标 = 行号 * 40 + 列号
    # `ys` 的 shape 是 (B,32,1)：第 b 个样本的第 i 行，源行号 = y0[b] + i。
    # `xs` 的 shape 是 (B,1,32)：第 j 列，源列号 = x0[b] + j。
    # 两者相加时触发**广播**，(B,32,1) + (B,1,32) → (B,32,32)，
    # 于是每个 (i,j) 位置都得到了自己的源下标 ys*40+xs。
    #
    # 关键一步：`(ys * 40 + xs)` 的形状是 (B,32,32)，只有「空间」两维；
    # 但 gather 需要每个通道有一份（三个通道的取样位置完全相同），
    # 所以先 unsqueeze(1) 变成 (B,1,32,32)，再 expand 到 (B,3,32,32)。
    # expand 不复制数据（只是把 stride 设为 0），比 repeat 省 3 倍内存。
    # 最后 reshape(b, 3, -1) 把每个通道摊平成 1024 个下标；
    # 因为 expand 后的张量不连续，reshape 内部会自动做一次拷贝，这没关系。
    ys = y0[:, None, None] + torch.arange(32)[None, :, None]
    xs = x0[:, None, None] + torch.arange(32)[None, None, :]
    idx = (ys * 40 + xs).unsqueeze(1).expand(b, 3, 32, 32).reshape(b, 3, -1)
    # torch.gather(输入, dim, 索引)：out[b,c,k] = 输入[b,c, idx[b,c,k]]。
    # 即在 dim=2（展平后的空间维）上，按 idx 逐个位置取像素。
    # 输入要先 reshape(b,3,-1) 成 (B,3,1600)，gather 完再 reshape 回 (B,3,32,32)。
    x = torch.gather(x.reshape(b, 3, -1), 2, idx).reshape(b, 3, 32, 32)

    # 水平翻转：torch.rand(b) 为每张图抽一个 [0,1) 的均匀随机数，
    # 小于 0.5 就翻（50% 概率）。得到一个 bool 掩码。
    flip = torch.rand(b) < 0.5
    # `if flip.any()` 是一个小优化：如果这一批恰好一张都不翻，就跳过整个切片操作。
    # 用布尔掩码 x[flip] 做「高级索引」，会先取出被选中的那些图（拷贝一份），
    # 对副本 .flip(-1)（沿最后一维 W 镜像），再赋值写回原张量。
    # 注意 `flip` 返回的是新张量（PyTorch 不支持负 stride 视图，所以不是原地操作），
    # 这也是为什么必须写成 `x[flip] = x[flip].flip(-1)` 而不是 `x[flip].flip(-1)`——
    # 后者算完就丢掉了，等于什么都没做（初学者常踩的坑）。
    if flip.any():
        x[flip] = x[flip].flip(-1)

    # Cutout：按概率在图上贴一个随机大小的黑色矩形，强迫模型别只盯着局部特征。
    if erase_p > 0:
        # 先一次性抽 mask：哪些图要做擦除。erase_p=0.25 表示平均每 4 张擦 1 张。
        do_erase = torch.rand(b) < erase_p
        # torch.nonzero(...).flatten().tolist() 把「所有 True 的位置」变成 Python 列表。
        # as_tuple=False 保证返回的是 (K,1) 的二维下标张量；
        # 因为我们只有一维，所以 flatten 成 (K,) 再用 tolist 转成 Python int 列表。
        # 顺带对照一下 numpy 的等价写法：`np.flatnonzero(mask)` 直接返回布尔数组里
        # 所有 True 的一维下标（比 np.nonzero(mask)[0] 更简洁）。
        # 训练脚本按类别切分 train/val 时就是这么用的：
        # `idx = np.flatnonzero(labels == cls)` 取出「标签等于 cls 的全部样本下标」。
        # 之所以要走 Python 循环，是因为每张图的矩形大小、位置都不一样，
        # 无法用一个统一的索引张量表达——好在命中的样本通常只占少数。
        for i in torch.nonzero(do_erase, as_tuple=False).flatten().tolist():
            # 擦除块的高/宽：在 [4, 10] 之间随机（randint 上界 11 是开区间）。
            h = int(torch.randint(4, 11, (1,)).item())
            w = int(torch.randint(4, 11, (1,)).item())
            # 左上角坐标：0 ~ 32-h（含），保证矩形完全落在图内，不会越界。
            top = int(torch.randint(0, 32 - h + 1, (1,)).item())
            left = int(torch.randint(0, 32 - w + 1, (1,)).item())
            # 切片赋值把这块区域的所有通道都置 0。
            # 注意此时 x 还是 [0,1] 的原始像素，填 0 等于填「纯黑」；
            # 标准化是在所有增强之后统一做的，所以这里的 0 是像素意义上的黑，
            # 而不是「标准化后的均值」——顺序不能反，反了就会擦出一块灰色。
            x[i, :, top : top + h, left : left + w] = 0.0

    # 最后统一标准化：此时 x 已是 (B,3,32,32) 的 float。
    return _normalize(x)


def normalize_batch(images: torch.Tensor) -> torch.Tensor:
    """把 uint8 NHWC 的图像 batch 转成「标准化后的 float NCHW」，不做任何随机增强。

    参数
    ----
    images : torch.Tensor
        shape=(B,32,32,3), dtype=uint8（也可以已经是 float 的 NHWC，只要值在 0~255）。

    返回
    ----
    torch.Tensor
        shape=(B,3,32,32), dtype=float32，已按 CIFAR-10 均值方差标准化。

    关键设计原因
    ------------
    验证/测试必须用**确定性**的预处理：只有「转置 + 缩放 + 标准化」，
    没有随机裁剪、翻转、擦除。否则同一张图两次评估喂进模型的内容不同，
    指标会莫名其妙地抖动，也没法和别人的结果比较。

    这里和 `augment_batch` 共享同一套 `permute → /255 → _normalize` 流程，
    保证训练和评估的输入分布在「几何/颜色」这一层面是一致的
    （训练多了随机增广，但基础变换完全相同）。
    """
    # 一次链式调用完成三件事：NHWC→NCHW、uint8→float32、[0,255]→[0,1]。
    # div_ 能原地做，是因为 .float() 已经生成了新张量。
    x = images.permute(0, 3, 1, 2).float().div_(255.0)
    return _normalize(x)


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """把标准化过的图像张量还原成 [0,1] 的可视化范围。

    参数
    ----
    x : torch.Tensor
        shape=(N,3,H,W) 的 float 张量，即 `_normalize` 的输出。

    返回
    ----
    torch.Tensor
        shape 相同，数值被裁剪到 [0,1]，可以直接 `imshow` 显示。

    关键设计原因
    ------------
    标准化是 `(x-mean)/std`，逆运算就是 `x*std + mean`。做完之后数值可能在
    [0,1] 之外（因为原始像素本来只在 0~1，但浮点误差或增强会带来轻微越界），
    所以最后 `clamp(0,1)`，否则 matplotlib 显示时会出现怪异的饱和色。

    同样用 `x.new_tensor(...)` 保证 mean/std 与输入同 dtype、同 device，
    避免「CPU 常数 × GPU 张量」这类设备不匹配的报错。
    """
    # 与 _normalize 完全对称：先乘 std 再加 mean。
    mean = x.new_tensor(CIFAR_MEAN).view(1, 3, 1, 1)
    std = x.new_tensor(CIFAR_STD).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0, 1)
