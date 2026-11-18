"""训练单个 CIFAR-10 ViT 变体，并把结果落盘成可复现的实验记录。

[脚本做什么]
    一次调用只训练**一个**模型（vit / dynamic / mobile 三选一），完整流程是：
        读 npy 缓存 -> 分层切出验证集 -> 建模型/损失/优化器/学习率调度器
        -> 逐 epoch 训练 + 验证 -> 按验证准确率保留最佳权重
        -> 用最佳权重在测试集上评估一次 -> 写 JSON / checkpoint / 日志。
    设计上刻意做成"一模型一进程"：这样每个实验的随机种子、线程数、内存占用
    互不干扰，失败重跑也只影响一个 tag。

[输入]
    数据（由 scripts/prepare_data.py 生成，路径写死为 <项目根>/data）：
        train_images.npy / train_labels.npy / test_images.npy / test_labels.npy / meta.json
    命令行：见 parse_args()。最关键的三个是 --model、--loss、--tag，
        其中 --tag 决定所有输出文件名，必须每次实验唯一。
    可选输入：--distill-logits 指向教师模型在**完整训练集**上的 logits（.pt），
        由另一个实验加 --export-logits 导出，用于知识蒸馏。
        --resume 指向上一次的 checkpoints/<tag>_last.pt，用于断点续训。

[输出]（都在 <项目根>/ 下）
    results/<tag>.log                训练日志（print 的同时写文件并 flush）
    results/<tag>.json               配置、超参、每 epoch 指标、测试指标、耗时
    checkpoints/<tag>_last.pt        每个 epoch 覆盖写，供 --resume 使用
    checkpoints/<tag>.pt             训练结束后保存的"最佳验证准确率"权重，
                                     供 measure_complexity.py / analyze.py 使用
    checkpoints/<tag>_train_logits.pt  仅 --export-logits 时生成（教师 logits）
    results/<tag>_test_pred.npy      仅 --save-predictions 时生成（测试集预测）
    results/<tag>_test_labels.npy    仅 --save-predictions 时生成（测试集标签）

[在实验流程中的位置]
    prepare_data.py 之后。典型顺序（对应 README）：
        baseline_ce  -> baseline_cb（加 --export-logits 当教师）
        -> dynamic_cb / mobile_cb（用 --distill-logits 跟教师学）
        -> measure_complexity.py / dynamic_tokens.py -> analyze.py -> build_report.py
    为什么顺序重要：蒸馏需要教师的 logits 文件先存在；而"基线 -> 轻量化"的顺序
    保证轻量模型能公平地对比同一教师。

[怎么运行]
    cd vit_cifar10_lab
    python scripts/train.py --model vit --loss ce --tag baseline_ce --epochs 20 --save-predictions
    python scripts/train.py --model vit --loss cb --tag baseline_cb --epochs 20 \
        --export-logits --save-predictions
    python scripts/train.py --model dynamic --loss cb --tag dynamic_cb --epochs 20 \
        --prune-after 0,1,2 --keep-ratios 0.75,0.5,0.25 \
        --distill-logits checkpoints/baseline_cb_train_logits.pt --save-predictions
    python scripts/train.py --model mobile --loss cb --tag mobile_cb --epochs 20 \
        --dim 72 --mobile-depth 2 --global-depth 1 --mobile-patch 4 \
        --distill-logits checkpoints/baseline_cb_train_logits.pt --save-predictions
    断点续训：给同一条命令再加 --resume checkpoints/<tag>_last.pt
    快速冒烟（几十秒跑通流程）：加 --limit-train 2000 --max-batches 5 --epochs 1

[阅读前需要知道的概念]
    * 分层抽样（stratified split）：按类别分别抽验证集，保证小类也有验证样本。
      若直接随机切 10%，某些小类可能只分到个位数验证图，验证准确率会剧烈跳动。
    * Class-Balanced Loss（CB）：按"有效样本数"给每类一个权重，缓解长尾不均衡。
      counts 必须只用训练集统计，绝不能把验证/测试集算进去（否则信息泄漏）。
    * 知识蒸馏（distillation）：学生除了拟合硬标签，还要拟合教师输出的软分布。
      本脚本用 KL 散度 + 温度 T 软化，T^2 用来补偿梯度尺度。
    * LambdaLR：学习率 = base_lr * lambda(step)，本实验用"线性 warmup + 余弦退火"。
      注意它是**按 step（batch）**更新，所以每个 batch 都要调一次 scheduler.step()。
    * checkpoint 里的 best_state 与 state_dict 不是一回事：前者是"验证最好那一轮"
      的深拷贝，后者是"当前最后一轮"的参数。测试必须用 best_state，否则等于拿
      训练最后一轮去测试，会破坏"用验证集选模型"的纪律。
    * 设备固定 CPU：本实验在无 GPU 环境完成，device 显式写出来是为了以后改 GPU 时
      只需要改这一处（其余 .to(device) 都已接好）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

# --- 让脚本能找到项目内的 src 包 ---------------------------------------------
# 直接 `python scripts/train.py` 时，解释器只把 scripts/ 放进 sys.path，
# 于是下面的 `from src.data import ...` 会 ModuleNotFoundError。
# __file__ -> 绝对路径；.parents[1] 取上两级（scripts -> vit_cifar10_lab），
# 得到项目根；插到 sys.path[0] 后，无论 cwd 在哪都能 import 到 src。
# 这只适合"脚本入口"，被 import 的库不应该改这种全局状态。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 以下 import 必须在 sys.path 修改之后，故带 noqa: E402（flake8 的豁免标记）。
# IndexedArrayDataset 与普通 Dataset 的区别：它的 __getitem__ 返回三元组
# (image, label, index)，那个 index 是"该样本在原始 train_images.npy 里的行号"，
# 蒸馏时要靠它去教师 logits 里取对应行。
from src.data import NUM_CLASSES, IndexedArrayDataset  # noqa: E402

# 训练/评估的公共组件：
#   build_optimizer        构造 AdamW，并对 bias/LayerNorm/cls_token/pos_embed 关闭 weight decay
#   cosine_lambda          返回一个"闭包函数"，LambdaLR 用它算学习率倍率
#   count_parameters       统计参数量（只用于日志/结果 JSON）
#   evaluate               在验证/测试集上算 acc、macro-F1、balanced acc、ECE、混淆矩阵
#   predict_logits         批量前向导出 logits（蒸馏的教师 logits 就靠它生成）
#   set_seed               固定 random / numpy / torch 的随机种子
#   train_one_epoch        一个 epoch 的训练循环（内部负责 zero_grad/backward/step）
from src.engine import (  # noqa: E402
    build_optimizer,
    cosine_lambda,
    count_parameters,
    evaluate,
    predict_logits,
    set_seed,
    train_one_epoch,
)
from src.losses import build_loss, effective_number_weights  # noqa: E402
from src.models import build_model as build_model_from_cfg  # noqa: E402
# 故意给 import 起别名 build_model_from_cfg：本文件里也有一个同名的 build_model()
# 包装函数，不加别名会互相覆盖，读代码时容易搞混"到底调的是哪个"。


def parse_args() -> argparse.Namespace:
    """声明并解析本脚本的全部命令行参数。

    作用
        把"实验配置"从代码里搬到命令行，使同一条命令可复现、不同实验可对比。
        每个参数都会原样进入 checkpoint 的 config 字段和结果 JSON，便于事后追溯。

    参数
        无（读取 sys.argv）。

    返回值
        argparse.Namespace，属性名即去掉 "--" 并把 "-" 换成 "_" 的参数名，
        例如 --batch-size 对应 args.batch_size。

    副作用
        none。不读文件、不打印；但 --help 会打印帮助并以 0 退出，
        参数非法时 argparse 会打印用法并以 2 退出。

    参数分组说明
        必填/标识：--model --tag
        优化相关：--loss --epochs --batch-size --lr --weight-decay --label-smoothing
                  --drop-path --warmup-epochs --num-threads
        数据相关：--seed --val-frac --limit-train --max-batches
        续训/蒸馏：--resume --distill-logits --distill-alpha --distill-temp
        导出开关：--export-logits --save-predictions
        结构几何：--patch-size --dim --depth --heads --mlp-ratio --prune-after
                  --keep-ratios（ViT/DynamicViT）与 --mobile-depth --global-depth
                  --mobile-patch（MobileViT）
    """
    p = argparse.ArgumentParser()

    # add_argument 的常用写法：
    #   "名字"        以 -- 开头是可选参数；required=True 才强制必须提供
    #   choices=[...] 限定取值，写错会立刻报错并列出合法值（比运行时 KeyError 友好）
    #   type=         解析时做类型转换（int/float），转换失败直接报错退出
    #   default=      用户未提供时的取值；有 default 就不再是"必填"
    # 这里 --model 有 choices 又有 required，等于"必须从三个里挑一个"。
    p.add_argument("--model", choices=["vit", "dynamic", "mobile"], required=True)

    # 损失函数：ce = 普通交叉熵，cb = Class-Balanced（按有效样本数加权）。
    # 默认 cb 是因为本实验的核心主题就是类别不均衡。
    p.add_argument("--loss", choices=["ce", "cb"], default="cb")

    # tag 是本次实验的唯一标识，直接拼进所有输出文件名，所以必填。
    # 建议写成 "<结构>_<损失>"，如 baseline_cb / dynamic_cb，方便 analyze.py 聚合。
    p.add_argument("--tag", type=str, required=True)

    # 训练轮数。小数据集 + 小模型，15~20 轮足够收敛。
    p.add_argument("--epochs", type=int, default=15)

    # 批大小 128：CPU 上兼顾吞吐与内存；再大容易触发频繁的线程同步开销。
    p.add_argument("--batch-size", type=int, default=128)

    # 学习率、权重衰减、标签平滑、drop path 都是 ViT 微调的常规默认值。
    # 5e-4 是 AdamW 训小 ViT 的稳妥起点；0.05 的 weight decay 偏大，但配合
    # "对 1 维参数/bias/位置编码关闭衰减"的 build_optimizer 才不会伤到它们。
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)

    # label smoothing 把 one-hot 目标软化（正确类 1-eps，其余 eps/(K-1)），
    # 抑制过度自信，通常能略微提升泛化与校准（ECE）。
    p.add_argument("--label-smoothing", type=float, default=0.1)

    # drop path（随机深度）：训练时以一定概率跳过整个残差分支，属于结构正则。
    p.add_argument("--drop-path", type=float, default=0.05)

    # warmup 用"epoch"为单位给出（2.0 表示前两个 epoch 线性升温），
    # 后面会被换算成 step 数。之所以按 epoch 配置，是因为它更贴近人的直觉。
    p.add_argument("--warmup-epochs", type=float, default=2.0)

    # 随机种子：同时控制数据划分、numpy/torch 初始化、DataLoader 的 shuffle 顺序。
    # 所有模型共用同一个 seed，保证它们看到完全相同的训练/验证划分。
    p.add_argument("--seed", type=int, default=42)

    # 验证集比例（分层抽取，每类各取 10%）。
    p.add_argument("--val-frac", type=float, default=0.1)

    # 线程数：控制 PyTorch intra-op 并行度，也影响 CPU 训练速度。
    p.add_argument("--num-threads", type=int, default=8)

    # 断点续训：指向 *_last.pt 即可从下一轮继续。默认空串表示不续训。
    p.add_argument("--resume", type=str, default="")

    # 冒烟测试用：只取前 N 张训练图（按类别比例），0 表示不限制。
    p.add_argument("--limit-train", type=int, default=0)

    # 冒烟测试用：每个 epoch 只跑前 N 个 batch，0 表示不限制。
    p.add_argument("--max-batches", type=int, default=0)

    # 蒸馏：教师 logits 文件路径 + 两个超参。
    # alpha 是蒸馏损失的权重（0.5 = 硬标签与软标签各占一半），
    # temp 是温度（越大分布越平，越关注"类间相似性"这种暗知识）。
    p.add_argument("--distill-logits", type=str, default="")
    p.add_argument("--distill-alpha", type=float, default=0.5)
    p.add_argument("--distill-temp", type=float, default=2.0)

    # action="store_true" 的含义：这个参数**不接收值**，命令行里出现即为 True，
    # 不出现即为 False（默认值自动是 False，所以不能再写 default=True）。
    # 与之相对的是 type= 参数，必须跟一个值。布尔开关用它最省事，
    # 比 --export-logits True 这种写法更符合习惯，也不会误把 "False" 当成真值。
    p.add_argument("--export-logits", action="store_true")

    # 保存测试集预测与标签，供 analyze.py 画混淆矩阵、算单类召回。
    p.add_argument("--save-predictions", action="store_true")

    # ViT / DynamicViT geometry
    # patch_size=4 -> 32/4=8，每边 8 个 patch，共 64 个 token（外加 1 个 cls）。
    # patch 越小 token 越多、分辨率越细，但注意力开销按 token 数平方增长。
    p.add_argument("--patch-size", type=int, default=4)

    # 嵌入维度、层数、注意力头数、MLP 扩张比：ViT 的四个核心超参。
    p.add_argument("--dim", type=int, default=96)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--mlp-ratio", type=float, default=2.0)

    # DynamicViT 专用。注意这两个参数声明成 str，因为命令行习惯写 "0,1,2" 这种
    # 逗号串；真正的解析在 src/models/factory.py 的 _as_tuple() 里按 "," 切分再转
    # int/float。所以这里不能写 type=list（argparse 会把字符串拆成单个字符）。
    # prune_after 是"在第几个 Block 之后剪枝"（0 基，且必须早于最后一个 Block）；
    # keep_ratios 是每一级保留多少比例的 patch token，两个列表长度必须一致。
    # 不传时用默认值 "0,1,2" / "0.75,0.5,0.25"，与 README 的实验设置一致。
    p.add_argument("--prune-after", type=str, default="0,1,2")
    p.add_argument("--keep-ratios", type=str, default="0.75,0.5,0.25")

    # MobileViT geometry
    # mobile_depth = 几个"局部注意力"的 MobileViT Block；
    # global_depth = 几个标准全局 Transformer Block；
    # mobile_patch = 局部分支内部做注意力的 patch 边长（4x4 的小窗口）。
    p.add_argument("--mobile-depth", type=int, default=2)
    p.add_argument("--global-depth", type=int, default=1)
    p.add_argument("--mobile-patch", type=int, default=4)

    # parse_args() 才真正读 sys.argv、做校验并返回 Namespace。
    # 必须放在所有 add_argument 之后；提前调用会漏掉后面声明的参数。
    return p.parse_args()


def split_train_val(labels: np.ndarray, val_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """按类别分层，把训练集索引切成"训练 / 验证"两部分。

    作用
        在**每个类别内部**各自随机抽 val_frac 比例做验证，再把各类的验证索引拼起来。
        这就是"分层抽样"（stratified sampling）。

    参数
        labels  : int64 数组，长度 N，第 i 个元素是第 i 张图的类别（0..9）。
        val_frac: 验证集比例，通常 0.1（每类各抽 10%）。
        seed    : 随机种子，决定打乱顺序；同一 seed 永远得到同一划分。

    返回值
        (train_idx, val_idx)：两个已排序的 int 索引数组，元素互不重叠，
        合起来正好覆盖 [0, N)。排序是为了让 DataLoader 的顺序稳定、便于复现与调试。

    副作用
        none（纯函数，不读文件、不打印）。使用独立的 RandomState，
        不会污染全局 numpy 随机状态，因此不影响后续的数据增广。

    为什么每个类各自抽 10%，而不是整体随机抽 10%
        本数据集是长尾不均衡的：大类可能有几千张，小类只有几百张。若整体随机抽，
        小类的验证样本数会随种子大幅波动（甚至可能只有个位数），验证准确率的
        方差就会很大，导致"最佳 epoch"的选择变得随机——相当于用噪声在选模型。
        按类等比例抽，能让每个类都稳定地贡献约 10% 的验证样本，
        宏平均 F1、balanced accuracy 这类"每类等权"的指标才可靠。
        附带好处：训练集与验证集的类别分布一致，验证指标更能代表测试集表现。
    """
    # RandomState 是"独立随机数发生器"：用自己的种子，不读也不改全局 np.random。
    # 这样即使训练过程中别处调用了 np.random，数据划分也不会漂移。
    rng = np.random.RandomState(seed)
    train_idx, val_idx = [], []

    # 逐类处理：cls 从 0 到 9。
    for cls in range(NUM_CLASSES):
        # flatnonzero 返回"值等于 cls 的所有下标"，即该类样本在原始数组里的行号。
        # 注意它返回的是新数组，所以下一行原地 shuffle 不会破坏 labels 本身。
        idx = np.flatnonzero(labels == cls)

        # 原地打乱这类样本的顺序，然后"前 n_val 个当验证、其余当训练"。
        # 打乱是关键：否则会按文件读入顺序切分（同类图片常按来源聚集），
        # 造成验证集与训练集分布不一致。
        rng.shuffle(idx)

        # round 后转 int：0.1 * 5000 = 500.0 -> 500。
        # 小类样本极少时这里可能算出 0，此时该类没有验证样本，属于可接受的极端情况。
        n_val = int(round(len(idx) * val_frac))

        # 切片不复制数据，但 concatenate 会复制，所以后面可以安全地排序。
        val_idx.append(idx[:n_val])
        train_idx.append(idx[n_val:])

    # concatenate 把 10 段拼成一整条；sort 让索引有序。
    # 排序的实际作用：DataLoader 按索引递增取样本，图像读取更接近顺序访问，
    # 也方便人工核对"第 k 个验证样本是哪张图"。
    return np.sort(np.concatenate(train_idx)), np.sort(np.concatenate(val_idx))


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    """按命令行配置构造模型（对 src.models.build_model 的薄包装）。

    作用
        把整个 Namespace 直接转成 dict 交给工厂函数，避免在这里重复列出十几个
        结构参数；工厂会根据 model 名字读取它需要的键，多余的键会被忽略。

    参数
        args: parse_args() 返回的 Namespace。

    返回值
        未初始化的 nn.Module（权重是随机的，尚未搬到 device）。

    副作用
        none。注意构造过程会消耗全局 torch 随机数，所以在 set_seed 之后调用
        才能保证不同实验的初始权重一致。
    """
    # vars(args) 把 Namespace 转成普通 dict（等价于 args.__dict__）。
    # factory 里用 cfg.get("patch_size", 4) 这类写法取值，因此多传的键无害。
    return build_model_from_cfg(args.model, vars(args))


def make_datasets(args: argparse.Namespace):
    """加载 npy 缓存、切分训练/验证、构造三个带索引的 Dataset。

    作用
        数据准备的总入口：读盘 -> 分层切分 ->（可选）按比例截取 -> 包装成
        IndexedArrayDataset -> 统计训练集类别分布。

    参数
        args: 需要 args.val_frac、args.seed、args.limit_train。

    返回值
        (train_ds, val_ds, test_ds, meta, counts, images) 六元组：
            train_ds / val_ds / test_ds : IndexedArrayDataset，返回 (图, 标签, 原始行号)
            meta    : meta.json 的内容（源目录、各类数量、类名等），会原样写进结果 JSON
            counts  : 训练集每个类别的样本数（长度 10 的 int 数组），供 CB 权重使用
            images  : 完整训练图像数组（未切分），导出教师 logits 时需要它

    副作用
        只读 data/ 下的 5 个文件；不写文件、不打印。
        如果 data/ 还没生成，会在 np.load 处抛 FileNotFoundError。

    为什么返回原始 images
        教师 logits 是按"完整训练集"导出的（行号 = 原文件行号），
        导出时必须用同一份未切分的数组，否则行号对不上。
    """
    # 数据目录固定为 <项目根>/data，与 prepare_data.py 的默认输出一致。
    data_dir = ROOT / "data"

    # 图像保持 uint8（NHWC），标签转成 int64（PyTorch 的交叉熵要求 long）。
    # .astype 会复制一份，避免后续误改缓存数组。
    images = np.load(data_dir / "train_images.npy")
    labels = np.load(data_dir / "train_labels.npy").astype(np.int64)
    test_images = np.load(data_dir / "test_images.npy")
    test_labels = np.load(data_dir / "test_labels.npy").astype(np.int64)

    # read_text 显式指定 utf-8（Windows 默认编码可能是 GBK）；
    # json.loads 把字符串解析成 dict。meta 里记录了源目录与各类数量，
    # 写进结果 JSON 后，几个月后回看实验也能知道数据从哪来。
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))

    # 分层切分：得到"训练行号"和"验证行号"。两者都指向原始 images 数组，
    # 也就是说验证集并没有复制图像，只是索引不同——省内存。
    train_idx, val_idx = split_train_val(labels, args.val_frac, args.seed)

    if args.limit_train > 0:
        # 冒烟测试分支：只在需要快速验证代码能跑通时使用。
        # 这里同样按类别等比例截取，而不是直接取前 N 个，
        # 否则前 N 张可能全属于前几个类，CB 权重与验证指标会完全失真。
        rng = np.random.RandomState(args.seed + 1)  # +1 让截取顺序与划分种子不同
        picked = []
        for cls in range(NUM_CLASSES):
            # 先取出训练索引中属于该类的那些行号，再原地打乱。
            cls_idx = train_idx[labels[train_idx] == cls]
            rng.shuffle(cls_idx)

            # 按该类在训练集中的占比分配名额：大类多给、小类少给，
            # 至少给 1 个（max(1, ...)），避免某个类直接消失。
            n = max(1, int(round(args.limit_train * len(cls_idx) / len(train_idx))))
            picked.append(cls_idx[:n])
        # 拼起来再排序，保持与完整训练集一致的索引递增顺序。
        train_idx = np.sort(np.concatenate(picked))

    # IndexedArrayDataset 只是"索引视图"：它持有完整数组 + 一个行号列表，
    # __getitem__(i) 返回 (第 indices[i] 张图, 它的标签, 真实行号)。
    # 三个 dataset 共享同一份 images 内存（train/val 共享，test 用另一份）。
    train_ds = IndexedArrayDataset(images, labels, train_idx)
    val_ds = IndexedArrayDataset(images, labels, val_idx)

    # 测试集不做切分：索引就是 0..N-1，用 arange 生成。
    # 这样"第 i 个测试样本"与"测试文件第 i 行"一一对应，
    # 保存下来的 predictions/labels 可以直接和原数据对齐。
    test_ds = IndexedArrayDataset(test_images, test_labels, np.arange(len(test_labels)))

    # bincount 统计每个类别出现次数，minlength 保证即使某个类为 0 也输出 10 个数。
    # 注意统计的是"切分后的训练索引"，绝不能用全量 labels，否则 CB 权重会泄漏验证集信息。
    counts = np.bincount(labels[train_idx], minlength=NUM_CLASSES)
    return train_ds, val_ds, test_ds, meta, counts, images


def main() -> None:
    """完整的单模型训练流程：从命令行到 JSON/checkpoint 落盘。

    作用
        本脚本的主控函数，把数据、模型、损失、优化器、调度器、训练循环、
        评估与结果持久化串起来，并在关键节点写日志。

    参数
        无（所有配置来自 parse_args 解析的命令行）。

    返回值
        None。训练失败会由异常中断（例如数据缺失、checkpoint 不匹配）；
        正常情况下即使准确率不理想也会正常退出并留下完整记录。

    副作用
        1. 创建 results/、checkpoints/ 目录（已存在则跳过）；
        2. 每个 epoch 覆盖写 checkpoints/<tag>_last.pt；
        3. 结束时写 results/<tag>.json 和 checkpoints/<tag>.pt；
        4. 可选写 results/<tag>_test_pred.npy / _test_labels.npy 和教师 logits；
        5. 持续向 stdout 与 results/<tag>.log 打印（log 内每次 flush）。

    整体步骤
        1) 解析参数；设置线程数与随机种子；确定 cpu 设备与输出目录；
        2) 打开日志文件并定义 log()（同时 print 与写文件）；
        3) 准备数据集与三个 DataLoader（只有训练集 shuffle）；
        4) 建模型、算参数量、打印规模与类别分布；
        5) 计算 CB 类别权重并构造损失函数；
        6) 若配置了蒸馏，加载并校验教师 logits；
        7) 建优化器与"warmup + 余弦"调度器（按 step 计）；
        8) 若 --resume，恢复模型/优化器/调度器/历史/最佳权重，并从下一轮继续；
        9) 逐 epoch：训练 -> 验证 -> 记录 -> 刷新最佳权重 -> 存 last checkpoint
           -> 记录内存；
        10) 载入最佳权重，在测试集上评估一次；
        11) 写结果 JSON、最佳模型 checkpoint、可选的预测与教师 logits；
        12) 打印总耗时并关闭日志。
    """
    # ---- 步骤 1：参数、线程、随机性、输出目录 ------------------------------
    args = parse_args()

    # set_num_threads 控制 intra-op（单个算子内部，如一次大矩阵乘）的线程数，
    # 是 CPU 训练速度的主要开关。设得比物理核多会因频繁上下文切换而变慢。
    torch.set_num_threads(args.num_threads)

    # set_num_interop_threads 控制 inter-op（多个算子之间）的并行线程数。
    # 训练时各算子之间有数据依赖，inter-op 并行收益很小，反而会抢走 intra-op 的核，
    # 所以固定为 1。重要限制：这个函数必须在任何并行计算开始**之前**调用，
    # 且一个进程只能设一次，否则会抛 RuntimeError——所以它只能放在这里（最先）。
    torch.set_num_interop_threads(1)

    # 固定 python random / numpy / torch 的种子。
    # 顺序上必须在建模型之前调用，否则初始权重每次都不一样。
    set_seed(args.seed)

    # 本实验全程 CPU。显式写成变量（而不是到处硬编码 "cpu"），
    # 以后要换 GPU 只改这一行即可，其余 .to(device) 都已接好。
    device = torch.device("cpu")

    # 输出目录固定为项目下的 results/ 与 checkpoints/；
    # mkdir(exist_ok=True) 表示已存在不报错。这里不建 parents，因为 ROOT 一定存在。
    out_dir = ROOT / "results"
    ckpt_dir = ROOT / "checkpoints"
    out_dir.mkdir(exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)

    # 以 "w" 模式打开日志：同名 tag 重跑会**清空**上一次日志（这是有意的，
    # 避免两次实验的日志混在一起难以分辨）。encoding 显式 utf-8 防止中文乱码。
    # 句柄在函数末尾 close()；中途抛异常则由进程退出时回收。
    log_file = (out_dir / f"{args.tag}.log").open("w", encoding="utf-8")

    def log(msg: str) -> None:
        """同时把一行消息打到终端和日志文件。

        作用
            训练可能跑几十分钟，既想实时看到进度（终端），又想事后留档（文件），
            所以在唯一入口处同时做两件事，避免调用方每次写两行。

        参数
            msg: 要输出的文本（调用方负责格式化，通常已带指标）。

        返回值
            None。

        副作用
            print 到 stdout；向 log_file 写入 msg + 换行；写后立即 flush。
            闭包捕获了外层的 log_file，因此只能在 main() 内部使用。

        为什么 flush
            Python 的文件写入默认有用户态缓冲，进程被中断（Ctrl+C、OOM）时
            缓冲区里的日志会丢失。每行 flush 保证"崩之前的最后一行"一定在磁盘上，
            这对排查长训练任务的中断原因很关键。代价是每行一次系统调用，
            但日志频率是每 epoch 几行，完全可以接受。
        """
        # flush=True 让终端也立即刷新：否则管道/重定向时输出可能成块出现，
        # 看不到"实时进度"。
        print(msg, flush=True)

        # 写文件并立刻落盘，理由见 docstring。
        log_file.write(msg + "\n")
        log_file.flush()

    # 记录整个脚本的墙钟起点，最后用来算 wall_time_s。
    t_start = time.time()

    # ---- 步骤 3：数据与 DataLoader ----------------------------------------
    # make_datasets 内部完成读盘与分层切分；counts 是训练集（不含验证集）的类别计数。
    train_ds, val_ds, test_ds, meta, counts, all_train_images = make_datasets(args)

    # 固定 DataLoader 的 shuffle 顺序：generator 只影响"每个 epoch 怎么打乱"。
    # 不传 generator 时 DataLoader 会用全局 torch RNG，那样一旦中途多消耗了随机数
    # （比如多建了一个模型），后续 epoch 的顺序就会变，实验无法精确复现。
    # 注意：数据增广用的是全局 RNG（augment_batch 里 torch.rand/randint），
    # 已由 set_seed 固定；generator 不负责增广的随机性。
    g = torch.Generator().manual_seed(args.seed)

    # 训练集必须 shuffle：否则同一类的样本连续出现，梯度的类别分布剧烈震荡，
    # BatchNorm/权重更新都会受影响（且容易记住"数据顺序"这一无关信息）。
    # num_workers=0：数据已经在内存里（numpy），取 batch 只是一次索引 + from_numpy，
    # 用子进程反而要付 Windows 下 spawn 与数组序列化的代价，得不偿失。
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, generator=g)

    # 验证/测试集不 shuffle：评估只关心汇总指标，顺序无关；
    # 不 shuffle 还能让保存下来的 predictions 与原始顺序一一对应。
    # batch 用 256（比训练大）是因为评估不需要存激活，可以更充分地利用内存。
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    # ---- 步骤 4：模型与规模 ------------------------------------------------
    # .to(device) 把参数搬到 CPU（此处是空操作，但换 GPU 时就是必需的一步）。
    model = build_model(args).to(device)

    # 参数量只统计一次：结构在训练中不变。
    n_params = count_parameters(model)

    # :, 是千分位分隔（如 311,000），比裸数字好读。
    log(f"[{args.tag}] model={args.model} loss={args.loss} params={n_params:,}")

    # 打印三个 split 的大小与训练集类别分布：这是判断"切分是否合理、是否长尾"的第一手信息。
    log(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} counts={counts.tolist()}")

    # ---- 步骤 5：类别权重与损失 -------------------------------------------
    # 有效样本数权重：E_n = (1 - beta^n)/(1 - beta)，w ∝ 1/E_n，再归一化到均值为 1。
    # beta=0.9999 接近"逆频率"但更平滑：样本极多的类不会把权重压到接近 0。
    # astype(np.float32)：PyTorch 的权重张量用 float32，提前转好省一次转换。
    class_weights = effective_number_weights(counts, beta=0.9999).astype(np.float32)

    # 按名字构造损失：ce 时 class_weights 不会被用到；cb 时必须传。
    # label_smoothing 对两种损失都生效（实现里传给了 F.cross_entropy）。
    criterion = build_loss(args.loss, class_weights, args.label_smoothing)

    # 只有用 CB 时才打印权重，CE 实验没必要刷屏。
    # :.3f 保留三位小数；", ".join 把 10 个数字拼成一行。
    if args.loss == "cb":
        log("class weights: " + ", ".join(f"{w:.3f}" for w in class_weights.tolist()))

    # ---- 步骤 6：教师 logits（可选） --------------------------------------
    teacher_logits = None
    if args.distill_logits:
        # weights_only=True：只允许反序列化 tensor 等白名单类型，拒绝任意 Python 对象。
        # 教师 logits 文件里就是一个纯 tensor，所以用 True 更安全（这也是 PyTorch 2.6+
        # 的默认值）。对比下面 --resume 的加载：那个 checkpoint 里有 dict/list/str，
        # 必须用 weights_only=False。map_location="cpu" 保证无 GPU 也能读。
        teacher_logits = torch.load(args.distill_logits, map_location="cpu", weights_only=True)

        # 行数必须等于完整训练集大小：训练时用 batch 里的原始行号去索引它，
        # 如果行数不一致（例如教师是在别的数据版本上导出的），索引会错位或越界，
        # 而且这种错误不会报错、只会让蒸馏"悄悄学错目标"，所以必须显式校验。
        if teacher_logits.shape[0] != len(all_train_images):
            raise ValueError("teacher logits do not match the training set size")

        log(f"distillation from {args.distill_logits} alpha={args.distill_alpha} T={args.distill_temp}")

    # ---- 步骤 7：优化器与学习率调度器 -------------------------------------
    # build_optimizer 返回 AdamW，并分成两组：需要 weight decay 的矩阵参数，
    # 以及关闭 decay 的 1 维参数（bias、LayerNorm 的 gamma/beta）、cls_token、pos_embed。
    # 对这些参数做衰减会损害表达能力，是 ViT 训练的常规做法。
    optimizer = build_optimizer(model, args.lr, args.weight_decay)

    # 一个 epoch 有多少个 batch。len(loader) = ceil(N / batch_size)（未设 drop_last）。
    # 注意这个数依赖 batch_size 与训练集大小，所以 total_steps 也随之变化。
    steps_per_epoch = len(train_loader)

    # 总步数与 warmup 步数：调度器以 **step（batch）** 为单位，不是 epoch。
    # warmup_epochs 允许小数（如 2.0 或 0.5），乘完再取整。
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(steps_per_epoch * args.warmup_epochs)

    # LambdaLR：学习率 = base_lr * lambda(step)，其中 lambda 由 cosine_lambda 返回。
    #
    # cosine_lambda 是一个"闭包工厂"：调用它时把 warmup_steps / total_steps 固定进
    # 内部函数 fn 的闭包里，返回 fn 本身。之后 LambdaLR 每一步都用当前 step 调 fn(step)
    # 得到倍率。为什么要闭包而不是传参？因为 LambdaLR 只接受"一个参数的函数"，
    # 而我们需要额外携带两个常量，闭包正好把常量"藏"进函数对象。
    #
    # 曲线形状：step < warmup 时线性从 0 升到 1；之后按余弦从 1 衰减到 min_lr_ratio(0.02)。
    # 先 warmup 是因为训练初期权重随机、梯度方向噪声大，用大学习率容易发散；
    # 后余弦衰减是为了后期小步收敛，避免在最优解附近来回跳。
    #
    # 易错点：LambdaLR 在**构造时**就会用 lambda(0) 设置一次初始学习率，而此时
    # fn(0) = 0/warmup = 0，所以第一个 batch 实际以 lr≈0 更新（等价于空转一步）。
    # 这是 warmup 的固有现象，不是 bug；随后的 scheduler.step() 会把 lr 抬起来。
    # 另一个易错点：调度器是"按 step"的，必须在每个 batch 末尾调用一次 step()，
    # 若像 StepLR 那样每 epoch 调一次，学习率会掉到接近 0。
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, cosine_lambda(warmup_steps, total_steps)
    )

    # ---- 步骤 8：断点续训状态（可选） -------------------------------------
    # history 记录每个 epoch 的指标，最终写进 JSON 并用于画曲线。
    history = []

    # best_val 初值 -1.0 保证第一个 epoch 一定刷新（准确率 >= 0 > -1）。
    # best_epoch 记录最佳轮次，best_state 保存那一轮的权重副本。
    best_val = -1.0
    best_epoch = -1
    best_state = None
    start_epoch = 1

    if args.resume:
        # weights_only=False：这里必须用 False。checkpoint 里除 tensor 外还有
        # history（list[dict]，含 str/float）、config（dict）、best_epoch（int）等
        # 普通 Python 对象，weights_only=True 会拒绝加载。风险提示：False 意味着
        # 完整反序列化，只能用于**自己生成的**可信文件，不要加载来路不明的 .pt。
        resume = torch.load(args.resume, map_location="cpu", weights_only=False)

        # 逐项恢复：模型权重、优化器动量、调度器步数。三者缺一不可——
        # 只恢复模型权重的话，AdamW 的一阶/二阶矩会重置，学习率也会从头 warmup，
        # 等于"权重是旧的、优化状态是新的"，loss 常常会先跳一下。
        model.load_state_dict(resume["state_dict"])
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])

        # 历史与"最佳"状态也要接着上次，否则续训后 best_val 从 -1 重来，
        # 可能用一个更差的 epoch 覆盖掉之前真正最好的权重。
        history = resume["history"]
        best_val = resume["best_val"]
        best_epoch = resume["best_epoch"]
        best_state = resume["best_state"]

        # 从"上次存盘的 epoch + 1"继续，既不重复也不跳轮。
        start_epoch = resume["epoch"] + 1
        log(f"resumed from {args.resume} at epoch {resume['epoch']}, best_val={best_val:.4f}")

    # ---- 步骤 9：逐 epoch 训练与验证 --------------------------------------
    for epoch in range(start_epoch, args.epochs + 1):
        # perf_counter 是单调高精度时钟，用来测一个 epoch 的墙钟耗时。
        t0 = time.perf_counter()

        # train_one_epoch 内部按固定顺序完成一个 epoch 的所有参数更新，
        # 这个顺序是深度学习训练最容易写错的地方，值得逐条记住：
        #
        #   1. optimizer.zero_grad(set_to_none=True)
        #        清空上一步的梯度。set_to_none=True 是把 .grad 置为 None 而不是填 0，
        #        更省一次全参数写内存；PyTorch 会在 backward 时重新分配。
        #        忘了清零 -> 梯度在 batch 之间累加 -> 等效学习率乱掉。
        #   2. logits = model(images)      前向，建立计算图
        #   3. loss = criterion(logits, labels)   算损失
        #   4. loss.backward()             反向传播，把梯度写进 param.grad
        #   5. clip_grad_norm_(params, 1.0)  梯度裁剪：把所有梯度当成一个长向量，
        #        若其 L2 范数超过 1.0 就整体等比缩小。作用是防止个别 batch 的
        #        异常大梯度把权重打飞（Transformer 训练常见）。必须在 step 之前。
        #   6. optimizer.step()            用梯度更新参数（AdamW 在这里做衰减）
        #   7. scheduler.step()            推进学习率。放在 step 之后、按 batch 调一次。
        #        顺序颠倒（先 scheduler.step 再 optimizer.step）会触发 PyTorch 警告，
        #        并且第一个 batch 用的学习率就错了。
        #
        # 关于蒸馏：teacher_logits 的形状是 [N_train, 10]，索引方式是
        #     teacher = teacher_logits[indices]
        # 其中 indices 来自 IndexedArrayDataset 的第三个返回值，是这批样本在
        # **原始 train_images.npy 里的行号**（不是它在 train_ds 里的位置！）。
        # 用"高级索引"一次取出对应行，形状变成 [batch, 10]，再与同一批的 student
        # logits 算 KL。为什么不按 batch 顺序取？因为 shuffle 后 batch 内的样本
        # 顺序是随机的，只有保留原始行号才能找回"这张图"对应的教师输出。
        # 另外注意训练时的图像经过随机增广，而教师 logits 是在未增广图像上算的：
        # 教师是固定目标，噪声只起正则作用，这是蒸馏的常见做法。
        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            device,
            teacher_logits=teacher_logits,
            distill_alpha=args.distill_alpha,
            distill_temp=args.distill_temp,
            # 0 表示"不限制"，这里转成 None 让内部用 `is not None` 判断更清晰。
            max_batches=args.max_batches if args.max_batches > 0 else None,
        )

        # 验证集评估：内部会切到 model.eval()（关 dropout/drop_path）并用 no_grad 前向，
        # 返回 acc / macro_f1 / balanced_accuracy / 混淆矩阵 / ECE 等。
        val_metrics = evaluate(model, val_loader, device, NUM_CLASSES)

        # 一个 epoch 的总墙钟（训练 + 验证），与纯训练时间 train_stats["seconds"] 区分。
        elapsed = time.perf_counter() - t0

        # 从优化器里读当前学习率（第 0 个 param_group）：
        # 调度器改的就是这个值，记下它才能画出真实的 lr 曲线、确认 warmup 生效。
        lr_now = optimizer.param_groups[0]["lr"]

        # 每个 epoch 一条结构化记录，字段名会被 analyze.py / build_report.py 依赖。
        rec = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_acc": train_stats["accuracy"],
            "train_seconds": train_stats["seconds"],
            "val_acc": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_balanced_acc": val_metrics["balanced_accuracy"],
            "lr": lr_now,
            "epoch_seconds": elapsed,
        }
        history.append(rec)

        # 单行日志：同时给训练损失/准确率、验证 acc/F1、训练秒数、总秒数、当前 lr。
        # :02d 是补零到两位（epoch 01/20），:2e 是科学计数法（学习率很小，用小数不好读）。
        log(
            f"epoch {epoch:02d}/{args.epochs:02d} loss={rec['train_loss']:.4f} "
            f"train_acc={rec['train_acc']:.4f} val_acc={rec['val_acc']:.4f} "
            f"val_f1={rec['val_macro_f1']:.4f} train_s={train_stats['seconds']:.1f} "
            f"total_s={elapsed:.1f} lr={lr_now:.2e}"
        )

        # 用**验证集准确率**选模型（绝不能用测试集，否则等于用测试集调参，结果虚高）。
        # 严格大于：并列时不覆盖，保留更早（更简单/更少过拟合）的那个 epoch。
        if val_metrics["accuracy"] > best_val:
            best_val = val_metrics["accuracy"]
            best_epoch = epoch

            # 这里是"深拷贝"，三个操作缺一不可：
            #   .detach()  断开计算图，避免把整个图（及其激活）留在内存里
            #   .cpu()     搬到内存，便于之后序列化/换设备
            #   .clone()   真正复制数值
            # 为什么必须 clone：state_dict() 返回的 tensor 与模型参数**共享存储**，
            # 不复制的话，下一轮 optimizer.step() 会原地修改这些数值，
            # 于是"最佳权重"会悄悄变成"最新权重"，最后测的就不是最佳模型了。
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        # 每个 epoch 覆盖写一份 last checkpoint，用于断点续训。
        # 注意它保存的是**当前** state_dict（最后一轮），而 best_state 是历史最佳。
        # 各字段用途：
        #   epoch       已完成的轮次，续训时 +1 得到 start_epoch
        #   state_dict  当前模型参数（不是最佳！）
        #   optimizer   优化器状态（AdamW 的动量/二阶矩），续训必需
        #   scheduler   调度器状态（已走多少 step），续训后学习率才能接上
        #   history     逐 epoch 指标，续训后继续追加
        #   best_val / best_epoch / best_state  历史最佳，防止续训把最佳记录丢掉
        #   config      本次的命令行配置，事后复现/排查用
        torch.save(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "history": history,
                "best_val": best_val,
                "best_epoch": best_epoch,
                "best_state": best_state,
                "config": vars(args),
            },
            ckpt_dir / f"{args.tag}_last.pt",
        )

        # 内存监控：psutil 是可选依赖（不在 requirements 的硬性范围内），
        # 而且某些受限环境里 Process().memory_info() 可能不可用。
        # 所以整段包在 try/except 里：监控失败绝不能让训练中断。
        # except Exception 之后直接 pass，是"尽力而为"的监控该有的写法。
        try:
            import psutil

            # virtual_memory().available 是系统当前可用内存；
            # Process().memory_info().rss 是本进程的常驻内存。
            # 除 2**30 把字节换成 GB（1 GiB = 2^30 B），便于人读。
            vm = psutil.virtual_memory()
            rss = psutil.Process().memory_info().rss / 2**30
            log(f"memory: available={vm.available/2**30:.2f}GB process_rss={rss:.2f}GB")
        except Exception:
            # 静默忽略：没有 psutil、权限不足、平台不支持都不影响训练。
            # 放在 epoch 末尾而不是每个 batch，是为了避免频繁采样干扰计时。
            pass

    # ---- 步骤 10：用最佳权重做最终测试 ------------------------------------
    # 防御性断言：如果 epochs <= 0，或续训文件的 best_state 为 None，
    # 循环体一次都没执行，best_state 仍是 None，下面 load_state_dict 会报错。
    # 这里提前断言，让失败点更明确。
    assert best_state is not None

    # 把"验证集上最好那一轮"的权重装回模型，再测测试集。
    # 这一步是实验纪律的核心：test 只用来报告最终成绩，不参与任何选择。
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device, NUM_CLASSES)

    # 汇总行：同时给出选择依据（val_acc）与最终成绩（test acc / F1 / balanced acc）。
    log(
        f"BEST epoch={best_epoch} val_acc={best_val:.4f} | TEST acc={test_metrics['accuracy']:.4f} "
        f"macro_f1={test_metrics['macro_f1']:.4f} bal_acc={test_metrics['balanced_accuracy']:.4f}"
    )

    # ---- 步骤 11：写结果 JSON 与最佳模型 ----------------------------------
    # 复制一份配置：后面不会再改 args，但复制能避免"结果里的 config 被意外修改"。
    config = vars(args).copy()

    result = {
        "tag": args.tag,
        "model": args.model,
        "loss": args.loss,
        "config": config,
        "params": n_params,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        "class_counts_train": counts.tolist(),
        # CE 实验没有类别权重，写 None 而不是省略字段，保持 JSON 结构统一，
        # 下游 analyze.py 读 key 时不用判断 key 是否存在。
        "class_weights": class_weights.tolist() if args.loss == "cb" else None,
        "dataset_meta": meta,
        "best_epoch": best_epoch,
        "best_val_acc": best_val,
        "history": history,
        # 从测试指标里剔除 predictions / labels：它们是几千个整数的列表，
        # 塞进 JSON 会让文件暴涨（几百 KB），而它们另有 .npy 专门保存。
        # 字典推导 + 集合差集实现"排除某些键"。
        "test": {k: v for k, v in test_metrics.items() if k not in {"predictions", "labels"}},
        # 三种耗时口径：所有 epoch 累计、纯训练每轮均值、纯训练每轮标准差。
        # 均值±标准差比单个总数更能说明"每轮大概多久、稳不稳"。
        "train_time_total_s": float(sum(h["epoch_seconds"] for h in history)),
        "train_epoch_mean_s": float(np.mean([h["train_seconds"] for h in history])),
        "train_epoch_std_s": float(np.std([h["train_seconds"] for h in history])),
        # 包含数据加载、评估、存盘在内的端到端耗时，与上面的纯训练时间区分。
        "wall_time_s": time.time() - t_start,
    }

    # json.dumps 的两个关键参数：
    #   indent=2        每层缩进 2 空格、一行一个字段。虽然比紧凑格式大，
    #                   但人可以直接读、git diff 能按行比较，实验记录值得这个代价。
    #   ensure_ascii=False  让非 ASCII 字符（meta 里的中文路径、类名等）原样输出，
    #                   否则会写成 \u4e2d\u6587 这种转义，人能读但很难受。
    # 必须配 encoding="utf-8"，否则 Windows 上可能按 GBK 写而出现乱码/异常。
    (out_dir / f"{args.tag}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 另存一份"可重建模型"的 checkpoint（不含优化器状态，所以比 last 小很多）。
    # 字段是 measure_complexity.py / dynamic_tokens.py 的接口约定：
    #   model_name + model_kwargs  -> build_model(name, kwargs) 重建结构
    #   state_dict                 -> 装入最佳权重
    #   test_metrics / params      -> 汇总表直接用，不必重新评估
    # 注意这里存的是 best_state（已 detach/cpu/clone），不是当前 state_dict。
    torch.save(
        {
            "tag": args.tag,
            "model_name": args.model,
            "model_kwargs": config,
            "state_dict": best_state,
            "test_metrics": result["test"],
            "params": n_params,
        },
        ckpt_dir / f"{args.tag}.pt",
    )

    # 可选：保存测试集预测与真实标签，供 analyze.py 画混淆矩阵、算单类召回。
    # np.save 会自动补 .npy 后缀；np.asarray 是因为 evaluate 返回的是 Python list。
    if args.save_predictions:
        np.save(out_dir / f"{args.tag}_test_pred.npy", np.asarray(test_metrics["predictions"]))
        np.save(out_dir / f"{args.tag}_test_labels.npy", np.asarray(test_metrics["labels"]))

    # 可选：导出教师 logits，供后续蒸馏实验使用。
    if args.export_logits:
        # torch.from_numpy 与 numpy 数组**共享内存**，不发生拷贝（几 GB 数据也能瞬间完成）。
        # 传入的是完整训练集 all_train_images（含验证集部分），行号 = 原始文件行号，
        # 这才与 IndexedArrayDataset 返回的 index 对得上，蒸馏时索引才不会错位。
        # 此时模型已装载 best_state，所以导出的是"最佳教师"的输出。
        # batch_size=256 只是为了控制峰值内存。
        # predict_logits 内部会处理 uint8 -> 归一化（normalize_batch），无需在这里手动转。
        logits = predict_logits(model, torch.from_numpy(all_train_images), device, batch_size=256)

        # 存成 .pt（tensor）而不是 npy：加载端用 weights_only=True 更安全，
        # 且能保留 dtype/device 信息。文件名固定为 <tag>_train_logits.pt，
        # 与 README 里 --distill-logits 的写法一致。
        torch.save(logits, ckpt_dir / f"{args.tag}_train_logits.pt")
        log(f"saved train logits {tuple(logits.shape)} -> {ckpt_dir / (args.tag + '_train_logits.pt')}")

    # 收尾：报告总耗时与结果 JSON 的位置，方便直接去查看。
    log(f"finished in {time.time() - t_start:.1f}s; results -> {out_dir / (args.tag + '.json')}")

    # 显式关闭日志句柄，确保缓冲区全部落盘（虽然每行都 flush 过，这里仍是好习惯）。
    log_file.close()


# 只有直接运行本文件时 __name__ == "__main__"，被 import 时不会自动开训。
if __name__ == "__main__":
    main()
