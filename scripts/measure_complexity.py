"""测量训练好的模型的参数量、FLOPs 与 CPU 推理速度，汇总成一份 JSON。

[脚本做什么]
    对 checkpoints/ 下选中的每个最终模型，做三件事：
        1. 重建模型并统计参数量（可训练参数总数）；
        2. 用 PyTorch 内置的 FlopCounterMode 统计"单张图片一次前向"的浮点运算量；
        3. 用固定输入张量反复前向，测 batch=1 的单张延迟与 batch=64 的吞吐。
    由于是在笔记本 CPU 上测量，单次长跑会受到频率/温度漂移的污染（先测的模型
    系统性偏快），所以本脚本采用"交错测量 + 多轮取中位数"的实验设计，让各模型
    在同一时间段内轮流被测，把漂移平摊到每个模型身上。两个同结构的 ViT 基线
    （CE 与 CB）测出来的吞吐应当在 ~1% 内一致，这可以作为测量是否可信的自检。

[输入]
    checkpoints/*.pt —— train.py 生成的最终模型（含 model_name / model_kwargs /
    state_dict / tag / test_metrics / params）。脚本会自动过滤掉：
        *_train_logits.pt  教师 logits，本质是 tensor，不是模型
        *smoke*.pt         冒烟测试产物
        *_last.pt          每 epoch 覆盖写的续训快照，不是"最佳模型"
    也可用 --tags 只测指定 tag（按文件名去掉 .pt 后的 stem 匹配）。

[输出]
    results/complexity.json（默认路径，可用 --out 改）：每个 tag 一条记录，包含
        model / tag / params / params_m / flops / gflops
        latency_b1  / latency_b64：batch=1 与 batch=64 的统计量（median/mean/min/max、
                                   每张毫秒、每秒张数）
        threads：测量时的线程数（换硬件或线程数后数值不可直接比较）

[在实验流程中的位置]
    在 train.py 跑完所有模型之后、analyze.py 画"复杂度—精度"权衡图之前。
    analyze.py 读这份 JSON 拿 FLOPs 与吞吐，和 train.py 写的准确率拼成一张表。

[怎么运行]
    cd vit_cifar10_lab
    python scripts/measure_complexity.py                 # 默认 8 线程、5 轮
    python scripts/measure_complexity.py --rounds 3      # 赶时间时少测几轮
    python scripts/measure_complexity.py --tags baseline_cb dynamic_cb
    python scripts/measure_complexity.py --out results/complexity_debug.json

[阅读前需要知道的概念]
    * 参数量（params）：权重与偏置的元素总数，决定模型文件大小与内存占用，
      但不直接决定速度。
    * FLOPs：前向一次需要的浮点运算次数，衡量"计算量"，与硬件无关，便于横向比较；
      注意 FLOPs 低不等于延迟低——小矩阵、内存带宽、算子实现都会影响实际速度。
    * 延迟 vs 吞吐：单张延迟（batch=1）受 CPU 调度噪声影响极大，常常不可复现；
      batch=64 的吞吐（img/s）稳定得多，所以报告以吞吐为主指标、单张延迟仅作参考。
    * 中位数（median）：把测得的 N 次耗时排序后取中间那个。相比平均值，它对偶发的
      长尾（系统抢占、降频、GC）不敏感，更能代表"典型一次要多久"。
    * 交错测量（interleaving）：不让一个模型连续测完再测下一个，而是每轮把所有模型
      各测一遍。这样温度/频率漂移对每个模型的影响近似相同，模型之间的**相对**比较
      才成立；只测一个模型时数值再准也没有对照意义。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# --- 把项目根目录加入模块搜索路径 -------------------------------------------
# 理由同其他脚本：`python scripts/measure_complexity.py` 时 sys.path[0] 是 scripts/，
# 不插这一行就 import 不到 src 包。__file__ -> 绝对路径 -> 上两级 = 项目根。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 这两行的 import 必须排在 sys.path 修改之后，故带 noqa: E402 豁免。
from src.engine import count_parameters  # noqa: E402
from src.models import build_model  # noqa: E402


def load_model(path: Path, device: torch.device):
    """从 checkpoint 文件恢复一个处于推理模式的模型。

    参数
        path    : .pt 文件路径，需包含 model_name / model_kwargs / state_dict。
        device  : 模型搬到的设备（本脚本固定 cpu）。

    返回值
        (model, ckpt)：model 是已加载权重、已 .eval() 的 nn.Module；
        ckpt 是原始 checkpoint 字典，调用方还要用它的 tag / model_name 等元信息。

    副作用
        只读磁盘并分配内存；不修改文件，不打印。
        注意：模型被置为 eval 模式（关闭 dropout/drop_path），这个状态会影响后续
        所有前向的速度与结果，正是测量想要的。
    """
    # weights_only=False：checkpoint 里除 state_dict 外还有 model_kwargs 这样的普通
    # dict（含 str/int/float/None），weights_only=True 会拒绝反序列化这类对象。
    # 这里读的是本仓库自己训练出的文件，可信；对下载来的 .pt 不要这样加载。
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    # 用"训练时存下的配置"重建结构，而不是当前代码里的默认值：
    # 这样即使以后改了默认超参，旧 checkpoint 依然能正确复现。
    model = build_model(ckpt["model_name"], ckpt["model_kwargs"]).to(device)

    # strict=True（默认）要求 state_dict 的键完全匹配，防止静默漏加载某些层。
    model.load_state_dict(ckpt["state_dict"])

    # eval() 有两个作用：(1) 关掉 dropout/drop_path 的随机性，否则每次前向的
    # 计算图不同、延迟抖动大；(2) 让 DynamicViT 的剪枝走确定性的 top-k 分支
    # （训练模式会额外加 Gumbel 噪声），保证测到的是部署时的真实路径。
    model.eval()
    return model, ckpt


def count_flops(model: torch.nn.Module, device: torch.device, img_size: int = 32) -> int:
    """用 PyTorch 内置的 FlopCounterMode 统计单张图片前向一次的 FLOPs。

    参数
        model    : 待测模型，应已处于 eval 模式。
        device   : 输入张量所在设备（与模型一致）。
        img_size : 输入边长，CIFAR-10 为 32。

    返回值
        int，前向一次的浮点运算次数（FLOPs，不是 FLOP/s）。除以 1e9 即 GFLOPs。

    副作用
        none（不写文件、不打印；FlopCounterMode 内部会临时给算子挂 hook，
        退出 with 时自动卸载）。会多跑一次前向，但不改变模型权重。

    说明
        相比 thop 这类只统计卷积/线性的工具，FlopCounterMode 会通过 aten 算子级
        拦截把注意力的 matmul 也算进去，并且按"实际张量形状"累加，因此能正确反映
        DynamicViT 剪枝后变短的序列长度——这是选它而不是 thop 的主要原因。
        只统计前向；训练还要反传，计算量约为前向的 3 倍，不在本脚本范围内。
    """
    # 延迟 import：这个模块在部分旧版 PyTorch 里不存在，放在函数内可以让
    # "只想要参数统计"的场景不至于在 import 阶段就失败。
    from torch.utils.flop_counter import FlopCounterMode

    # batch=1 的随机输入：得到的是"单张前向"的 FLOPs，与报告口径一致。
    # 输入内容不影响 FLOPs（卷积/矩阵乘的尺寸由形状决定），所以随机数够用。
    x = torch.randn(1, 3, img_size, img_size, device=device)

    # with 语句：FlopCounterMode 是上下文管理器，__enter__ 时注册算子 hook，
    # __exit__ 时卸载并汇总。必须让前向发生在 with 块内部，否则统计不到。
    # display=False 表示不要在退出时打印"按模块拆分的 FLOPs 表"，我们只要总数。
    # torch.inference_mode() 比 no_grad 更彻底：不记录版本计数、产出的是 inference
    # tensor，能进一步降低测量本身的开销，适合纯推理场景（此处的 tensor 不参与训练）。
    with torch.inference_mode():
        with FlopCounterMode(display=False) as counter:
            model(x)

    # get_total_flops() 返回 Python int（可能很大），再 int() 一次确保 JSON 可序列化。
    return int(counter.get_total_flops())


@torch.inference_mode()
def benchmark(
    model: torch.nn.Module,
    batch_size: int,
    warmup: int,
    iters: int,
) -> float:
    """测一个模型在给定 batch 下的单次前向耗时（毫秒），取 iters 次的中位数。

    参数
        model      : 已 eval 的模型。
        batch_size : 1 表示测单张延迟，64 表示测吞吐。
        warmup     : 预热次数，结果不计入统计。
        iters      : 正式计时次数，取其耗时中位数作为返回值。

    返回值
        float，毫秒/批（不是毫秒/张）。要换算成每张需再除以 batch_size。

    副作用
        无文件/输出副作用；但会连续跑 warmup+iters 次前向，占用 CPU 数秒。
        装饰器 @torch.inference_mode() 让函数体内所有前向都不建计算图。

    为什么需要预热（warmup）
        第一次前向要分配中间缓冲区、让 oneDNN/MKL 选定并打包算子实现、把权重
        搬进缓存。这些一次性开销若混进计时，会把结果拉高几倍。先空跑若干次，
        等进入稳态后再计时，测到的才是"持续推理"的速度。

    为什么取中位数而不是平均
        CPU 上偶发的调度抢占、降频、其他进程干扰会产生少量很长的样本；平均值
        会被这些长尾拖高，而中位数只反映"典型一次"，更稳定也更能复现。
    """
    # 输入只在 CPU 上创建一次并复用：避免把"分配 64 张图"的时间算进推理耗时。
    # 也不做归一化，因为这里只关心计算量/耗时，不关心数值正确性。
    x = torch.randn(batch_size, 3, 32, 32)

    # 预热：不记录时间。循环变量用 _ 表示"这个值不用"。
    for _ in range(warmup):
        model(x)

    times = []

    # 计时区间只包住一次 model(x)，把 Python 循环与列表操作的开销降到最低。
    # perf_counter() 是单调高精度时钟；乘 1000 把秒换算成毫秒。
    for _ in range(iters):
        t0 = time.perf_counter()
        model(x)
        times.append((time.perf_counter() - t0) * 1000.0)

    # np.median 对偶数个样本取中间两个的平均值，返回 numpy 标量，再 float() 转成
    # 原生 float 以便写 JSON（json 不认识 np.float64）。
    return float(np.median(times))


def summarize(times, batch_size: int) -> dict:
    """把一串耗时样本汇总成便于比较的统计量字典。

    参数
        times      : 每轮测得的"毫秒/批"样本，可为 list 或 numpy 数组。
        batch_size : 该组样本对应的批大小，用于换算每张耗时与吞吐。

    返回值
        dict，含：
            batch_size / rounds            批大小与样本数
            median_ms_per_batch            中位数（主指标）
            mean_ms_per_batch / min / max  均值与极值，用来看抖动范围
            median_ms_per_image            中位数 / batch_size
            images_per_second              batch_size * 1000 / 中位数

    副作用
        none（纯函数，只做 numpy 计算）。

    为什么同时给出 median 与 min/max
        中位数用于下结论；min 接近"无干扰时的最好成绩"，max 反映最坏抖动。
        三者的差距本身就是测量质量的信号——差距过大说明机器当时很吵，
        这组数字只能作参考。
    """
    # asarray 统一输入类型并指定 float64，避免整数列表在除法时被截断。
    times = np.asarray(times, dtype=np.float64)

    # 中位数是主指标：抗长尾干扰。
    median = float(np.median(times))
    return {
        "batch_size": batch_size,
        # 样本数 = 测了几轮。轮数越多，中位数越稳，但总耗时线性增长。
        "rounds": len(times),
        "median_ms_per_batch": median,
        "mean_ms_per_batch": float(np.mean(times)),
        "min_ms_per_batch": float(np.min(times)),
        "max_ms_per_batch": float(np.max(times)),
        # 批内平均到每张图。batch=1 时它就等于单张延迟。
        "median_ms_per_image": median / batch_size,
        # 吞吐：每批 batch_size 张、耗时 median 毫秒，故 张/秒 = batch*1000/median。
        # 这个换算把"毫秒"换成"秒"，单位不要漏。
        "images_per_second": batch_size * 1000.0 / median,
    }


def main() -> None:
    """选出 checkpoint、逐个测量、交错跑多轮、汇总写出 complexity.json。

    作用
        测量入口：负责"选哪些模型、测几轮、写到哪里"，把单模型测量交给
        load_model / count_flops / benchmark，把统计交给 summarize。

    参数
        无（通过命令行配置）：
            --threads  intra-op 线程数，默认 8，必须与训练时一致才可比；
            --rounds   交错测量轮数，默认 5；
            --out      输出 JSON 路径，默认 results/complexity.json；
            --tags     只测这些 tag（按 checkpoint 文件名去后缀匹配），默认全部。

    返回值
        None。

    副作用
        1. 读 checkpoints/*.pt；
        2. 覆盖写 --out 指定的 JSON；
        3. 每加载一个模型打印一行 [complexity] loaded ...，最后每个模型打印一行指标。

    整体步骤
        1) 解析参数，设置线程数；
        2) glob 出所有 .pt，过滤掉 logits/smoke/_last，再按 --tags 收窄；
        3) 逐个加载：记录参数量、FLOPs，并准备好空的计时列表；
        4) 交错跑 rounds 轮：偶数轮正序、奇数轮逆序，每个模型测 b1 与 b64；
        5) 对每个模型汇总统计，组装结果字典；
        6) json.dumps 写文件并打印。
    """
    # argparse：参数声明 + parse_args 的用法见 prepare_data.py 的注释。
    parser = argparse.ArgumentParser()

    # 线程数。PyTorch 的 CPU 推理靠 intra-op 并行（一次矩阵乘内部切成多块），
    # 所以这个值直接改变吞吐：8 线程的数字不能和 4 线程的数字横向比较。
    parser.add_argument("--threads", type=int, default=8)

    # 交错测量的轮数。轮数越多中位数越稳，代价是耗时线性增加。
    parser.add_argument("--rounds", type=int, default=5)

    # 输出路径默认 results/complexity.json（results/ 由 train.py 创建）。
    parser.add_argument("--out", type=str, default=str(ROOT / "results" / "complexity.json"))

    # nargs="*" 表示这个参数可以接 0 个或多个值，收集成 list；
    # 不写 --tags 时取 default=None，用于区分"没指定"和"指定了空列表"。
    # 用法：--tags baseline_cb dynamic_cb（空格分隔，不带 .pt）
    parser.add_argument("--tags", type=str, nargs="*", default=None)

    args = parser.parse_args()

    # set_num_threads 控制 intra-op（单个算子内部）的线程数，是影响 CPU 前向速度
    # 的主要开关。注意它与训练脚本里设置的线程数应保持一致，否则吞吐没法对比。
    # 这里刻意没有调用 torch.set_num_interop_threads：inter-op 并行指的是"多个算子
    # 同时跑"，对单个小模型前向几乎没有收益，而且该函数必须在任何并行工作开始前
    # 调用（否则可能抛 RuntimeError），在测量脚本里设置它反而容易踩坑。
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")

    ckpt_dir = ROOT / "checkpoints"

    # Path.glob("*.pt")：在 checkpoints 目录下做非递归模式匹配，返回"路径生成器"。
    # 用 sorted 包一层，得到按文件名排序的稳定列表——顺序会影响交错测量的次序，
    # 固定下来才能让多次运行的结果可比。
    paths = sorted(ckpt_dir.glob("*.pt"))

    # 列表推导式过滤：p.stem 是"去掉最后一个后缀的文件名"，
    # 例如 baseline_cb_train_logits.pt -> "baseline_cb_train_logits"。
    #   "logits" not in p.stem   排除教师 logits 文件（里面是 tensor，没有模型结构，
    #                            load_model 会因缺少 model_name 键而 KeyError）
    #   "smoke" not in p.stem    排除冒烟测试产生的小模型
    #   not endswith("_last")    排除每 epoch 覆盖写的续训快照，只测最终 best
    # 三个条件用 and 连接，全部满足才保留。
    paths = [
        p
        for p in paths
        if "logits" not in p.stem and "smoke" not in p.stem and not p.stem.endswith("_last")
    ]

    # --tags 收窄范围：用 set 做 O(1) 成员判断；比较的是 stem（不带 .pt）。
    if args.tags:
        wanted = set(args.tags)
        paths = [p for p in paths if p.stem in wanted]

    # 一个都没选出来时直接退出：继续跑只会写出空 JSON，容易让人误以为测过了。
    # SystemExit 会以非零状态码终止脚本。
    if not paths:
        raise SystemExit("no checkpoints selected")

    # 用 dict 而不是 list 保存：后面要按 tag 反复查找，且要保持插入顺序（Python 3.7+
    # 的 dict 有序），这样交错测量与最终输出的顺序都稳定。
    loaded = {}
    for path in paths:
        model, ckpt = load_model(path, device)
        loaded[path.stem] = {
            "model": model,
            # 保留整个 ckpt，后面要取 tag 与 model_name 写进结果。
            "ckpt": ckpt,
            # 参数量与 FLOPs 只与结构有关，测一次即可，不必每轮重复。
            "params": count_parameters(model),
            "flops": count_flops(model, device),
            # 两个批大小的计时样本容器，每轮 append 一个数。
            "b1": [],
            "b64": [],
        }
        print(f"[complexity] loaded {path.stem}")

    # 交错测量模型；每轮反转顺序，以抵消 CPU 频率/温度的单调漂移。
    # 若固定顺序，第一个模型总是"冷机"状态，会系统性偏快；反转后每个模型
    # 在各轮中处于"先测/后测"的次数近似相等，趋势被平均掉。
    for r in range(args.rounds):
        order = list(loaded)
        # r % 2 == 1 的奇数轮倒序。list(reversed(...)) 复制一份，避免改到 loaded 的键序。
        if r % 2 == 1:
            order = list(reversed(order))
        for tag in order:
            entry = loaded[tag]
            # batch=1：单次前向极短（毫秒级），噪声占比高，所以 warmup 60、iters 400，
            # 用大量重复让中位数稳定。结果作为"单张延迟"参考值。
            entry["b1"].append(benchmark(entry["model"], 1, warmup=60, iters=400))
            # batch=64：单次前向已较长，warmup 10、iters 80 就够；测的是吞吐。
            # 两个 batch 的迭代次数不同，是为了让每组测量总时长接近，兼顾稳定与耗时。
            entry["b64"].append(benchmark(entry["model"], 64, warmup=10, iters=80))

    results = {}
    for tag, entry in loaded.items():
        lat1 = summarize(entry["b1"], 1)
        lat64 = summarize(entry["b64"], 64)
        results[tag] = {
            # 从 checkpoint 取模型名与 tag，而不是从文件名猜，保证与训练记录一致。
            "model": entry["ckpt"]["model_name"],
            "tag": entry["ckpt"]["tag"],
            "params": entry["params"],
            # 同时给原始值和人类友好单位（M/G），下游画图时不必再除。
            "params_m": entry["params"] / 1e6,
            "flops": entry["flops"],
            "gflops": entry["flops"] / 1e9,
            "latency_b1": lat1,
            "latency_b64": lat64,
            # 记录线程数：脱离线程数的吞吐数字没有意义。
            "threads": args.threads,
        }
        # :16s 左对齐补到 16 字符，让多行输出对齐；:6.3f 表示总宽 6、保留 3 位小数。
        print(
            f"{tag:16s} params={entry['params']/1e6:6.3f}M flops={entry['flops']/1e9:6.3f}G "
            f"b1={lat1['median_ms_per_image']:.2f}ms/img b64={lat64['images_per_second']:.0f}img/s"
        )

    # indent=2 让 JSON 一行一个字段，便于人读与 git diff。
    # 这里没加 ensure_ascii=False，因为键和值都是 ASCII（模型名/tag 都是英文）。
    # 写文件显式 utf-8，避免 Windows 默认编码带来的乱码。
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"written -> {args.out}")


# 直接运行才执行 main；被 import 时（例如写单元测试）不会触发测量。
if __name__ == "__main__":
    main()
