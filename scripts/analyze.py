"""实验汇总分析脚本：把各实验的 JSON 结果整理成报告用的图与表。

===============================================================================
一、这个脚本做什么
===============================================================================
``scripts/train.py`` 每训练完一个模型，会把该次实验的配置、逐 epoch 训练历史、
测试集指标（accuracy / macro_f1 / balanced_accuracy / ece / confusion_matrix /
per_class_recall 等）写进 ``results/<tag>.json``；
``scripts/measure_complexity.py`` 把参数量、FLOPs、batch=64 延迟写进
``results/complexity.json``；``scripts/dynamic_tokens.py`` 把 DynamicViT 每个阶段
保留的 token 数写进 ``results/dynamic_tokens.json``。

本脚本不重新训练、也不碰图片像素，只做“读数 -> 画图 / 制表”：

* 生成 6 张报告插图（PNG，200 dpi）；
* 在终端打印一张汇总表，方便快速核对数字；
* 把同一张汇总表落成 CSV 与 Markdown 两个文件，供 DOCX 报告和 README 使用。

二、输入（全部由脚本自己按固定路径定位，不通过命令行参数传入）
-------------------------------------------------------------------------------
* ``data/meta.json`` —— 训练集 / 测试集每个类别的样本数
* ``results/{baseline_ce,baseline_cb,dynamic_cb,mobile_cb}.json`` —— 四组实验
* ``results/complexity.json`` —— 可选；缺失时参数量/FLOPs 相关的图会退化
* ``results/dynamic_tokens.json`` —— 可选；缺失时直接跳过 token 数图

三、输出
-------------------------------------------------------------------------------
* ``figures/fig_dataset.png``   训练集（不均衡）vs 测试集（均衡）的类别分布
* ``figures/fig_curves.png``    训练损失 / 验证准确率随 epoch 的曲线
* ``figures/fig_confusion.png`` 4 个模型按真实类归一化的混淆矩阵（含共用色标）
* ``figures/fig_per_class.png`` 各模型 10 类召回率的分组柱状图
* ``figures/fig_tradeoff.png``  准确率—参数量、准确率—FLOPs 两个权衡散点图
* ``figures/fig_tokens.png``    DynamicViT 逐阶段保留的 token 数
* ``results/summary.csv``       汇总表（utf-8-sig，Excel 双击不乱码）
* ``results/summary.md``        同一张表的 Markdown 版本

四、在实验流程中的位置
-------------------------------------------------------------------------------
prepare_data.py -> train.py（4 个 tag）-> measure_complexity.py -> dynamic_tokens.py
-> 【analyze.py】-> build_report.py（把图和表拼进 DOCX）

也就是说，本脚本是“实验跑完之后”的收尾环节：输入是前面几步产出的 JSON，
输出是报告素材。它不做任何训练，在 CPU 上几秒钟就能跑完，可以反复重跑。

五、怎么运行
-------------------------------------------------------------------------------
在项目根目录 ``vit_cifar10_lab`` 下执行（脚本用 ``__file__`` 定位项目根目录，
所以从其他工作目录调用也能找到文件）::

    python scripts/analyze.py

读代码前的两个小提示：

* 本文件所有画图函数的返回值都是 ``None``，它们的效果体现在磁盘上的 PNG；
* 画哪些实验、按什么顺序画，由全局常量 TAGS 统一决定（四个 tag 与 README
  里的四组实验一一对应），各函数只是消费它。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

# 【为什么必须先 use("Agg") 再 import pyplot】
# pyplot 在第一次被导入时就会锁定一个后端（有桌面时通常是 TkAgg / QtAgg）。
# 若顺序反了，部分环境下会报 "cannot switch backend"，甚至仍然尝试弹窗。
# Agg 是纯软件光栅化后端：只把像素写进文件，不依赖显示器，因此服务器、CI、
# 远程 SSH，以及本课程“无 GPU + 无显示器”的实验环境都能稳定保存 PNG。
# 本脚本从不调用 plt.show()，用 Agg 正合适。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 【项目根目录 = 本文件的上一级的上一级】
# __file__ 是当前脚本路径；.resolve() 先转成绝对路径（消除相对路径和符号链接），
# .parents[1] 再从 scripts/analyze.py 往上数两级：scripts/ 的上一级就是项目根。
# 这样无论用户从哪个工作目录执行，读写 data/、results/、figures/ 都不会跑偏。
ROOT = Path(__file__).resolve().parents[1]
# scripts/ 目录默认不在 sys.path 里，直接 import src.* 会失败；把项目根目录插到
# 模块搜索路径最前面，下一行才能正常导入 src.data。这是“脚本放在子目录里”的
# 标准处理方式，比让用户手动设置 PYTHONPATH 更省事。
sys.path.insert(0, str(ROOT))

# 这里引入 src.data 里的英文类名常量，主要用于对照“类别编号 -> 类别”的权威顺序；
# 本脚本画图时用的是下面 CN_CLASSES 的中文名，两者顺序必须完全一致，否则柱子
# 和混淆矩阵的行列会错位。由于上面调整过 sys.path，这行 import 不在文件顶部，
# flake8 的 E402（模块级导入不在顶部）会报警，行尾 noqa 表示“有意为之”。
from src.data import CLASS_NAMES  # noqa: E402

# 【中文字体】rcParams 是 matplotlib 的全局配置字典。font.sans-serif 是一个候选
# 列表，matplotlib 会按顺序挑系统里第一个能用的字体：微软雅黑 -> 黑体 -> 等线。
# 不设置的话，默认的 DejaVu Sans 没有汉字，图中的中文标题会显示成一串“豆腐块”。
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DengXian"]
# 【负号】不少中文字体缺少 U+2212（真正的数学减号），坐标轴上的 -0.2 会变成方块。
# 置 False 让 matplotlib 改用 ASCII 的 "-"，牺牲一点排版美观，换来正确显示。
plt.rcParams["axes.unicode_minus"] = False
# 屏幕预览 150 dpi、保存 200 dpi：报告插图要清晰，同时控制 PNG 体积。
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 200
# 只保留左、下两条坐标轴边框，去掉上、右，是报告里常用的简洁风格。
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

# 参与汇总 / 绘图的四个实验 tag。顺序决定折线图图例、分组柱状图的组顺序，
# 也决定 summary 表的行顺序；想加新实验时，在这里加 tag 并补 LABELS/SHORT/COLORS。
TAGS = ["baseline_ce", "baseline_cb", "dynamic_cb", "mobile_cb"]
# 图例和汇总表里展示的“人类可读”名字。
LABELS = {
    "baseline_ce": "ViT 基线 (CE)",
    "baseline_cb": "ViT + CB 损失",
    "dynamic_cb": "DynamicViT + CB",
    "mobile_cb": "MobileViT-Lite + CB",
}
# 散点图标注、分组柱状图图例空间有限，用更短的名字，避免文字互相压盖。
SHORT = {
    "baseline_ce": "ViT-CE",
    "baseline_cb": "ViT-CB",
    "dynamic_cb": "DynamicViT",
    "mobile_cb": "MobileViT-Lite",
}
# 每个模型固定一种颜色，跨图表保持一致；读者看曲线图时建立的“灰色=基线”印象，
# 在散点图、柱状图里依然成立，这是数据可视化里很有用的一致性。
COLORS = {
    "baseline_ce": "#8c8c8c",
    "baseline_cb": "#2f6db3",
    "dynamic_cb": "#d9822b",
    "mobile_cb": "#3a9d6e",
}
# CIFAR-10 十个类别的中文名，顺序与 src.data.CLASS_NAMES 完全对应：
# 0 飞机、1 汽车、2 鸟、3 猫、4 鹿、5 狗、6 蛙、7 马、8 船、9 卡车。
CN_CLASSES = ["飞机", "汽车", "鸟", "猫", "鹿", "狗", "蛙", "马", "船", "卡车"]


def load_results() -> dict:
    """读取所有已完成的实验结果 JSON，返回 ``{tag: 结果字典}``。

    每个 JSON 由 ``scripts/train.py`` 写出，顶层包含 ``tag`` / ``model`` /
    ``loss`` / ``params`` / ``history`` / ``train_epoch_mean_s`` / ``test`` 等键；
    本脚本只读取其中一部分（具体见各 ``fig_*`` 与 ``build_summary``）。

    参数:
        无。路径固定为 ``<项目根>/results/<tag>.json``，tag 取自全局 TAGS。

    返回:
        dict: 键是实验 tag（如 ``"baseline_ce"``），值是 ``json.loads`` 得到的
        嵌套 dict；其中 ``history`` 是“每个 epoch 一个 dict”的列表，
        ``test`` 里还含有 confusion_matrix、per_class_recall 等。

    异常:
        SystemExit: 四个 JSON 一个都不存在时主动退出，提示先跑训练脚本。
    """
    out = {}
    # 只加载存在的 JSON：允许“只跑了部分实验”时也能出图，而不是直接报错。
    for tag in TAGS:
        path = ROOT / "results" / f"{tag}.json"
        if path.exists():
            # 显式 encoding="utf-8"：Windows 默认编码是 GBK，而 JSON 里有中文，
            # 不写编码很容易读出乱码甚至 UnicodeDecodeError。
            out[tag] = json.loads(path.read_text(encoding="utf-8"))
    # 一个结果都没有时，后面所有图都无从谈起，直接给用户一条可操作的提示。
    if not out:
        raise SystemExit("no result JSON files found")
    return out


def load_complexity() -> dict:
    """读取 ``results/complexity.json``；文件不存在时返回空 dict。

    参数:
        无。

    返回:
        dict: ``{tag: {"params":..., "gflops":..., "latency_b64": {...}}}``。
        文件缺失时返回 ``{}``，调用方需要用 ``comp.get(tag, {})`` 这类写法
        容忍“复杂度还没测”的情况。
    """
    path = ROOT / "results" / "complexity.json"
    # 条件表达式一行完成“存在就读、不存在就给空字典”，这样后续代码不必到处判空。
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def fig_dataset() -> None:
    """画训练集 / 测试集的类别分布对比图，写入 ``figures/fig_dataset.png``。

    左图是训练集（不均衡，最多 4513 张、最少 2023 张），右图是测试集
    （每类 487 张，完全均衡）。并排放是为了让学生一眼看出：“不均衡只发生在
    训练集”。所以后文讨论 accuracy 与 macro-F1 / 单类召回的差异时，不能把
    测试集分布当成借口——测试集本身是均衡的。

    参数:
        无。读取 ``data/meta.json``（由 ``scripts/prepare_data.py`` 生成）。

    生成图片:
        ``figures/fig_dataset.png``

    返回:
        None。图片直接落盘，不返回 Figure 对象。
    """
    # meta.json 里的 counts 形如 {"train": {"0": 4435, ...}}；JSON 的对象键
    # 只能是字符串，所以下面取数时必须写 str(i) 而不是 i。
    meta = json.loads((ROOT / "data" / "meta.json").read_text(encoding="utf-8"))
    # 列表推导式按 0..9 的顺序取出 10 个类别的样本数，保证与 CN_CLASSES 对齐；
    # np.array 再把 Python list 变成向量，方便后面做 max/min 等整体运算。
    train = np.array([meta["counts"]["train"][str(i)] for i in range(10)])
    test = np.array([meta["counts"]["test"][str(i)] for i in range(10)])
    # plt.subplots(1, 2) 返回 (fig, axes)：fig 是整张画布，axes 是“子图数组”，
    # 这里形状为 (2,)，即 axes[0]、axes[1]。注意只有画单个子图时 ax 才是单个
    # Axes 对象；多子图一律是 ndarray，所以下面用 zip 遍历它而不是 ax.plot。
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    # zip 把多个可迭代对象按位置配对，长度以最短的为准；axes 有 2 个元素、
    # 后面三个列表也各 2 个元素，于是每轮循环恰好拿到“一个子图 + 一份数据 +
    # 一个标题 + 一种颜色”。这种写法比手写下标可读得多。
    for ax, values, title, color in zip(
        axes, [train, test], ["训练集（不均衡）", "测试集（均衡）"], ["#4c78a8", "#59a14f"]
    ):
        # ax.bar 返回一个 BarContainer，它是可迭代的，元素是每根柱子的 Rectangle；
        # 留着 bars 才能在下面读取每根柱子的左边界和宽度来放数值标签。
        bars = ax.bar(CN_CLASSES, values, color=color)
        ax.set_title(title)
        ax.set_ylabel("样本数")
        # 中文类别名较长，旋转 30 度避免互相重叠。
        ax.tick_params(axis="x", rotation=30)
        # 逐个柱子在柱顶写样本数：b.get_x() 是柱子左边界，加上半个宽度即中心；
        # v + max(values)*0.015 把文字抬高“最大值的 1.5%”，留出一点呼吸空间，
        # 这里用的是 y 轴数据坐标（不是像素），所以柱高变化时偏移量会自动缩放。
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v + max(values) * 0.015, str(v),
                    ha="center", va="bottom", fontsize=7)
    # 用“最大类 / 最小类”的样本数之比量化不均衡程度，越接近 1 越均衡。
    ratio = train.max() / train.min()
    fig.suptitle(f"类别分布：训练集最大/最小类样本数之比 = {ratio:.2f}", fontsize=12)
    # tight_layout 自动收缩子图间距，避免标题、坐标轴标签互相压盖。
    fig.tight_layout()
    # bbox_inches="tight" 在保存时裁掉画布四周多余的空白，报告排版更紧凑；
    # 它和 tight_layout 分工不同：前者管“整张图的边缘”，后者管“子图之间”。
    fig.savefig(ROOT / "figures" / "fig_dataset.png", bbox_inches="tight")
    # 关闭 figure、释放内存。脚本一次要画六张图，不关的话 matplotlib 会一直
    # 持有所有 Figure，内存随图数量增长。
    plt.close(fig)


def fig_curves(results: dict) -> None:
    """画四个实验的训练损失 / 验证准确率曲线，写入 ``figures/fig_curves.png``。

    参数:
        results: ``load_results()`` 的返回值，``{tag: 实验 JSON 字典}``。

    生成图片:
        ``figures/fig_curves.png``（左：训练损失；右：验证集准确率）

    返回:
        None。图例用 ``LABELS`` 里的完整模型名，颜色用 ``COLORS``。
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    # 字典在 Python 3.7+ 保持插入顺序；load_results 按 TAGS 顺序插入，所以
    # 这里的遍历顺序与 TAGS 一致，图例顺序稳定、可复现。
    for tag, res in results.items():
        # history 是“一个 epoch 一个 dict”的列表，先取出公共的 epoch 序列，
        # 再用列表推导抽出每条曲线的 y 值。
        hist = res["history"]
        ep = [h["epoch"] for h in hist]
        axes[0].plot(ep, [h["train_loss"] for h in hist], label=LABELS[tag], color=COLORS[tag])
        axes[1].plot(ep, [h["val_acc"] for h in hist], label=LABELS[tag], color=COLORS[tag])
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("训练损失")
    axes[0].set_title("训练损失曲线")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("验证集准确率")
    axes[1].set_title("验证集准确率曲线")
    # 遍历 ndarray 得到的是逐个 Axes；右图准确率单位是比例，网格线能帮读者读数。
    for ax in axes:
        ax.grid(alpha=0.25, linestyle="--")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "fig_curves.png", bbox_inches="tight")
    plt.close(fig)


def fig_confusion(results: dict) -> None:
    """画四个模型的按真实类归一化混淆矩阵，写入 ``figures/fig_confusion.png``。

    每个面板对应一个已有结果的实验：横轴是预测类别、纵轴是真实类别，
    颜色深浅表示“该真实类别的样本被预测成各类的比例”，对角线值就是单类召回率。
    面板标题还附了整体准确率，方便把“总体指标”和“逐类错误”对照着看。

    参数:
        results: ``load_results()`` 的返回值。

    生成图片:
        ``figures/fig_confusion.png``（2×2 子图 + 右侧共用色标）

    返回:
        None。若结果不足 4 个，多出来的空面板会被隐藏。
    """
    # 只画确实存在结果的实验，并保持 TAGS 的固定顺序，这样即使只跑了部分实验，
    # 面板的排列也是稳定的，不会因为 JSON 字典顺序而变。
    tags = [t for t in TAGS if t in results]
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 10.6))
    # 手动留白：右边留出约 16% 给色标，并调整子图间距。这里不用 tight_layout，
    # 因为下面要用 add_axes 手动把色标放进右侧空白，布局需要精确可控。
    fig.subplots_adjust(left=0.08, right=0.84, top=0.93, bottom=0.07, hspace=0.45, wspace=0.28)
    # axes 是 (2, 2) 的 ndarray；.flat 把它按“行优先”摊平成一维迭代器
    # （左上 -> 右上 -> 左下 -> 右下），于是可以用一个循环顺序取到四个面板，
    # 不必写 axes[0, 0]、axes[0, 1] 这样的下标。zip 再把面板和 tag 配对。
    for ax, tag in zip(axes.flat, tags):
        # JSON 里是嵌套 list，先转成 ndarray，才能做矩阵除法和 cm[i, j] 索引。
        cm = np.asarray(results[tag]["test"]["confusion_matrix"])
        # 按“真实类别”归一化：axis=1 表示沿列方向对每一行求和（每行 = 一个真实类
        # 被分成各类的次数）。keepdims=True 让结果保持 (10, 1) 而不是 (10,)，这样
        # 广播时每一行都除以自己那一行的样本数，得到“行和为 1”的比例矩阵。
        # 为什么要按真实类归一化？因为这样对角线元素恰好等于该类的 recall，
        # 稀有类（如卡车）不会被数量占优的类淹没；如果统一除以 4870，行和就不再
        # 是 1，比较不同类的错误率会很别扭。
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        # imshow 把矩阵当图片画。vmin=0、vmax=1 固定色彩映射范围：如果让每张图
        # 各自自动缩放，同一个深蓝在两张图里会代表不同的比例，跨模型就没法比。
        # imshow 默认 origin="upper"，第 0 行画在最上面，正好让“真实类别”从上到下
        # 按 0..9 排列，符合阅读习惯。
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        # 标题里带整体准确率；pad=12 把标题往上推，避免压住第一行刻度标签。
        ax.set_title(f"{LABELS[tag]}\nacc={results[tag]['test']['accuracy']:.3f}", fontsize=10, pad=12)
        # 默认刻度是像素坐标，先设成 0..9 的整数位置，再贴上中文类别名。
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.set_xticklabels(CN_CLASSES, fontsize=6.8, rotation=0, ha="center")
        ax.set_yticklabels(CN_CLASSES, fontsize=7)
        ax.set_xlabel("预测类别", fontsize=8)
        ax.set_ylabel("真实类别", fontsize=8)
        # 在每个格子里写比例。只标 >= 0.05 的格子：否则 100 个数字挤在一起，
        # 反而看不清主要错误模式。字色按背景深浅切换（深蓝底用白字）保证可读性。
        for i in range(10):
            for j in range(10):
                if cm_norm[i, j] >= 0.05:
                    ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                            fontsize=5.8, color="white" if cm_norm[i, j] > 0.55 else "black")
    # 结果不足 4 个时，剩余面板要关掉坐标轴，否则会露出空白坐标系。
    # ndarray.flat 支持切片，axes.flat[len(tags):] 正好是“没被用到的那些面板”。
    for ax in axes.flat[len(tags):]:
        ax.axis("off")
    # add_axes 的参数是“图坐标”（figure fraction，左下角为原点，范围 0~1）：
    # [left, bottom, width, height]。这里把色标放在右侧留白里（right=0.84 之后）。
    cax = fig.add_axes([0.865, 0.20, 0.016, 0.62])
    # fig.colorbar 需要知道“哪个可映射对象（ScalarMappable）的颜色规则”，也就是
    # im。四个面板共用同一个 cmap 和 vmin/vmax，所以取最后一次循环的 im 建一个
    # 共用色标即可；cax 指定色标画在哪个 Axes 里。
    cb = fig.colorbar(im, cax=cax)
    # 色标本身也是一个 Axes，可以设标题；pad 控制标题与色条的距离。
    cb.ax.set_title("比例", fontsize=8, pad=6)
    fig.savefig(ROOT / "figures" / "fig_confusion.png", bbox_inches="tight")
    plt.close(fig)


def fig_per_class(results: dict) -> None:
    """画各模型在十个类别上的召回率分组柱状图，写入 ``figures/fig_per_class.png``。

    参数:
        results: ``load_results()`` 的返回值。

    生成图片:
        ``figures/fig_per_class.png``

    返回:
        None。横轴 10 个类别，每个类别下并排若干根柱子（每个模型一根）。
    """
    tags = [t for t in TAGS if t in results]
    # np.arange(10) 生成 [0, 1, ..., 9]，作为 10 组柱子的“基准 x 坐标”。
    # 用整数位置而不是类别名，是因为要在此基础上做数值偏移。
    x = np.arange(10)
    # 所有模型共享总宽度 0.8（<1，于是组与组之间留出空隙），每个模型分 0.8/n。
    # max(len(tags), 1) 防止没有结果时除以 0。
    width = 0.8 / max(len(tags), 1)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for i, tag in enumerate(tags):
        recalls = results[tag]["test"]["per_class_recall"]
        # 分组柱状图的关键就是偏移：第 i 根柱子从基准位置向左/右平移。
        # (i - (n-1)/2) 让 n 根柱子关于基准 x 对称，再乘 width 换算成坐标偏移，
        # 例如 n=4 时偏移系数是 -1.5、-0.5、+0.5、+1.5，整组正好以 x 为中心。
        ax.bar(x + (i - (len(tags) - 1) / 2) * width, recalls, width,
               label=SHORT[tag], color=COLORS[tag])
    # 刻度要落在基准 x 上（组中心），而不是某根柱子上，否则标签会偏。
    ax.set_xticks(x)
    ax.set_xticklabels(CN_CLASSES)
    ax.set_ylabel("单类召回率")
    # 上限略高于 1，给柱顶和图例留出空间，避免贴边。
    ax.set_ylim(0, 1.05)
    ax.set_title("各模型在均衡测试集上的单类召回率")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "fig_per_class.png", bbox_inches="tight")
    plt.close(fig)


def fig_tradeoff(results: dict, comp: dict) -> None:
    """画“准确率—参数量”“准确率—FLOPs”两个权衡散点图，写入 ``figures/fig_tradeoff.png``。

    参数:
        results: ``load_results()`` 的返回值（提供测试准确率）。
        comp: ``load_complexity()`` 的返回值（提供参数量、FLOPs、吞吐）。

    生成图片:
        ``figures/fig_tradeoff.png``。每个点是一个模型，右图还在点旁标注
        batch=64 的吞吐（img/s）。

    返回:
        None。若没有任何 tag 同时具备训练结果与复杂度数据，函数直接返回，
        不生成图片（这是“可选输入缺失时优雅降级”的写法）。
    """
    # 必须同时有训练结果和复杂度测量结果，才能算出一个散点。
    tags = [t for t in TAGS if t in results and t in comp]
    if not tags:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for tag in tags:
        # 参数量除以 1e6 换算成“百万（M）”，轴上数字更短、更好读。
        p = comp[tag]["params"] / 1e6
        g = comp[tag]["gflops"]
        a = results[tag]["test"]["accuracy"]
        tput = comp[tag]["latency_b64"]["images_per_second"]
        # scatter 的 s 是标记面积（单位是 point^2，不是直径）；edgecolor="white"
        # 给点描一圈白边，点重叠时也能分清；zorder=3 让点画在网格线之上。
        axes[0].scatter(p, a, s=90, color=COLORS[tag], edgecolor="white", zorder=3)
        # annotate 的 xy 是“被标注的数据点”，textcoords="offset points" 表示
        # xytext=(6, 6) 的单位是“点（point，1/72 英寸）”而不是数据坐标：
        # 文字永远相对数据点向右上偏 6 pt，不会因为坐标轴量纲变化而飞走。
        # 这正是它比 ax.text 适合做散点标签的原因。
        axes[0].annotate(SHORT[tag], (p, a), textcoords="offset points", xytext=(6, 6), fontsize=8)
        axes[1].scatter(g, a, s=90, color=COLORS[tag], edgecolor="white", zorder=3)
        # 第二行文字标注吞吐；xytext=(6, -2) 让标签落在点的右下方，避免压住别的点。
        axes[1].annotate(f"{SHORT[tag]}\n{tput:.0f} img/s", (g, a), textcoords="offset points",
                         xytext=(6, -2), fontsize=7)
    axes[0].set_xlabel("参数量 (M)")
    axes[0].set_ylabel("测试准确率")
    axes[0].set_title("准确率—参数量")
    axes[1].set_xlabel("FLOPs (G)")
    axes[1].set_ylabel("测试准确率")
    axes[1].set_title("准确率—计算量（标注为 batch=64 吞吐）")
    # margins 在数据范围外再加一圈留白，防止 annotate 的文字被坐标轴边缘裁掉。
    for ax in axes:
        ax.grid(alpha=0.25, linestyle="--")
        ax.margins(x=0.18, y=0.12)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "fig_tradeoff.png", bbox_inches="tight")
    plt.close(fig)


def fig_tokens() -> None:
    """画 DynamicViT 逐阶段保留的 token 数，写入 ``figures/fig_tokens.png``。

    参数:
        无。读取 ``results/dynamic_tokens.json``（由 ``scripts/dynamic_tokens.py``
        生成）。文件不存在时直接返回，不生成图片，也不报错。

    生成图片:
        ``figures/fig_tokens.png``。横轴是四个阶段（输入、Block 1/2/3 之后），
        纵轴是保留的 token 数（包含 1 个 CLS token）。

    返回:
        None。
    """
    path = ROOT / "results" / "dynamic_tokens.json"
    # 这是“可选材料”：没跑 dynamic_tokens.py 时跳过，而不是让整个分析失败。
    if not path.exists():
        return
    info = json.loads(path.read_text(encoding="utf-8"))
    stages = info["stages"]
    counts = info["kept_tokens"]
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    # 四个阶段给四种颜色，配合下面柱顶的数字，一眼看出 65 -> 49 -> 25 -> 7 的下降。
    # 颜色列表长度必须与柱数一致；多出的颜色会被忽略，少了会循环复用。
    bars = ax.bar(stages, counts, color=["#4c78a8", "#f58518", "#e45756", "#72b7b2"])
    # 柱顶标数字；这里 c + 1.5 是 y 轴数据坐标下的固定偏移（token 数都是整数，
    # 1.5 个单位足够把文字抬到柱顶之上），与 fig_dataset 的做法思路相同。
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c + 1.5, str(c), ha="center", fontsize=9)
    ax.set_ylabel("保留 token 数（含 cls）")
    ax.set_title("DynamicViT 逐阶段 token 数")
    # 纵轴上限放宽到最大值的 1.18 倍，避免最高的柱子顶到图框、数字被裁掉。
    ax.set_ylim(0, max(counts) * 1.18)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "fig_tokens.png", bbox_inches="tight")
    plt.close(fig)


def build_summary(results: dict, comp: dict) -> pd.DataFrame:
    """把四个实验的关键指标汇总成一张表，并写成 CSV 与 Markdown。

    参数:
        results: ``load_results()`` 的返回值。
        comp: ``load_complexity()`` 的返回值；某个 tag 没有复杂度数据时，
            对应单元格用 ``np.nan`` 占位，表格结构仍然完整。

    生成表格 / 文件:
        * ``results/summary.csv`` —— ``encoding="utf-8-sig"``，Excel 直接打开不乱码
        * ``results/summary.md``  —— 手写的 Markdown 表格，供 README / 汇报使用

    返回:
        pandas.DataFrame: 与写入文件的内容一致，供 ``main()`` 在终端打印。
        列顺序就是下面每个 row 字典的键顺序。
    """
    rows = []
    # 按 TAGS 的固定顺序组装行，保证表格行序稳定；缺哪个实验就跳过哪一行。
    for tag in TAGS:
        if tag not in results:
            continue
        r = results[tag]
        # 没有复杂度数据时给空 dict，后面用 .get 取值就不会 KeyError。
        c = comp.get(tag, {})
        # 每行是一个 dict：键会变成 DataFrame 的列名，所以这里的顺序就是列顺序。
        # 数值统一乘 100 或除以 1e6，是为了让表格里是“人类习惯”的单位（%、M）。
        rows.append(
            {
                "模型": LABELS[tag],
                "损失": r["loss"].upper(),
                "参数量(M)": c.get("params", r["params"]) / 1e6,
                "FLOPs(G)": c.get("gflops", np.nan),
                "测试准确率(%)": r["test"]["accuracy"] * 100,
                "宏平均F1(%)": r["test"]["macro_f1"] * 100,
                "平衡准确率(%)": r["test"]["balanced_accuracy"] * 100,
                "ECE(%)": r["test"]["ece"] * 100,
                "训练秒/epoch": r["train_epoch_mean_s"],
                "批延迟(ms/64)": c.get("latency_b64", {}).get("median_ms_per_batch", np.nan),
                "吞吐(img/s)": c.get("latency_b64", {}).get("images_per_second", np.nan),
            }
        )
    # 用“列表套字典”构造 DataFrame：每个字典是一行，键自动成为列名。
    df = pd.DataFrame(rows)
    # index=False 很关键：否则 CSV 会多出一列没有名字的行号。
    # encoding="utf-8-sig" 会写入 UTF-8 BOM：Windows 版 Excel 靠 BOM 判断编码，
    # 只用 utf-8 的话中文列名常被显示成乱码（这是教学里最常踩的坑之一）。
    df.to_csv(ROOT / "results" / "summary.csv", index=False, encoding="utf-8-sig")
    headers = list(df.columns)
    # 手写 Markdown 表格的三段式：表头行、分隔行（--- 表示左对齐）、数据行。
    # 这里没有用 df.to_markdown()，是为了不额外依赖 tabulate 库。
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    # iterrows() 返回 (行号, 行 Series)；行号用不上，按惯例写成 _ 表示“丢弃”。
    for _, row in df.iterrows():
        cells = []
        for h in headers:
            v = row[h]
            # 保留两位小数；isinstance 同时检查 Python float 和 numpy 浮点类型，
            # 因为 pandas 取出的标量往往是 np.float64，只写 float 会漏掉。
            # 注意：某列一旦含有 NaN，pandas 会把整列提升为 float，于是原本的
            # 整数指标也会走“两位小数”这个分支——这是可接受的显示差异。
            cells.append(f"{v:.2f}" if isinstance(v, (float, np.floating)) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    (ROOT / "results" / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return df


def main() -> None:
    """按固定顺序跑完整个分析流水线：准备目录 -> 读数据 -> 画六张图 -> 输出汇总表。

    整体步骤：
        1. 确保 ``figures/`` 目录存在（``exist_ok=True`` 使重复运行不报错）；
        2. 读四个实验的 JSON 与可选的复杂度 JSON；
        3. 依次生成数据集分布图、训练曲线、混淆矩阵、单类召回、权衡散点、token 数；
        4. 生成汇总表并打印到终端，方便和报告里的数字对账。

    参数:
        无。

    返回:
        None。副作用是写出 ``figures/*.png``、``results/summary.csv`` 与
        ``results/summary.md``。
    """
    # 输出目录可能还不存在（比如刚 clone 下来只跑了训练）；exist_ok=True 让
    # “目录已存在”不抛异常，脚本可以反复运行。
    (ROOT / "figures").mkdir(exist_ok=True)
    results = load_results()
    comp = load_complexity()
    # 下面每一步都是“读已有结果 -> 画图 -> 存盘”，彼此独立，顺序只影响日志观感。
    fig_dataset()
    fig_curves(results)
    fig_confusion(results)
    fig_per_class(results)
    fig_tradeoff(results, comp)
    fig_tokens()
    df = build_summary(results, comp)
    # index=False 与 CSV 同理：终端表格不需要左侧的行号列。
    print(df.to_string(index=False))
    print("figures ->", ROOT / "figures")


if __name__ == "__main__":
    # 只有“直接运行本文件”时才执行；被其他脚本 import 时不会自动画图，
    # 这样 build_report.py 之类的脚本可以安全地复用上面的函数。
    main()
