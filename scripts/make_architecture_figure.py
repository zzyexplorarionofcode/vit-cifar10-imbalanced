"""绘制报告用的模型结构示意图：一张图里并排对比 ViT / DynamicViT / MobileViT-Lite。

===============================================================================
一、这个脚本做什么
===============================================================================
用 matplotlib 的“补丁（Patch）+ 文本”在归一化坐标里手工摆放方框和箭头，
画出一张三段式的结构示意图：

* (a) ViT 基线的前向流程：输入 -> Patch Embed -> Token 序列 -> 位置编码 ->
      Transformer Block x4 -> 分类头 -> 10 类 softmax；
* (b) DynamicViT：在 Block 1/2/3 之后逐级剪枝，token 数 65 -> 49 -> 25 -> 7；
* (c) MobileViT-Lite：卷积 stem + 局部表示 + Unfold + patch 内自注意力 +
      Fold/残差 + 全局 Block。

它不读任何实验结果，也不训练模型，只是把“结构长什么样”画成图，属于报告里
“方法”一节的插图；图里的数字（4x4 patch、4 个 Block、75%/50%/25% 保留率、
65/49/25/7 个 token）都与 src/ 里的实现和 README 一致，改结构时要同步改这里。

二、输入
-------------------------------------------------------------------------------
无。所有坐标、文字、颜色都写死在代码里（示意图本来就该人工排版），
因此不需要 data/、results/、checkpoints/ 中的任何文件。

三、输出
-------------------------------------------------------------------------------
``figures/fig_architecture.png`` —— 一张 12 x 5.6 英寸、220 dpi 的 PNG，保存时
用 ``bbox_inches="tight"`` 裁掉四周空白。

四、在实验流程中的位置
-------------------------------------------------------------------------------
它不依赖实验顺序，任何时候都能跑；通常在训练/分析完成后、由 build_report.py
把这张图插进 DOCX 报告之前生成。若报告里想改结构画法，只改本文件即可。

五、怎么运行
-------------------------------------------------------------------------------
在项目根目录 ``vit_cifar10_lab`` 下执行::

    python scripts/make_architecture_figure.py

脚本用 ``__file__`` 定位项目根目录，所以从其他工作目录调用也能把图存到
正确位置；运行成功会打印图片的绝对路径。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

# 【为什么用 Agg 后端】pyplot 导入时就会锁定后端，所以必须在 import pyplot
# 之前调用 use()。Agg 是纯软件光栅化后端，只写文件、不开窗口，因此在没有
# 显示器的服务器 / CI / 远程 SSH 环境中也能稳定生成 PNG。本脚本从不 show()。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# 这两个 Patch 类是本图的两大“画笔”：FancyBboxPatch 画圆角矩形，
# FancyArrowPatch 画带箭头的连线。它们都来自 matplotlib.patches 子模块。
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# 项目根目录 = 本文件的上一级的上一级（scripts/ 的父目录）。用 __file__ 而不是
# 当前工作目录，保证从任何位置运行都能把图写进项目的 figures/。
ROOT = Path(__file__).resolve().parents[1]
# 【中文字体】按候选顺序找系统里可用的中文字体：微软雅黑 -> 黑体 -> 等线；
# 不设置的话默认字体没有汉字，图中所有中文都会变成方块。
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DengXian"]
# 【负号】中文字体常缺 U+2212 减号，若将来打开坐标轴刻度，负数会显示成方块；
# 置 False 改用 ASCII "-"。本图隐藏了刻度，这里保留该设置是为了与 analyze.py
# 保持一致，以后加坐标轴也不会踩坑。
plt.rcParams["axes.unicode_minus"] = False


def box(ax, xy, w, h, text, fc="#e8f0fb", ec="#2f6db3", fs=9):
    """在 ax 上画一个圆角矩形方框，并在框中心写一段文字。

    这是本脚本最基础的“积木”，用于画出流程图里的每一个步骤节点。

    参数:
        ax: matplotlib 的 Axes 对象（由 main 里的 plt.subplots 创建）。
        xy: 二元组 (x, y)，方框【左下角】的位置，单位是数据坐标（本图 0~1）。
        w, h: 方框的宽和高，同样是数据坐标。
        text: 方框内要显示的文字；可以含 ``\\n`` 手动换行。
        fc: facecolor，填充色，默认浅蓝。
        ec: edgecolor，边框色，默认深蓝。
        fs: fontsize，字号，单位是点（point），默认 9。

    在图片中生成:
        一个 FancyBboxPatch（圆角矩形）和一段居中的 Text，最终出现在
        ``figures/fig_architecture.png`` 里。

    返回:
        None。补丁对象直接被加到 ax 上，不返回句柄。
    """
    # FancyBboxPatch 的 xy 是矩形【左下角】，不是中心——这是最容易记错的一点；
    # 文字想居中就得自己加 w/2、h/2（见下一行）。
    p = FancyBboxPatch(
        xy, w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor=fc, edgecolor=ec, linewidth=1.1
    )
    # 与 ax.plot 不同，Patch 对象不会自动出现在图上，必须显式 add_patch，
    # 否则画了等于没画（初学者常见的“代码跑了但图是空的”）。
    ax.add_patch(p)
    # 文本坐标也必须是同一个数据坐标系，所以用“左下角 + 半个宽高”得到中心；
    # ha/va 都设 center，让文字围绕该点居中，而不是从点向右上排。
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, xy1, xy2, color="#555555"):
    """在 ax 上画一条从 xy1 指向 xy2 的带箭头连线。

    参数:
        ax: matplotlib 的 Axes 对象。
        xy1: 起点 (x, y)，数据坐标。
        xy2: 终点 (x, y)，数据坐标；箭头画在这一端。
        color: 线条和箭头颜色，默认深灰。

    在图片中生成:
        一个 FancyArrowPatch，最终出现在 ``figures/fig_architecture.png`` 里。

    返回:
        None。同样需要 add_patch 才会渲染。
    """
    # FancyArrowPatch 的坐标也是数据坐标，和方框共用一套 0~1 坐标系，所以只要
    # 把端点对齐到方框边界，箭头就会“贴”在框上，改布局时不用换算像素。
    # arrowstyle="-|>" 表示实心三角箭头；mutation_scale 控制箭头头部大小（单位
    # 是点，不是数据单位），linewidth 只影响箭杆粗细。
    ax.add_patch(FancyArrowPatch(xy1, xy2, arrowstyle="-|>", mutation_scale=9,
                                 linewidth=1.0, color=color))


def main() -> None:
    """搭建画布并依次画出 (a)(b)(c) 三段结构示意图，最后保存为 PNG。

    整体步骤：
        1. 建一张 12 x 5.6 英寸的画布，把 x/y 都限制在 0~1 并隐藏坐标轴，
           形成“归一化坐标系”；
        2. 画 (a) ViT 基线的七个方框和六个箭头；
        3. 画 (b) DynamicViT 的四个阶段（65 -> 49 -> 25 -> 7 tokens）及 prune 标注；
        4. 画 (c) MobileViT-Lite 的六个模块；
        5. 以 220 dpi、``bbox_inches="tight"`` 保存到 ``figures/fig_architecture.png``。

    参数:
        无。所有坐标、文字、颜色都在函数体内写死（示意图需要人工排版，
        用常量控制比从配置读取更直观）。

    返回:
        None。副作用是写出 ``figures/fig_architecture.png`` 并打印其路径。
    """
    # 单个子图时 plt.subplots 返回 (fig, ax)，ax 是 Axes 对象（不是数组）；
    # 只有多子图时第二个返回值才是 ndarray。figsize 单位是英寸。
    fig, ax = plt.subplots(figsize=(12, 5.6))
    # 【坐标系统】把 x、y 都限制为 0~1 的“归一化坐标”，之后所有方框、文字、
    # 箭头都用这个尺度的数据坐标摆放。好处是布局与 figsize、dpi 解耦：改画布
    # 大小只会整体缩放，相对位置不变。注意 y 轴向上为正（左下角是 (0,0)）。
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # 关掉坐标轴（刻度、边框、背景全隐藏），这是画示意图而不是数据图的标准操作。
    ax.axis("off")

    # ---------------- (a) ViT 基线前向流程 ----------------
    # 一条水平“行”的基准 y；方框从 y-0.06 画到 y+0.07，因此框的垂直中心在 y+0.005。
    y = 0.76
    # 七个方框各自的【左边界】x 坐标。宽度单独放在下面的 widths 里，便于微调。
    xs = [0.015, 0.14, 0.29, 0.445, 0.61, 0.775, 0.885]
    labels = [
        "输入\n32×32×3",
        "Patch Embed\n4×4, stride 4",
        "Token 序列\n[CLS]+64",
        "+ 位置编码\nPre-LN",
        "Transformer\nBlock ×4",
        "LN + 线性\n分类头",
        "10 类\nsoftmax",
    ]
    # 与 xs 一一对应的宽度。最后两个框窄一些，因为文字短且要给右侧留边。
    widths = [0.10, 0.125, 0.13, 0.13, 0.14, 0.095, 0.085]
    # 三元 zip：同时拿到“左边界、宽度、文字”三个等长列表的一组值；
    # 方框左下角是 (x, y-0.06)，高度 0.13，于是垂直方向正好以 y 为中心。
    for x, w, t in zip(xs, widths, labels):
        box(ax, (x, y - 0.06), w, 0.13, t, fs=8.5)
    # 相邻框之间画箭头：从第 i 个框的右边界 xs[i]+widths[i] 指向第 i+1 个框的
    # 左边界 xs[i+1]，y 用框中心 y+0.005，保证箭头水平且居中对齐。
    for i in range(len(xs) - 1):
        arrow(ax, (xs[i] + widths[i], y + 0.005), (xs[i + 1], y + 0.005))

    # 段落标题，放在这一行的上方。
    ax.text(0.015, 0.95, "(a) ViT 基线前向流程", fontsize=11, fontweight="bold")

    # ---------------- (b) DynamicViT 渐进剪枝 ----------------
    # DynamicViT strip
    # 第二段的基准 y 比第一段低，保证两段之间有空隙，不互相压盖。
    y2 = 0.44
    ax.text(0.015, 0.565, "(b) DynamicViT：在 Block 1/2/3 后渐进剪枝（保留 75%/50%/25%）",
            fontsize=10.5, fontweight="bold")
    # 65 = 1 个 CLS + 64 个 patch；之后每次按保留率对 patch token 取 top-k：
    # 64*0.75=48 -> 49（含 CLS）；48*0.5=24 -> 25；24*0.25=6 -> 7。
    # 这些数字与 src/models/dynamic_vit.py 的 keep_ratios 实现一致。
    stages = ["65 tokens", "49 tokens", "25 tokens", "7 tokens"]
    # 四个阶段的方框左边界；间距比 (a) 大，因为只有四个框、文字也短。
    xs2 = [0.06, 0.30, 0.54, 0.78]
    # enumerate 同时给出下标 i 和值；i 只用来判断“是不是第一阶段”和“要不要画箭头”。
    for i, (x, s) in enumerate(zip(xs2, stages)):
        # 三元条件：i=0（输入阶段）用蓝色，其余三个“剪枝后”阶段用橙色，
        # 颜色本身就在传达“这里发生了剪枝”。
        box(ax, (x, y2 - 0.05), 0.15, 0.10, s, fc="#fff2e2" if i else "#e8f0fb",
            ec="#d9822b" if i else "#2f6db3")
        # 最后一段后面没有“下一段”，用 i < 3 避免索引越界。
        if i < 3:
            # 从当前框右边界连到下一个框的左边界，颜色用橙色强调剪枝步骤。
            arrow(ax, (x + 0.15, y2), (xs2[i + 1], y2), color="#d9822b")
            # “prune”标注在箭头上方（y2 + 0.025），x 略微右移让文字居中于箭头段。
            ax.text(x + 0.165, y2 + 0.025, "prune", fontsize=7.5, color="#d9822b")
    # 补充说明放在最后一个框下方；va="center" 让文字块以其坐标点为垂直中心。
    # 注意这一行在图的最右侧，x=0.78 与最后一个框的左边界对齐。
    ax.text(0.78, y2 - 0.095, "推理 FLOPs 显著下降，参数量基本不变",
            fontsize=8.2, color="#d9822b", va="center")

    # ---------------- (c) MobileViT-Lite 结构 ----------------
    # MobileViT block
    ax.text(0.015, 0.25, "(c) MobileViT-Lite：卷积 stem + 局部/全局混合块",
            fontsize=10.5, fontweight="bold")
    # 第三段基准 y；方框高 0.10，所以框占据 y3 ~ y3+0.10，中心是 y3+0.05。
    y3 = 0.10
    # 每个元素是 (左边界 x, 方框文字)。用列表而不是两个平行列表，是为了让
    # “这个框在哪儿、写什么”保持在一起，增删模块时不容易错位。
    mb = [
        (0.02, "卷积 stem\n32→8×8, d=72"),
        (0.20, "局部表示\nDWConv 3×3 + PW"),
        (0.38, "Unfold\npatch 4×4"),
        (0.55, "patch 内\n自注意力"),
        (0.70, "Fold + PW\n+ 残差"),
        (0.85, "全局 Block\n+ 平均池化"),
    ]
    # enumerate 提供下标 i 用于判断是否是最后一个模块。
    for i, (x, t) in enumerate(mb):
        box(ax, (x, y3), 0.13, 0.10, t, fc="#eaf6ef", ec="#3a9d6e", fs=8)
        # 最后一个模块后面不画箭头，否则箭头会指向画布外。
        if i < len(mb) - 1:
            # 起点是当前框右边界 x+0.13，终点是下一个模块的左边界 mb[i+1][0]；
            # y3+0.05 正好是框的垂直中心。
            arrow(ax, (x + 0.13, y3 + 0.05), (mb[i + 1][0], y3 + 0.05), color="#3a9d6e")

    # dpi=220 提高输出分辨率（PNG 像素尺寸 ≈ figsize * dpi）；bbox_inches="tight"
    # 裁掉画布四周的空白，避免报告里插图带一大圈白边。注意这个脚本没有调用
    # tight_layout，因为坐标轴已隐藏、所有元素都是绝对定位，不需要再自动排布。
    fig.savefig(ROOT / "figures" / "fig_architecture.png", bbox_inches="tight", dpi=220)
    # 释放 Figure，避免交互式后端下窗口越开越多（Agg 下也建议保持这个习惯）。
    plt.close(fig)
    print("written ->", ROOT / "figures" / "fig_architecture.png")


if __name__ == "__main__":
    # 直接运行本文件时才画图；被 import 时不会产生副作用。
    main()
