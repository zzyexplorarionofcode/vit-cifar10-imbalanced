"""绘制“代码阅读脉络图”，帮助初学者按依赖顺序阅读本项目。

图中不展示实验结果，只回答三个问题：
1. 先读哪些文件；
2. 每个文件负责哪一段流程；
3. 数据形状如何从图像变成分类 logits。

运行：
    python scripts/make_code_roadmap.py
输出：
    figures/fig_code_roadmap.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

# Agg 后端不依赖图形界面，适合在服务器/无显示器环境生成图片。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
# 中文字体：Windows 上优先微软雅黑，其次黑体；避免中文显示成方框。
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DengXian"]
plt.rcParams["axes.unicode_minus"] = False


def card(ax, x, y, w, h, title, files, question, desc, color, number):
    """画一个圆角卡片：编号 + 阶段标题 + 关键文件 + 一句话说明。

    参数使用 0–100 的“画布坐标”，这样调整布局时不需要关心像素。
    """
    # 卡片主体：圆角矩形，facecolor 用浅色，edgecolor 用阶段主色。
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.35,rounding_size=1.2",
        facecolor=color[0], edgecolor=color[1], linewidth=1.4,
    )
    ax.add_patch(box)

    # 左上角编号圆点：让“阅读顺序”一眼可见。
    ax.add_patch(plt.Circle((x + 2.2, y + h - 2.4), 1.35, color=color[1], zorder=5))
    ax.text(x + 2.2, y + h - 2.4, str(number), ha="center", va="center",
            color="white", fontsize=9, fontweight="bold", zorder=6)

    # 阶段标题与文件清单。
    ax.text(x + 4.2, y + h - 2.45, title, ha="left", va="center",
            fontsize=11, fontweight="bold", color=color[1])
    ax.text(x + 2.0, y + h - 5.6, files, ha="left", va="top",
            fontsize=8.2, color="#1f1f1f", linespacing=1.35)
    # 中间一行“读这个文件先回答什么问题”，避免初学者迷失在语法细节里。
    ax.text(x + 2.0, y + 9.2, question, ha="left", va="center",
            fontsize=7.6, color=color[1], style="italic")
    ax.text(x + 2.0, y + 2.0, desc, ha="left", va="bottom",
            fontsize=7.8, color="#444444", linespacing=1.3)


def arrow(ax, xy1, xy2, color="#666666", style="-|>", ls="-"):
    """画带箭头的连线；ls='--' 表示“辅助/可选”的阅读关系。"""
    ax.add_patch(
        FancyArrowPatch(
            xy1, xy2, arrowstyle=style, mutation_scale=12,
            linewidth=1.2, color=color, linestyle=ls,
            connectionstyle="arc3,rad=0.0",
        )
    )


def main() -> None:
    """创建画布、放置 6 张阶段卡片和箭头，并保存为 PNG。"""
    fig, ax = plt.subplots(figsize=(13.2, 8.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # 四个阶段的配色：(浅底色, 主色)。
    data_c = ("#e8f0fb", "#2f6db3")
    model_c = ("#eaf6ef", "#2e7d4f")
    train_c = ("#fff2e2", "#c46a12")
    eval_c = ("#f2ecf8", "#6a4c93")

    ax.text(2, 96, "代码阅读脉络：从一张 CIFAR 图像到实验报告", fontsize=17,
            fontweight="bold", color="#17365D")
    ax.text(2, 92.2, "建议按 1 → 6 阅读；每个文件顶部、函数 docstring 和复杂语法处都有中文注释。",
            fontsize=9.2, color="#555555")

    # 第一行：数据 → 模型 → 损失。
    card(ax, 2, 62, 28, 24, "数据准备",
         "scripts/prepare_data.py\nsrc/data.py",
         "先问：为什么把 JPEG 先转成 npy？",
         "JPEG 文件夹 → uint8 数组\nDataset / 随机裁剪 / 翻转 / Cutout", data_c, 1)
    card(ax, 36, 62, 28, 24, "模型基础",
         "src/models/vit.py",
         "先问：一张图怎样变成 64 个 token？",
         "Patch Embedding → [CLS]+位置编码\n→ 4×Transformer Block → 分类头", model_c, 2)
    card(ax, 70, 62, 28, 24, "损失函数",
         "src/losses.py",
         "先问：CB 权重为什么能帮少数类？",
         "CE / Class-Balanced Loss\n教师-学生日志蒸馏 KL", data_c, 3)

    # 第二行：训练 → 轻量化 → 分析（从右往左，形成蛇形顺序）。
    card(ax, 70, 24, 28, 24, "训练与评估",
         "src/engine.py\nscripts/train.py",
         "先问：一个 epoch 里各组件怎么配合？",
         "训练循环 / 验证选模 / 测试\ncheckpoint / 教师 logits", train_c, 4)
    card(ax, 36, 24, 28, 24, "两种轻量化",
         "src/models/dynamic_vit.py\nsrc/models/mobile_vit.py\nsrc/models/factory.py",
         "先问：剪枝/卷积怎样减少计算？",
         "DynamicViT：token 剪枝\nMobileViT-Lite：卷积+局部注意力", model_c, 5)
    card(ax, 2, 24, 28, 24, "复杂度与报告",
         "scripts/measure_complexity.py\nscripts/analyze.py\nscripts/build_report.py",
         "先问：参数、FLOPs、吞吐说明什么？",
         "参数量 / FLOPs / 吞吐\n曲线 / 混淆矩阵 / DOCX 报告", eval_c, 6)

    # 主阅读顺序：1→2→3→4→5→6。
    arrow(ax, (30.5, 74), (35.2, 74))
    arrow(ax, (64.5, 74), (69.2, 74))
    arrow(ax, (84, 61.2), (84, 48.8), color="#c46a12")
    arrow(ax, (69.2, 36), (64.5, 36), color="#2e7d4f")
    arrow(ax, (35.2, 36), (30.5, 36), color="#6a4c93")

    # 辅助说明：如果时间有限，先读这三个文件。
    ax.add_patch(FancyBboxPatch((2, 7.0), 96, 10.5,
                                boxstyle="round,pad=0.3,rounding_size=1.0",
                                facecolor="#fffbe8", edgecolor="#d4a72c",
                                linewidth=1.0))
    ax.text(4, 14.0, "时间有限时的“最小闭环”", fontsize=9.5, fontweight="bold", color="#8a5a00")
    ax.text(4, 10.2,
            "先读  src/models/vit.py  （模型怎么搭）  →  scripts/train.py  （模型怎么训）  "
            "→  src/engine.py  （指标怎么算）。  "
            "再看  src/losses.py  理解 CB 损失，最后看两个轻量化模型。",
            fontsize=8.3, color="#4a3a00")

    # 底部形状流：把“张量形状”作为阅读时的抓手。
    ax.text(2, 1.6,
            "形状流：  [B, 32, 32, 3] 图像  →  [B, 3, 32, 32] 张量  →  [B, 64, 96] patch token  "
            "→  [B, 65, 96] 加 [CLS]  →  [B, 10] logits",
            fontsize=7.6, color="#666666")

    fig.savefig(ROOT / "figures" / "fig_code_roadmap.png", bbox_inches="tight", dpi=220)
    plt.close(fig)
    print("written ->", ROOT / "figures" / "fig_code_roadmap.png")


if __name__ == "__main__":
    main()
