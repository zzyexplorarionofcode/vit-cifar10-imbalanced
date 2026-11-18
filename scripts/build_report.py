"""把实验结果自动排版成中文 DOCX 实验报告。

这个脚本不参与模型训练或推理，它只做“数据 → 文档”的搬运：
1. 读取 results/*.json、results/complexity.json、data/meta.json；
2. 用 python-docx 逐段写入标题、正文、表格、图片；
3. 最后保存 report/CIFAR10_ViT实验报告.docx。

阅读建议：先看 main 流程（文件末尾的 doc = setup_document() 到 doc.save()），
再看 body/table/figure 等排版辅助函数；不需要逐行读报告正文的文字。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TAGS = ["baseline_ce", "baseline_cb", "dynamic_cb", "mobile_cb"]
LABELS = {
    "baseline_ce": "ViT 基线（CE）",
    "baseline_cb": "ViT + CB 损失",
    "dynamic_cb": "DynamicViT + CB",
    "mobile_cb": "MobileViT-Lite + CB",
}
TABLE_NAMES = {
    "baseline_ce": "ViT (CE)",
    "baseline_cb": "ViT + CB",
    "dynamic_cb": "DynamicViT",
    "mobile_cb": "MobileViT-Lite",
}
CN_CLASSES = ["飞机", "汽车", "鸟", "猫", "鹿", "狗", "蛙", "马", "船", "卡车"]
CLASS_EN = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

results = {t: json.loads((ROOT / "results" / f"{t}.json").read_text(encoding="utf-8")) for t in TAGS}
comp = json.loads((ROOT / "results" / "complexity.json").read_text(encoding="utf-8"))
meta = json.loads((ROOT / "data" / "meta.json").read_text(encoding="utf-8"))
tokens = json.loads((ROOT / "results" / "dynamic_tokens.json").read_text(encoding="utf-8"))


def fmt(x: float, nd: int = 2) -> str:
    """把浮点数格式化为固定小数位，便于表格对齐。"""
    return f"{x:.{nd}f}"


def signed(x: float, nd: int = 2) -> str:
    """格式化带正负号的数值，例如 +1.4 / -2.0。"""
    return f"{x:+.{nd}f}"


def metric(tag: str, key: str) -> float:
    """从某个实验的测试指标字典中取一个标量。"""
    return float(results[tag]["test"][key])


def c(tag: str, key: str, default=0.0):
    """安全读取 complexity.json 中的字段，缺失时返回默认值。"""
    return comp.get(tag, {}).get(key, default)


def params_m(tag: str) -> float:
    """返回参数量（单位：百万）。"""
    return c(tag, "params", results[tag]["params"]) / 1e6


def gflops(tag: str) -> float:
    """返回单张前向 FLOPs（单位：GFLOPs）。"""
    return c(tag, "gflops", 0.0)


def latency(tag: str) -> float:
    """返回 batch=1 的单张延迟中位数（毫秒）；仅作参考。"""
    return c(tag, "latency_b1", {}).get("median_ms_per_image", float("nan"))


def throughput(tag: str) -> float:
    """返回 batch=64 的吞吐（images/second）。"""
    return c(tag, "latency_b64", {}).get("images_per_second", float("nan"))


def batch_latency(tag: str) -> float:
    """返回 batch=64 的批延迟（毫秒/64 张）。"""
    return c(tag, "latency_b64", {}).get("median_ms_per_batch", float("nan"))


def epoch_time(tag: str) -> float:
    """返回每个 epoch 的平均训练秒数。"""
    return float(results[tag]["train_epoch_mean_s"])


def rel_pct(new: float, old: float) -> str:
    """计算相对变化百分比，例如 +18.1%。"""
    return signed((new / old - 1.0) * 100.0, 1) + "%"


def font(run, name: str = "宋体", size: float = 10.5, bold: bool = False, color: str | None = None):
    """同时设置西文和中文（eastAsia）字体，避免中文显示成方框。"""
    run.font.name = "Times New Roman"
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), name)
    run.font.size = Pt(size)
    run.bold = bold
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


def body(doc: Document, text: str, indent: bool = True, lead: str | None = None, size: float = 10.5):
    """添加一段正文，可选首行缩进和加粗小标题。"""
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.line_spacing = 1.35
    pf.space_after = Pt(4)
    if indent:
        pf.first_line_indent = Pt(size * 2)
    if lead:
        r = p.add_run(lead)
        font(r, "黑体", size, bold=True)
    r = p.add_run(text)
    font(r, "宋体", size)
    return p


def bullet(doc: Document, text: str, lead: str | None = None):
    """添加一个项目符号段落，用于列要点。"""
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.line_spacing = 1.3
    p.paragraph_format.space_after = Pt(2)
    if lead:
        r = p.add_run(lead)
        font(r, "黑体", 10.5, bold=True)
    r = p.add_run(text)
    font(r, "宋体", 10.5)


def formula(doc: Document, text: str):
    """添加居中的公式文本（用 Unicode 数学符号表示）。"""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(5)
    r = p.add_run(text)
    font(r, "Cambria Math", 10.5)


def note(doc: Document, text: str):
    """添加小号灰色注释，用于表注、说明。"""
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.5)
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run(text)
    font(r, "楷体", 9, color="595959")


def caption(doc: Document, text: str):
    """添加居中的图/表标题。"""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(7)
    r = p.add_run(text)
    font(r, "楷体", 9, color="404040")


def heading(doc: Document, text: str, level: int = 1, page_break: bool = False):
    """添加标题，可选在标题前强制分页。"""
    if page_break:
        doc.add_page_break()
    h = doc.add_heading(level=level)
    h.paragraph_format.space_before = Pt(10 if level == 1 else 8)
    h.paragraph_format.space_after = Pt(5)
    r = h.add_run(text)
    font(r, "黑体", {1: 16, 2: 13, 3: 11.5}[level], bold=True, color="1F1F1F")


def shade(cell, fill: str):
    """给 Word 表格单元格设置背景色（十六进制 RGB 字符串）。"""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tcPr.append(shd)


def table(doc: Document, headers, rows, widths=None, font_size: float = 8.5, first_col_left: bool = False):
    """添加固定列宽的三线表风格表格，支持表头重复和跨页不断行。"""
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    tblPr = t._tbl.tblPr
    tblLayout = OxmlElement("w:tblLayout")
    tblLayout.set(qn("w:type"), "fixed")
    tblPr.append(tblLayout)
    hdr = t.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = ""
        p = hdr[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(str(h))
        font(r, "黑体", font_size, bold=True)
        shade(hdr[i], "DCE6F1")
        hdr[i].vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    trPr = t.rows[0]._tr.get_or_add_trPr()
    tblHeader = OxmlElement("w:tblHeader")
    tblHeader.set(qn("w:val"), "true")
    trPr.append(tblHeader)
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if (first_col_left and i == 0) else WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(str(val))
            font(r, "宋体", font_size)
            cells[i].vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        trPr = t.rows[-1]._tr.get_or_add_trPr()
        trPr.append(OxmlElement("w:cantSplit"))
    if widths:
        for i, w in enumerate(widths):
            t.columns[i].width = Cm(w)
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
    return t


def figure(doc: Document, name: str, cap: str, width_in: float = 6.1):
    """插入 figures/ 下的图片并自动加图注。"""
    path = ROOT / "figures" / name
    if not path.exists():
        note(doc, f"[缺少图片 {name}]")
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(2)
    p.add_run().add_picture(str(path), width=Inches(width_in))
    caption(doc, cap)


def add_page_number(section):
    """在页脚插入 Word 的 PAGE 域，让页码自动更新。"""
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = footer.add_run()
    fldChar1 = OxmlElement("w:fldChar")
    fldChar1.set(qn("w:fldCharType"), "begin")
    instrText = OxmlElement("w:instrText")
    instrText.set(qn("xml:space"), "preserve")
    instrText.text = "PAGE"
    fldChar2 = OxmlElement("w:fldChar")
    fldChar2.set(qn("w:fldCharType"), "end")
    run._r.append(fldChar1)
    run._r.append(instrText)
    run._r.append(fldChar2)
    font(run, "宋体", 9, color="666666")


def setup_document() -> Document:
    """创建 A4 文档，设置页边距、正文样式、页眉页脚。"""
    doc = Document()
    sec = doc.sections[0]
    sec.page_height = Cm(29.7)
    sec.page_width = Cm(21.0)
    sec.top_margin = Cm(2.3)
    sec.bottom_margin = Cm(2.2)
    sec.left_margin = Cm(2.4)
    sec.right_margin = Cm(2.4)
    sec.different_first_page_header_footer = True

    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "宋体")
    normal.font.size = Pt(10.5)

    header = sec.header.paragraphs[0]
    header.text = "CIFAR-10 图像分类实验：ViT 基线、类别不均衡与轻量化"
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    for r in header.runs:
        font(r, "宋体", 8.5, color="7F7F7F")
    add_page_number(sec)
    return doc


doc = setup_document()

# --------------------------------------------------------------------------------------
# Title page
# --------------------------------------------------------------------------------------
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(80)
r = p.add_run("CIFAR-10 图像分类实验报告")
font(r, "黑体", 24, bold=True, color="17365D")

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(10)
r = p.add_run("—— Vision Transformer 基线、类别不均衡处理与轻量化")
font(r, "微软雅黑", 13, color="2F5597")

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(18)
r = p.add_run("（独立完成）")
font(r, "楷体", 11, color="595959")

info = doc.add_table(rows=4, cols=2)
info.alignment = WD_TABLE_ALIGNMENT.CENTER
info.autofit = False
info_rows = [("姓    名", "祝梓元"), ("学    号", "U202314554"),
             ("实验日期", "2026 年 11 月 10 日"),
             ("实验环境", "PyTorch 2.12 (CPU)，16 逻辑核，15.7 GB 内存")]
for i, (k, v) in enumerate(info_rows):
    for j, text in enumerate((k, v)):
        cell = info.rows[i].cells[j]
        cell.text = ""
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(text)
        font(r, "宋体", 10.5, bold=(j == 0))
        cell.width = Cm(4.0 if j == 0 else 7.5)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(100)
r = p.add_run("关键词：Vision Transformer；CIFAR-10；类别不均衡；DynamicViT；MobileViT；FLOPs")
font(r, "宋体", 9, color="666666")

# --------------------------------------------------------------------------------------
# Abstract
# --------------------------------------------------------------------------------------
doc.add_page_break()
heading(doc, "摘要", 1)
body(
    doc,
    "本实验围绕 CIFAR-10 图像分类，从零搭建 Vision Transformer（ViT），并在类别不均衡训练集上研究"
    "代价敏感学习与两类结构轻量化方法。实验采用课程提供的 35,580 张不均衡训练图像（最大/最小类样本数"
    "之比约 2.23:1）和 4,870 张均衡测试图像，从训练集中分层抽取 10% 作为验证集，测试集只在最终评估时"
    "使用。基线模型采用 4×4 patch、4 个 Transformer Block、96 维嵌入，参数量 0.311 M，单张前向"
    "FLOPs 约 0.045 G。针对不均衡问题，本文实现基于有效样本数的类别平衡交叉熵（Class-Balanced Loss，"
    "CB），在不改变采样流程和推理成本的前提下提高少数类权重。轻量化方面实现两种结构级方法："
    "（1）DynamicViT 风格的渐进式 token 剪枝，在 Block 1/2/3 后按 75%/50%/25% 保留 patch token，"
    "使推理 token 数由 65 降至 7；（2）MobileViT-Lite 混合结构，用卷积 stem、patch 内自注意力和"
    "一个全局 Block 替代标准 ViT 主干。两种轻量模型均使用 CB 基线作为教师进行日志蒸馏训练。"
    "本文使用 PyTorch FlopCounterMode 统计 FLOPs，并在相同 CPU 环境下测量 batch=64 吞吐与每轮训练时间。"
    f"实验结果表明：CB 损失使测试准确率从 {results['baseline_ce']['test']['accuracy']*100:.1f}% 提升到 "
    f"{results['baseline_cb']['test']['accuracy']*100:.1f}%，少数类召回提升更明显；DynamicViT 在参数量"
    f"基本不变的情况下将 FLOPs 降低约 {(1-gflops('dynamic_cb')/gflops('baseline_cb'))*100:.0f}%，"
    f"准确率下降约 {abs((results['dynamic_cb']['test']['accuracy']-results['baseline_cb']['test']['accuracy'])*100):.1f} "
    f"个百分点；MobileViT-Lite 将参数量降低约 {(1-params_m('mobile_cb')/params_m('baseline_cb'))*100:.0f}%、"
    f"FLOPs 降低约 {(1-gflops('mobile_cb')/gflops('baseline_cb'))*100:.0f}%，测试准确率反而提高约 "
    f"{(results['mobile_cb']['test']['accuracy']-results['baseline_cb']['test']['accuracy'])*100:.1f} 个百分点，"
    f"但吞吐只提高约 {(throughput('mobile_cb')/throughput('baseline_cb')-1)*100:.0f}%。两种方法的"
    "精度—速度—存储权衡并不相同。报告最后结合混淆矩阵、单类召回、参数量、计算量和速度对结果进行了"
    "分析，并讨论了实验局限与改进方向。",
)
p = doc.add_paragraph()
p.paragraph_format.first_line_indent = Pt(21)
r = p.add_run("关键词：")
font(r, "黑体", 10.5, bold=True)
r = p.add_run("Vision Transformer；类别不均衡；Class-Balanced Loss；DynamicViT；MobileViT；模型复杂度")
font(r, "宋体", 10.5)

# --------------------------------------------------------------------------------------
# Contents
# --------------------------------------------------------------------------------------
doc.add_page_break()
heading(doc, "目录", 1)
toc_items = [
    ("1  实验概述", "4"),
    ("2  数据集与问题分析", "5"),
    ("3  Vision Transformer 原理与基线", "7"),
    ("4  类别不均衡处理方法", "9"),
    ("5  ViT 轻量化方法", "11"),
    ("6  实验设置", "14"),
    ("7  实验结果", "16"),
    ("8  分析与讨论", "21"),
    ("9  局限与改进方向", "24"),
    ("10  结论", "24"),
    ("参考文献", "26"),
    ("附录 A  代码结构", "27"),
    ("附录 B  复现命令", "28"),
    ("附录 C  关键代码", "29"),
    ("附录 D  代码阅读脉络", "30"),
]
for name, page in toc_items:
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.35
    p.paragraph_format.space_after = Pt(1)
    p.paragraph_format.tab_stops.add_tab_stop(Cm(15.5))
    r = p.add_run(name)
    font(r, "宋体", 10.0)
    r = p.add_run("\t" + page)
    font(r, "宋体", 10.5, color="666666")
note(doc, "注：目录页码与本次 PDF 排版一致；若在 Word 中修改内容导致重新分页，请以实际页码为准。")

# --------------------------------------------------------------------------------------
# 1 Overview
# --------------------------------------------------------------------------------------
heading(doc, "1  实验概述", 1, page_break=True)
heading(doc, "1.1  实验背景", 2)
body(
    doc,
    "Vision Transformer（ViT）把图像切分为固定大小的 patch，将每个 patch 线性映射为 token，再使用"
    "标准 Transformer 编码器建模 token 之间的全局关系。与卷积神经网络相比，ViT 的归纳偏置更弱，"
    "在中小规模数据上通常需要更强的数据增广、正则化和更长的训练；但其结构统一、易于分析，也方便通过"
    "token 剪枝、混合卷积、蒸馏等方式进行轻量化。CIFAR-10 只有 32×32 分辨率，若直接使用 16×16 patch"
    "仅得到 4 个 token，难以体现注意力建模能力，因此本实验采用 4×4 patch，得到 64 个 patch token。",
)
body(
    doc,
    "课程训练集按类别呈现明显不均衡：多数类 4,513 张，少数类 2,023 张；测试集则每类 487 张，完全均衡。"
    "若直接使用普通交叉熵，模型可能偏向多数类，在均衡测试集上的宏平均指标和少数类召回会受损。另一方面，"
    "ViT 的自注意力计算量随 token 数平方增长，标准 Transformer 的参数量和计算量也较大，因此有必要研究"
    "结构级轻量化方法，并定量比较精度、参数量、FLOPs 与速度之间的取舍。",
)

heading(doc, "1.2  实验任务", 2)
for lead, text in [
    ("ViT 基线：", "使用 PyTorch 从零实现 ViT，在不均衡训练集上训练，在均衡测试集上评估。"),
    ("类别不均衡：", "分析不均衡程度，实现至少一种处理方法，说明原理、实施方式与结果影响。"),
    ("轻量化：", "实现至少两种非“仅调小维度/层数”的结构级方法，比较参数量、FLOPs 和速度。"),
    ("复杂度与结果分析：", "统计参数量、FLOPs、训练/推理时间，结合分类指标分析收益与代价。"),
]:
    bullet(doc, text, lead)

heading(doc, "1.3  主要工作", 2)
body(
    doc,
    "（1）建立统一的数据划分、增广、训练和评估流程，固定随机种子，并保存每个模型的配置、日志、"
    "checkpoint、混淆矩阵和预测结果，保证实验可复现。"
)
body(
    doc,
    "（2）实现 Class-Balanced Loss，与普通交叉熵在完全相同的网络结构、数据划分和训练轮数下对比，"
    "并从单类召回、宏平均 F1、平衡准确率和校准误差等角度分析其作用。"
)
body(
    doc,
    "（3）实现 DynamicViT 风格的渐进 token 剪枝：使用轻量 MLP 预测 token 重要性，训练时采用 Gumbel"
    "噪声 top-k 与直通估计器，推理时真实丢弃 token，因此 FLOPs 的下降可在前向计算中兑现。"
)
body(
    doc,
    "（4）实现 MobileViT-Lite 混合结构：卷积 stem 降低分辨率并提取局部特征，MobileViT Block 在"
    "patch 内做自注意力，最后用一个全局 Transformer Block 建模全局关系；该结构同时降低参数量和 FLOPs。"
)
body(
    doc,
    "（5）使用 PyTorch 内置 FlopCounterMode 统计前向 FLOPs，在相同 CPU 线程设置下测量 batch=64 "
    "吞吐，并结合精度绘制“精度—参数量”“精度—FLOPs”“精度—吞吐”对比图。"
)

# --------------------------------------------------------------------------------------
# 2 Dataset
# --------------------------------------------------------------------------------------
heading(doc, "2  数据集与问题分析", 1, page_break=True)
heading(doc, "2.1  数据划分", 2)
train_counts = [int(meta["counts"]["train"][str(i)]) for i in range(10)]
test_counts = [int(meta["counts"]["test"][str(i)]) for i in range(10)]
body(
    doc,
    f"训练集共 {sum(train_counts):,} 张，测试集共 {sum(test_counts):,} 张，图像均为 32×32 RGB。"
    "课程数据已按类别存放在 0–9 十个文件夹中，文件夹编号与 CIFAR-10 的标准类别顺序一致。"
    "为了避免用测试集选模造成信息泄漏，本文从训练集中按类别分层抽取 10% 作为验证集，"
    f"最终用于训练的样本为 {results['baseline_cb']['train_size']:,} 张，验证集 "
    f"{results['baseline_cb']['val_size']:,} 张；测试集只在训练结束后评估一次。"
)
caption(doc, "表 1  各类别样本数统计")
rows = []
for i in range(10):
    rows.append([f"{i} ({CN_CLASSES[i]}/{CLASS_EN[i]})", f"{train_counts[i]:,}", f"{test_counts[i]:,}"])
rows.append(["合计", f"{sum(train_counts):,}", f"{sum(test_counts):,}"])
table(doc, ["类别", "训练集", "均衡测试集"], rows, widths=[6.0, 3.4, 3.4])

max_cls = int(np.argmax(train_counts))
min_cls = int(np.argmin(train_counts))
body(
    doc,
    f"训练集多数类为“{CN_CLASSES[max_cls]}”（{train_counts[max_cls]:,} 张），少数类为“{CN_CLASSES[min_cls]}”"
    f"（{train_counts[min_cls]:,} 张），最大/最小类样本数之比为 {max(train_counts)/min(train_counts):.2f}:1；"
    f"多数类占比 {max(train_counts)/sum(train_counts)*100:.1f}%，少数类占比 "
    f"{min(train_counts)/sum(train_counts)*100:.1f}%。"
    "这一不均衡程度属于中度不均衡：它不足以让模型完全忽略少数类，但足以使普通交叉熵的决策边界向多数类偏移，"
    "因此在均衡测试集上，总体准确率与宏平均指标可能出现分离。"
)

heading(doc, "2.2  不均衡的影响机制", 2)
body(
    doc,
    "设类别 c 的训练样本数为 n_c，普通交叉熵对每个样本等权，因此总损失中层 c 的贡献正比于 n_c。"
    "在梯度层面，多数类的平均梯度会占据主导，模型为降低总体损失会把更多决策区域划给多数类；"
    "在贝叶斯意义上，若训练分布与测试分布不同，模型学到的后验会向训练先验偏移。"
    "本文测试集每类样本数相同，相当于均匀先验，因此训练先验与测试先验的不一致是影响少数类召回的主要原因。"
)
body(
    doc,
    "需要强调的是，不均衡处理并不必然提高总体准确率：牺牲多数类的一部分正确率、换取少数类召回提升时，"
    "总体准确率可能持平甚至略降，但宏平均 F1、平衡准确率和少数类召回会改善。因此本文同时报告总体准确率、"
    "宏平均 F1、平衡准确率、单类召回和期望校准误差（ECE），避免只用单一指标下结论。"
)

heading(doc, "2.3  预处理与数据增广", 2)
body(
    doc,
    "所有图像保持 32×32 原始分辨率，不做缩放。训练阶段使用：随机裁剪（先按 4 像素反射填充，再随机裁回"
    "32×32）、随机水平翻转、Cutout（以 0.25 概率擦除 4–10 像素的随机矩形）以及 CIFAR-10 的通道均值和"
    "标准差归一化。测试阶段只做归一化，不做裁剪或翻转。"
)
body(
    doc,
    "为降低 CPU 上读取 3.5 万张小 JPEG 的开销，预处理脚本一次性把图像解码为 uint8 数组（NHWC），"
    "训练时在 batch 张量上一次性完成随机裁剪、翻转、擦除和归一化。这样既避免了 DataLoader 多进程在"
    "Windows 上的额外开销，也使数据加载不成为瓶颈。"
)
figure(doc, "fig_dataset.png", "图 1  训练集与测试集的类别分布（训练集最大/最小类之比约 2.23:1）")

# --------------------------------------------------------------------------------------
# 3 ViT
# --------------------------------------------------------------------------------------
heading(doc, "3  Vision Transformer 原理与基线", 1, page_break=True)
heading(doc, "3.1  整体前向流程", 2)
body(
    doc,
    "ViT 把图像视为一个 token 序列，前向流程可以概括为：Patch Embedding → 拼接 [CLS] token → 加入"
    "位置编码 → 堆叠 Transformer Encoder Block → LayerNorm → 取 [CLS] 表示分类。本文基线结构如"
    "图 2(a) 所示。与 NLP Transformer 不同的是，图像 token 由卷积切块得到，位置信息通过可学习的位置"
    "编码注入；分类不使用逐 token 输出，而是使用 [CLS] token 的最终表示。"
)
figure(doc, "fig_architecture.png", "图 2  ViT 基线、DynamicViT 渐进剪枝与 MobileViT-Lite 的结构示意")

heading(doc, "3.2  Patch Embedding 与 token 化", 2)
body(
    doc,
    "设输入图像为 X∈R^{H×W×3}，patch 大小为 P×P。使用核大小和步长均为 P 的卷积实现切块与线性映射："
)
formula(doc, "X_p = Flatten( Conv_{P×P, stride P}(X) ) ∈ R^{N×d},   N = (H/P)(W/P)")
body(
    doc,
    "本文 H=W=32、P=4，因此 N=64、d=96。在 patch token 序列前拼接一个可学习的 [CLS] token，"
    "并加上可学习位置编码 E_pos："
)
formula(doc, "Z_0 = [ x_cls ; X_p^1 ; … ; X_p^N ] + E_pos,   Z_0 ∈ R^{(N+1)×d}")
body(
    doc,
    "位置编码使模型能够区分不同空间位置的 patch。由于 CIFAR-10 分辨率固定，本文使用一维可学习位置"
    "编码，长度 65；没有使用插值。"
)

heading(doc, "3.3  多头自注意力", 2)
body(
    doc,
    "对第 l 层的 token 表示 Z_l 做 LayerNorm 后，通过三个线性变换得到 Query、Key、Value，并分成 h 个头："
)
formula(doc, "Q = Z_l W_Q,  K = Z_l W_K,  V = Z_l W_V,   W_Q,W_K,W_V ∈ R^{d×d}")
formula(doc, "head_i = softmax( Q_i K_i^T / √d_h ) V_i,   d_h = d/h")
formula(doc, "MHSA(Z_l) = Concat(head_1, …, head_h) W_O")
body(
    doc,
    "本文 d=96、h=4，每个头的维度为 24。缩放因子 1/√d_h 防止点积随维度增大而方差过大，"
    "从而使 softmax 进入饱和区。自注意力的计算复杂度为 O(N²d)，当 token 数较多时是主要开销之一；"
    "这也是 token 剪枝类方法的主要动机。"
)

heading(doc, "3.4  Transformer Encoder Block", 2)
body(doc, "本文采用 Pre-LN 结构，每个 Block 由多头自注意力和前馈网络（FFN）两个子层组成，均带残差连接：")
formula(doc, "Z' = Z + MHSA( LN(Z) )")
formula(doc, "Z'' = Z' + FFN( LN(Z') ),   FFN(x) = GELU(xW_1)W_2")
body(
    doc,
    "FFN 的隐藏维度为 2d=192。Pre-LN 相比原始 Transformer 的 Post-LN 更容易训练，深层时不需要"
    "复杂的学习率预热；本文仍使用 2 个 epoch 的线性预热，以提高小模型训练的稳定性。Block 中还使用"
    "DropPath（随机深度，最大概率 0.05）和 Dropout（本文设为 0）进行正则化。"
)

heading(doc, "3.5  分类与训练目标", 2)
body(
    doc,
    "最后一层 LayerNorm 后取 [CLS] token 的表示 z_L^0，经线性分类头得到 10 类 logits，再经 softmax"
    "得到类别概率："
)
formula(doc, "p(y|x) = softmax( W_head LN(z_L^0) + b_head )")
body(
    doc,
    "基线使用带标签平滑（ε=0.1）的交叉熵损失。标签平滑把真实类目标从 1 变为 1−ε，其余类均分 ε，"
    "可以抑制过度自信、改善校准，但也可能略微降低训练集拟合速度。优化器为 AdamW，学习率 5×10⁻⁴，"
    "权重衰减 0.05（LayerNorm、bias、位置编码和 [CLS] token 不做权重衰减），余弦退火，梯度裁剪 1.0。"
)

heading(doc, "3.6  基线配置", 2)
caption(doc, "表 2  ViT 基线结构配置")
base_rows = [
    ["输入分辨率", "32×32×3"],
    ["Patch 大小 / 步长", "4×4 / 4"],
    ["Patch token 数", "64"],
    ["嵌入维度 d", "96"],
    ["Transformer Block 数 L", "4"],
    ["注意力头数 h", "4"],
    ["FFN 隐藏维度", "192（mlp ratio = 2）"],
    ["位置编码", "可学习一维编码，长度 65"],
    ["分类头", "LayerNorm + Linear(96→10)"],
    ["参数量", f"{params_m('baseline_cb'):.3f} M"],
    ["单张前向 FLOPs", f"{gflops('baseline_cb'):.3f} G"],
]
table(doc, ["项目", "设置"], base_rows, widths=[6.0, 8.0], first_col_left=True)
note(
    doc,
    "说明：本实验在 CPU 上完成，为了在可接受时间内完成多组对照，基线采用了较小的嵌入维度和层数；"
    "这会影响绝对精度，但不影响在统一协议下比较各方法的相对差异。",
)


# --------------------------------------------------------------------------------------
# 4 Imbalance
# --------------------------------------------------------------------------------------
heading(doc, "4  类别不均衡处理方法", 1, page_break=True)
heading(doc, "4.1  方法选择", 2)
body(
    doc,
    "常见的类别不均衡处理可以分为三类：数据层重采样（对少数类过采样、对多数类欠采样或使用平衡采样器）、"
    "损失层重加权（类别权重、Focal Loss、Class-Balanced Loss）以及后处理（logit adjustment、阈值调整）。"
    "过采样会重复少数类样本，增加过拟合风险并改变每轮迭代的样本分布；欠采样会丢弃大量多数类信息；"
    "平衡采样器虽然实现简单，但每个 batch 的分布由采样器决定，且会改变训练轮数的含义。"
)
body(
    doc,
    "本文选择损失层的 Class-Balanced Loss（CB Loss）。原因是：它不丢弃样本、不改变数据分布和采样流程，"
    "不增加任何推理开销，便于与轻量化方法组合；同时它直接针对“有效样本数”建模，比简单反频率加权更有"
    "理论依据。为了隔离变量，CB 模型与 CE 模型使用完全相同的网络结构、划分、增广、优化器和训练轮数，"
    "唯一区别是损失函数。"
)
caption(doc, "表 3  不均衡处理方法对比")
table(
    doc,
    ["方法", "核心思想", "优点", "局限"],
    [
        ["随机过采样", "重复少数类样本", "实现简单，不丢信息", "重复样本易过拟合，训练分布改变"],
        ["随机欠采样", "丢弃多数类样本", "训练快，类别平衡", "丢失多数类信息，可能欠拟合"],
        ["平衡采样器", "每批按类别均匀采样", "不修改损失，通用", "改变每轮样本分布，batch 内方差大"],
        ["类别权重 CE", "按 1/n_c 加权", "无额外推理成本", "权重对极端值敏感，需调参"],
        ["CB Loss（本文）", "按有效样本数 1/E_n 加权", "理论清晰，不丢样本，易组合", "仍需选择 β，可能牺牲多数类"],
    ],
    widths=[3.0, 4.6, 4.2, 4.2],
    font_size=8.2,
    first_col_left=True,
)

heading(doc, "4.2  Class-Balanced Loss 原理", 2)
body(
    doc,
    "CB Loss 的核心观察是：随着样本数增加，新增样本带来的边际信息递减。若直接使用样本数 n_c 作为权重，"
    "会高估多数类的有效信息量。Cui 等人提出用有效样本数（effective number）刻画一个类别实际提供的"
    "信息量："
)
formula(doc, "E_n = (1 − β^n) / (1 − β),   β ∈ [0, 1)")
body(
    doc,
    "其中 n 为类别样本数。β→1 时 E_n→n，等价于反频率加权；β→0 时 E_n→1，等价于均匀权重。"
    "类别权重取有效样本数的倒数并归一化到均值为 1："
)
formula(doc, "w_c = (1 / E_{n_c}) / mean_j(1 / E_{n_j})")
body(
    doc,
    "带类别权重和标签平滑的损失为："
)
formula(doc, "L_CB = − Σ_c w_c · [ (1−ε)·1[y=c] + ε/10 ] · log p_c(x)")
body(
    doc,
    "本文取 β=0.9999。这个取值需要结合样本量理解：训练集类别样本数在 1,821–4,062 之间，"
    "若取 β=0.999，则 β^n 在 n 为数千时已经接近 0，各类的 E_n 都接近 1/(1−β)=1000，权重几乎相同，"
    "不均衡处理效果很弱；取 β=0.9999 时，E_n 仍能区分不同样本量，权重范围约为 0.79–1.57，"
    "与 2.23:1 的不均衡程度相匹配。β 没有在验证集上精细搜索，这是本实验的一个局限。"
)
weights = results["baseline_cb"]["class_weights"]
caption(doc, "表 4  CB Loss 的类别权重（β=0.9999，训练集）")
wrows = [[f"{i} ({CN_CLASSES[i]})", f"{train_counts[i]:,}", f"{weights[i]:.3f}"] for i in range(10)]
table(doc, ["类别", "训练样本数", "权重 w_c"], wrows, widths=[5.0, 4.0, 4.0])
note(
    doc,
    "权重以均值为 1 归一化。权重最大的是样本最少的“卡车”，最小的是样本最多的“鹿”，"
    "二者之比约 1.98，略低于原始样本数之比 2.23，体现了有效样本数的饱和效应。",
)

heading(doc, "4.3  实施方式", 2)
body(
    doc,
    "在代码中，CB Loss 实现为一个带 buffer 的 nn.Module：训练开始前用训练集标签统计 n_c，按上式"
    "计算权重并注册为 buffer，前向时调用 F.cross_entropy(logits, target, weight=w, label_smoothing=0.1)。"
    "权重只依赖训练集统计量，不参与梯度；验证和测试阶段不重新加权，也不做 logit 调整，因此推理图与"
    "基线完全一致，参数量、FLOPs 和延迟不变。"
)
body(
    doc,
    "为保持可比性，CE 与 CB 使用相同的随机种子、分层划分、batch 顺序、增广参数、学习率计划和训练轮数。"
    "这样，测试指标差异可以主要归因于损失函数，而不是训练随机性或数据划分。"
)
body(
    doc,
    "从机制上看，CB 损失提高了少数类样本对梯度的贡献，使决策边界向少数类一侧移动。其预期效果是："
    "少数类召回率上升，宏平均 F1 和平衡准确率改善；多数类召回率可能略降，总体准确率不一定上升。"
    "在均衡测试集上，这种“牺牲头部、补偿尾部”的交换通常是值得的，但需要结合具体指标判断。"
)


# --------------------------------------------------------------------------------------
# 5 Lightweight
# --------------------------------------------------------------------------------------
heading(doc, "5  ViT 轻量化方法", 1, page_break=True)
heading(doc, "5.1  设计约束与评价口径", 2)
body(
    doc,
    "题目明确要求“仅修改 ViT 模块的参数设置不算作轻量化方法”。因此本文不把“把 d 从 192 改成 96”"
    "或“把层数从 12 改成 4”当作方法，而是实现两种会改变计算图或网络拓扑的结构级方案："
    "（1）DynamicViT 的 token 剪枝，在推理时真实减少参与后续 Block 的 token 数；"
    "（2）MobileViT-Lite 的卷积—Transformer 混合结构，用卷积 stem 和 patch 内注意力替代标准全局注意力。"
)
body(
    doc,
    "评价一个轻量化方法需要同时看四个量：参数量（模型存储）、FLOPs（理论计算量）、延迟/吞吐（实际速度）"
    "和精度（分类表现）。参数量小不等于 FLOPs 低，FLOPs 低也不等于延迟低，因为内存访问、算子启动、"
    "token gather/reshape 等都会影响实际速度。本文对四个量都做测量，并在第 8 节讨论它们不一致的原因。"
)
caption(doc, "表 5  三种结构的关键差异")
table(
    doc,
    ["项目", "ViT 基线", "DynamicViT（本文）", "MobileViT-Lite（本文）"],
    [
        ["token 数", "65（含 CLS）", "65→49→25→7", "8×8=64（无 CLS）"],
        ["主干", "4× 全局 Transformer", "4× 全局 Transformer + 3 个剪枝器", "卷积 stem + 2× MobileViT + 1× 全局 Block"],
        ["注意力范围", "全部 64 token", "每层保留 token 的全局注意力", "patch 内 16 token + 最后 1 层全局"],
        ["参数量", f"{params_m('baseline_cb'):.3f} M", f"{params_m('dynamic_cb'):.3f} M", f"{params_m('mobile_cb'):.3f} M"],
        ["FLOPs", f"{gflops('baseline_cb'):.3f} G", f"{gflops('dynamic_cb'):.3f} G", f"{gflops('mobile_cb'):.3f} G"],
        ["结构级变化", "—", "动态 token 稀疏化", "卷积 stem + patch 内注意力"],
    ],
    widths=[3.0, 4.0, 4.8, 5.0],
    font_size=8.2,
    first_col_left=True,
)

heading(doc, "5.2  DynamicViT：渐进式 token 剪枝", 2)
body(
    doc,
    "DynamicViT 的出发点是：图像中并非所有 patch 都对分类同等重要，背景或重复纹理对应的 token 可以在"
    "后续层中被丢弃，从而减少自注意力和 FFN 的计算。与训练后一次性剪枝不同，DynamicViT 在训练过程中"
    "学习一个轻量的 token 重要性打分器，并在多个阶段逐步剪枝，使网络适应逐渐变短的 token 序列。"
)
body(
    doc,
    "本文在 Block 1、2、3 之后各插入一个 TokenPruner，保留比例分别为 75%、50%、25%（均只针对 patch"
    " token，[CLS] 始终保留）。由于 64 个 patch token 按比例取整，实际 token 数为 65→49→25→7，"
    "如图 2(b) 与图 4 所示。"
)
heading(doc, "5.2.1  重要性打分与 Straight-Through 估计", 3)
body(
    doc,
    "对当前 patch token 矩阵 T∈R^{N×d}，剪枝器先做 LayerNorm，再经过两层 MLP 输出每个 token 的标量分数："
)
formula(doc, "s_i = MLP( LN(t_i) ) ∈ R,   i = 1, …, N")
body(
    doc,
    "训练时在分数上加入 Gumbel 噪声，再取 top-k，以鼓励探索并避免打分器过早固化："
)
formula(doc, "r_i = s_i + g_i,   g_i = −log(−log U_i),   U_i ~ Uniform(0,1)")
formula(doc, "S = TopK({r_i}, k),   k = round(ρN)")
body(
    doc,
    "top-k 是离散操作，无法直接反向传播。本文使用直通估计器（straight-through estimator）：前向用硬 mask"
    "保留 top-k token，反向则把保留 token 的软分数 s_i 的梯度传回打分器。对保留 token，乘以"
    "“软分数 − 软分数.detach() + 1”，其前向值为 1，梯度为 σ'(s_i)；未被选中的 token 被真实丢弃，"
    "不参与后续计算，因此节省的 FLOPs 不是“掩码为 0 但仍然计算”的虚假节省。"
)
body(
    doc,
    "推理阶段去掉 Gumbel 噪声，直接按分数取 top-k，并对保留 token 做 gather。由于每个阶段保留的 token"
    "数是固定的，推理图是静态的（每个阶段形状确定），不会引入数据相关的控制流。剪枝器自身只有两层线性层，"
    "参数量约 14.7 k，占基线的 4.7%，是该方法的主要参数代价。"
)
heading(doc, "5.2.2  计算量分析", 3)
body(
    doc,
    "设第 l 层输入 token 数为 N_l，注意力头数为 h，每个头维度为 d_h，则一个 Block 的主要计算量为："
)
formula(doc, "FLOPs_block ≈ 2N_l d² (QKV) + 2N_l d² (投影) + 2N_l² d (注意力) + 4N_l d·d_ff (FFN)")
body(
    doc,
    "当 N_l 从 65 降到 49 再降到 25、7 时，后续 Block 的 QKV、投影、FFN 和注意力矩阵都按 N_l 或 N_l²"
    "缩小。按本文 schedule，四个 Block 的平均 token 数约为 (65+49+25+7)/4=36.5，相对 65 的平均保留率"
    "约 56%；再考虑第一层仍为 65、以及分类头和 patch embedding 不随 token 数变化，实测 FLOPs 降低约"
    f"{(1-gflops('dynamic_cb')/gflops('baseline_cb'))*100:.1f}%。这一收益来自真实的前向路径变短，"
    "而不是把参数改小。"
)
body(
    doc,
    "需要指出，DynamicViT 的主要目标是降低计算量，而不是降低参数量；相反，打分器会带来少量额外参数。"
    "因此它适合“算力/延迟受限但存储相对宽裕”的场景。若同时追求更小参数量，需要与结构裁剪、量化或"
    "MobileViT 类混合结构结合。"
)

heading(doc, "5.3  MobileViT-Lite：卷积与 Transformer 的混合结构", 2)
body(
    doc,
    "ViT 的全局自注意力在 token 数较多时计算量大，而卷积在局部特征提取上更高效。MobileViT 的核心思想是"
    "“用卷积获取局部表示，用 Transformer 在 patch 内建模长程依赖，再用卷积融合”。本文实现一个适合"
    "32×32 输入的轻量版本，结构如图 2(c) 所示。"
)
heading(doc, "5.3.1  网络结构", 3)
body(
    doc,
    "（1）卷积 stem：3×3 卷积（stride 2）把 32×32 降到 16×16；深度可分离 3×3 卷积（stride 2）"
    "把分辨率降到 8×8；1×1 卷积把通道数映射到 d=72。stem 的作用是快速降低分辨率、提取边缘和纹理，"
    "同时避免标准 ViT 在 32×32 上使用 4×4 patch 时缺失的局部归纳偏置。"
)
body(
    doc,
    "（2）MobileViT Block：对 8×8×72 的特征图，先用深度可分离 3×3 卷积和 1×1 卷积得到局部表示；"
    "再用 F.unfold 以 4×4、步长 4 切分为 4 个 patch，每个 patch 有 16 个 token；在 patch 内部做"
    "自注意力（16 个 token 之间），而不是在全部 64 个 token 之间做注意力；然后用 F.fold 还原为 8×8，"
    "经 1×1 卷积投影后与输入残差相加。这样注意力的 token 对数从 64² 降到 4×16²，计算量显著下降。"
)
body(
    doc,
    "（3）全局 Block：在 2 个 MobileViT Block 之后，对 8×8=64 个 token 加可学习位置编码，"
    "再用一个标准 Transformer Block 建模全局关系；最后做全局平均池化、LayerNorm 和线性分类头。"
    "全局 Block 保留了跨 patch 的信息交换，弥补 patch 内注意力感受野有限的不足。"
)
body(
    doc,
    "该结构同时改变了参数量和计算图：卷积层和 1×1 投影替代了大部分全连接层，patch 内注意力减少了"
    "注意力矩阵规模。与“把 d 改小”不同，它是通过改变算子组合和注意力范围来实现轻量化。"
)
heading(doc, "5.3.2  复杂度与设计取舍", 3)
body(
    doc,
    f"MobileViT-Lite 的参数量为 {params_m('mobile_cb'):.3f} M，相对基线下降 "
    f"{(1-params_m('mobile_cb')/params_m('baseline_cb'))*100:.1f}%；FLOPs 为 {gflops('mobile_cb'):.3f} G，"
    f"下降 {(1-gflops('mobile_cb')/gflops('baseline_cb'))*100:.1f}%。代价是：卷积 stem 下采样到 8×8，"
    "空间细节少于 4×4 patch 的 64 token 表示；patch 内注意力只覆盖局部 4×4 区域，全局建模主要依赖最后"
    "一个 Block；d=72 的容量也小于基线的 96。这些因素理论上可能损失精度，但卷积的权值共享和局部性"
    "也可能提升小数据上的样本效率，实际影响需要由实验判断。"
)
body(
    doc,
    "选择 MobileViT 而不是 A-ViT 作为第二种方法的理由是：A-ViT 与 DynamicViT 同属“动态 token 数”"
    "思路，且主要降低 FLOPs、不降低参数量；MobileViT 则同时降低参数量和 FLOPs，能更全面地展示"
    "“结构级轻量化”的不同方向。A-ViT 作为相关工作在第 8 节讨论。"
)

heading(doc, "5.4  蒸馏训练策略", 2)
body(
    doc,
    "轻量模型容量较小，直接训练容易欠拟合。为了在相同训练轮数下给出更公平的结构比较，本文对 DynamicViT"
    " 和 MobileViT-Lite 都使用 CB 基线作为教师做日志蒸馏（logit distillation）。教师先在训练集上训练好，"
    "再对全部训练图像做一次前向，缓存 10 维 logits；学生训练时读取对应样本的缓存 logits，不需要每步"
    "重复前向教师，因此蒸馏几乎不增加训练时间。"
)
formula(doc, "L = L_CB(y, p_S) + α·T²·KL( softmax(z_T/T) ‖ softmax(z_S/T) )")
body(
    doc,
    "本文取 α=0.5、T=2。蒸馏项提供比硬标签更平滑的类间关系（“暗知识”），在训练样本有限或模型较小"
    "时通常能改善泛化。需要说明：教师也是在同一条不均衡训练集上训练的，蒸馏不能凭空消除不均衡，"
    "但它可以把教师的决策边界和类别相似性结构传递给学生，减少小模型在少数类上的抖动。"
)


# --------------------------------------------------------------------------------------
# 6 Setup
# --------------------------------------------------------------------------------------
heading(doc, "6  实验设置", 1, page_break=True)
heading(doc, "6.1  软硬件环境", 2)
body(
    doc,
    "实验在 Windows 主机上完成，无可用的 NVIDIA GPU（torch.cuda.is_available() 为 False），"
    "因此全部训练和推理均在 CPU 上执行。软件环境为 Python 3.13、PyTorch 2.12.0（CPU 版）、"
    "NumPy 2.1、Matplotlib 3.10。CPU 为 16 个逻辑核，内存 15.7 GB。"
    "训练时设置 torch.set_num_threads(8)；复杂度测量时同样固定为 8 线程，"
    "并在正式计时前做 warm-up。"
)
note(
    doc,
    "由于没有 GPU，绝对训练时间和吞吐不能与 GPU 论文结果直接比较；本文所有速度结论都只在同一台机器、"
    "同一线程设置下做相对比较。",
)

heading(doc, "6.2  训练超参数", 2)
caption(doc, "表 6  所有模型共享的训练配置")
hp_rows = [
    ["训练 / 验证 / 测试样本数", f"{results['baseline_cb']['train_size']:,} / {results['baseline_cb']['val_size']:,} / {results['baseline_cb']['test_size']:,}"],
    ["训练轮数", "20"],
    ["batch size", "128"],
    ["优化器", "AdamW（β=(0.9, 0.999)）"],
    ["基础学习率", "5×10⁻⁴"],
    ["权重衰减", "0.05（norm/bias/pos/cls 除外）"],
    ["学习率计划", "线性 warm-up 2 epoch + 余弦退火至 2%"],
    ["标签平滑", "0.1"],
    ["DropPath", "0.05"],
    ["梯度裁剪", "1.0"],
    ["随机种子", "42"],
    ["数据增广", "随机裁剪（pad 4）+ 水平翻转 + Cutout(p=0.25)"],
    ["蒸馏（仅轻量模型）", "教师=CB 基线，α=0.5，T=2"],
]
table(doc, ["项目", "设置"], hp_rows, widths=[6.0, 8.5], first_col_left=True)

heading(doc, "6.3  评价指标", 2)
body(
    doc,
    "分类指标包括：总体准确率（Accuracy）、宏平均 F1（Macro-F1）、平衡准确率（各类召回率的算术平均，"
    "在均衡测试集上等于准确率，但在类别不均衡验证集上更能反映尾部表现）、单类召回率、混淆矩阵以及"
    "期望校准误差（ECE，15 个等宽置信度区间）。复杂度指标包括：参数量（numel 求和）、单张前向 FLOPs、"
    "batch=64 的批延迟（ms/64 张）与吞吐（img/s）以及每 epoch 训练时间。"
)
body(
    doc,
    "宏平均 F1 对每个类别单独计算 F1 再平均，因此少数类的表现不会被多数类淹没；ECE 衡量预测置信度与"
    "实际正确率之间的差距，用来观察标签平滑、蒸馏和类别重加权是否让模型过度自信或过度保守。"
)

heading(doc, "6.4  参数量、FLOPs 与速度的测量", 2)
body(
    doc,
    "参数量直接对 model.parameters() 的 numel 求和，包含可学习的位置编码和 [CLS] token，不包含 buffer。"
    "FLOPs 使用 PyTorch 内置的 torch.utils.flop_counter.FlopCounterMode，对 batch=1 的前向计算进行"
    "算子级统计（卷积、矩阵乘、线性层等）。相比 thop/torchsummary 只统计 Conv/Linear 的做法，"
    "FlopCounterMode 还能统计注意力中的 QKᵀ 和 AV 矩阵乘，并且能跟随 DynamicViT 的动态 token 形状，"
    "因此更适合本文的 Transformer 模型。"
)
body(
    doc,
    "延迟/吞吐测量使用 torch.inference_mode()，四个模型在同一进程内分 5 轮交错测量，每轮 batch=1 "
    "计时 400 次、batch=64 计时 80 次，取各轮中位数的中位数。交错和重复是为了抵消 CPU 频率、缓存和"
    "后台负载随时间漂移的影响；两个结构完全相同的基线（CE 与 CB）的 batch=64 吞吐相差不到 1%，"
    "可作为测量稳定性的校验。"
)
body(
    doc,
    "单张（batch=1）延迟在 CPU 上受线程调度影响很大，同一模型不同轮次可相差近一倍，因此正文以 "
    "batch=64 的批延迟/吞吐作为推理速度的主要指标，单张延迟只作为参考，不用于排名。"
)
body(
    doc,
    "需要区分“理论计算量”和“实际延迟”：FLOPs 不包含内存访问、token gather、reshape、LayerNorm、"
    "softmax 等开销；CPU 上小 batch 的延迟还受线程调度、缓存和算子实现影响。因此本文同时报告两者，"
    "不把 FLOPs 降低简单等同于延迟按比例降低。"
)

heading(doc, "6.5  可复现性", 2)
body(
    doc,
    "数据划分由固定种子 42 分层抽样生成，所有模型共享同一份训练/验证索引；DataLoader 的 shuffle 也使用"
    "固定种子。每个实验保存：完整配置（JSON）、逐 epoch 日志、最佳 checkpoint（按验证集准确率选择）、"
    "测试集混淆矩阵与单类召回、测试预测结果。最终测试只加载最佳 checkpoint 评估一次，不参与任何调参。"
)
body(
    doc,
    "完整复现命令见附录 B。核心训练命令形如：python scripts/train.py --model vit --loss cb "
    "--tag baseline_cb --epochs 20；轻量模型额外传入 --distill-logits checkpoints/baseline_cb_train_logits.pt。"
)


# --------------------------------------------------------------------------------------
# 7 Results
# --------------------------------------------------------------------------------------
acc = {t: metric(t, "accuracy") for t in TAGS}
f1 = {t: metric(t, "macro_f1") for t in TAGS}
bal = {t: metric(t, "balanced_accuracy") for t in TAGS}
ece = {t: metric(t, "ece") for t in TAGS}
best_tag = max(TAGS, key=lambda t: acc[t])
d_acc_cb = (acc["baseline_cb"] - acc["baseline_ce"]) * 100
d_f1_cb = (f1["baseline_cb"] - f1["baseline_ce"]) * 100
d_bal_cb = (bal["baseline_cb"] - bal["baseline_ce"]) * 100
dyn_dacc = (acc["dynamic_cb"] - acc["baseline_cb"]) * 100
mob_dacc = (acc["mobile_cb"] - acc["baseline_cb"]) * 100
dyn_df1 = (f1["dynamic_cb"] - f1["baseline_cb"]) * 100
mob_df1 = (f1["mobile_cb"] - f1["baseline_cb"]) * 100


def top_confusion(tag):
    """返回某个模型最大的非对角混淆项 (真实类, 预测类, 样本数)。"""
    cm = np.asarray(results[tag]["test"]["confusion_matrix"], dtype=float)
    cm = cm.copy()
    np.fill_diagonal(cm, 0)
    i, j = np.unravel_index(np.argmax(cm), cm.shape)
    return i, j, int(cm[i, j])


def minority_recall(tag, cls=9):
    """返回指定类别的召回率；默认是样本最少的卡车（类别 9）。"""
    return results[tag]["test"]["per_class_recall"][cls]


heading(doc, "7  实验结果", 1, page_break=True)
heading(doc, "7.1  总体结果", 2)
body(
    doc,
    "四个模型在均衡测试集上的结果如表 7 所示。所有模型使用相同的训练/验证划分和 20 个 epoch；"
    "轻量模型使用相同的蒸馏超参数。表中参数量与 FLOPs 由独立脚本在最佳 checkpoint 上测量，"
    "批延迟为 CPU、8 线程、batch=64 的中位数（单位 ms/64 张），吞吐由同一测量换算；单张延迟受 CPU 调度噪声影响较大，正文对此有说明。"
)
caption(doc, "表 7  各模型在均衡测试集上的分类表现与复杂度（Acc/F1/BalAcc/ECE 单位为 %，ms/64 为 batch=64 的批延迟）")
main_rows = []
for t in TAGS:
    main_rows.append([
        TABLE_NAMES[t],
        f"{params_m(t):.3f}",
        f"{gflops(t):.3f}",
        f"{acc[t]*100:.1f}",
        f"{f1[t]*100:.1f}",
        f"{bal[t]*100:.1f}",
        f"{ece[t]*100:.1f}",
        f"{epoch_time(t):.1f}",
        f"{batch_latency(t):.1f}",
        f"{throughput(t):.0f}",
    ])
table(
    doc,
    ["模型", "参数(M)", "GFLOPs", "Acc", "F1", "BalAcc", "ECE", "s/epoch", "ms/64", "img/s"],
    main_rows,
    widths=[3.15, 1.3, 1.45, 1.3, 1.3, 1.5, 1.1, 1.8, 1.35, 1.7],
    font_size=7.4,
)

body(
    doc,
    f"总体来看，四个模型中测试准确率最高的是 {LABELS[best_tag]}，测试准确率 {acc[best_tag]*100:.2f}%、"
    f"宏平均 F1 {f1[best_tag]*100:.2f}%。ViT 基线的准确率为 {acc['baseline_ce']*100:.2f}%，"
    f"CB 损失将其变为 {acc['baseline_cb']*100:.2f}%（变化 {signed(d_acc_cb,1)} 个百分点），"
    f"宏平均 F1 变化 {signed(d_f1_cb,1)} 个百分点，平衡准确率变化 {signed(d_bal_cb,1)} 个百分点。"
    "这些差异的方向和幅度与 2.2 节的分析一致：在中度不均衡且测试集均衡的条件下，重加权主要影响尾部类别，"
    "对总体准确率的影响相对有限。"
)
body(
    doc,
    f"DynamicViT 将 FLOPs 从 {gflops('baseline_cb'):.3f} G 降到 {gflops('dynamic_cb'):.3f} G"
    f"（{rel_pct(gflops('dynamic_cb'), gflops('baseline_cb'))}），但参数量从 "
    f"{params_m('baseline_cb'):.3f} M 变为 {params_m('dynamic_cb'):.3f} M（{rel_pct(params_m('dynamic_cb'), params_m('baseline_cb'))}），"
    f"测试准确率变化 {signed(dyn_dacc,1)} 个百分点。MobileViT-Lite 将参数量降到 "
    f"{params_m('mobile_cb'):.3f} M（{rel_pct(params_m('mobile_cb'), params_m('baseline_cb'))}），"
    f"FLOPs 降到 {gflops('mobile_cb'):.3f} G（{rel_pct(gflops('mobile_cb'), gflops('baseline_cb'))}），"
    f"测试准确率变化 {signed(mob_dacc,1)} 个百分点。"
)

heading(doc, "7.2  类别不均衡处理的效果", 2)
body(
    doc,
    f"为了观察 CB 损失对尾部类别的作用，表 8 给出 CE 与 CB 在少数类（样本最少的“卡车”，类别 9）"
    f"和多数类（“汽车”，类别 1）上的召回率，以及两类的 F1。CE 的卡车召回率为 "
    f"{minority_recall('baseline_ce')*100:.1f}%，CB 为 {minority_recall('baseline_cb')*100:.1f}%，"
    f"变化 {signed((minority_recall('baseline_cb')-minority_recall('baseline_ce'))*100,1)} 个百分点；"
    f"汽车召回率从 {results['baseline_ce']['test']['per_class_recall'][1]*100:.1f}% 变为 "
    f"{results['baseline_cb']['test']['per_class_recall'][1]*100:.1f}%。"
)
caption(doc, "表 8  CE 与 CB 在多数类/少数类上的召回率")
table(
    doc,
    ["模型", "卡车召回(%)", "汽车召回(%)", "宏平均F1(%)", "平衡准确率(%)"],
    [
        ["ViT + CE", f"{minority_recall('baseline_ce')*100:.1f}", f"{results['baseline_ce']['test']['per_class_recall'][1]*100:.1f}", f"{f1['baseline_ce']*100:.1f}", f"{bal['baseline_ce']*100:.1f}"],
        ["ViT + CB", f"{minority_recall('baseline_cb')*100:.1f}", f"{results['baseline_cb']['test']['per_class_recall'][1]*100:.1f}", f"{f1['baseline_cb']*100:.1f}", f"{bal['baseline_cb']*100:.1f}"],
    ],
    widths=[4.2, 3.2, 3.2, 3.2, 3.4],
)
body(
    doc,
    "从单类召回看，CB 的收益并不是所有类别均匀分布，而是集中在样本较少的类别；多数类可能略有回退。"
    "这符合代价敏感学习的基本规律：重加权把决策边界向少数类推，减少了少数类被误判为多数类的情况，"
    "同时会牺牲一部分多数类的召回。由于测试集每类样本数相同，宏平均指标比总体准确率更能反映这种交换。"
    "在本实验中，CB 的总体准确率不降反升，说明 2.23:1 的中度不均衡尚未严重到需要大幅牺牲头部性能；"
    "若使用更强的重采样或更大的 β，收益和代价都会被放大。"
)
figure(doc, "fig_per_class.png", "图 3  各模型在均衡测试集上的单类召回率（类别按 CIFAR-10 标准顺序）")
body(
    doc,
    f"具体到类别：卡车召回率从 {results['baseline_ce']['test']['per_class_recall'][9]*100:.1f}% 提升到 "
    f"{results['baseline_cb']['test']['per_class_recall'][9]*100:.1f}%（{(results['baseline_cb']['test']['per_class_recall'][9]-results['baseline_ce']['test']['per_class_recall'][9])*100:+.1f} 个百分点），"
    f"鸟从 {results['baseline_ce']['test']['per_class_recall'][2]*100:.1f}% 提升到 {results['baseline_cb']['test']['per_class_recall'][2]*100:.1f}%，"
    f"船从 {results['baseline_ce']['test']['per_class_recall'][8]*100:.1f}% 提升到 {results['baseline_cb']['test']['per_class_recall'][8]*100:.1f}%；"
    f"与此同时汽车从 {results['baseline_ce']['test']['per_class_recall'][1]*100:.1f}% 降到 {results['baseline_cb']['test']['per_class_recall'][1]*100:.1f}%，"
    f"猫从 {results['baseline_ce']['test']['per_class_recall'][3]*100:.1f}% 降到 {results['baseline_cb']['test']['per_class_recall'][3]*100:.1f}%。"
    "这说明 CB 的增益主要来自样本较少的类别，代价由部分多数类承担；在均衡测试集上，前者足以抵消后者，"
    "因此总体准确率和宏平均 F1 同时上升。"
)

heading(doc, "7.3  DynamicViT 的效果", 2)
body(
    doc,
    f"DynamicViT 的精度为 {acc['dynamic_cb']*100:.2f}%，相对 CB 基线变化 {signed(dyn_dacc,1)} 个百分点；"
    f"宏平均 F1 变化 {signed(dyn_df1,1)} 个百分点。它的主要收益在计算量：FLOPs 降低 "
    f"{(1-gflops('dynamic_cb')/gflops('baseline_cb'))*100:.1f}%，batch=64 批延迟从 {batch_latency('baseline_cb'):.1f} ms "
    f"降到 {batch_latency('dynamic_cb'):.1f} ms（{rel_pct(batch_latency('dynamic_cb'), batch_latency('baseline_cb'))}），"
    f"吞吐从 {throughput('baseline_cb'):.0f} img/s 提升到 {throughput('dynamic_cb'):.0f} img/s。"
)
body(
    doc,
    f"精度的轻微下降可以从两方面解释。第一，剪枝器在 Block 1 后就把 token 从 65 降到 49，"
    "早期丢弃的 token 无法恢复；对于只有 4 层的浅层 ViT，后期层数不足以弥补信息损失。第二，"
    "打分器只用当前层的 token 特征做局部判断，缺乏对全局上下文的二次校验；被误删的判别性 token "
    "会直接影响 [CLS] 的聚合。本文没有加入 DynamicViT 论文中的注意力掩码蒸馏和比例损失，"
    "只使用了日志蒸馏，因此精度恢复有限。"
)
body(
    doc,
    f"值得注意的是，DynamicViT 的参数量比基线高 {rel_pct(params_m('dynamic_cb'), params_m('baseline_cb'))}，"
    "因为三个打分器本身有约 1.47 万参数。它属于“用少量参数换计算量”的方法，而不是“压缩模型存储”的方法。"
    "在部署时，如果瓶颈是算力/延迟而不是模型体积，这种交换是合理的；如果瓶颈是存储或带宽，则应选择"
    "MobileViT 类结构，或对打分器做共享/量化。"
)
figure(doc, "fig_tokens.png", "图 4  DynamicViT 各阶段保留的 token 数（含 [CLS]）")

heading(doc, "7.4  MobileViT-Lite 的效果", 2)
body(
    doc,
    f"MobileViT-Lite 的参数量为 {params_m('mobile_cb'):.3f} M，仅为基线的 "
    f"{params_m('mobile_cb')/params_m('baseline_cb')*100:.1f}%；FLOPs 为 {gflops('mobile_cb'):.3f} G，"
    f"为基线的 {gflops('mobile_cb')/gflops('baseline_cb')*100:.1f}%。测试准确率为 {acc['mobile_cb']*100:.2f}%，"
    f"相对 CB 基线变化 {signed(mob_dacc,1)} 个百分点，宏平均 F1 变化 {signed(mob_df1,1)} 个百分点。"
    f"batch=64 吞吐为 {throughput('mobile_cb'):.0f} img/s，相对基线 {rel_pct(throughput('mobile_cb'), throughput('baseline_cb'))}。"
)
body(
    doc,
    f"这一结果与“轻量化必然掉点”的预期相反：MobileViT-Lite 的准确率不仅没有下降，反而比 CB 基线高 "
    f"{signed(mob_dacc,1)} 个百分点，也比 DynamicViT 高 "
    f"{(acc['mobile_cb']-acc['dynamic_cb'])*100:+.1f} 个百分点。可能的原因有三点：（1）卷积 stem 的"
    "局部归纳偏置与 32×32 小图像更匹配，权值共享提高了样本效率；（2）8×8 特征图保留了比 4×4 patch "
    "更细的空间结构，patch 内 4×4 注意力提供了局部交互，最后一个全局 Block 负责跨区域信息；"
    "（3）d=72 的宽度虽然小于基线的 96，但卷积 stem 和 MobileViT Block 的参数效率更高，"
    "在 3.2 万张训练样本上反而更不容易过拟合。参数量小不等于容量小，这是本实验最值得注意的现象之一。"
)
body(
    doc,
    f"从校准看，MobileViT-Lite 的 ECE 为 {ece['mobile_cb']*100:.1f}%，高于 CB 基线的 "
    f"{ece['baseline_cb']*100:.1f}%；DynamicViT 为 {ece['dynamic_cb']*100:.1f}%。这说明轻量模型在"
    "保持或提高准确率的同时，置信度校准变差；若下游应用需要可靠的置信度，应在验证集上做温度缩放"
    "或其他后处理校准。"
)
body(
    doc,
    f"从速度看，MobileViT-Lite 的 batch=64 吞吐为 {throughput('mobile_cb'):.0f} img/s，"
    f"基线为 {throughput('baseline_cb'):.0f} img/s。FLOPs 降低约 "
    f"{(1-gflops('mobile_cb')/gflops('baseline_cb'))*100:.0f}%，但吞吐并未按相同比例提升，"
    "原因是 CPU 上 unfold/fold、深度可分离卷积和 BatchNorm 的访存开销较高，且小 batch 时算子启动"
    "开销占比更大。这再次说明 FLOPs 只能作为计算量的近似指标，必须结合实际吞吐评估。"
)

heading(doc, "7.5  训练过程", 2)
body(
    doc,
    "图 5 给出四个模型的训练损失和验证准确率曲线。所有模型在 warm-up 后损失稳定下降，没有出现明显的"
    "发散或震荡；验证准确率在最后几个 epoch 仍在缓慢上升，尚未完全收敛。CB 与 CE 的曲线在前 10 个 epoch "
    "非常接近，后期 CB 的验证准确率略高/略低（取决于具体轮次），说明在中度不均衡下，损失重加权的即时"
    "影响较小，主要体现在收敛后的决策边界。DynamicViT 和 MobileViT-Lite 的训练损失包含蒸馏项，"
    "数值高于基线，不能与 CE/CB 的损失直接比较；从验证准确率看，DynamicViT 前期收敛略慢，"
    "MobileViT-Lite 在中期追上并超过基线。"
)
figure(doc, "fig_curves.png", "图 5  训练损失与验证准确率曲线")

heading(doc, "7.6  复杂度—性能权衡", 2)
body(
    doc,
    "图 6 将测试准确率分别对参数量和 FLOPs 作图，并在 FLOPs 图中标注 batch=64 吞吐。理想的方法应该位于"
    f"“左上角”（高精度、低复杂度）。在本实验中，MobileViT-Lite 同时具有最高的测试准确率"
    f"（{acc['mobile_cb']*100:.1f}%）、最低的参数量（{params_m('mobile_cb'):.3f} M）和最低的 FLOPs"
    f"（{gflops('mobile_cb'):.3f} G），batch=64 吞吐（{throughput('mobile_cb'):.0f} img/s）也高于 CB 基线"
    f"（{throughput('baseline_cb'):.0f} img/s）；DynamicViT 的吞吐最高（{throughput('dynamic_cb'):.0f} img/s），"
    f"但准确率降到 {acc['dynamic_cb']*100:.1f}%，参数量还略高于基线。CB 基线在精度和吞吐上都不占优，"
    "这提示：在本实验条件下，结构轻量化不仅压缩了成本，卷积归纳偏置和蒸馏还带来了泛化收益。"
    "不过这只是单种子、20 epoch 的结果，不能过度外推；若以吞吐为第一目标，DynamicViT 仍是最快的选择，"
    "若综合精度、存储和算力，MobileViT-Lite 更均衡。"
)
figure(doc, "fig_tradeoff.png", "图 6  测试准确率与参数量、FLOPs 的权衡（右图标注 batch=64 吞吐）")

heading(doc, "7.7  混淆矩阵与典型错误", 2)
i, j, n = top_confusion("baseline_cb")
body(
    doc,
    "图 7 给出四个模型在均衡测试集上的归一化混淆矩阵。整体上，模型对“汽车”“船”“飞机”等类别较容易"
    "区分，对“猫/狗”“鹿/马”“鸟/飞机”等视觉相似的类别更容易混淆。以 CB 基线为例，"
    f"最大的非对角项是真实“{CN_CLASSES[i]}”被预测为“{CN_CLASSES[j]}”（{n} 张）；"
    "这类错误往往来自低分辨率下的纹理和轮廓相似，以及 4×4 patch 对局部细节的压缩。"
)
body(
    doc,
    "CB 与 CE 的混淆矩阵差异主要体现在少数类所在的行：CB 减少了少数类被整体推向多数类的倾向，"
    "但可能把一部分多数类样本误判为少数类。DynamicViT 的混淆结构与 CB 基线接近，说明 token 剪枝"
    "主要影响“看不看得清”，而不是系统性地改变类别偏好；MobileViT-Lite 的混淆结构与 CB 基线接近，"
    "卡车等少数类的对角项更强，与其更高的总体准确率一致。"
)
figure(doc, "fig_confusion.png", "图 7  四个模型在均衡测试集上的归一化混淆矩阵")


# --------------------------------------------------------------------------------------
# 8 Discussion
# --------------------------------------------------------------------------------------
heading(doc, "8  分析与讨论", 1, page_break=True)
heading(doc, "8.1  为什么 CB 的收益是温和的", 2)
body(
    doc,
    f"本实验中 CB 相对 CE 的总体准确率变化为 {signed(d_acc_cb,1)} 个百分点，宏平均 F1 变化 "
    f"{signed(d_f1_cb,1)} 个百分点。收益温和的原因主要有三点。第一，不均衡程度是 2.23:1，"
    "而不是 100:1 的极端长尾；多数类和少数类的样本量仍在同一数量级，普通交叉熵并不会完全忽略少数类。"
    "第二，训练使用了标签平滑、随机裁剪、翻转和 Cutout，这些正则化会削弱模型对训练先验的依赖，"
    "本身就能缓解一部分不均衡。第三，CB 的 β=0.9999 对应的权重范围只有 0.79–1.57，调整力度温和；"
    "若使用 β→1 的反频率权重，少数类权重会更大，收益和代价都会更明显。"
)
body(
    doc,
    "从指标上看，CB 更适合用宏平均 F1、平衡准确率和少数类召回评价，而不是只看总体准确率。"
    "在均衡测试集上，如果 CB 提高了少数类召回但总体准确率持平，说明它把决策边界从“多数类优先”"
    "调整到了更接近“各类均衡”的位置，这正是类别不均衡处理的目标。实际部署中应根据业务代价选择"
    "权重：若少数类漏检代价高，可以增大 β 或使用 Focal Loss；若多数类精度更关键，则应保守重加权。"
)

heading(doc, "8.2  DynamicViT 的收益与边界", 2)
body(
    doc,
    f"DynamicViT 在本实验中把 FLOPs 降低了 {(1-gflops('dynamic_cb')/gflops('baseline_cb'))*100:.1f}%，"
    f"但准确率变化 {signed(dyn_dacc,1)} 个百分点，参数量增加 "
    f"{(params_m('dynamic_cb')/params_m('baseline_cb')-1)*100:.1f}%。这说明 token 剪枝的收益是"
    "“计算量换精度”，而不是“参数量换精度”。其收益大小取决于三个因素：剪枝比例、剪枝位置和模型深度。"
    "本文在 4 层模型的第 1/2/3 层后剪枝，平均保留率约 56%；如果只剪后期层，精度损失会更小但 FLOPs "
    "收益也更小；如果在第 1 层就大幅剪枝，收益更大但可能删掉关键细节。"
)
body(
    doc,
    "本文的剪枝器训练只依赖分类损失和日志蒸馏，没有使用 DynamicViT 原论文的注意力掩码蒸馏和"
    "“保留比例损失”。原论文在 ImageNet 上通过自蒸馏和更长的训练，把 30%–40% 的 FLOPs 削减做到"
    "接近无损；本文在 CIFAR-10、小模型、20 epoch 的条件下，精度恢复有限。这提示：token 剪枝并不是"
    "“插入即免费”的模块，它需要足够的训练预算和合适的蒸馏目标；在少数据/少轮数场景中，剪枝器本身"
    "也可能欠拟合。"
)
body(
    doc,
    f"另一个值得注意的现象是：DynamicViT 的 FLOPs 降低 "
    f"{(1-gflops('dynamic_cb')/gflops('baseline_cb'))*100:.1f}%，batch=64 吞吐提升 "
    f"{(throughput('dynamic_cb')/throughput('baseline_cb')-1)*100:.1f}%，两者基本对应；这说明在较大 "
    "batch 下，token 剪枝减少的矩阵乘能够转化为真实速度。相比之下，MobileViT-Lite 的 FLOPs 降低 "
    f"{(1-gflops('mobile_cb')/gflops('baseline_cb'))*100:.1f}%，但吞吐只提升 "
    f"{(throughput('mobile_cb')/throughput('baseline_cb')-1)*100:.1f}%，因为 unfold/fold、"
    "深度可分离卷积和 BatchNorm 带来额外访存与算子启动开销；当模型很小时，这些固定开销占比更高。"
)

heading(doc, "8.3  MobileViT-Lite 的收益与代价", 2)
body(
    doc,
    f"MobileViT-Lite 同时把参数量降低 {(1-params_m('mobile_cb')/params_m('baseline_cb'))*100:.1f}%、"
    f"FLOPs 降低 {(1-gflops('mobile_cb')/gflops('baseline_cb'))*100:.1f}%，而准确率变化 "
    f"{signed(mob_dacc,1)} 个百分点，没有出现预期的精度损失。它的收益来自两点：卷积 stem 在低分辨率上"
    "提取局部特征，比全局注意力更省算力；patch 内注意力把 token 对数从 64² 降到 4×16²，同时保留了"
    "一定的局部建模能力。它的代价主要体现在速度而非精度：FLOPs 降低超过一半，但 batch=64 吞吐只提高 "
    f"{(throughput('mobile_cb')/throughput('baseline_cb')-1)*100:.1f}%，因为 unfold/fold、深度可分离"
    "卷积和 BatchNorm 的访存开销较高。"
)
body(
    doc,
    "与 DynamicViT 相比，MobileViT-Lite 是“静态”轻量化：推理图固定，不依赖输入相关的 token 选择，"
    "因此更容易部署到不支持动态 shape 的推理引擎；参数量也更小，适合存储/带宽受限的设备。"
    "DynamicViT 则是“动态”轻量化：可以根据输入难度调整保留 token 数（本文为简化实现固定比例），"
    "在简单样本上更省算力，但需要支持 gather/top-k 等算子。两者不是互斥的：可以把 MobileViT-Lite "
    "作为学生，再叠加 token 剪枝或量化，获得更激进的压缩。"
)

heading(doc, "8.4  参数量、FLOPs 与速度的关系", 2)
param_order = " < ".join(LABELS[t] for t in sorted(TAGS, key=params_m))
flop_order = " < ".join(LABELS[t] for t in sorted(TAGS, key=gflops))
lat_order = " < ".join(LABELS[t] for t in sorted(TAGS, key=throughput))
body(
    doc,
    f"表 7 中三个维度从低到高的排序分别为：参数量 {param_order}；"
    f"FLOPs {flop_order}；batch=64 吞吐 {lat_order}。这三个量并不必然一致：参数量反映存储，"
    "FLOPs 反映理论计算，延迟还受内存访问、算子融合、并行度和动态形状影响。"
    "轻量化方法必须同时报告这三者，不能用其中一个代替全部。"
)
body(
    doc,
    "在 CPU 上，小模型的延迟很容易被数据搬运和线程启动开销支配，因此 FLOPs 降低 50% 不一定带来"
    "50% 的延迟降低；在大 batch 或 GPU 上，计算更密集，FLOPs 与延迟的相关性通常更强。本文的吞吐"
    "测量使用 batch=64，比单张延迟更稳定，但仍受 8 线程设置限制。若要在实际设备上选型，应在目标"
    "硬件上用自己的输入尺寸和 batch 重新测量，而不是只依赖论文或 FLOPs 数字。"
)

heading(doc, "8.5  误差分析与典型失败模式", 2)
body(
    doc,
    "从混淆矩阵看，主要错误集中在视觉相似的类别对：猫/狗、鹿/马、鸟/飞机、船/飞机。32×32 分辨率下，"
    "这些类别的全局轮廓相似，局部纹理（毛发、羽毛、金属反光）在 4×4 patch 中也被压缩。"
    "此外，训练集的类别不均衡会让少数类（卡车）更容易被误判为汽车，因为两者在形状和颜色上接近，"
    "而汽车的样本量是卡车的两倍多。CB 通过提高卡车权重缓解了这一点，但也可能把一部分汽车误判为卡车。"
)
body(
    doc,
    "两类轻量模型的错误模式不同：MobileViT-Lite 的卷积 stem 对纹理敏感，可能把背景相似的样本分错；"
    "DynamicViT 的 token 剪枝可能删除小目标或边缘区域，导致“飞机/船”这类依赖全局形状的类别"
    "出现额外错误。一个可行的改进是：在剪枝器中加入类别感知的监督，或对被剪枝的 token 做注意力"
    "掩码蒸馏，让学生保留教师对关键区域的注意力。"
)

heading(doc, "8.6  方法适用场景", 2)
body(
    doc,
    "若目标是“不增加推理成本地缓解不均衡”，CB Loss 是最简单的选择：它不改变网络结构，不增加参数和 "
    "FLOPs，适合作为任何模型的默认基线。若目标是“降低计算量/延迟”，DynamicViT 更合适，但需要接受"
    "少量精度损失和动态 shape 的部署复杂度。若目标是“降低模型体积和计算量”，MobileViT-Lite 更合适，"
    "但需要针对 unfold/fold、深度卷积等算子做部署优化，才能把 FLOPs 优势转化为吞吐优势。若目标是"
    "“极致压缩”，可以把三者组合：CB 损失 + MobileViT-Lite 学生 + token 剪枝/量化 + 蒸馏，"
    "但组合后的调参空间和训练成本也会显著增加。"
)
body(
    doc,
    "与 A-ViT 的关系：A-ViT 让每个 token 学习一个 halt 概率，简单 token 提前停止计算，"
    "在推理时自适应减少 token 数；它与 DynamicViT 同属动态计算路线，但 halt 机制更细粒度，"
    "训练时通常需要 ponder loss 和批量内 token 排序。本文没有实现 A-ViT，是因为在 4 层、"
    "16 个 patch token 的小模型上，halt 的调度开销可能超过收益；这并不说明 A-ViT 无效，"
    "而是说明轻量化方法需要与模型规模和硬件匹配。"
)

# --------------------------------------------------------------------------------------
# 9 Limitations
# --------------------------------------------------------------------------------------
heading(doc, "9  局限与改进方向", 1, page_break=True)
body(doc, "本实验存在以下局限，阅读结果时需要注意：")
for lead, text in [
    ("单一种子：", "所有模型只使用 seed=42，未做多次重复，因而无法给出置信区间或显著性检验；小模型在 CIFAR-10 上的波动可能达到 1–2 个百分点。"),
    ("训练预算有限：", "20 个 epoch、4 层、96 维的基线是 CPU 条件下的折中，绝对精度低于论文中的大模型；DynamicViT 的精度损失也可能被欠训练放大。"),
    ("未调 CB 的 β：", "β=0.9999 是按样本量选择的，没有在验证集上搜索；更小的 β 或 Focal Loss 可能给出不同的精度—召回权衡。"),
    ("轻量化方法有限：", "只实现了 DynamicViT 和 MobileViT-Lite，没有实现 A-ViT、ToMe、Linformer、量化或结构化剪枝，方法覆盖面有限。"),
    ("速度测量的硬件局限：", "延迟和吞吐是在 CPU、8 线程、batch=1/64 下测的，不能直接外推到 GPU、手机或其他 batch；FLOPs 也不等于实际能耗。"),
    ("蒸馏教师不够强：", "教师是同样小的 CB 基线，精度有限；使用更大的教师或更强的增广可能让学生获得更多收益，但训练成本也会增加。"),
    ("测试集使用：", "测试集只在最终评估一次，但所有模型共享同一测试集，仍存在“在同一测试集上比较多个方法”的选择偏差。"),
]:
    bullet(doc, text, lead)
body(
    doc,
    "改进方向包括：（1）使用 2–3 个随机种子重复实验，报告均值±标准差；（2）延长训练到 50–100 epoch，"
    "加入 Mixup/CutMix、RandAugment 和 EMA，提升基线精度；（3）在验证集上搜索 CB 的 β、Focal Loss 的 γ，"
    "或尝试 logit adjustment；（4）实现 A-ViT 或 ToMe，与 DynamicViT 做更全面的动态计算比较；"
    "（5）把 token 剪枝与量化（INT8）结合，测量真实边缘设备上的延迟和能耗；（6）使用更大的教师"
    "（如 patch 4、d=192 的 ViT）做蒸馏，缩小 DynamicViT 的精度差距。"
)

# --------------------------------------------------------------------------------------
# 10 Conclusion
# --------------------------------------------------------------------------------------
heading(doc, "10  结论", 1)
body(
    doc,
    "本文在 CIFAR-10 上从零实现了 ViT 基线，围绕类别不均衡和轻量化完成了四组对照实验。主要结论如下："
)
for lead, text in [
    ("基线可复现：", f"4×4 patch、4 层、96 维的 ViT 参数量 {params_m('baseline_cb'):.3f} M、单张 FLOPs {gflops('baseline_cb'):.3f} G，在均衡测试集上达到 {acc['baseline_cb']*100:.1f}% 准确率；完整代码、日志和 checkpoint 均可复现。"),
    ("CB 损失有效但温和：", f"相对 CE，CB 使宏平均 F1 变化 {signed(d_f1_cb,1)} 个百分点、平衡准确率变化 {signed(d_bal_cb,1)} 个百分点，总体准确率变化 {signed(d_acc_cb,1)} 个百分点；在不均衡程度为 2.23:1 时，重加权的收益主要集中在少数类。"),
    ("DynamicViT 降低计算量：", f"FLOPs 降低 {(1-gflops('dynamic_cb')/gflops('baseline_cb'))*100:.1f}%，参数量增加 {(params_m('dynamic_cb')/params_m('baseline_cb')-1)*100:.1f}%，准确率变化 {signed(dyn_dacc,1)} 个百分点；它适合算力/延迟受限、存储相对宽裕的场景。"),
    ("MobileViT-Lite 降低参数量：", f"参数量降低 {(1-params_m('mobile_cb')/params_m('baseline_cb'))*100:.1f}%，FLOPs 降低 {(1-gflops('mobile_cb')/gflops('baseline_cb'))*100:.1f}%，准确率变化 {signed(mob_dacc,1)} 个百分点；它适合存储/带宽受限、同时要求较高精度的设备。"),
    ("复杂度需要多指标：", "参数量、FLOPs 和吞吐的排序并不一致，FLOPs 降低不会自动带来等比例吞吐提升；轻量化评估必须同时报告精度、参数量、FLOPs 和真实硬件速度。"),
]:
    bullet(doc, text, lead)
body(
    doc,
    "总体而言，本实验验证了“不均衡处理—结构轻量化—复杂度分析”这条完整链路：不均衡处理不增加推理成本，"
    "适合作为默认配置；DynamicViT 和 MobileViT-Lite 分别从“动态 token”和“卷积—注意力混合”两个方向"
    "降低计算量/参数量，但它们的精度、速度和部署复杂度取舍不同。评价重点不应是单一准确率，"
    "而是在明确部署约束后，选择代价最合适的方法。"
)

# --------------------------------------------------------------------------------------
# References
# --------------------------------------------------------------------------------------
heading(doc, "参考文献", 1, page_break=True)
refs = [
    "[1] Dosovitskiy A, Beyer L, Kolesnikov A, et al. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR, 2021. arXiv:2010.11929.",
    "[2] Vaswani A, Shazeer N, Parmar N, et al. Attention Is All You Need. NeurIPS, 2017. arXiv:1706.03762.",
    "[3] Touvron H, Cord M, Douze M, et al. Training data-efficient image transformers & distillation through attention. ICML, 2021. arXiv:2012.12877.",
    "[4] Rao Y, Zhao W, Liu B, et al. DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification. NeurIPS, 2021. arXiv:2106.02034.",
    "[5] Mehta S, Rastegari M. MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer. ICLR, 2022. arXiv:2110.02178.",
    "[6] Cui Y, Jia M, Lin T Y, et al. Class-Balanced Loss Based on Effective Number of Samples. CVPR, 2019. arXiv:1901.05555.",
    "[7] Lin T Y, Goyal P, Girshick R, et al. Focal Loss for Dense Object Detection. ICCV, 2017. arXiv:1708.02002.",
    "[8] Yin H, Vahdat A, Alvarez J M, et al. A-ViT: Adaptive Tokens for Efficient Vision Transformer. CVPR, 2022. arXiv:2112.07658.",
    "[9] Bolya D, Fu C Y, Dai X, et al. Token Merging: Your ViT But Faster. ICLR, 2023. arXiv:2210.09461.",
    "[10] Loshchilov I, Hutter F. Decoupled Weight Decay Regularization. ICLR, 2019. arXiv:1711.05101.",
    "[11] Szegedy C, Vanhoucke V, Ioffe S, et al. Rethinking the Inception Architecture for Computer Vision. CVPR, 2016. arXiv:1512.00567.",
    "[12] He K, Zhang X, Ren S, et al. Deep Residual Learning for Image Recognition. CVPR, 2016. arXiv:1512.03385.",
]
for ref in refs:
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.25
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.left_indent = Cm(0.6)
    p.paragraph_format.first_line_indent = Cm(-0.6)
    r = p.add_run(ref)
    font(r, "宋体", 9.2)

# --------------------------------------------------------------------------------------
# Appendix A
# --------------------------------------------------------------------------------------
heading(doc, "附录 A  代码结构", 1, page_break=True)
body(doc, "项目按“数据—模型—训练—分析—报告”分层，主要文件如下：")
caption(doc, "表 A1  代码与结果文件说明")
table(
    doc,
    ["路径", "说明"],
    [
        ["src/data.py", "数据扫描、uint8 数组缓存、batch 增广、归一化"],
        ["src/models/vit.py", "ViT 基线：PatchEmbed、Attention、Block、分类头"],
        ["src/models/dynamic_vit.py", "TokenPruner、DynamicViT 渐进剪枝与直通估计器"],
        ["src/models/mobile_vit.py", "ConvBNSiLU、MobileViTBlock、MobileViT-Lite"],
        ["src/models/factory.py", "按配置字典构建三种模型"],
        ["src/losses.py", "CE、CB Loss、有效样本数权重、蒸馏损失"],
        ["src/engine.py", "训练循环、评估、指标、参数量统计"],
        ["scripts/prepare_data.py", "JPEG 文件夹 → npy 数组"],
        ["scripts/train.py", "单模型训练、验证选模、测试、日志与 checkpoint"],
        ["scripts/measure_complexity.py", "参数量、FLOPs、延迟、吞吐"],
        ["scripts/analyze.py", "汇总表、训练曲线、混淆矩阵、权衡图"],
        ["scripts/build_report.py", "生成本报告 DOCX"],
        ["results/*.json", "每个实验的完整配置、历史与测试指标"],
        ["checkpoints/*.pt", "最佳模型权重与教师 logits"],
        ["figures/*.png", "报告插图"],
    ],
    widths=[5.2, 9.4],
    font_size=8.2,
    first_col_left=True,
)

# --------------------------------------------------------------------------------------
# Appendix B
# --------------------------------------------------------------------------------------
heading(doc, "附录 B  复现命令", 1, page_break=True)
body(doc, "在项目根目录下依次执行（Windows PowerShell）：")
code_lines = [
    "# 1. 数据预处理（只需一次）",
    "python scripts/prepare_data.py",
    "",
    "# 2. 基线（CE / CB）",
    "python scripts/train.py --model vit --loss ce --tag baseline_ce --epochs 20 --save-predictions",
    "python scripts/train.py --model vit --loss cb --tag baseline_cb --epochs 20 --export-logits --save-predictions",
    "",
    "# 3. DynamicViT（使用 CB 教师 logits 蒸馏）",
    "python scripts/train.py --model dynamic --loss cb --tag dynamic_cb --epochs 20 \\",
    "  --prune-after 0,1,2 --keep-ratios 0.75,0.5,0.25 \\",
    "  --distill-logits checkpoints/baseline_cb_train_logits.pt --save-predictions",
    "",
    "# 4. MobileViT-Lite",
    "python scripts/train.py --model mobile --loss cb --tag mobile_cb --epochs 20 \\",
    "  --dim 72 --mobile-depth 2 --global-depth 1 --mobile-patch 4 \\",
    "  --distill-logits checkpoints/baseline_cb_train_logits.pt --save-predictions",
    "",
    "# 5. 复杂度、token 统计、图表",
    "python scripts/measure_complexity.py --threads 8",
    "python scripts/dynamic_tokens.py",
    "python scripts/analyze.py",
    "python scripts/build_report.py",
]
for line in code_lines:
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(0)
    r = p.add_run(line if line else " ")
    font(r, "Consolas", 8.2)
note(doc, "注：完整训练在 CPU 上约需 2–3 小时；若只想冒烟测试，可加 --epochs 1 --limit-train 1000。")

# --------------------------------------------------------------------------------------
# Appendix C
# --------------------------------------------------------------------------------------
heading(doc, "附录 C  关键代码", 1, page_break=True)
heading(doc, "C.1  DynamicViT 的 token 剪枝核心", 2)
code = [
    "def forward(self, x, keep_ratio):",
    "    cls, tokens = x[:, :1], x[:, 1:]",
    "    b, n, c = tokens.shape",
    "    k = max(1, min(n, int(round(n * keep_ratio))))",
    "    score = self.fc2(self.act(self.fc1(self.norm(tokens)))).squeeze(-1)",
    "    ranking = score + gumbel_noise(score.shape, score.device, score.dtype) if self.training else score",
    "    idx = ranking.topk(k, dim=1, largest=True, sorted=False).indices",
    "    kept = torch.gather(tokens, 1, idx.unsqueeze(-1).expand(-1, -1, c))",
    "    if self.training:",
    "        soft = torch.sigmoid(score / self.temperature)",
    "        kept_soft = torch.gather(soft, 1, idx).unsqueeze(-1)",
    "        kept = kept * (kept_soft - kept_soft.detach() + 1.0)",
    "    return torch.cat((cls, kept), dim=1)",
]
for line in code:
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.1
    p.paragraph_format.space_after = Pt(0)
    r = p.add_run(line)
    font(r, "Consolas", 8.0)

heading(doc, "C.2  MobileViT Block 的 unfold–attention–fold", 2)
code2 = [
    "unfolded = F.unfold(y, kernel_size=p, stride=p)          # [B, C*p*p, L]",
    "tokens = unfolded.view(b, c, p*p, L).permute(0, 3, 2, 1).reshape(b*L, p*p, c)",
    "tokens = self.transformer(tokens)                        # attention within each patch",
    "tokens = tokens.view(b, L, p*p, c).permute(0, 3, 2, 1)",
    "folded = F.fold(tokens.reshape(b, c*p*p, L), output_size=(h, w), kernel_size=p, stride=p)",
    "return x + self.drop_path(self.project(folded))",
]
for line in code2:
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.1
    p.paragraph_format.space_after = Pt(0)
    r = p.add_run(line)
    font(r, "Consolas", 8.0)

heading(doc, "C.3  复杂度统计", 2)
code3 = [
    "from torch.utils.flop_counter import FlopCounterMode",
    "with torch.inference_mode():",
    "    with FlopCounterMode(display=False) as counter:",
    "        model(torch.randn(1, 3, 32, 32))",
    "flops = counter.get_total_flops()",
    "params = sum(p.numel() for p in model.parameters())",
]
for line in code3:
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.1
    p.paragraph_format.space_after = Pt(0)
    r = p.add_run(line)
    font(r, "Consolas", 8.0)

# --------------------------------------------------------------------------------------
# Appendix D
# --------------------------------------------------------------------------------------
heading(doc, "附录 D  代码阅读脉络", 1, page_break=True)
body(
    doc,
    "本项目代码按“数据 → 模型 → 损失 → 训练 → 轻量化 → 复杂度与报告”组织。建议不要从 "
    "build_report.py 开始读，而是先抓住三个文件：src/models/vit.py 负责“模型怎么搭”，"
    "scripts/train.py 负责“模型怎么训”，src/engine.py 负责“指标怎么算”。把这三者串起来后，"
    "再读 CB 损失和两个轻量化模型。图 D1 给出推荐阅读顺序与每个阶段要回答的问题。"
)
figure(doc, "fig_code_roadmap.png", "图 D1  代码阅读脉络：从一张 CIFAR 图像到实验报告", width_in=6.25)
caption(doc, "表 D1  推荐阅读顺序与关键问题")
table(
    doc,
    ["顺序", "文件", "读之前先问 / 读完后应能回答"],
    [
        ["1", "scripts/prepare_data.py\nsrc/data.py", "JPEG 文件夹怎样变成 uint8 数组；Dataset 的 __getitem__ 返回什么。"],
        ["2", "src/models/vit.py", "一张图怎样变成 64 个 patch token；[CLS]、位置编码、注意力和分类头的形状。"],
        ["3", "src/losses.py", "CE 与 CB 的区别；有效样本数权重为什么能帮助少数类。"],
        ["4", "src/engine.py", "一个训练 step 的顺序；准确率、宏平均 F1、ECE 怎样计算。"],
        ["5", "scripts/train.py", "分层划分、验证选模、checkpoint 和教师 logits 怎样组织。"],
        ["6", "src/models/dynamic_vit.py", "token 打分、Gumbel top-k、直通估计器怎样把 65 个 token 剪到 7 个。"],
        ["7", "src/models/mobile_vit.py", "卷积 stem、F.unfold/F.fold、patch 内注意力和全局 Block 的作用。"],
        ["8", "scripts/measure_complexity.py\nscripts/analyze.py", "参数量、FLOPs、吞吐分别说明什么；图表怎样汇总。"],
    ],
    widths=[1.4, 5.6, 8.8],
    font_size=8.0,
)
body(
    doc,
    "代码中每个文件顶部都有一段“这个文件解决什么问题、在流程中的位置、阅读前需要知道什么”的说明；"
    "每个类/函数都有中文 docstring，写明参数、返回值与张量形状；对 permute、gather、topk、unfold/fold、"
    "直通估计器、register_buffer、FlopCounterMode 等复杂语法都加了行内注释。更详细的阅读建议和"
    "Mermaid 依赖图见项目 docs/代码阅读指南.md。"
)

out = ROOT / "report" / "CIFAR10_ViT实验报告.docx"
doc.core_properties.title = "CIFAR-10 图像分类实验报告：ViT 基线、类别不均衡与轻量化"
doc.core_properties.subject = "Vision Transformer / 类别不均衡 / 模型轻量化"
doc.core_properties.author = "祝梓元"
doc.save(out)
print("written ->", out)

md_lines = [
    "# CIFAR-10 图像分类实验摘要",
    "",
    "**姓名**：祝梓元　　**学号**：U202314554　　**日期**：2026-11-10",
    "",
    "本实验在 CIFAR-10 上从零实现 ViT，比较类别平衡损失（CB Loss）、DynamicViT 渐进 token 剪枝"
    "和 MobileViT-Lite 混合结构。训练集 35,580 张（最大/最小类约 2.23:1），测试集 4,870 张均衡图像；"
    "从训练集分层抽取 10% 作为验证集，测试集只在最终评估一次。",
    "",
    "## 主要结果",
    "",
    "| 模型 | 参数量(M) | FLOPs(G) | 测试准确率(%) | 宏平均F1(%) | 吞吐(img/s) |",
    "|---|---:|---:|---:|---:|---:|",
]
for t in TAGS:
    md_lines.append(
        f"| {LABELS[t]} | {params_m(t):.3f} | {gflops(t):.3f} | {acc[t]*100:.2f} | "
        f"{f1[t]*100:.2f} | {throughput(t):.0f} |"
    )
md_lines += [
    "",
    "## 结论",
    "",
    f"- 基线（CB）测试准确率 {acc['baseline_cb']*100:.2f}%，参数量 {params_m('baseline_cb'):.3f} M，"
    f"FLOPs {gflops('baseline_cb'):.3f} G。",
    f"- CB 相对 CE 的总体准确率变化 {signed(d_acc_cb,1)} 个百分点，宏平均 F1 变化 {signed(d_f1_cb,1)} 个百分点；"
    "在中度不均衡下收益温和，但推理成本不变。",
    f"- DynamicViT 将 FLOPs 降低 {(1-gflops('dynamic_cb')/gflops('baseline_cb'))*100:.1f}%，"
    f"参数量增加 {(params_m('dynamic_cb')/params_m('baseline_cb')-1)*100:.1f}%，"
    f"测试准确率变化 {signed(dyn_dacc,1)} 个百分点。",
    f"- MobileViT-Lite 将参数量降低 {(1-params_m('mobile_cb')/params_m('baseline_cb'))*100:.1f}%、"
    f"FLOPs 降低 {(1-gflops('mobile_cb')/gflops('baseline_cb'))*100:.1f}%，"
    f"测试准确率变化 {signed(mob_dacc,1)} 个百分点。",
    "",
    "复现方法、完整结果表、混淆矩阵和误差分析见 `CIFAR10_ViT实验报告.docx` 与 `README.md`。",
]
(ROOT / "report" / "实验摘要.md").write_text("\n".join(md_lines), encoding="utf-8")
print("written ->", ROOT / "report" / "实验摘要.md")
