"""加载训练好的 DynamicViT，记录每个 token 剪枝阶段的保留数量并写成 JSON。

[脚本做什么]
    DynamicViT 会在若干个 Transformer Block 之后按比例丢弃"信息量低"的 patch token，
    从而真实地减少后续计算量（FLOPs 下降来自计算路径变短，不是掩码）。
    本脚本把训练好的 DynamicViT 跑一次前向，读取模型内部记录的 token_counts，
    于是能拿到"输入时多少 token、每个剪枝点之后还剩多少 token"，连同各阶段名称、
    配置的保留比例、最终保留比例一起写进 results/dynamic_tokens.json。
    这是一份"结构证据"：报告里说"token 逐级减少"时，引用的就是它。

[输入]
    checkpoints/dynamic_cb.pt —— 由 scripts/train.py --model dynamic --tag dynamic_cb
    训练产物。脚本内部把路径写死了，没有命令行参数，所以 tag 必须是 dynamic_cb。
    该 checkpoint 需包含四个键：model_name / model_kwargs / state_dict / tag。

[输出]
    results/dynamic_tokens.json（UTF-8，缩进 2）字段：
        kept_tokens   [65, 49, ...] 各阶段"序列总长度"，含 1 个 cls token
        stages        ["输入", "Block 1 后", ...] 与 kept_tokens 一一对应
        keep_ratios   模型配置的每级保留比例，如 [0.75, 0.5, 0.25]
        patch_tokens  kept_tokens 减 1，只数 patch token（不含 cls）
        final_ratio   counts[-1] / counts[0]，最终还剩多少比例
    同时在终端打印整个 dict 与输出路径。

[在实验流程中的位置]
    train.py（产出 dynamic_cb.pt）之后、analyze.py / build_report.py 之前。
    它不参与训练，也不改模型，只做"事后读取 + 记录"，因此非常便宜（秒级）。

[怎么运行]
    cd vit_cifar10_lab
    python scripts/dynamic_tokens.py
    运行前必须已经训练出 checkpoints/dynamic_cb.pt，且 results/ 目录存在
    （results/ 由 train.py 创建；本脚本用的是 Path.write_text，不会自动建目录）。

[阅读前需要知道的概念]
    * patch token 数：输入 32x32、patch 大小 4，则每边 32/4=8 个 patch，
      共 8*8=64 个 patch token；再加上 1 个用于分类的 cls token，序列长度为 65。
    * 剪枝比例是"复利式"的：每一级按 keep_ratios 对"当前剩余的 patch token"
      再乘一次比例，而不是相对最初的 64 个。以 0.75/0.5/0.25 为例，
      64 -> 48 -> 24 -> 6，因此最终只剩 6/64 ≈ 9.4% 的 patch token，
      远低于只乘一次 0.25 的直觉值。看图/写报告时要注意这个差别。
    * train 与 eval 的差别：训练时 TokenPruner 会给打分加 Gumbel 噪声以探索，
      推理时用确定性的打分。所以必须先 .eval()，否则同一张图两次运行结果可能不同。
    * cls token 永远保留，且始终排在第 0 位（动态剪枝只动 patch token），
      因此最后分类头取的是 x[:, 0]。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

# --- 为什么每个脚本都要重复这两行 -------------------------------------------
# Python 执行 `python scripts/dynamic_tokens.py` 时，sys.path[0] 是 scripts/，
# 而不是项目根目录，所以 `from src.models import ...` 会找不到包。
# 这里用 __file__ 反推出项目根（scripts 的上一级）并插到搜索路径最前面。
# 注意这行必须出现在 `from src... import ...` 之前，否则不起作用。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# noqa: E402 是给 linter 的豁免：这行 import 故意不在文件顶部，因为要先改 sys.path。
from src.models import build_model  # noqa: E402


def main() -> None:
    """重建 DynamicViT、跑一次前向、把各阶段 token 数导出为 JSON。

    作用
        从固定的 checkpoint 还原模型结构与权重，触发一次前向让模型内部记录
        token_counts，然后把"剪枝过程"整理成一份人类可读的 JSON 供报告引用。

    参数
        无。checkpoint 路径与输出路径都写死在函数内部，属于一次性分析脚本的
        简化做法（换成通用脚本时才会加 argparse）。

    返回值
        None。

    副作用
        1. 读取 checkpoints/dynamic_cb.pt；
        2. 覆盖写 results/dynamic_tokens.json；
        3. 向 stdout 打印 info 字典与输出路径；
        4. 会临时修改模型状态（设为 eval 模式并跑一次前向），进程结束即丢弃。

    整体步骤
        1) 拼出 checkpoint 路径并 torch.load；
        2) 用 checkpoint 里存的 model_name + model_kwargs 重建模型（保证结构一致），
           再 load_state_dict，并切换到 eval 模式；
        3) 用一个随机张量做一次前向，触发 forward_features 里对 token_counts 的记录；
        4) 拷贝出 token_counts / prune_after / keep_ratios，拼出阶段名；
        5) 组织成 info 字典 -> json.dumps -> 写文件 -> 打印。
    """
    # 路径写死为 dynamic_cb：本脚本专门用于报告里的 DynamicViT 那一节，
    # 想分析别的实验就改这一行（或以后加 --tag 参数）。
    path = ROOT / "checkpoints" / "dynamic_cb.pt"

    # weights_only=False：checkpoint 里除了 state_dict，还有 model_kwargs 这类
    # 普通 Python 对象（str/int/float/None 组成的 dict）。weights_only=True 只允许
    # 加载 tensor 等白名单类型，遇到这些会报 UnpicklingError。这里加载的是自己
    # 训练产出的本地文件，可信；千万不要对来源不明的 .pt 用 False，因为反序列化
    # 可以执行任意代码。map_location="cpu" 保证在无 GPU 机器上也能加载。
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    # 用 checkpoint 里保存的配置重建模型，而不是用当前的默认超参：
    # 否则一旦代码默认值变了，权重与结构就对不上，load_state_dict 会报形状不匹配。
    # .eval() 很关键：关闭 dropout/drop_path，并让剪枝用确定性打分（不加 Gumbel 噪声）。
    model = build_model(ckpt["model_name"], ckpt["model_kwargs"]).eval()

    # strict=True（默认）要求键集合完全一致；它能挡住"加载了半个模型"这类静默错误。
    model.load_state_dict(ckpt["state_dict"])

    # torch.no_grad()：这次前向只为触发记录，不需要梯度，关掉 autograd 可省内存与时间。
    # 输入用 torch.randn(1, 3, 32, 32)：数值本身没有意义（未做归一化），
    # 我们只关心形状——batch=1、3 通道、32x32，与真实数据一致即可。
    # 注意 DynamicViT 的 token 数由 keep_ratios 决定，与输入内容无关，所以随机图足够。
    with torch.no_grad():
        model(torch.randn(1, 3, 32, 32))

    # forward_features 每次前向都会重建 token_counts（先写输入长度，再在每个剪枝点
    # append 一次），所以上面那次前向之后它才有值；不跑前向直接读会得到空列表。
    # list(...) 复制一份：模型下次前向会就地覆盖这个属性，直接引用会读到"未来"的值。
    counts = list(model.token_counts)

    # prune_after 是"在第几个 Block（0 基）之后剪枝"，例如 (0, 1, 2)。
    # 阶段名对使用者用 1 基更自然，所以下面 i + 1。
    prune_after = list(model.prune_after)

    # 阶段名的长度必须与 counts 相同：counts[0] 是"输入"，之后每个剪枝点对应一项。
    # 这里用列表推导 + f-string 拼出 ["输入", "Block 1 后", "Block 2 后", ...]。
    stages = ["输入"] + [f"Block {i + 1} 后" for i in prune_after]

    info = {
        # 各阶段序列总长度，含 cls token。理想情况下应严格递减。
        "kept_tokens": counts,
        # 与 kept_tokens 逐项对应的中文阶段名，方便画图/写报告时直接当横轴。
        "stages": stages,
        # 模型配置里每一级的保留比例（相对"当前"patch token 数，见模块说明）。
        "keep_ratios": list(model.keep_ratios),
        # 减 1 去掉 cls token，得到纯 patch token 数。classifier 只用到 cls，
        # 但报告里关心的是"参与注意力的图像 token"有多少，所以单独给一列。
        "patch_tokens": [c - 1 for c in counts],
        # 最终保留比例：用最后一个阶段除以输入阶段，衡量整体压缩程度。
        # 注意分子分母都含 cls token，所以这个比值会略高于纯 patch 的比值。
        "final_ratio": counts[-1] / counts[0],
    }

    out = ROOT / "results" / "dynamic_tokens.json"

    # json.dumps 的两个常用参数：
    #   indent=2          每个层级缩进 2 空格、每行一个字段，便于人读和 git diff；
    #                     省略它则输出一行紧凑 JSON，机器读没问题、人读很痛苦。
    #   ensure_ascii=False 允许直接输出非 ASCII 字符。stages 里有中文"输入"、
    #                     "Block 1 后"，默认 True 会写成 \u8f93\u5165 这种转义。
    # 写文件时显式指定 encoding="utf-8"，否则 Windows 默认可能是 GBK 导致乱码。
    out.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

    # 先打印结构（调试时最直观），再打印落盘位置，方便直接从终端复制路径。
    print(info)
    print("written ->", out)


# 直接运行才执行；被 import 时不会触发读写文件。
if __name__ == "__main__":
    main()
