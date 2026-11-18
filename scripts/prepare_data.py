"""把课程提供的 JPEG 文件夹数据集，一次性转换成便于快速读取的 numpy 数组缓存。

[脚本做什么]
    课程附件里的 CIFAR-10 是"每个类别一个文件夹、文件夹里全是 .jpg"的原始布局。
    如果每个 epoch 都用 PIL 现场解码四万张小图，在 CPU 上比模型前向本身还慢，
    I/O 与解码会成为瓶颈。所以这里做一次性的离线预处理：把训练集/测试集全部
    读进内存，拼成两个连续的大数组（图像 + 标签），用 numpy 的 .npy 存到磁盘。
    之后训练脚本只需 np.load(...) 一次，就能按索引切片，不必再碰 JPEG。

[输入]
    --source：数据根目录。默认是本项目的上一级目录（即 .../ViT），其下应当能
              找到 CIFAR10_unbalance/ 与 CIFAR10_balanced/ 这两类目录。
              具体"在哪个子目录里找 0..9 这十个类文件夹"的规则，见
              src/data.py 的 _find_split()，它按目录名关键词匹配训练/测试划分。
    --workers：读图线程数，默认 16。
    --out：缓存输出目录，默认是本项目的 data/。

[输出]（全部写到 --out 目录，由 src/data.py 的 build_arrays() 负责落盘）
    train_images.npy   uint8，形状 [N_train, 32, 32, 3]（NHWC，通道在最后一维）
    train_labels.npy   int64，形状 [N_train]，取值 0..9
    test_images.npy    uint8，形状 [N_test, 32, 32, 3]
    test_labels.npy    int64，形状 [N_test]
    meta.json          记录源目录、每个类别的样本数、类名、图像尺寸
    同时会在终端打印每个 split 的样本数与类别分布。

[在实验流程中的位置]
    整个实验的第 0 步，必须最先执行，且只需要成功跑一次：
        prepare_data.py  ->  train.py  ->  measure_complexity.py / dynamic_tokens.py
                                      ->  analyze.py  ->  build_report.py
    训练脚本默认这些 .npy 已经存在，自己不会再调用本脚本。

[怎么运行]
    cd vit_cifar10_lab
    python scripts/prepare_data.py                  # 全部使用默认路径
    python scripts/prepare_data.py --workers 8      # 机械硬盘或核数较少时降低并发
    python scripts/prepare_data.py --source "D:\\claude-code\\ViT" --out data
    重复运行会覆盖同名 .npy 文件，属于幂等操作。

[阅读前需要知道的概念]
    * uint8 图像：像素保持 0~255 的原始整数，不做归一化。好处是缓存小（一张
      32x32x3 只占 3KB）、不丢信息；归一化与数据增广留给训练时按 batch 做，
      这样也方便以后改 normalize 的均值方差而不用重新预处理。
    * NHWC 布局：numpy 与 PIL 习惯 [高, 宽, 通道]，而 PyTorch 卷积要 NCHW。
      这里按 HWC 存，把 permute 集中到 src/data.py 的加载/增广函数里做一次，
      避免在磁盘格式里引入框架细节。
    * 为什么用线程池而不是多进程：读 JPEG 是 I/O 密集任务，而且 PIL 的 C 解码
      会释放 GIL，线程池足以吃满磁盘；用多进程反而要额外把几 GB 数组 pickle
      回主进程，得不偿失。
    * .npy 不是压缩格式，它原样存 dtype/shape/数据，所以加载几乎是 memcpy，
      代价是文件较大（本项目约 100MB 量级），但换来每个 epoch 的启动开销接近 0。
"""

from __future__ import annotations

# __future__ 这行让"类型注解"延迟求值（PEP 563）：写成 list[int] | None 这类
# 新语法时，旧解释器也不会在运行时立刻检查，注解只作为字符串保存。
# 本文件其实没写注解，但项目里所有模块统一加这一行，保持一致；注意它必须是
# 模块 docstring 之后的第一条语句，否则会 SyntaxError。

import argparse
import sys
import time
from pathlib import Path

# --- 定位项目根目录，并把它塞进模块搜索路径 -----------------------------------
# __file__            当前脚本的路径（可能是相对路径）
# .resolve()          转成绝对路径，同时解析符号链接与 ".."
# .parents[1]         取上两级目录：scripts/prepare_data.py -> scripts -> vit_cifar10_lab
# 因此 ROOT 恒等于项目根目录，与"你从哪个目录启动脚本"（cwd）无关。
#
# 为什么必须改 sys.path：直接执行 `python scripts/prepare_data.py` 时，Python 只把
# scripts/ 加入 sys.path，于是下面 `from src.data import ...` 会报 ModuleNotFoundError。
# 把 ROOT 插到 sys.path[0]，相当于临时告诉解释器"项目根目录也是个包搜索起点"。
# 这类写法只适合脚本入口；被 import 的库不应该这样改全局状态。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 这行 import 必须写在修改 sys.path 之后，因此违反 flake8 的 E402
#（module level import not at top of file）。行尾 # noqa: E402 是给 linter 的
# 豁免标记，不是给 Python 看的——删掉它代码照样能跑，只是 lint 会报警。
from src.data import build_arrays  # noqa: E402


def main() -> None:
    """解析命令行参数，然后调用 build_arrays() 完成 JPEG -> npy 的转换。

    作用
        本脚本唯一入口：把命令行给出的源目录与输出目录交给 src.data.build_arrays，
        并在结束时打印总耗时，方便判断这次预处理值不值得缓存。

    参数
        无（不接收函数参数）；所有配置都来自命令行，见下面的 add_argument。

    返回值
        None。成功时通过副作用产生输出文件；失败时由下层函数抛异常
        （如定位不到数据目录会抛 FileNotFoundError），脚本以非零码退出。

    副作用
        1. 在 args.out 目录写入 5 个文件（4 个 .npy + meta.json），必要时创建目录；
        2. 向 stdout 打印每个 split 的样本数（由 build_arrays 打印）与最终耗时。

    整体步骤
        1) 建 ArgumentParser，声明三个可选参数；
        2) parse_args() 把 sys.argv 解析成 Namespace；
        3) 记录起始时间 t0；
        4) 调 build_arrays 扫描目录、并发读图、堆叠数组、save；
        5) 打印耗时与输出路径。
    """
    # argparse 是标准库的命令行解析器：先声明"有哪些参数、类型、默认值"，
    # 再让 parse_args() 统一读取 sys.argv[1:]、做类型转换与合法性检查，
    # 顺带生成 --help。相比手写 sys.argv 下标，好处是：自动报错、自带帮助、
    # 参数顺序任意、支持 --key=value 与 --key value 两种写法。
    parser = argparse.ArgumentParser()

    # add_argument 的三个常见参数：
    #   "参数名"   以 -- 开头的是可选参数；不带 -- 的才是位置参数（必填）
    #   type=      解析后做的类型转换（str/int/float），转换失败会直接报错退出
    #   default=   用户没提供时使用的值
    # 这里 default 用 str(ROOT.parent)：默认去项目上一级找数据，正好对应
    # README 里"把课程附件放在 vit_cifar10_lab 的上一级"的约定。
    parser.add_argument("--source", type=str, default=str(ROOT.parent))

    # 输出目录默认是 <项目根>/data；用 ROOT 拼接而不是相对路径 "data"，
    # 这样无论从哪个 cwd 启动，产物位置都一致。
    parser.add_argument("--out", type=str, default=str(ROOT / "data"))

    # 并发读图的线程数。默认 16 是"够用且不会把机械硬盘打满"的经验值；
    # 注意它是 type=int，命令行写 --workers abc 会在解析阶段就报错。
    parser.add_argument("--workers", type=int, default=16)

    # parse_args() 才真正读取命令行：返回 argparse.Namespace 对象，
    # 之后用 args.source / args.out 这样的属性访问即可（等价于字典取值）。
    # 若用户传了未声明的参数，或缺少必填参数，它会打印用法并以状态码 2 退出。
    args = parser.parse_args()

    # perf_counter 是单调递增的高精度计时器（不受系统时间调整影响），
    # 适合测"经过了多少秒"；不要用 time.time() 做耗时统计。
    t0 = time.time()

    # 真正干活的一步：扫描目录 -> 按类收集文件 -> 线程池解码 -> 堆叠 -> np.save。
    # 它内部会打印每个 split 的样本数，也会在输出目录不存在时创建它。
    build_arrays(args.source, args.out, workers=args.workers)

    # :.1f 表示保留 1 位小数。打印的是"墙钟时间"，包含磁盘 I/O，不只是 CPU 时间。
    print(f"done in {time.time() - t0:.1f}s -> {args.out}")


# 只有"直接运行本文件"时 __name__ 才等于 "__main__"；被别的模块 import 时
# 这个条件为假，main() 不会执行，方便测试时单独 import 函数而不触发副作用。
if __name__ == "__main__":
    main()
