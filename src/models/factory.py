"""模型工厂：根据名字和配置字典构建 ViT / DynamicViT / MobileViT-Lite。

这个文件解决什么问题
--------------------
训练脚本、检查点恢复脚本和复杂度测量脚本都需要“按配置创建模型”。如果每个
脚本都写一长串 ``if model == ...`` 并手动拼 ``ViTConfig``，配置项很容易不一致。
本文件把这件事集中到 ``build_model(name, cfg)``：调用方只需要给一个模型名字和
一个普通字典 ``cfg``，工厂负责读取默认值、做类型转换、选择配置类并实例化。

在“数据→模型→训练→评估”流程中的位置
------------------------------------
- 上游：``scripts/train.py`` 解析命令行参数，把参数整理成字典；
- **本文件**：根据 ``--model`` 的值选择具体模型，返回一个未训练的 ``nn.Module``；
- 下游：``src/engine.py`` 把模型放到 device 上训练/评估，``torch.save`` 保存
  ``state_dict``。因为这里只依赖普通字典，检查点或 JSON 配置也能直接复用。

阅读前需要知道的概念
--------------------
- **``Mapping[str, Any]``**：只读字典接口。传普通 ``dict``、``argparse.Namespace``
  转成的字典都可以；工厂只调用 ``.get``，不修改原配置。
- **``dict.get(key, default)``**：键存在时返回其值，键不存在时返回默认值。
  注意它只在“键缺失”时用默认值；如果键存在但值是 ``None``，不会回退。
- **``**kwargs``**：函数签名写成 ``def f(**kwargs)`` 时，调用方任意数量的关键字
  参数会被打包成一个 dict。本文件刻意没有在 ``build_model`` 上使用
  ``**kwargs``，而是显式接收一个 ``cfg`` 映射：这样配置来源清晰、拼写错误更早
  暴露，也方便把整份 JSON 配置原样传入。若上层想写
  ``build_model("vit", **cfg_dict)``，需要把签名改成
  ``build_model(name, **kwargs)``，再把 ``kwargs`` 当作这里的 ``cfg`` 使用；
  当前实现不这样做。
- **字符串转 tuple**：命令行/JSON 里的 ``"0,1,2"`` 是字符串，而模型需要
  ``(0, 1, 2)`` 这样的整数元组；``_as_tuple`` 负责按逗号切分并逐项转换。
- **配置键别名**：历史/CLI 配置使用 ``heads``、``mobile_patch`` 这类短名字，
  模型 dataclass 使用 ``num_heads``、``patch_size``；工厂负责在两者之间映射。

主要类/函数
-----------
- ``_as_tuple``：把字符串或可迭代对象转换成指定元素类型的 tuple。
- ``build_model``：工厂入口，返回三种模型之一。
"""

from __future__ import annotations  # 延迟类型注解求值；本文件的函数签名不需要在导入时解析具体类型

from typing import Any, Mapping  # Any 表示任意值；Mapping 是只读字典接口，用来接收配置字典

from .dynamic_vit import DynamicViT
from .mobile_vit import MobileViTConfig, MobileViTLite
from .vit import ViTConfig, VisionTransformer


def _as_tuple(value: Any, cast):
    """把配置值统一转换成 tuple，并逐元素做类型转换。

    支持两种输入：
    - 字符串：按逗号切分，忽略空字符串，例如 ``"0,1,2" -> (0,1,2)``；
    - 可迭代对象（list/tuple 等）：逐元素转换，例如 ``[0.75, 0.5] -> (0.75,0.5)``。

    参数
    ----
    value : Any
        原始配置值。通常是字符串（来自命令行）或 tuple/list（来自 Python 配置）。
        如果传入 ``None``，会在迭代时触发 ``TypeError``；调用方的默认值应避免
        把 ``None`` 传进来。
    cast : callable
        元素转换函数，例如 ``int`` 或 ``float``。这里的名字是“转换器”，不是
        ``typing.cast``（那个只影响类型检查，运行时不做转换）。

    返回
    ----
    tuple
        转换后的不可变序列，长度与输入元素个数一致。

    关键设计：统一返回 tuple 而不是保留 list，既能让下游把它当作不可变配置，
    也方便打印和比较；空字符串会被丢弃，避免 ``"0,1,"`` 这种尾逗号产生空项。
    """
    if isinstance(value, str):  # 字符串是“一个可迭代对象”，必须最先单独处理，否则会被逐字符转换
        return tuple(cast(x) for x in value.split(",") if x != "")  # "0,1,2" -> ["0","1","2"] -> (0,1,2)
    return tuple(cast(x) for x in value)  # list/tuple 等：逐元素转换后转成 tuple


# 类型注解说明：`name: str`、`cfg: Mapping[str, Any]` 只是给读者/静态检查的提示，
# 运行时不会强制校验；这里不写返回类型，是因为根据 name 可能返回三种不同模型。
# 本函数不用 **kwargs 收集参数，而是显式接收 cfg 映射：JSON/argparse 配置可原样传入，键名更可控。
def build_model(name: str, cfg: Mapping[str, Any]):
    """根据模型名字和配置映射构建模型。

    参数
    ----
    name : str
        模型名，大小写不敏感；支持 ``"vit"``、``"dynamic"``、``"mobile"``。
    cfg : Mapping[str, Any]
        配置字典。所有键都是可选的，缺失时使用下面的默认值；值可以来自
        JSON/命令行，因此工厂会用 ``int``/``float`` 做显式转换。常见键：
        ``img_size``、``in_chans``、``num_classes``、``dim``、``heads``、
        ``depth``、``mlp_ratio``、``drop_path``、``patch_size``、
        ``mobile_patch``、``mobile_depth``、``global_depth``、``prune_after``、
        ``keep_ratios``。

    返回
    ----
    nn.Module
        具体类型取决于 ``name``：``"vit"`` 返回 ``VisionTransformer``，
        ``"dynamic"`` 返回 ``DynamicViT``，``"mobile"`` 返回 ``MobileViTLite``。
        模型处于训练模式、参数已初始化，但尚未训练。

    异常
    ----
    ValueError
        名字不在支持列表中，或者 DynamicViT 的剪枝配置不合法时抛出。
    """
    name = name.lower()  # 允许 "ViT"/"VIT" 等写法；字符串不可变，所以把结果赋回 name
    if name == "mobile":
        return MobileViTLite(
            MobileViTConfig(
                img_size=int(cfg.get("img_size", 32)),  # 键不存在才用 32；显式 int 兼容 "32" 这类字符串
                in_chans=int(cfg.get("in_chans", 3)),
                num_classes=int(cfg.get("num_classes", 10)),
                dim=int(cfg.get("dim", 64)),  # MobileViT 的默认宽度比 ViT 小
                num_heads=int(cfg.get("heads", 4)),  # 配置键叫 heads，dataclass 字段叫 num_heads
                mobile_depth=int(cfg.get("mobile_depth", 2)),
                global_depth=int(cfg.get("global_depth", 1)),
                patch_size=int(cfg.get("mobile_patch", 4)),  # 配置键叫 mobile_patch，避免和 ViT 的 patch_size 混淆
                mlp_ratio=float(cfg.get("mlp_ratio", 2.0)),
                drop_path=float(cfg.get("drop_path", 0.05)),
            )
        )  # 注意这里 return 了，所以不会再走下面构建 ViTConfig 的代码
    vit_cfg = ViTConfig(
        img_size=int(cfg.get("img_size", 32)),
        patch_size=int(cfg.get("patch_size", 4)),
        in_chans=int(cfg.get("in_chans", 3)),
        num_classes=int(cfg.get("num_classes", 10)),
        embed_dim=int(cfg.get("dim", 96)),  # ViT 的默认宽度是 96；配置统一用 dim 表示
        depth=int(cfg.get("depth", 4)),
        num_heads=int(cfg.get("heads", 4)),
        mlp_ratio=float(cfg.get("mlp_ratio", 2.0)),
        drop_path=float(cfg.get("drop_path", 0.05)),
    )  # 先构建共享的 ViT 配置：dynamic 和 vit 都基于它，省去重复读取
    if name == "dynamic":
        return DynamicViT(
            vit_cfg,
            prune_after=_as_tuple(cfg.get("prune_after", (0, 1, 2)), int),  # 默认在 Block 0/1/2 后剪枝
            keep_ratios=_as_tuple(cfg.get("keep_ratios", (0.75, 0.5, 0.25)), float),  # 对应 75%/50%/25%
        )
    if name == "vit":
        return VisionTransformer(vit_cfg)
    raise ValueError(f"unknown model: {name}")  # 放在最后：只有名字既不是 dynamic 也不是 vit 才会到这里
