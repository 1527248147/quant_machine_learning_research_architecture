# View Building 使用指南（视图构建）

> 本文件是 Phase 3 第一步（view building）的完整教程。
> 它教你：如何运行已有的 view、如何从零写一个新的 view builder、框架提供了哪些工具帮你读数据。
>
> 前置阅读：`guidance/guidance2_label_target_building/TARGET_BUILDING_GUIDE.md`

---

## 1. 什么是 View？

View 是**模型能直接消费的输入格式**。不同模型需要不同的 view：

| 模型类型 | View 格式 | 例子 |
|----------|-----------|------|
| LSTM / Transformer | 3D 时序张量 `[T, N, D]` | memmap 文件 |
| LightGBM / XGBoost | 2D 特征矩阵 `[样本数, 特征数]` | parquet 或 numpy |
| 图神经网络 | 图结构 + 节点特征 | 自定义格式 |

View builder 的职责就是：**读 panel_base + target_blocks → 转换成模型需要的格式 → 保存到磁盘**。

### 1.1 在 pipeline 中的位置

```
Phase 1:  raw data → build_assets → build_panel_base → panel_base
                                                          ↓
Phase 2:  panel_base → build_target → target_block(s)
                                                          ↓
Phase 3:  panel_base + target_block(s) → build_view → view 文件
                                                          ↓
          view 文件 → train.py → model → evaluation
```

View builder 直接从 `panel_base.parquet` + `target_block.parquet` 读取所需列，
不需要先合成 `panel_labeled`。这样避免了 14GB+ 的全量拼接。

### 1.2 运行

```bash
# 根据 config 中的 model.name 自动选择对应的 view builder
python scripts/build_view.py --config configs/default.yaml

# 强制重建（即使已存在）
python scripts/build_view.py --config configs/default.yaml --force
```

脚本做的事情：
1. 读取 `config["model"]["name"]`（如 `"lstm_mtl"`）
2. 从 registry 获取对应的 `ViewBuilderClass`
3. 调用 `view_builder.build(paths, config)` → 产出对应的 view 文件

### 1.3 前置条件

| 文件 | 来源 |
|------|------|
| `data/processed/panel/panel_base.parquet` | `scripts/build_panel_base.py` |
| `data/processed/targets/<name>/target_block.parquet` | `scripts/build_target.py` |

### 1.4 输入数据格式

写 view builder 之前，你必须了解你将读到的数据长什么样。

#### panel_base.parquet

统一底板，包含所有股票所有交易日的行情、因子、状态信息。**不含标签**。

```
Index: [date, sid]（存为 parquet 索引，读出来是 pandas MultiIndex）
行数:  ~7,704,699（1691 交易日 × ~4500 只股票/日）
列数:  496
大小:  ~14 GB
```

列按前缀分为 4 组：

| 前缀 | 数量 | 类型 | 说明 |
|------|------|------|------|
| `market.*` | 7 | float | 行情数据：open, high, low, close, volume, amount, adj_factor |
| `feature.*` | 473 | float | 因子/特征值（如 `feature.KMID`, `feature.MA5`）。可能有 NaN（停牌/新股等） |
| `status.*` | 12 | bool/int | 状态标记（如 `status.is_listed`, `status.is_suspended`, `status.sample_usable_for_feature`） |
| `meta.*` | 2 | string | 元信息：`meta.symbol`, `meta.list_date` |

**对 view builder 最重要的列**：

- `feature.*`（473 列）— 模型的输入特征。用 `read_feature_chunks()` 分批读取
- `status.sample_usable_for_feature`（bool）— 该行是否可用于建模。`True` 表示该股票当天有有效的因子数据
- `market.close` 等行情列 — 部分 view 可能需要（如回测用）

**注意**：`feature.*` 列中的 NaN 表示该因子在该天对该股票不可用（通常是停牌或数据缺失），
不是 bug。你的 view builder 需要决定如何处理它（LSTM view 用 isna flag 标记 + 填 0）。

> 详细列名清单见 `data/processed/panel/COLUMN_REFERENCE.md`
> 详细结构说明见 `guidance/guidance1_panel_base_and_assets/PANEL_BASE_GUIDE.md`

#### target_block.parquet

每个 target 一个文件，包含标签值和有效性标记。

```
Index: [date, sid]（与 panel_base 完全对齐，同样的行数和行顺序）
列数:  3
大小:  ~60 MB
```

| 列名 | 类型 | 说明 |
|------|------|------|
| `label.<target_name>` | float32 / int | 标签值。无效时为 NaN |
| `label_valid.<target_name>` | bool | 该标签是否有效。`True` = 可用于训练 |
| `label_reason.<target_name>` | string | 无效原因（有效时为空字符串 `""`） |

以 `return_c0c1` 为例：

| 列名 | 例子值 | 说明 |
|------|--------|------|
| `label.return_c0c1` | `0.0234` | 1 日前向收益率 |
| `label_valid.return_c0c1` | `True` | 标签有效 |
| `label_reason.return_c0c1` | `""` | 无原因（有效） |

以 `momentum_cls` 为例：

| 列名 | 例子值 | 说明 |
|------|--------|------|
| `label.momentum_cls` | `3` | 5 类动量分类（0-4） |
| `label_valid.momentum_cls` | `False` | 标签无效 |
| `label_reason.momentum_cls` | `"insufficient_forward_window"` | 前向窗口不足 |

**关键特性**：
- target_block 和 panel_base **行完全对齐**（相同的 `[date, sid]` 索引，相同的行顺序），所以读出来的数组可以直接按行对应
- `label_valid=False` 的行，`label` 值为 NaN。训练时用 mask 过滤，不删行
- 用 `read_target_labels(paths, "return_c0c1")` 读取时，只会返回 `label` 和 `label_valid` 两列（`label_reason` 不需要加载到 view 中）

> 详细标签说明见 `guidance/guidance2_label_target_building/TARGET_BUILDING_GUIDE.md`

---

## 2. 从零写一个新的 View Builder（教程）⭐⭐⭐

假设你要为一个新模型 `lgb`（LightGBM）写 view builder。完整流程分 3 步。

### 2.1 第 1 步：创建文件

在 `engine/views/` 下新建子目录（**目录名 = model name**）：

```
engine/views/
├── base.py                  # 抽象基类（不要修改）
├── lstm_mtl/                # 已有
│   ├── __init__.py
│   └── view.py
└── lgb/                     # 你要新建的 ↓
    ├── __init__.py          # 空文件
    └── view.py              # view builder 代码
```

### 2.2 第 2 步：写 view.py

#### 2.2.1 继承 `BaseViewBuilder`

你的 view builder **必须**继承 `BaseViewBuilder` 并实现 4 个方法：

```python
from engine.views.base import BaseViewBuilder

class LGBViewBuilder(BaseViewBuilder):
    name = "lgb"  # ← 必须和 model name、目录名一致

    def required_columns(self) -> List[str]: ...
    def build(self, paths: PathManager, config: dict) -> Path: ...
    def get_dataset(self, view_dir, day_start, day_end, config) -> Any: ...
    def build_dataloader(self, dataset, config, shuffle=True) -> Any: ...
```

#### 2.2.2 用 `panel_reader` 工具读数据（重要！）

框架在 `engine/io/panel_reader.py` 中提供了 4 个封装好的读取工具，
**你不需要自己写 PyArrow 代码**：

```python
from engine.io.panel_reader import (
    read_panel_index,       # 读 date/sid 索引 + 任意附加列
    read_feature_chunks,    # 分 chunk 迭代读 feature.* 列
    read_target_labels,     # 读单个 target 的 label + valid
    read_feature_columns,   # 从 schema 读 feature 列名（不加载数据）
)
```

**① `read_panel_index(paths, extra_columns=[])`** — 读索引

返回 `PanelIndex` 对象，包含 date/sid 映射和你要求的附加列：

```python
pidx = read_panel_index(paths, extra_columns=["status.sample_usable_for_feature"])

pidx.T              # 交易日数量
pidx.N              # 股票数量
pidx.row_t          # int32 数组，每行对应的日期索引 t
pidx.row_n          # int32 数组，每行对应的股票索引 n
pidx.date_strings   # ["2019-01-02", "2019-01-03", ...]
pidx.sid_strings    # ["000001.SZ", "000002.SZ", ...]
pidx.num_rows       # 总行数（~770 万）
pidx.extra_columns["status.sample_usable_for_feature"]  # numpy 数组
```

峰值内存：~500 MB（只读 date + sid + 附加列）。

**② `read_feature_chunks(paths, chunk_size=50)`** — 分块读特征

每次只读 50 列，避免一次性加载全部 473 个 feature（~14 GB）：

```python
for chunk_start, chunk_end, col_names, values in read_feature_chunks(paths, chunk_size=50):
    # chunk_start: 起始列索引（如 0, 50, 100, ...）
    # chunk_end:   结束列索引（如 50, 100, 150, ...）
    # col_names:   本 chunk 的列名列表
    # values:      numpy float32 数组，shape = [num_rows, chunk_size]

    # 处理这个 chunk 的特征...
```

峰值内存：~2 GB（一次持有 50 列）。

**③ `read_target_labels(paths, "return_c0c1")`** — 读标签

```python
labels = read_target_labels(paths, "return_c0c1")

labels.found         # bool: target_block 文件是否存在
labels.label_values  # float32 数组，标签值（NaN = 无效）
labels.valid_mask    # bool 数组，标签是否有效
```

峰值内存：~60 MB。

**④ `read_feature_columns(paths.panel_base_path)`** — 读列名

不加载任何数据，只读 parquet schema：

```python
feat_cols = read_feature_columns(paths.panel_base_path)
# ['feature.BETA10', 'feature.BETA20', ..., 'feature.fund__xxx']
# 共 473 个
```

#### 2.2.3 完整示例：LightGBM View Builder

```python
"""
LGB view builder: 将 panel_base + target_blocks 转换为 LightGBM 需要的 2D 矩阵。
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, List

import numpy as np

from engine.io.paths import PathManager
from engine.io.panel_reader import (
    read_panel_index,
    read_feature_columns,
    read_feature_chunks,
    read_target_labels,
)
from engine.views.base import BaseViewBuilder

logger = logging.getLogger(__name__)


class LGBViewBuilder(BaseViewBuilder):
    """View builder for LightGBM."""

    name = "lgb"

    def required_columns(self) -> List[str]:
        return ["feature.*", "status.sample_usable_for_feature"]

    def build(self, paths: PathManager, config: dict) -> Path:
        out_dir = paths.memmap_dir(self.name)
        meta_path = out_dir / "meta.json"

        # 幂等：已构建则跳过
        if meta_path.exists():
            logger.info("Already built at %s — skipping", out_dir)
            return out_dir

        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()

        # --- 读索引 ---
        pidx = read_panel_index(paths, extra_columns=["status.sample_usable_for_feature"])
        usable = pidx.extra_columns["status.sample_usable_for_feature"].astype(bool)

        # --- 读特征（分 chunk 拼接）---
        feat_cols = read_feature_columns(paths.panel_base_path)
        all_feats = np.empty((pidx.num_rows, len(feat_cols)), dtype=np.float32)
        for start, end, cols, vals in read_feature_chunks(paths, feat_columns=feat_cols):
            all_feats[:, start:end] = vals

        # --- 读标签 ---
        label_name = config.get("model", {}).get("label_roles", {}).get("regression", "return_c0c1")
        labels = read_target_labels(paths, label_name)

        # --- 构建训练 mask ---
        mask = usable & labels.valid_mask
        X = all_feats[mask]
        y = labels.label_values[mask]
        dates = pidx.row_t[mask]  # 日期索引，用于时间分割

        # --- 保存 ---
        np.save(str(out_dir / "X.npy"), X)
        np.save(str(out_dir / "y.npy"), y)
        np.save(str(out_dir / "dates.npy"), dates)

        elapsed = time.time() - t0
        meta = {
            "num_samples": int(mask.sum()),
            "num_features": len(feat_cols),
            "feat_cols": feat_cols,
            "label_name": label_name,
            "build_time_s": round(elapsed, 1),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        logger.info("LGB view built: %d samples, %d features, %.1fs", X.shape[0], X.shape[1], elapsed)
        return out_dir

    def get_dataset(self, view_dir: Path, day_start: int, day_end: int, config: dict) -> Any:
        X = np.load(str(view_dir / "X.npy"))
        y = np.load(str(view_dir / "y.npy"))
        dates = np.load(str(view_dir / "dates.npy"))
        mask = (dates >= day_start) & (dates <= day_end)
        return {"X": X[mask], "y": y[mask]}

    def build_dataloader(self, dataset: Any, config: dict, shuffle: bool = True) -> Any:
        return dataset  # LightGBM 不需要 DataLoader，直接返回 dict
```

### 2.3 第 3 步：注册

编辑 `engine/models/registry.py`，在 `_register_builtins()` 中添加：

```python
def _register_builtins() -> None:
    from engine.models.impl.lstm_mtl.model import LSTMMTLModel
    from engine.views.impl.lstm_mtl.view import LSTMViewBuilder
    from engine.training.impl.lstm_mtl.trainer import LSTMMTLTrainer
    register_model("lstm_mtl", LSTMMTLModel, LSTMViewBuilder, LSTMMTLTrainer)

    # 新增 ↓
    from engine.models.impl.lgb.model import LGBModel
    from engine.views.impl.lgb.view import LGBViewBuilder
    from engine.training.impl.lgb.trainer import LGBTrainer
    register_model("lgb", LGBModel, LGBViewBuilder, LGBTrainer)
```

然后改 config：

```yaml
model:
  name: "lgb"   # ← build_view.py 和 train.py 会自动使用 LGBViewBuilder
  label_roles:
    regression: "return_c0c1"
  view: {}       # lgb 不需要额外 view 参数
```

运行：
```bash
python scripts/build_view.py --config configs/lgb.yaml
```

**不需要修改 `build_view.py` 或 `train.py`**。

---

## 3. 规则清单（必须遵守）

写新的 view builder 时，以下规则**必须**遵守：

### 3.1 命名规则

| 项目 | 规则 | 例子 |
|------|------|------|
| 目录名 | `engine/views/<model_name>/` | `engine/views/lgb/` |
| 文件名 | `view.py` | — |
| 类的 `name` 属性 | 等于 model name | `name = "lgb"` |
| model 类的 `name` 属性 | 同上 | `name = "lgb"` |

**view name = model name = config 中的 `model.name`**。三者必须一致。

### 3.2 `build()` 方法的规则

| 规则 | 说明 |
|------|------|
| **幂等** | `meta.json` 存在时必须跳过构建。`--force` 由 `build_view.py` 处理（删除 `meta.json` 后再调用）。 |
| **不接受 DataFrame** | `build()` 的签名是 `(self, paths: PathManager, config: dict) -> Path`。不要让调用方传 DataFrame。 |
| **自行读数据** | 用 `panel_reader` 工具读取 panel_base 和 target_block，只读你需要的列。 |
| **控制内存** | panel_base 有 14 GB。用 `read_feature_chunks()` 分批读取，不要一次性全量加载。 |
| **必须输出 `meta.json`** | 这是构建完成的标志。至少包含 build_time_s 和你 `get_dataset()` 需要的元信息。 |
| **返回 out_dir** | 返回构建产物的目录路径。 |

### 3.3 `get_dataset()` 和 `build_dataloader()` 的规则

| 规则 | 说明 |
|------|------|
| **惰性加载** | 对大文件用 memmap 或分片读取，不要全部加载到内存。 |
| **day_start / day_end** | 这是 memmap 的日期索引（整数），由 trainer 根据 config 中的时间分割计算。 |
| **返回类型** | `get_dataset()` 返回 `Any`，你的 model 和 trainer 能消费就行。PyTorch 模型通常返回 `torch.utils.data.Dataset`。 |

### 3.4 不要做的事

```python
# ❌ 不要全量读取 panel_base
panel = pd.read_parquet(str(paths.panel_base_path))  # 14 GB!

# ❌ 不要手写 PyArrow — 用 panel_reader
import pyarrow.parquet as pq
table = pq.read_table(str(paths.panel_base_path), columns=[...])

# ❌ 不要期望接收 panel_labeled
def build(self, panel_labeled: pd.DataFrame, ...):  # 错，这个参数已经不存在了

# ❌ 不要硬编码 label 名
label_col = "label.return_c0c1"  # 错，应该从 config["model"]["label_roles"] 读取
```

---

## 4. 框架提供的工具一览

### 4.1 数据读取 — `engine/io/panel_reader.py`

| 函数 | 输入 | 输出 | 内存 |
|------|------|------|------|
| `read_panel_index(paths, extra_columns=)` | PathManager + 附加列名 | `PanelIndex`（date/sid 映射 + 附加列数据） | ~500 MB |
| `read_feature_chunks(paths, chunk_size=50)` | PathManager + chunk 大小 | Iterator: (start, end, col_names, values_f32) | ~2 GB 峰值 |
| `read_target_labels(paths, target_name)` | PathManager + target 名 | `TargetLabels`（label_values + valid_mask） | ~60 MB |
| `read_feature_columns(panel_base_path)` | 文件路径 | `List[str]` 特征列名 | ~0 |

### 4.2 路径管理 — `engine/io/paths.py`

| 方法 | 返回 |
|------|------|
| `paths.panel_base_path` | `data/processed/panel/panel_base.parquet` |
| `paths.target_block_path("return_c0c1")` | `data/processed/targets/return_c0c1/target_block.parquet` |
| `paths.memmap_dir("lgb")` | `data/processed/views/lgb/memmap/` |
| `paths.memmap_meta_path("lgb")` | `data/processed/views/lgb/memmap/meta.json` |

### 4.3 基类 — `engine/views/base.py`

```python
class BaseViewBuilder(ABC):
    name: str                                          # 必须设置

    def required_columns(self) -> List[str]: ...       # 声明需要的列（文档/调试用）
    def build(self, paths, config) -> Path: ...        # 构建 view 文件
    def get_dataset(self, view_dir, day_start, day_end, config) -> Any: ...  # 给 trainer 的接口
    def build_dataloader(self, dataset, config, shuffle) -> Any: ...         # 给 trainer 的接口
```

---

## 5. 已有的 View Builder

### 5.1 lstm_mtl（LSTM 多任务）

- 代码：`engine/views/lstm_mtl/view.py`
- 输出格式：3D memmap `[T, N, D]`，D = 2F + 1（特征 + isna 标记 + row_present）
- 标签：回归 + 分类两个 target，分别写入独立的 memmap 文件
- Dataset：`MemmapDayWindowDataset` — 按天采样 K 只股票，返回 lookback 窗口

输出文件：

```
data/processed/views/lstm_mtl/memmap/
├── X_f16.mmap           # [T=1691, N=5351, D=947] float16
├── y_ret_f32.mmap       # [T, N] float32 — 回归标签
├── y_mom_i8.mmap        # [T, N] int8 — 分类标签（0-4，无效=-1）
├── ret_mask_u8.mmap     # [T, N] uint8 — 回归有效 mask
├── mom_mask_u8.mmap     # [T, N] uint8 — 分类有效 mask
├── both_mask_u8.mmap    # [T, N] uint8 — 两者都有效
├── meta.json            # 元信息
├── VIEW_REPORT.md       # 质量报告
└── COLUMN_REFERENCE.md  # 维度-列名对照表
```

X 张量的维度布局 (D = 2F + 1)：

| 维度范围 | 数量 | 说明 |
|----------|------|------|
| `[0, 472]` | 473 | 特征值（原始 NaN → 0.0） |
| `[473, 945]` | 473 | isna 标记（1.0 = 原始值为 NaN，0.0 = 有值） |
| `[946]` | 1 | row_present（来自 `status.sample_usable_for_feature`） |

isna 的意义：模型可以区分"特征值真的是 0"和"缺失被填成 0"。

---

## 6. 架构参考

### 6.1 代码文件对照表

| 文件 | 作用 |
|------|------|
| `engine/views/base.py` | `BaseViewBuilder` 抽象基类 |
| `engine/views/lstm_mtl/view.py` | LSTM 多任务 view builder |
| `engine/io/panel_reader.py` | 数据读取工具（`read_panel_index`、`read_feature_chunks` 等） |
| `engine/io/paths.py` | `PathManager`：路径管理 |
| `engine/models/registry.py` | 注册表：`register_model(name, ModelClass, ViewClass)` |
| `scripts/build_view.py` | CLI 入口：构建 view |

### 6.2 build_view.py 的调用流程

```
build_view.py
    │
    ├─ 读取 config["model"]["name"]         # "lstm_mtl"
    │
    ├─ get_view_class("lstm_mtl")           # → LSTMViewBuilder (from registry)
    │
    ├─ view_builder = LSTMViewBuilder()
    │
    └─ view_builder.build(paths, config)    # → 读 parquet, 写 memmap
                │
                │  内部使用 panel_reader 工具:
                ├─ read_panel_index(paths)          # 读索引
                ├─ read_feature_chunks(paths)       # 分块读特征
                ├─ read_target_labels(paths, ...)   # 读标签
                │
                │  输出到:
                └─ paths.memmap_dir("lstm_mtl")     # data/processed/views/lstm_mtl/memmap/
```

### 6.3 完整 pipeline 命令

```bash
# Phase 1: 构建底板
python scripts/build_assets.py --config configs/default.yaml
python scripts/build_panel_base.py --config configs/default.yaml

# Phase 2: 构建标签
python scripts/build_target.py --config configs/default.yaml

# Phase 3: 构建 view → 训练
python scripts/build_view.py --config configs/default.yaml
python scripts/train.py --config configs/default.yaml
```

---

## 7. 常见问题

**Q：需要先跑 `build_panel_labeled.py` 吗？**

A：不需要。View builder 用 `panel_reader` 直接读 `panel_base` + `target_block`。

**Q：修改了 target（比如换了 `return_5d`），需要重建 view 吗？**

A：需要。`python scripts/build_view.py --config configs/default.yaml --force`

**Q：只改了训练参数（lr、epochs），需要重建 view 吗？**

A：不需要。View 只包含特征和标签数据。

**Q：不同 model 可以复用同一个 view builder 吗？**

A：可以。在 registry 中注册时指向同一个 ViewBuilder 类即可：
```python
register_model("model_a", ModelA, SharedViewBuilder)
register_model("model_b", ModelB, SharedViewBuilder)
```

**Q：panel_base 有 14 GB，怎么避免内存爆掉？**

A：用 `read_feature_chunks(paths, chunk_size=50)` 分批读，峰值内存约 2 GB。
绝对不要 `pd.read_parquet(paths.panel_base_path)` 全量加载。

**Q：如何查看所有已注册的 model/view？**

A：
```python
from engine.models.registry import list_models
print(list_models())  # ['lstm_mtl']
```
