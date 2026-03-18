# Target Building 使用指南（标签构建）

> 本文件是 Phase 2 的完整使用手册。
> 它覆盖：如何运行已有的标签构建、如何添加新标签、输出数据的格式、以及 panel_labeled 的使用方法。
>
> 前置阅读：`guidance/guidance1_panel_base_and_assets/PANEL_BASE_GUIDE.md`

---

## 1. 整体概览

### 1.1 Phase 2 在整个 pipeline 中的位置

```
Phase 1:  raw data → build_assets → build_panel_base → panel_base
                                                          ↓
Phase 2:  panel_base → build_target → target_block(s)
                                          ↓
          panel_base + target_block(s) → build_panel_labeled → panel_labeled
                                                                   ↓
Phase 3:  panel_labeled → view → model → trainer → backtest
```

### 1.2 核心数据结构

| 名称 | 格式 | 说明 |
|------|------|------|
| **panel_base** | `index=[date, sid]`, 496 列 | 统一底板：行情 + 因子 + 状态（不含标签） |
| **target_block** | `index=[date, sid]`, 3 列/target | 标签表：每个 target 产出 3 列 |
| **panel_labeled** | `index=[date, sid]`, 496 + 3×N 列 | panel_base + 所有 target_block 的列拼接 |

### 1.3 target_block 的 3 列合约

**每个 target 必须且只产出以下 3 列**：

| 列名 | 类型 | 说明 |
|------|------|------|
| `label.<target_name>` | float / int | 标签值。无效时为 NaN |
| `label_valid.<target_name>` | bool | 该标签是否有效。`True` = 可用于训练 |
| `label_reason.<target_name>` | string | 无效原因。有效时为空字符串 `""` |

**为什么需要 label_valid 和 label_reason？**

在量化场景中，很多样本的标签天然无法计算（数据集末尾没有足够前向窗口、停牌导致无价格等）。
传统做法是直接删掉这些行，但这会破坏面板结构。我们的做法是**保留所有行**，用 `label_valid` 标记哪些可用，用 `label_reason` 记录不可用的原因。这样：
- 面板结构完整（LSTM 等序列模型需要连续行）
- 下游训练时用 mask 过滤，而非删行
- 可以统计和审计无效样本的分布

---

## 2. 运行方法

### 2.1 运行 build_target（生成 target_block）

```bash
# 构建 configs/default.yaml 中定义的所有 target
python scripts/build_target.py --config configs/default.yaml

# 只构建单个 target
python scripts/build_target.py --config configs/default.yaml --target return_c0c1
```

**内存优化**：脚本不会加载完整的 panel_base（~14 GB）。
engine 会根据 recipe 的 `required_columns()` 声明，只从 parquet 读取所需的列（通常只有 ~120 MB）。

### 2.2 运行 build_panel_labeled（合并 panel_base + target_block）

```bash
# 合并所有 target
python scripts/build_panel_labeled.py --config configs/default.yaml

# 只合并单个 target
python scripts/build_panel_labeled.py --config configs/default.yaml --target return_c0c1
```

### 2.3 配置文件格式

在 `configs/default.yaml` 中：

```yaml
targets:
  - name: "return_c0c1"        # 今日收盘 → 明日收盘
  - name: "momentum_cls"
    params:
      gap: 4
      line_len: 7
      price_col: "market.close"
```

- `name`：target 的唯一标识名（用于输出目录、label 列名等）。如果不在 registry 中，需要同时在 registry 中注册
- `params`：传递给 recipe 的参数，覆盖 TargetSpec 中的默认参数。只写你想改的，其余用默认值

---

## 3. 输出文件说明

### 3.1 目录结构

```
data/processed/
├── panel/
│   └── panel_base.parquet                    # Phase 1 输出
├── targets/
│   ├── return_c0c1/
│   │   ├── target_block.parquet              # target_block 数据
│   │   ├── TARGET_REPORT_return_c0c1.md        # 质量报告
│   │   └── COLUMN_REFERENCE_return_c0c1.md     # 列名参考
│   └── momentum_cls/
│       ├── target_block.parquet
│       ├── TARGET_REPORT_momentum_cls.md
│       └── COLUMN_REFERENCE_momentum_cls.md
└── panel_labeled/
    └── return_c0c1+momentum_cls/
        └── panel_labeled.parquet             # panel_base + 所有 target 合并
```

### 3.2 target_block.parquet 格式

以 `return_c0c1` 为例：

| 列名 | 类型 | 说明 |
|------|------|------|
| date | datetime | 交易日（index） |
| sid | string | 股票标识（index） |
| `label.return_c0c1` | float32 | 收益率 = (close_{t+1} - close_t) / close_t |
| `label_valid.return_c0c1` | bool | 标签是否有效 |
| `label_reason.return_c0c1` | string | 无效原因 |

以 `momentum_cls` 为例：

| 列名 | 类型 | 说明 |
|------|------|------|
| date | datetime | 交易日（index） |
| sid | string | 股票标识（index） |
| `label.momentum_cls` | float32 | 5 类动量分类 (0-4) |
| `label_valid.momentum_cls` | bool | 标签是否有效 |
| `label_reason.momentum_cls` | string | 无效原因 |

### 3.3 panel_labeled.parquet 格式

panel_labeled 是 panel_base 和所有 target_block 的**列拼接**（行对齐）。

```
panel_labeled 的列 = panel_base 的全部列 + 每个 target 的 3 列

具体示例（return_c0c1 + momentum_cls）：
  index: [date, sid]
  列数: 496 (panel_base) + 3 (return_c0c1) + 3 (momentum_cls) = 502 列

  panel_base 原有列:
    market.open, market.high, ..., market.adj_factor      (7 列)
    feature.KMID, feature.KLEN, ..., feature.fund__xxx    (473 列)
    status.is_listed, status.is_suspended, ...            (12 列)
    meta.symbol, meta.list_date                           (2 列)

  新增 target 列:
    label.return_c0c1, label_valid.return_c0c1, label_reason.return_c0c1
    label.momentum_cls, label_valid.momentum_cls, label_reason.momentum_cls
```

### 3.4 质量报告（TARGET_REPORT_xxx.md）

每个 target 构建完后自动生成，包含：
- 基本信息（任务类型、有效率）
- 无效原因分布
- 标签分布（回归：统计量/分位数；分类：各类别数量和占比）
- 按年分布
- 质量检查（有效率、NaN 检查、类别平衡度）

---

## 4. 已内置的 Target（标签）

### 4.1 ReturnRecipe（灵活收益率 recipe）

**任务类型**：regression

**核心概念**：使用量化标准的 `o/c` 记法指定入场和出场价格点：

```
formula = "{entry_type}{entry_day}{exit_type}{exit_day}"

    entry_type / exit_type:  'o' = 开盘价 (open),  'c' = 收盘价 (close)
    entry_day / exit_day:    相对 t 的交易日偏移 (0 = 当天)
```

**计算公式**：
```
label = (exit_price - entry_price) / entry_price
```
所有日偏移都是**交易日**（不是自然日），通过 trading_calendar 定位。

**常用公式示例**：

| formula | 含义 | 典型用途 |
|---------|------|---------|
| `c0c1` | 今日收盘 → 明日收盘 | 最基础的日收益 |
| `o1c1` | 明日开盘 → 明日收盘 | 日内收益 |
| `c0o1` | 今日收盘 → 明日开盘 | 隔夜收益 |
| `o1c2` | 明日开盘 → 后日收盘 | 短期持仓（避免 T 日涨停无法买入） |
| `o1c5` | 明日开盘 → 第 5 日收盘 | 约 1 周持仓 |
| `c0c20` | 今日收盘 → 20 日后收盘 | 月度收益 |

> **为什么推荐 `o1c2` 而非 `c0c1`？** 在 A 股 T+1 规则下，信号产生于 t 日收盘后，
> 最早只能在 t+1 日开盘买入。用 `c0c1` 计算的收益包含了 t→t+1 的涨跌，而实际上交易者无法在 t 日收盘价买入。
> `o1c2` 更贴近真实可获得的收益。

**参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `formula` | 无（必填，除非用 `window` 兼容） | 价格公式，如 `"o1c2"` |
| `open_col` | `"market.open"` | panel_base 中开盘价列名 |
| `close_col` | `"market.close"` | panel_base 中收盘价列名 |
| `window` | 无 | **向后兼容**：`window=N` 等价于 `formula="c0cN"` |
| `price_col` | 无 | **向后兼容**：设置 close_col |

**预注册的 target**：

| 注册名 | formula | 说明 |
|--------|---------|------|
| `return_c0c1` | `c0c1` | 今日收盘→明日收盘（与参考模型一致） |
| `return_5d` | `c0c5` | 5 日 close-to-close |
| `return_o1c5` | `o1c5` | 次日开盘→5 日后收盘 |
| `return_20d` | `c0c20` | 月度收益 |

**无效原因**：

| reason | 说明 |
|--------|------|
| `insufficient_history` | entry_day 指向数据范围之前 |
| `insufficient_forward` | exit_day 超出数据集末尾 |
| `missing_entry_price` | 入场价格为 NaN 或 ≤ 0（停牌等） |
| `missing_exit_price` | 出场价格为 NaN 或 ≤ 0 |

### 4.2 momentum_cls（5 类动量分类）

**任务类型**：classification

**计算方法**：

对每个 (date=t, sid)，构建一条动量线（momentum line）：
```
m_line[k] = close[t + k] - close[t - GAP]    k = 0, 1, ..., LINE_LEN-1
```
然后根据动量线的零穿越（zero-crossing）模式分类：

| 类别 | 名称 | 含义 |
|------|------|------|
| 4 | Bounce (负→正) | 动量线从负穿越到正（恰好 1 次穿零） |
| 3 | Positive (全正) | 动量线全程 > 0 |
| 2 | Volatile (震荡) | 多次穿零或信号混杂 |
| 1 | Negative (全负) | 动量线全程 < 0 |
| 0 | Sink (正→负) | 动量线从正穿越到负（恰好 1 次穿零） |

**参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `gap` | 4 | 回看间隔 GAP（交易日数） |
| `line_len` | 7 | 动量线长度 LINE_LEN（采样点数） |
| `price_col` | `"market.close"` | 使用的价格列 |

**无效原因**：

| reason | 说明 |
|--------|------|
| `insufficient_history` | 日期前不足 GAP 个交易日 |
| `insufficient_forward_window` | 日期后不足 LINE_LEN-1 个交易日 |
| `missing_close_in_window` | 所需窗口内任何一天的收盘价为 NaN 或 ≤ 0 |

---

## 5. 如何添加新的 Target（如何添加新的label）⭐⭐⭐
为了打好label，你一定要先去看看data\processed\panel\COLUMN_REFERENCE.md和data\processed\panel\PANEL_BASE_GUIDE.md，那里面有panel_base的具体的列名以及对panel_base的具体的讲解

添加一个新标签只需 3 步：**写 recipe → 注册 → 加配置**。不需要修改 engine、脚本或其他模块。

### 5.1 第 1 步：创建 Recipe 文件

在 `engine/targets/recipes/` 下新建一个 `.py` 文件。

**文件命名**：用蛇形命名法描述标签，如 `volatility_20d.py`、`sector_rank.py`。

**Recipe 的标准模板**：

```python
"""
<Recipe 名>: 一句话描述这个标签做什么。

<详细公式或算法描述>

Invalid-label reasons
---------------------
- "reason_a" : 什么情况下会出现
- "reason_b" : 什么情况下会出现
"""
from __future__ import annotations
from typing import Dict, List

import numpy as np
import pandas as pd

from engine.targets.base import BaseTargetRecipe


class MyNewRecipe(BaseTargetRecipe):
    """一句话描述。"""

    # ================================================================
    # 1. __init__: 声明参数
    # ================================================================
    def __init__(
        self,
        name: str,                                    # 必须有，engine 会传入
        # ... 你的自定义参数 ...
        my_param: int = 10,
        price_col: str = "market.close",
        trading_calendar: pd.DataFrame | None = None, # engine 自动传入
        **kwargs,                                     # 必须有，吸收多余参数
    ):
        self.name = name
        self.my_param = my_param
        self.price_col = price_col
        self.trading_calendar = trading_calendar

    # ================================================================
    # 2. required_columns: 声明需要 panel_base 的哪些列
    #    engine 会只加载这些列（+ date, sid），而不是全部 496 列
    # ================================================================
    def required_columns(self) -> List[str]:
        return [self.price_col]
        # 如果还需要成交量: return [self.price_col, "market.volume"]
        # 如果需要某个因子: return [self.price_col, "feature.MA5"]

    # ================================================================
    # 3. class_names (可选): 仅分类任务需要
    #    返回 {整数类别: 显示名称}，用于质量报告的自动展示
    # ================================================================
    def class_names(self) -> Dict[int, str]:
        return {}  # 回归任务返回空字典即可

    # ================================================================
    # 4. compute: 核心计算逻辑
    #    输入: panel_base (index=[date, sid], 只含你声明的列)
    #    输出: target_block (index=[date, sid], 恰好 3 列)
    # ================================================================
    def compute(self, panel_base: pd.DataFrame) -> pd.DataFrame:
        label_col = f"label.{self.name}"
        valid_col = f"label_valid.{self.name}"
        reason_col = f"label_reason.{self.name}"

        idx = panel_base.index  # 直接复用原 index，不要 copy 整个 DataFrame

        # --- 你的计算逻辑 ---
        # 提取数据
        values = panel_base[self.price_col].values.astype(np.float64)

        # 计算标签
        label = np.full(len(values), np.nan, dtype=np.float64)
        # label = ... 你的公式 ...

        # 判断有效性
        valid = np.ones(len(values), dtype=bool)
        reason = np.full(len(values), "", dtype=object)

        # 标记无效样本
        # bad_mask = ...
        # valid[bad_mask] = False
        # reason[bad_mask] = "your_reason"
        # label[~valid] = np.nan

        # --- 组装输出 ---
        target_block = pd.DataFrame(
            {
                label_col: label.astype(np.float32),  # 或 np.int16 用于分类
                valid_col: valid,
                reason_col: reason,
            },
            index=idx,  # 直接复用 panel_base 的 index
        )
        return target_block
```

### 5.2 关键注意事项

#### 不要拷贝整个 panel_base

```python
# 错误 ❌ — 会拷贝整个 DataFrame（即使只有几列也会触发不必要的内存分配）
df = panel_base.copy()
df = df.reset_index()

# 正确 ✅ — 直接从 index 取 date/sid，从列取数据
idx = panel_base.index
dates = idx.get_level_values("date")
sids = idx.get_level_values("sid")
close = panel_base["market.close"].values
```

#### 使用交易日历做日期偏移（不要用自然日）

```python
# 错误 ❌ — 自然日偏移会包含周末和假日
future_date = current_date + pd.Timedelta(days=5)

# 正确 ✅ — 用交易日历找第 N 个交易日
trade_dates = self.trading_calendar["date"].sort_values().values
# date_to_idx 映射 → 偏移 N → 取 trade_dates[idx + N]
```

#### 标签值和有效性必须一致

```python
# 规则: label_valid=True 的行，label 不能是 NaN
# 规则: label_valid=False 的行，label 必须是 NaN，reason 不能为空

label[~valid] = np.nan  # 一定要在最后设置
```

#### 所有日期偏移都要处理边界

对于任何使用前向窗口的 recipe，数据集末尾的若干行一定会因为"窗口不够"而无效。
这不是 bug，是正常行为。用 `label_reason = "insufficient_forward_window"` 标记即可。

### 5.3 第 2 步：注册 Target

编辑 `engine/targets/registry.py`，在 `_register_builtins()` 中添加：

```python
from engine.targets.recipes.my_new_recipe import MyNewRecipe

register_target(
    "my_target_name",           # 唯一标识，用于配置文件和输出目录
    MyNewRecipe,                # Recipe 类
    TargetSpec(
        name="my_target_name",
        task_type="regression",  # 或 "classification"
        family="my_family",      # 可选，用于分组
        horizon=10,              # 可选，前向窗口天数
        dtype="float32",         # 输出类型
        params={                 # 默认参数，可被 config 覆盖
            "my_param": 10,
            "price_col": "market.close",
        },
    ),
)
```

**TargetSpec 各字段说明**：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `name` | str | 是 | 唯一名称，和 `register_target` 第一个参数一致 |
| `task_type` | str | 是 | `"regression"` 或 `"classification"`。影响质量报告的展示方式 |
| `family` | str | 否 | 族名，如 `"return"`, `"momentum"`。用于分组展示 |
| `horizon` | int | 否 | 前向窗口天数。仅用于展示 |
| `dtype` | str | 否 | 输出数据类型。仅用于展示 |
| `params` | dict | 否 | 默认参数。会被 `configs/default.yaml` 中的 `params` 覆盖 |

### 5.4 第 3 步：添加配置

编辑 `configs/default.yaml`：

```yaml
targets:
  - name: "return_c0c1"        # 今日收盘 → 明日收盘
  - name: "momentum_cls"       # 5 类动量分类
  # 新增 ↓
  - name: "my_target_name"
    params:
      my_param: 10
```

然后运行：
```bash
python scripts/build_target.py --config configs/default.yaml --target my_target_name
```

---

## 6. panel_labeled 的使用方法

### 6.1 读取

```python
import pandas as pd

# 读取完整 panel_labeled
panel = pd.read_parquet('data/processed/panel_labeled/return_c0c1+momentum_cls/panel_labeled.parquet')

print(panel.index.names)    # ['date', 'sid']
print(panel.shape)           # (7,704,699, ~502)

# 只读需要的列（推荐，节省内存）
panel = pd.read_parquet(
    'data/processed/panel_labeled/return_c0c1+momentum_cls/panel_labeled.parquet',
    columns=['market.close', 'label.return_c0c1', 'label_valid.return_c0c1',
             'status.sample_usable_for_feature', 'status.is_suspended']
)
```

### 6.2 构建训练 mask

```python
# 典型的训练 mask 构建方式
train_mask = (
    panel['status.sample_usable_for_feature']      # 因子可用
    & panel['label_valid.return_c0c1']               # 标签有效
    & ~panel['status.is_suspended']                # 非停牌（可选）
)

# 提取训练数据
feature_cols = [c for c in panel.columns if c.startswith('feature.')]
X_train = panel.loc[train_mask, feature_cols]
y_train = panel.loc[train_mask, 'label.return_c0c1']

print(f"训练样本: {len(X_train):,} / {len(panel):,} ({len(X_train)/len(panel)*100:.1f}%)")
```

### 6.3 多标签联合 mask

```python
# 如果模型需要 return_c0c1 和 momentum_cls 都有效
joint_mask = (
    panel['status.sample_usable_for_feature']
    & panel['label_valid.return_c0c1']
    & panel['label_valid.momentum_cls']
)

# 多任务训练
X = panel.loc[joint_mask, feature_cols]
y_reg = panel.loc[joint_mask, 'label.return_c0c1']          # 回归目标
y_cls = panel.loc[joint_mask, 'label.momentum_cls'].astype(int)  # 分类目标
```

### 6.4 时间分割

```python
# 根据 config 中的 valid_start / test_start 做时间分割
dates = panel.index.get_level_values('date')

train_mask = joint_mask & (dates < '2022-01-01')
valid_mask = joint_mask & (dates >= '2022-01-01') & (dates < '2024-01-01')
test_mask  = joint_mask & (dates >= '2024-01-01')

X_train = panel.loc[train_mask, feature_cols]
X_valid = panel.loc[valid_mask, feature_cols]
X_test  = panel.loc[test_mask, feature_cols]
```

---

## 7. 架构参考

### 7.1 代码文件对照表

| 文件 | 作用 |
|------|------|
| `engine/targets/base.py` | `BaseTargetRecipe` 抽象基类 |
| `engine/targets/specs.py` | `TargetSpec` 数据类 |
| `engine/targets/registry.py` | 注册表：`register_target`, `get_recipe`, `get_spec`, `list_targets` |
| `engine/targets/engine.py` | 构建引擎：`build_target_block`, `build_target_block_from_path` |
| `engine/targets/report_generator.py` | 质量报告 + 列名参考生成器 |
| `engine/targets/recipes/return_nd.py` | 灵活收益率 recipe（支持 o/c 记法） |
| `engine/targets/recipes/momentum_cls.py` | 动量分类 recipe |
| `engine/panel/assemble_labeled_panel.py` | panel_labeled 合并逻辑 |
| `scripts/build_target.py` | CLI：构建 target_block |
| `scripts/build_panel_labeled.py` | CLI：合并 panel_labeled |

### 7.2 数据流图

```
                    ┌─────────────────────┐
                    │  configs/default.yaml │
                    │  targets:             │
                    │    - return_c0c1        │
                    │    - momentum_cls     │
                    └─────────┬────────────┘
                              │
                    ┌─────────▼────────────┐
                    │  build_target.py      │
                    │                       │
                    │  对每个 target:        │
                    │    1. 创建 recipe      │
                    │    2. 读取 required_columns │
                    │    3. 从 panel_base 只加载所需列 │
                    │    4. recipe.compute() │
                    │    5. 保存 target_block │
                    │    6. 生成质量报告     │
                    └─────────┬────────────┘
                              │
            ┌─────────────────┼─────────────────┐
            ▼                                   ▼
┌───────────────────┐               ┌───────────────────┐
│ targets/return_c0c1/ │               │ targets/momentum_cls/ │
│  target_block.parquet │           │  target_block.parquet │
│  TARGET_REPORT.md │               │  TARGET_REPORT.md │
│  COLUMN_REFERENCE.md │           │  COLUMN_REFERENCE.md │
└─────────┬─────────┘               └─────────┬─────────┘
          │                                   │
          └────────────┬──────────────────────┘
                       ▼
            ┌─────────────────────┐
            │ build_panel_labeled.py │
            │                       │
            │ panel_base             │
            │   + target_block(s)   │
            │   = panel_labeled     │
            └─────────┬────────────┘
                      ▼
            ┌─────────────────────┐
            │ panel_labeled/       │
            │  return_c0c1+momentum_cls/ │
            │    panel_labeled.parquet │
            └──────────────────────┘
```

### 7.3 参数传递流程

```
default.yaml 中的 params
        ↓ 被 build_target.py 解析
        ↓
与 TargetSpec.params 合并（yaml 优先覆盖 spec 默认值）
        ↓
传入 recipe.__init__(**params)
        ↓
recipe 内部使用 self.my_param 等属性
```

---

## 8. 必须处理的边界情况

构建任何标签时，以下情况都可能出现。你的 recipe 必须显式处理它们，
通过设置 `label_valid=False` 和 `label_reason="具体原因"` 来标记。

| 场景 | 典型原因 | label_reason 示例 |
|------|----------|-------------------|
| 标签窗口超出数据范围 | 距数据集末尾不足 N 个交易日 | `insufficient_forward_window` |
| 回看窗口超出数据范围 | 距数据集开头不足 N 个交易日 | `insufficient_history` |
| 停牌导致无价格 | t+N 日该股停牌，close = NaN | `missing_close_t_plus_n` |
| 当天无行情 | 当天停牌，close = NaN | `missing_close_t` |
| 窗口内有缺失价格 | 窗口内某天停牌 | `missing_close_in_window` |
| 当天数据完全缺失 | OHLCV 和因子都没有 | `no_source_data` |

**关键原则**：不删行、只标记。保持 target_block 和 panel_base 行数完全一致。

---

## 9. 常见问题

**Q：为什么 target_block 的行数和 panel_base 一样？**

A：因为 target_block 的 index 直接复用 panel_base 的 index。即使某些行的标签无效（`label_valid=False`），也保留该行。这保证了下游拼接 panel_labeled 时行对齐。

**Q：为什么只加载了 market.close 一列，内存那么小？**

A：engine 的 `build_target_block_from_path()` 会根据 recipe 的 `required_columns()` 声明，使用 PyArrow 只读取所需列。panel_base 有 496 列 ~14 GB，但只读 1 列 + index 只需 ~120 MB。

**Q：我想用复权价计算收益率怎么办？**

A：在 config 中把 `price_col` 改成 `"market.adj_factor"` 相关的列，或者在 recipe 中计算复权价。记得在 `required_columns()` 中声明所有需要的列。

**Q：我的 recipe 需要多个 panel_base 的列（比如 close + volume），怎么办？**

A：在 `required_columns()` 中声明所有需要的列即可：
```python
def required_columns(self) -> List[str]:
    return ["market.close", "market.volume"]
```
engine 会自动加载这两列。

**Q：怎么查看所有已注册的 target？**

A：
```python
from engine.targets.registry import list_targets
print(list_targets())  # ['return_c0c1', 'return_o1c2', 'return_5d', 'return_o1c5', 'return_20d', 'momentum_cls']
```

**Q：可以同时注册多个同类型但不同参数的 target 吗？**

A：可以。比如注册 `return_c0c1`（formula=c0c1）、`return_5d`（formula=c0c5）、`return_o1c5`（formula=o1c5），它们都用 `ReturnRecipe`，只是 formula 不同。每个 target 有独立的名称、spec 和输出目录。target 名称可以自由命名，不一定要包含公式名。
