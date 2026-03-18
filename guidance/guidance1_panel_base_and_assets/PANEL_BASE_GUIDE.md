# Panel Base 使用指南

> 自动生成于 2026-03-16 18:12:30
>
> 本文件由 `python scripts/build_panel_base.py` 在构建完成后自动输出。
> 它记录了 **输入数据格式、输出文件结构、每一列的含义，以及下游使用方法**。
> 如果你要为 panel_base 构建标签（target），请先阅读本文件。

---
## 1. 整体概览

panel_base 是整个量化框架的**统一底板**。它回答一个问题：

> 在每一个交易日，每一只处于上市状态的股票，它的行情、因子、状态分别是什么？

panel_base 的核心特征：

- **行空间**由 active_session_index 决定（不由 OHLCV 或因子表单独决定）
- **只表达已知事实**，不包含任何标签（label）列
- 停牌股票、缺数据的股票**都有行**，通过 status 列标记状态
- 下游的 target_block、model view、训练、回测都建立在这张表之上

### 本次构建摘要

| 指标 | 值 |
|---|---|
| 总行数 | 7,704,699 |
| 总列数 | 496 |
| 文件大小 | 14191.1 MB |
| market 列数 | 7 |
| feature 列数 | 473 （其中 alpha158: 158, 基本面: 315）|
| status 列数 | 12 |
| meta 列数 | 2 |
| index 列 | date, sid |

---
## 2. 输入数据

系统唯一的两个标准输入：

### 2.1 OHLCV 表（每日每股行情）

每个 parquet 文件对应一年，文件名格式 `year=YYYY.parquet`。

| 字段 | 类型 | 说明 |
|---|---|---|
| datetime / date | date | 交易日 |
| order_book_id / symbol | string | 股票代码（如 600519.XSHG），会被标准化为 600519.SH |
| open | float | 开盘价 |
| high | float | 最高价 |
| low | float | 最低价 |
| close | float | 收盘价 |
| volume | float | 成交量 |
| amount | float | 成交额（可选） |
| adj_factor | float | 复权因子（可选） |

**关键特点**：停牌股票在停牌日**不会出现在 OHLCV 表中**（整行缺失）。

### 2.2 因子表（每日每股因子）

同样是年度 parquet 文件。

| 字段 | 类型 | 说明 |
|---|---|---|
| datetime / date | date | 交易日 |
| order_book_id / instrument | string | 股票代码 |
| 其他所有数值列 | float | 因子值，进入系统后统一加 `feature.` 前缀 |

本数据集共 473 个因子，分为两大类：
- **alpha158 量价因子**（158 个）：如 KMID, MA5, ROC10, CORR20 等
- **基本面因子**（315 个）：如 fund\_\_pe\_ratio\_ttm, fund\_\_market\_cap 等

---
## 3. 构建流程

```
步骤 1: python scripts/build_assets.py --config configs/default.yaml
        ↓ 读取原始 OHLCV + 因子 → 生成 5 张标准化资产表

步骤 2: python scripts/build_panel_base.py --config configs/default.yaml
        ↓ 读取资产表 → 生成 active_session_index + panel_base
        ↓ 输出本指南 + 质量报告
```

---
## 4. 输出文件清单

### 4.1 资产层（`data/processed/assets/`）

| 文件 | 行数 | 大小 | 说明 |
|---|---|---|---|
| instrument_master.parquet | 5,351 | 0.1 MB | 股票基本信息 |
| trading_calendar.parquet | 1,691 | 0.0 MB | 交易日历 |
| daily_bars.parquet | — | — | 标准化行情（panel_base 生成后可删除） |
| factor_values.parquet | — | — | 标准化因子（panel_base 生成后可删除） |

### 4.2 面板层（`data/processed/panel/`）

| 文件 | 行数 | 大小 | 说明 |
|---|---|---|---|
| active_session_index.parquet | 7,704,699 | 10.9 MB | 行空间索引 |
| panel_base.parquet | 7,704,699 | 14191.1 MB | 统一底板 |
| PANEL_BASE_GUIDE.md | — | — | 本文件 |
| COLUMN_REFERENCE.md | — | — | 完整列名参考 |

---
## 5. 各输出文件详解

### 5.1 instrument_master（股票基本信息）

| 列名 | 类型 | 含义 |
|---|---|---|
| sid | string | 股票唯一标识，如 `600519.SH` |
| symbol | string | 展示用代码，当前版本等于 sid |
| list_date | datetime | 上市日期（实际是该股在 OHLCV 数据中**首次出现**的日期） |
| delist_date | datetime / NaT | 退市日期。如果最后交易日距数据集末尾 ≤90 天，视为仍在上市，填 NaT |

**注意**：list_date 是从 OHLCV 数据推导的近似值，不是真实 IPO 公告日期。
如果原始数据从 2005 年开始，那么 2005 年前就上市的股票的 list_date 会被记为 ~2005-01-04。

### 5.2 trading_calendar（交易日历）

| 列名 | 类型 | 含义 |
|---|---|---|
| date | datetime | 日期 |
| is_open | bool | 该日是否为交易日（当前表中只包含 is_open=True 的行） |

**后续阶段的关键用途**：
- 构建 return_5d 标签时，需要用 trading_calendar 找到**未来第 5 个交易日**的日期
- 回测时按交易日遍历
- 任何需要「往前/往后数 N 个交易日」的操作都依赖它

```python
# 示例：构建 trading_day → offset 的查找表
import pandas as pd
cal = pd.read_parquet('data/processed/assets/trading_calendar.parquet')
cal = cal.sort_values('date').reset_index(drop=True)
# 每个交易日向后移 N 天
cal['date_5d_later'] = cal['date'].shift(-5)
# 用这张表做 left join 就能拿到未来第 5 个交易日的日期
```

### 5.3 active_session_index（行空间索引）

| 列名 | 类型 | 含义 |
|---|---|---|
| date | datetime | 交易日 |
| sid | string | 股票标识 |

**生成逻辑**：trading_calendar 的所有交易日 × instrument_master 的所有股票 → 笛卡尔积 → 按上市窗口过滤。

保留条件：`date >= list_date AND (delist_date 为空 OR date <= delist_date)`

**为什么需要它**：如果直接以 OHLCV 为行空间，停牌日就会丢行。active_session_index 保证：
只要股票在那天处于上市状态，就一定有一行——即使 OHLCV 和因子表都没有数据。

### 5.4 panel_base（统一底板）⭐ 核心文件

这是整个框架最重要的输出。所有下游操作（打标签、训练、回测）都基于这张表。

**索引**：`[date, sid]`（MultiIndex）

**列总数**：496 列，分为 5 组：

#### 5.4.1 索引列（index）

| 列名 | 类型 | 含义 |
|---|---|---|
| date | datetime | 交易日 |
| sid | string | 股票标识，如 `000001.SZ` |

#### 5.4.2 元信息列（meta.*）

| 列名 | 类型 | 含义 |
|---|---|---|
| meta.symbol | string | 展示用证券代码 |
| meta.list_date | datetime | 上市日期（近似值） |

#### 5.4.3 行情列（market.*）

这些列来自 OHLCV 数据的 left join。如果该 (date, sid) 在 OHLCV 中不存在（例如停牌），这些列全部为 NaN。

| 列名 | 类型 | 含义 |
|---|---|---|
| market.open | float64 | 开盘价 |
| market.high | float64 | 最高价 |
| market.low | float64 | 最低价 |
| market.close | float64 | 收盘价 |
| market.volume | float64 | 成交量 |
| market.amount | float64 | 成交额 |
| market.adj_factor | float64 | 复权因子 |

**注意**：停牌日这些列为 NaN，不是 0。判断停牌应该用 `status.is_suspended`，不要用 `market.volume == 0`。

#### 5.4.4 因子列（feature.*）

共 473 列，全部为 float32 类型。这些列来自因子数据的 left join。

**分为两大类**：

**（A）alpha158 量价因子**（158 个）

基于价格和成交量计算的技术因子，按时间窗口分组：

| 因子族 | 窗口 | 示例 | 含义 |
|---|---|---|---|
| KMID/KLEN/KUP/KLOW/KSFT | — | KMID, KLEN | K 线形态特征 |
| ROC | 5,10,20,30,60 | ROC5, ROC60 | 变化率 (Rate of Change) |
| MA | 5,10,20,30,60 | MA5, MA60 | 移动平均 |
| STD | 5,10,20,30,60 | STD5, STD60 | 波动率（标准差） |
| BETA/RSQR/RESI | 5,10,20,30,60 | BETA5 | 市场回归系数 |
| MAX/MIN | 5,10,20,30,60 | MAX5, MIN60 | 窗口最高/最低 |
| QTLU/QTLD | 5,10,20,30,60 | QTLU5 | 上/下分位数 |
| RANK | 5,10,20,30,60 | RANK5 | 窗口内排名 |
| RSV | 5,10,20,30,60 | RSV5 | 相对强弱值 |
| IMAX/IMIN/IMXD | 5,10,20,30,60 | IMAX5 | 最高/最低所在位置 |
| CORR/CORD | 5,10,20,30,60 | CORR5 | 价量相关系数 |
| CNTP/CNTN/CNTD | 5,10,20,30,60 | CNTP5 | 上涨/下跌天数比 |
| SUMP/SUMN/SUMD | 5,10,20,30,60 | SUMP5 | 上涨/下跌幅度总和 |
| VMA/VSTD/WVMA | 5,10,20,30,60 | VMA5 | 成交量统计 |
| VSUMP/VSUMN/VSUMD | 5,10,20,30,60 | VSUMP5 | 成交量方向统计 |
| OPEN0/HIGH0/LOW0/VWAP0 | — | OPEN0 | 当日标准化价格 |

**（B）基本面因子**（315 个）

以 `feature.fund__` 开头，包含估值、盈利、杠杆、成长性等维度。后缀含义：

| 后缀 | 含义 |
|---|---|
| `_ttm` | 滚动 12 个月 (Trailing Twelve Months) |
| `_lyr` | 最近完整年报 (Last Year Report) |
| `_lf` | 最新财报（Latest Filing，可能是季报） |
| `_ly0/_ly1/_ly2` | 最近第 0/1/2 年的值 |
| `_ttm0/_ttm1/_ttm2` | 滚动 12 个月的第 0/1/2 期 |

**基本面因子可能为 NaN 的常见原因**：
- 新上市公司还没有完整年报
- 某些因子只适用于特定行业（如 fund\_\_market\_cap 对所有股票都有，但 fund\_\_dividend\_yield\_ttm 对从未分红的公司为 NaN）
- 这正是 `status.factor_missing_ratio > 0` 的主要原因

#### 5.4.5 状态列（status.*）⭐ 必读

这 12 列描述了每个 (date, sid) 样本的**数据到达情况**和**业务状态**。
下游构建标签时，必须根据这些列来决定哪些样本可以使用。

##### 布尔型状态列

| 列名 | 类型 | 含义 | 判定逻辑 |
|---|---|---|---|
| status.is_listed | bool | 是否处于上市状态 | 恒为 True（active_session_index 只包含上市窗口内的样本） |
| status.is_suspended | bool | 是否停牌 | `is_listed AND NOT has_market_record`（停牌日 OHLCV 无行） |
| status.has_market_record | bool | OHLCV 表中是否存在该行 | 只看行是否存在，不看字段是否完整 |
| status.has_factor_record | bool | 因子表中是否存在该行 | 只看行是否存在，不看因子是否完整 |
| status.bar_missing | bool | 行情数据是否不可用 | `NOT has_market_record OR market.close 为 NaN` |
| status.factor_row_missing | bool | 因子整行是否缺失 | 等价于 `NOT has_factor_record` |
| status.feature_all_missing | bool | 所有 feature 是否全为 NaN | `factor_missing_ratio >= 1.0` |
| status.sample_usable_for_feature | bool | 可否作为特征输入 | `has_factor_record AND NOT feature_all_missing` |

##### 数值型状态列

| 列名 | 类型 | 含义 | 取值范围 |
|---|---|---|---|
| status.factor_missing_ratio | float32 | feature 列的缺失比例 | 0.0（无缺失）~ 1.0（全缺失） |

##### 分类型状态列

| 列名 | 类型 | 可能的值 |
|---|---|---|
| status.market_state | category | OK, SUSPENDED, PARTIAL_MISSING, MISSING |
| status.factor_state | category | OK, ROW_MISSING, PARTIAL_MISSING, ALL_MISSING |
| status.sample_state | category | NORMAL, SUSPENDED, MARKET_ONLY, FACTOR_ONLY, NO_SOURCE_RECORD, PARTIAL_FACTOR_MISSING, INVALID_BASE_SAMPLE |

**sample_state 枚举值详解**：

| 值 | 含义 | has_market | has_factor |
|---|---|---|---|
| NORMAL | 行情+因子都完整 | True | True（且 fmr=0） |
| PARTIAL_FACTOR_MISSING | 行情+因子都有，但部分因子为 NaN | True | True（fmr>0） |
| SUSPENDED | 停牌（无 OHLCV 记录） | False | True 或 False |
| MARKET_ONLY | 只有行情，无因子 | True | False |
| FACTOR_ONLY | 只有因子，无行情 | False | True |
| NO_SOURCE_RECORD | 两个源都无记录 | False | False |
| INVALID_BASE_SAMPLE | 异常样本 | — | — |

---
## 6. 本次构建质量报告

### sample_state 分布

| 状态 | 行数 | 占比 |
|---|---|---|
| PARTIAL_FACTOR_MISSING | 7,682,978 | 99.72% |
| SUSPENDED | 21,527 | 0.28% |
| NO_SOURCE_RECORD | 194 | 0.00% |

### 布尔状态列统计

| 列名 | True 数量 | False 数量 | True 占比 |
|---|---|---|---|
| status.bar_missing | 21,721 | 7,682,978 | 0.2819% |
| status.factor_row_missing | 194 | 7,704,505 | 0.0025% |
| status.feature_all_missing | 194 | 7,704,505 | 0.0025% |
| status.has_factor_record | 7,704,505 | 194 | 99.9975% |
| status.has_market_record | 7,682,978 | 21,721 | 99.7181% |
| status.is_listed | 7,704,699 | 0 | 100.0000% |
| status.is_suspended | 21,721 | 7,682,978 | 0.2819% |
| status.sample_usable_for_feature | 7,704,505 | 194 | 99.9975% |

### 质量检查：全部通过 ✓

---
## 7. 下游使用指南 ⭐⭐⭐

### 7.1 读取 panel_base

```python
import pandas as pd

# 读取完整 panel_base（注意：文件较大，建议指定 columns 参数）
panel = pd.read_parquet('data/processed/panel/panel_base.parquet')
print(panel.index.names)   # ['date', 'sid']
print(panel.shape)          # (7,704,699, 494)

# 只读取行情和状态列（省内存）
panel_lite = pd.read_parquet(
    'data/processed/panel/panel_base.parquet',
    columns=['market.close', 'market.volume', 'status.is_suspended',
             'status.has_market_record', 'status.sample_usable_for_feature']
)
```

### 7.2 构建标签的通用模式

所有标签都遵循相同的输出 schema：

```python
# target_block 必须输出这 3 列：
#   label.<target_name>        — 标签值（float）
#   label_valid.<target_name>   — 该标签是否有效（bool）
#   label_reason.<target_name>  — 无效原因（string）
```

### 7.3 示例：构建 return_5d（5 日收益率）标签

```python
import pandas as pd
import numpy as np

# 1. 读取所需列
panel = pd.read_parquet(
    'data/processed/panel/panel_base.parquet',
    columns=['market.close', 'status.is_suspended', 'status.has_market_record']
)

# 2. 用交易日历构建 date → 5日后的date 映射
cal = pd.read_parquet('data/processed/assets/trading_calendar.parquet')
cal = cal.sort_values('date').reset_index(drop=True)
date_map = pd.DataFrame({
    'date': cal['date'],
    'date_5d': cal['date'].shift(-5),  # 第5个交易日
})

# 3. 计算 return_5d
close = panel['market.close'].unstack('sid')        # date × sid 宽表
close_5d = close.reindex(date_map.set_index('date')['date_5d'])
# ... 或者更简单的方式：
close_by_sid = panel.reset_index()
close_by_sid = close_by_sid.merge(date_map, on='date', how='left')
# ... join 5日后的 close 价格，计算收益率
```

### 7.4 必须处理的边界情况

构建任何标签时，以下情况都必须显式处理：

#### 情况 1：标签窗口超出数据范围

例如构建 return_60d，但当前日期距离数据集末尾只有 30 个交易日。

```python
# 解决方案：用 trading_calendar 检查未来是否有足够的交易日
cal_dates = cal['date'].values
last_valid_date = cal_dates[-60]  # 倒数第60个交易日
# date > last_valid_date 的样本 → label_valid = False, label_reason = 'INSUFFICIENT_FORWARD_WINDOW'
```

#### 情况 2：标签窗口内股票尚未上市

例如构建 return_60d，但该股票 30 天前才上市，没有足够的历史数据。

```python
# 解决方案：用 meta.list_date 检查
panel = pd.read_parquet('data/processed/panel/panel_base.parquet',
    columns=['market.close', 'meta.list_date'])
days_since_ipo = (panel.index.get_level_values('date') - panel['meta.list_date']).dt.days
too_young = days_since_ipo < 60
# too_young 的样本 → label_valid = False, label_reason = 'STOCK_TOO_YOUNG'
```

#### 情况 3：标签窗口内股票退市

例如构建 return_5d，但该股票 3 天后退市。

```python
# 解决方案：检查 5 日后该 sid 是否还在 active_session_index 中
# 或者：用 instrument_master 的 delist_date 做判断
im = pd.read_parquet('data/processed/assets/instrument_master.parquet')
# 如果 date_5d > delist_date → label_valid = False, label_reason = 'DELISTED_IN_WINDOW'
```

#### 情况 4：停牌导致窗口内无价格

例如构建 return_5d，但该股票在第 3-5 天停牌，第 5 天没有收盘价。

```python
# 解决方案 A：标记为无效
# 如果 5 日后的 market.close 为 NaN → label_valid = False

# 解决方案 B：用复牌后第一个有效价格替代（更宽松）
# 从第 5 天开始向后搜索第一个非 NaN 的 close
# 这种方式需要在 label_reason 中注明 'FORWARD_FILLED_DUE_TO_SUSPENSION'
```

#### 情况 5：当天停牌，无法作为特征输入

```python
# 直接用 status 列判断
panel['status.is_suspended']          # True = 停牌
panel['status.sample_usable_for_feature']  # 因子是否可用

# 注意：停牌日通常仍有因子数据（因子表的行在停牌日可能存在）
# 所以 is_suspended=True 且 sample_usable_for_feature=True 是完全可能的
```

#### 情况 6：源数据完全缺失

```python
# 某些 (date, sid) 在 OHLCV 和因子表中都不存在
# 此时 sample_state = 'NO_SOURCE_RECORD'
# 这些样本的 market.* 和 feature.* 全部为 NaN
# 它们不应该参与任何标签计算或训练
no_data = panel['status.sample_state'] == 'NO_SOURCE_RECORD'
```

### 7.5 推荐的标签有效性判定模板

```python
def compute_label_validity(panel, label_values, target_name, forward_days=5):
    """
    通用标签有效性判定。返回 label_valid 和 label_reason。
    """
    n = len(panel)
    label_valid = pd.Series(True, index=panel.index)
    label_reason = pd.Series('', index=panel.index, dtype='object')

    # 规则 1：标签值本身为 NaN
    is_nan = label_values.isna()
    label_valid[is_nan] = False
    label_reason[is_nan] = 'LABEL_IS_NAN'

    # 规则 2：停牌（取决于你的策略 —— 有些模型允许停牌样本）
    # is_susp = panel['status.is_suspended']
    # label_valid[is_susp] = False
    # label_reason[is_susp] = 'SUSPENDED'

    # 规则 3：没有足够的前向窗口
    # （由调用方在计算 label_values 之前处理）

    return label_valid, label_reason
```

### 7.6 训练阶段的 mask 组合

panel_base 中只有一个可用性标志：`status.sample_usable_for_feature`。
训练可用性需要在训练阶段动态生成：

```python
# 典型 train_mask 构建方式
train_mask = (
    panel['status.sample_usable_for_feature']   # 因子可用
    & label_valid['return_5d']                  # 标签有效
    & ~panel['status.is_suspended']             # 非停牌（可选）
)

X_train = panel.loc[train_mask, feature_cols]
y_train = labels.loc[train_mask, 'label.return_5d']
```

---
## 8. 常见问题

**Q：为什么 sample_state 几乎全是 PARTIAL_FACTOR_MISSING？**

A：因为当前因子表有 473 个因子，包括大量基本面因子。
某些基本面因子对特定股票天然为 NaN（如未分红公司的 dividend_yield）。
只要有 1 个因子缺失，就会被标记为 PARTIAL_FACTOR_MISSING。这是正常的。
真正影响可用性的是 `status.sample_usable_for_feature`（只要不是全缺失就可用）。

**Q：panel_base 生成后，daily_bars 和 factor_values 还需要保留吗？**

A：不需要。它们的数据已经完全被 panel_base 吸收（left join 进去了）。
可以删除释放磁盘空间。但 `trading_calendar` 和 `instrument_master` 要保留，后续阶段需要用。

**Q：如何只读取特定年份或特定股票的数据？**

```python
# 方法 1：用 pyarrow 过滤（不会加载全量数据到内存）
import pyarrow.parquet as pq
table = pq.read_table(
    'data/processed/panel/panel_base.parquet',
    filters=[('date', '>=', '2023-01-01'), ('date', '<', '2024-01-01')]
)
panel_2023 = table.to_pandas()

# 方法 2：先读再过滤（文件全部加载）
panel = pd.read_parquet('data/processed/panel/panel_base.parquet')
panel_2023 = panel.loc['2023']
stock = panel.loc[(slice(None), '600519.SH'), :]
```

**Q：panel_base 的行顺序是什么？**

A：按 `[date, sid]` 排序。同一天的所有股票按 sid 字母序排列。
