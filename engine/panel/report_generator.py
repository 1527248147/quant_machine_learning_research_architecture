"""
Auto-generate detailed usage guide and column reference after panel_base is built.

Called at the end of scripts/build_panel_base.py. Produces:
  - data/processed/panel/PANEL_BASE_GUIDE.md
  - data/processed/panel/COLUMN_REFERENCE.md
"""
from __future__ import annotations
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


# ======================================================================
# dtype display helper
# ======================================================================
_DTYPE_DISPLAY = {
    "double": "float64",
    "float": "float32",
    "bool": "bool",
    "string": "string",
    "timestamp[ns]": "datetime",
}


def _arrow_type_str(t) -> str:
    s = str(t)
    if s.startswith("dictionary"):
        return "category(string)"
    return _DTYPE_DISPLAY.get(s, s)


# ======================================================================
# Read parquet metadata (no data loaded)
# ======================================================================
def _parquet_info(path: Path) -> dict:
    """Return schema, row count, file size from parquet metadata only."""
    if not path.exists():
        return {"exists": False}
    pf = pq.ParquetFile(str(path))
    schema = pf.schema_arrow
    cols = []
    for i in range(len(schema)):
        f = schema.field(i)
        cols.append({"name": f.name, "type": _arrow_type_str(f.type)})
    return {
        "exists": True,
        "rows": pf.metadata.num_rows,
        "size_mb": os.path.getsize(path) / 1024 / 1024,
        "columns": cols,
    }


# ======================================================================
# Main generation functions
# ======================================================================
def generate_guide(
    panel_dir: Path,
    assets_dir: Path,
    stats_dict: dict,
    config_summary: Optional[dict] = None,
) -> str:
    """Generate PANEL_BASE_GUIDE.md content and write to panel_dir."""

    # Gather metadata
    panel_info = _parquet_info(panel_dir / "panel_base.parquet")
    asi_info = _parquet_info(panel_dir / "active_session_index.parquet")
    im_info = _parquet_info(assets_dir / "instrument_master.parquet")
    tc_info = _parquet_info(assets_dir / "trading_calendar.parquet")

    # Classify panel_base columns
    market_cols = [c for c in panel_info.get("columns", []) if c["name"].startswith("market.")]
    feature_cols = [c for c in panel_info.get("columns", []) if c["name"].startswith("feature.")]
    status_cols = [c for c in panel_info.get("columns", []) if c["name"].startswith("status.")]
    meta_cols = [c for c in panel_info.get("columns", []) if c["name"].startswith("meta.")]
    index_cols = [c for c in panel_info.get("columns", []) if c["name"] in ("date", "sid")]

    # Feature sub-groups
    alpha158_features = [c for c in feature_cols if not c["name"].startswith("feature.fund__")]
    fund_features = [c for c in feature_cols if c["name"].startswith("feature.fund__")]

    total_rows = stats_dict.get("total_rows", panel_info.get("rows", 0))

    md = []
    md.append("# Panel Base 使用指南")
    md.append("")
    md.append(f"> 自动生成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append(">")
    md.append("> 本文件由 `python scripts/build_panel_base.py` 在构建完成后自动输出。")
    md.append("> 它记录了 **输入数据格式、输出文件结构、每一列的含义，以及下游使用方法**。")
    md.append("> 如果你要为 panel_base 构建标签（target），请先阅读本文件。")
    md.append("")

    # ========== 1. Overview ==========
    md.append("---")
    md.append("## 1. 整体概览")
    md.append("")
    md.append("panel_base 是整个量化框架的**统一底板**。它回答一个问题：")
    md.append("")
    md.append("> 在每一个交易日，每一只处于上市状态的股票，它的行情、因子、状态分别是什么？")
    md.append("")
    md.append("panel_base 的核心特征：")
    md.append("")
    md.append("- **行空间**由 active_session_index 决定（不由 OHLCV 或因子表单独决定）")
    md.append("- **只表达已知事实**，不包含任何标签（label）列")
    md.append("- 停牌股票、缺数据的股票**都有行**，通过 status 列标记状态")
    md.append("- 下游的 target_block、model view、训练、回测都建立在这张表之上")
    md.append("")
    md.append("### 本次构建摘要")
    md.append("")
    md.append(f"| 指标 | 值 |")
    md.append(f"|---|---|")
    md.append(f"| 总行数 | {total_rows:,} |")
    if panel_info.get("exists"):
        md.append(f"| 总列数 | {len(panel_info['columns'])} |")
        md.append(f"| 文件大小 | {panel_info['size_mb']:.1f} MB |")
    md.append(f"| market 列数 | {len(market_cols)} |")
    md.append(f"| feature 列数 | {len(feature_cols)} （其中 alpha158: {len(alpha158_features)}, 基本面: {len(fund_features)}）|")
    md.append(f"| status 列数 | {len(status_cols)} |")
    md.append(f"| meta 列数 | {len(meta_cols)} |")
    md.append(f"| index 列 | date, sid |")
    md.append("")

    # ========== 2. Input ==========
    md.append("---")
    md.append("## 2. 输入数据")
    md.append("")
    md.append("系统唯一的两个标准输入：")
    md.append("")
    md.append("### 2.1 OHLCV 表（每日每股行情）")
    md.append("")
    md.append("每个 parquet 文件对应一年，文件名格式 `year=YYYY.parquet`。")
    md.append("")
    md.append("| 字段 | 类型 | 说明 |")
    md.append("|---|---|---|")
    md.append("| datetime / date | date | 交易日 |")
    md.append("| order_book_id / symbol | string | 股票代码（如 600519.XSHG），会被标准化为 600519.SH |")
    md.append("| open | float | 开盘价 |")
    md.append("| high | float | 最高价 |")
    md.append("| low | float | 最低价 |")
    md.append("| close | float | 收盘价 |")
    md.append("| volume | float | 成交量 |")
    md.append("| amount | float | 成交额（可选） |")
    md.append("| adj_factor | float | 复权因子（可选） |")
    md.append("")
    md.append("**关键特点**：停牌股票在停牌日**不会出现在 OHLCV 表中**（整行缺失）。")
    md.append("")
    md.append("### 2.2 因子表（每日每股因子）")
    md.append("")
    md.append("同样是年度 parquet 文件。")
    md.append("")
    md.append("| 字段 | 类型 | 说明 |")
    md.append("|---|---|---|")
    md.append("| datetime / date | date | 交易日 |")
    md.append("| order_book_id / instrument | string | 股票代码 |")
    md.append("| 其他所有数值列 | float | 因子值，进入系统后统一加 `feature.` 前缀 |")
    md.append("")
    md.append(f"本数据集共 {len(feature_cols)} 个因子，分为两大类：")
    md.append(f"- **alpha158 量价因子**（{len(alpha158_features)} 个）：如 KMID, MA5, ROC10, CORR20 等")
    md.append(f"- **基本面因子**（{len(fund_features)} 个）：如 fund\\_\\_pe\\_ratio\\_ttm, fund\\_\\_market\\_cap 等")
    md.append("")

    # ========== 3. Pipeline ==========
    md.append("---")
    md.append("## 3. 构建流程")
    md.append("")
    md.append("```")
    md.append("步骤 1: python scripts/build_assets.py --config configs/default.yaml")
    md.append("        ↓ 读取原始 OHLCV + 因子 → 生成 5 张标准化资产表")
    md.append("")
    md.append("步骤 2: python scripts/build_panel_base.py --config configs/default.yaml")
    md.append("        ↓ 读取资产表 → 生成 active_session_index + panel_base")
    md.append("        ↓ 输出本指南 + 质量报告")
    md.append("```")
    md.append("")

    # ========== 4. Output Files ==========
    md.append("---")
    md.append("## 4. 输出文件清单")
    md.append("")
    md.append("### 4.1 资产层（`data/processed/assets/`）")
    md.append("")
    md.append("| 文件 | 行数 | 大小 | 说明 |")
    md.append("|---|---|---|---|")
    if im_info.get("exists"):
        md.append(f"| instrument_master.parquet | {im_info['rows']:,} | {im_info['size_mb']:.1f} MB | 股票基本信息 |")
    if tc_info.get("exists"):
        md.append(f"| trading_calendar.parquet | {tc_info['rows']:,} | {tc_info['size_mb']:.1f} MB | 交易日历 |")
    md.append("| daily_bars.parquet | — | — | 标准化行情（panel_base 生成后可删除） |")
    md.append("| factor_values.parquet | — | — | 标准化因子（panel_base 生成后可删除） |")
    md.append("")
    md.append("### 4.2 面板层（`data/processed/panel/`）")
    md.append("")
    md.append("| 文件 | 行数 | 大小 | 说明 |")
    md.append("|---|---|---|---|")
    if asi_info.get("exists"):
        md.append(f"| active_session_index.parquet | {asi_info['rows']:,} | {asi_info['size_mb']:.1f} MB | 行空间索引 |")
    if panel_info.get("exists"):
        md.append(f"| panel_base.parquet | {panel_info['rows']:,} | {panel_info['size_mb']:.1f} MB | 统一底板 |")
    md.append("| PANEL_BASE_GUIDE.md | — | — | 本文件 |")
    md.append("| COLUMN_REFERENCE.md | — | — | 完整列名参考 |")
    md.append("")

    # ========== 5. Detail: instrument_master ==========
    md.append("---")
    md.append("## 5. 各输出文件详解")
    md.append("")
    md.append("### 5.1 instrument_master（股票基本信息）")
    md.append("")
    md.append("| 列名 | 类型 | 含义 |")
    md.append("|---|---|---|")
    md.append("| sid | string | 股票唯一标识，如 `600519.SH` |")
    md.append("| symbol | string | 展示用代码，当前版本等于 sid |")
    md.append("| list_date | datetime | 上市日期（实际是该股在 OHLCV 数据中**首次出现**的日期） |")
    md.append("| delist_date | datetime / NaT | 退市日期。如果最后交易日距数据集末尾 ≤90 天，视为仍在上市，填 NaT |")
    md.append("")
    md.append("**注意**：list_date 是从 OHLCV 数据推导的近似值，不是真实 IPO 公告日期。")
    md.append("如果原始数据从 2005 年开始，那么 2005 年前就上市的股票的 list_date 会被记为 ~2005-01-04。")
    md.append("")

    # ========== 5.2 trading_calendar ==========
    md.append("### 5.2 trading_calendar（交易日历）")
    md.append("")
    md.append("| 列名 | 类型 | 含义 |")
    md.append("|---|---|---|")
    md.append("| date | datetime | 日期 |")
    md.append("| is_open | bool | 该日是否为交易日（当前表中只包含 is_open=True 的行） |")
    md.append("")
    md.append("**后续阶段的关键用途**：")
    md.append("- 构建 return_5d 标签时，需要用 trading_calendar 找到**未来第 5 个交易日**的日期")
    md.append("- 回测时按交易日遍历")
    md.append('- 任何需要「往前/往后数 N 个交易日」的操作都依赖它')
    md.append("")
    md.append("```python")
    md.append("# 示例：构建 trading_day → offset 的查找表")
    md.append("import pandas as pd")
    md.append("cal = pd.read_parquet('data/processed/assets/trading_calendar.parquet')")
    md.append("cal = cal.sort_values('date').reset_index(drop=True)")
    md.append("# 每个交易日向后移 N 天")
    md.append("cal['date_5d_later'] = cal['date'].shift(-5)")
    md.append("# 用这张表做 left join 就能拿到未来第 5 个交易日的日期")
    md.append("```")
    md.append("")

    # ========== 5.3 active_session_index ==========
    md.append("### 5.3 active_session_index（行空间索引）")
    md.append("")
    md.append("| 列名 | 类型 | 含义 |")
    md.append("|---|---|---|")
    md.append("| date | datetime | 交易日 |")
    md.append("| sid | string | 股票标识 |")
    md.append("")
    md.append("**生成逻辑**：trading_calendar 的所有交易日 × instrument_master 的所有股票 → 笛卡尔积 → 按上市窗口过滤。")
    md.append("")
    md.append("保留条件：`date >= list_date AND (delist_date 为空 OR date <= delist_date)`")
    md.append("")
    md.append("**为什么需要它**：如果直接以 OHLCV 为行空间，停牌日就会丢行。active_session_index 保证：")
    md.append("只要股票在那天处于上市状态，就一定有一行——即使 OHLCV 和因子表都没有数据。")
    md.append("")

    # ========== 5.4 panel_base (CORE) ==========
    md.append("### 5.4 panel_base（统一底板）⭐ 核心文件")
    md.append("")
    md.append("这是整个框架最重要的输出。所有下游操作（打标签、训练、回测）都基于这张表。")
    md.append("")
    md.append("**索引**：`[date, sid]`（MultiIndex）")
    md.append("")
    md.append(f"**列总数**：{len(panel_info.get('columns', []))} 列，分为 5 组：")
    md.append("")

    # --- 5.4.1 index ---
    md.append("#### 5.4.1 索引列（index）")
    md.append("")
    md.append("| 列名 | 类型 | 含义 |")
    md.append("|---|---|---|")
    md.append("| date | datetime | 交易日 |")
    md.append("| sid | string | 股票标识，如 `000001.SZ` |")
    md.append("")

    # --- 5.4.2 meta ---
    md.append("#### 5.4.2 元信息列（meta.*）")
    md.append("")
    md.append("| 列名 | 类型 | 含义 |")
    md.append("|---|---|---|")
    md.append("| meta.symbol | string | 展示用证券代码 |")
    md.append("| meta.list_date | datetime | 上市日期（近似值） |")
    md.append("")

    # --- 5.4.3 market ---
    md.append("#### 5.4.3 行情列（market.*）")
    md.append("")
    md.append("这些列来自 OHLCV 数据的 left join。如果该 (date, sid) 在 OHLCV 中不存在（例如停牌），这些列全部为 NaN。")
    md.append("")
    md.append("| 列名 | 类型 | 含义 |")
    md.append("|---|---|---|")
    md.append("| market.open | float64 | 开盘价 |")
    md.append("| market.high | float64 | 最高价 |")
    md.append("| market.low | float64 | 最低价 |")
    md.append("| market.close | float64 | 收盘价 |")
    md.append("| market.volume | float64 | 成交量 |")
    md.append("| market.amount | float64 | 成交额 |")
    md.append("| market.adj_factor | float64 | 复权因子 |")
    md.append("")
    md.append("**注意**：停牌日这些列为 NaN，不是 0。判断停牌应该用 `status.is_suspended`，不要用 `market.volume == 0`。")
    md.append("")

    # --- 5.4.4 feature ---
    md.append("#### 5.4.4 因子列（feature.*）")
    md.append("")
    md.append(f"共 {len(feature_cols)} 列，全部为 float32 类型。这些列来自因子数据的 left join。")
    md.append("")
    md.append("**分为两大类**：")
    md.append("")
    md.append(f"**（A）alpha158 量价因子**（{len(alpha158_features)} 个）")
    md.append("")
    md.append("基于价格和成交量计算的技术因子，按时间窗口分组：")
    md.append("")
    md.append("| 因子族 | 窗口 | 示例 | 含义 |")
    md.append("|---|---|---|---|")
    md.append("| KMID/KLEN/KUP/KLOW/KSFT | — | KMID, KLEN | K 线形态特征 |")
    md.append("| ROC | 5,10,20,30,60 | ROC5, ROC60 | 变化率 (Rate of Change) |")
    md.append("| MA | 5,10,20,30,60 | MA5, MA60 | 移动平均 |")
    md.append("| STD | 5,10,20,30,60 | STD5, STD60 | 波动率（标准差） |")
    md.append("| BETA/RSQR/RESI | 5,10,20,30,60 | BETA5 | 市场回归系数 |")
    md.append("| MAX/MIN | 5,10,20,30,60 | MAX5, MIN60 | 窗口最高/最低 |")
    md.append("| QTLU/QTLD | 5,10,20,30,60 | QTLU5 | 上/下分位数 |")
    md.append("| RANK | 5,10,20,30,60 | RANK5 | 窗口内排名 |")
    md.append("| RSV | 5,10,20,30,60 | RSV5 | 相对强弱值 |")
    md.append("| IMAX/IMIN/IMXD | 5,10,20,30,60 | IMAX5 | 最高/最低所在位置 |")
    md.append("| CORR/CORD | 5,10,20,30,60 | CORR5 | 价量相关系数 |")
    md.append("| CNTP/CNTN/CNTD | 5,10,20,30,60 | CNTP5 | 上涨/下跌天数比 |")
    md.append("| SUMP/SUMN/SUMD | 5,10,20,30,60 | SUMP5 | 上涨/下跌幅度总和 |")
    md.append("| VMA/VSTD/WVMA | 5,10,20,30,60 | VMA5 | 成交量统计 |")
    md.append("| VSUMP/VSUMN/VSUMD | 5,10,20,30,60 | VSUMP5 | 成交量方向统计 |")
    md.append("| OPEN0/HIGH0/LOW0/VWAP0 | — | OPEN0 | 当日标准化价格 |")
    md.append("")
    md.append(f"**（B）基本面因子**（{len(fund_features)} 个）")
    md.append("")
    md.append("以 `feature.fund__` 开头，包含估值、盈利、杠杆、成长性等维度。后缀含义：")
    md.append("")
    md.append("| 后缀 | 含义 |")
    md.append("|---|---|")
    md.append("| `_ttm` | 滚动 12 个月 (Trailing Twelve Months) |")
    md.append("| `_lyr` | 最近完整年报 (Last Year Report) |")
    md.append("| `_lf` | 最新财报（Latest Filing，可能是季报） |")
    md.append("| `_ly0/_ly1/_ly2` | 最近第 0/1/2 年的值 |")
    md.append("| `_ttm0/_ttm1/_ttm2` | 滚动 12 个月的第 0/1/2 期 |")
    md.append("")
    md.append("**基本面因子可能为 NaN 的常见原因**：")
    md.append("- 新上市公司还没有完整年报")
    md.append("- 某些因子只适用于特定行业（如 fund\\_\\_market\\_cap 对所有股票都有，但 fund\\_\\_dividend\\_yield\\_ttm 对从未分红的公司为 NaN）")
    md.append("- 这正是 `status.factor_missing_ratio > 0` 的主要原因")
    md.append("")

    # --- 5.4.5 status ---
    md.append("#### 5.4.5 状态列（status.*）⭐ 必读")
    md.append("")
    md.append("这 12 列描述了每个 (date, sid) 样本的**数据到达情况**和**业务状态**。")
    md.append("下游构建标签时，必须根据这些列来决定哪些样本可以使用。")
    md.append("")

    # Bool status cols
    md.append("##### 布尔型状态列")
    md.append("")
    md.append("| 列名 | 类型 | 含义 | 判定逻辑 |")
    md.append("|---|---|---|---|")
    md.append("| status.is_listed | bool | 是否处于上市状态 | 恒为 True（active_session_index 只包含上市窗口内的样本） |")
    md.append("| status.is_suspended | bool | 是否停牌 | `is_listed AND NOT has_market_record`（停牌日 OHLCV 无行） |")
    md.append("| status.has_market_record | bool | OHLCV 表中是否存在该行 | 只看行是否存在，不看字段是否完整 |")
    md.append("| status.has_factor_record | bool | 因子表中是否存在该行 | 只看行是否存在，不看因子是否完整 |")
    md.append("| status.bar_missing | bool | 行情数据是否不可用 | `NOT has_market_record OR market.close 为 NaN` |")
    md.append("| status.factor_row_missing | bool | 因子整行是否缺失 | 等价于 `NOT has_factor_record` |")
    md.append("| status.feature_all_missing | bool | 所有 feature 是否全为 NaN | `factor_missing_ratio >= 1.0` |")
    md.append("| status.sample_usable_for_feature | bool | 可否作为特征输入 | `has_factor_record AND NOT feature_all_missing` |")
    md.append("")

    # Numeric status
    md.append("##### 数值型状态列")
    md.append("")
    md.append("| 列名 | 类型 | 含义 | 取值范围 |")
    md.append("|---|---|---|---|")
    md.append("| status.factor_missing_ratio | float32 | feature 列的缺失比例 | 0.0（无缺失）~ 1.0（全缺失） |")
    md.append("")

    # Categorical status
    md.append("##### 分类型状态列")
    md.append("")
    md.append("| 列名 | 类型 | 可能的值 |")
    md.append("|---|---|---|")
    md.append("| status.market_state | category | OK, SUSPENDED, PARTIAL_MISSING, MISSING |")
    md.append("| status.factor_state | category | OK, ROW_MISSING, PARTIAL_MISSING, ALL_MISSING |")
    md.append("| status.sample_state | category | NORMAL, SUSPENDED, MARKET_ONLY, FACTOR_ONLY, NO_SOURCE_RECORD, PARTIAL_FACTOR_MISSING, INVALID_BASE_SAMPLE |")
    md.append("")

    md.append("**sample_state 枚举值详解**：")
    md.append("")
    md.append("| 值 | 含义 | has_market | has_factor |")
    md.append("|---|---|---|---|")
    md.append("| NORMAL | 行情+因子都完整 | True | True（且 fmr=0） |")
    md.append("| PARTIAL_FACTOR_MISSING | 行情+因子都有，但部分因子为 NaN | True | True（fmr>0） |")
    md.append("| SUSPENDED | 停牌（无 OHLCV 记录） | False | True 或 False |")
    md.append("| MARKET_ONLY | 只有行情，无因子 | True | False |")
    md.append("| FACTOR_ONLY | 只有因子，无行情 | False | True |")
    md.append("| NO_SOURCE_RECORD | 两个源都无记录 | False | False |")
    md.append("| INVALID_BASE_SAMPLE | 异常样本 | — | — |")
    md.append("")

    # ========== 6. Quality Report ==========
    md.append("---")
    md.append("## 6. 本次构建质量报告")
    md.append("")
    # sample_state distribution
    state_counts = stats_dict.get("sample_state_counts", {})
    if state_counts:
        md.append("### sample_state 分布")
        md.append("")
        md.append("| 状态 | 行数 | 占比 |")
        md.append("|---|---|---|")
        for state in sorted(state_counts, key=lambda s: -state_counts[s]):
            cnt = state_counts[state]
            pct = cnt / total_rows * 100 if total_rows else 0
            md.append(f"| {state} | {cnt:,} | {pct:.2f}% |")
        md.append("")

    # Bool rates
    bool_counts = stats_dict.get("bool_true_counts", {})
    if bool_counts:
        md.append("### 布尔状态列统计")
        md.append("")
        md.append("| 列名 | True 数量 | False 数量 | True 占比 |")
        md.append("|---|---|---|---|")
        for col in sorted(bool_counts):
            true_cnt = bool_counts[col]
            false_cnt = total_rows - true_cnt
            rate = true_cnt / total_rows * 100 if total_rows else 0
            md.append(f"| {col} | {true_cnt:,} | {false_cnt:,} | {rate:.4f}% |")
        md.append("")

    # Sanity issues
    sanity_issues = stats_dict.get("sanity_issues", [])
    if sanity_issues:
        md.append("### 质量检查告警")
        md.append("")
        for issue in sanity_issues:
            md.append(f"- ⚠️ {issue}")
        md.append("")
    else:
        md.append("### 质量检查：全部通过 ✓")
        md.append("")

    # ========== 7. Usage Guide ==========
    md.append("---")
    md.append("## 7. 下游使用指南 ⭐⭐⭐")
    md.append("")
    md.append("### 7.1 读取 panel_base")
    md.append("")
    md.append("```python")
    md.append("import pandas as pd")
    md.append("")
    md.append("# 读取完整 panel_base（注意：文件较大，建议指定 columns 参数）")
    md.append("panel = pd.read_parquet('data/processed/panel/panel_base.parquet')")
    md.append("print(panel.index.names)   # ['date', 'sid']")
    md.append(f"print(panel.shape)          # ({total_rows:,}, {len(panel_info.get('columns', [])) - 2})")
    md.append("")
    md.append("# 只读取行情和状态列（省内存）")
    md.append("panel_lite = pd.read_parquet(")
    md.append("    'data/processed/panel/panel_base.parquet',")
    md.append("    columns=['market.close', 'market.volume', 'status.is_suspended',")
    md.append("             'status.has_market_record', 'status.sample_usable_for_feature']")
    md.append(")")
    md.append("```")
    md.append("")

    md.append("### 7.2 构建标签的通用模式")
    md.append("")
    md.append("所有标签都遵循相同的输出 schema：")
    md.append("")
    md.append("```python")
    md.append("# target_block 必须输出这 3 列：")
    md.append("#   label.<target_name>        — 标签值（float）")
    md.append("#   label_valid.<target_name>   — 该标签是否有效（bool）")
    md.append("#   label_reason.<target_name>  — 无效原因（string）")
    md.append("```")
    md.append("")

    md.append("### 7.3 示例：构建 return_5d（5 日收益率）标签")
    md.append("")
    md.append("```python")
    md.append("import pandas as pd")
    md.append("import numpy as np")
    md.append("")
    md.append("# 1. 读取所需列")
    md.append("panel = pd.read_parquet(")
    md.append("    'data/processed/panel/panel_base.parquet',")
    md.append("    columns=['market.close', 'status.is_suspended', 'status.has_market_record']")
    md.append(")")
    md.append("")
    md.append("# 2. 用交易日历构建 date → 5日后的date 映射")
    md.append("cal = pd.read_parquet('data/processed/assets/trading_calendar.parquet')")
    md.append("cal = cal.sort_values('date').reset_index(drop=True)")
    md.append("date_map = pd.DataFrame({")
    md.append("    'date': cal['date'],")
    md.append("    'date_5d': cal['date'].shift(-5),  # 第5个交易日")
    md.append("})")
    md.append("")
    md.append("# 3. 计算 return_5d")
    md.append("close = panel['market.close'].unstack('sid')        # date × sid 宽表")
    md.append("close_5d = close.reindex(date_map.set_index('date')['date_5d'])")
    md.append("# ... 或者更简单的方式：")
    md.append("close_by_sid = panel.reset_index()")
    md.append("close_by_sid = close_by_sid.merge(date_map, on='date', how='left')")
    md.append("# ... join 5日后的 close 价格，计算收益率")
    md.append("```")
    md.append("")

    md.append("### 7.4 必须处理的边界情况")
    md.append("")
    md.append("构建任何标签时，以下情况都必须显式处理：")
    md.append("")

    md.append("#### 情况 1：标签窗口超出数据范围")
    md.append("")
    md.append("例如构建 return_60d，但当前日期距离数据集末尾只有 30 个交易日。")
    md.append("")
    md.append("```python")
    md.append("# 解决方案：用 trading_calendar 检查未来是否有足够的交易日")
    md.append("cal_dates = cal['date'].values")
    md.append("last_valid_date = cal_dates[-60]  # 倒数第60个交易日")
    md.append("# date > last_valid_date 的样本 → label_valid = False, label_reason = 'INSUFFICIENT_FORWARD_WINDOW'")
    md.append("```")
    md.append("")

    md.append("#### 情况 2：标签窗口内股票尚未上市")
    md.append("")
    md.append("例如构建 return_60d，但该股票 30 天前才上市，没有足够的历史数据。")
    md.append("")
    md.append("```python")
    md.append("# 解决方案：用 meta.list_date 检查")
    md.append("panel = pd.read_parquet('data/processed/panel/panel_base.parquet',")
    md.append("    columns=['market.close', 'meta.list_date'])")
    md.append("days_since_ipo = (panel.index.get_level_values('date') - panel['meta.list_date']).dt.days")
    md.append("too_young = days_since_ipo < 60")
    md.append("# too_young 的样本 → label_valid = False, label_reason = 'STOCK_TOO_YOUNG'")
    md.append("```")
    md.append("")

    md.append("#### 情况 3：标签窗口内股票退市")
    md.append("")
    md.append("例如构建 return_5d，但该股票 3 天后退市。")
    md.append("")
    md.append("```python")
    md.append("# 解决方案：检查 5 日后该 sid 是否还在 active_session_index 中")
    md.append("# 或者：用 instrument_master 的 delist_date 做判断")
    md.append("im = pd.read_parquet('data/processed/assets/instrument_master.parquet')")
    md.append("# 如果 date_5d > delist_date → label_valid = False, label_reason = 'DELISTED_IN_WINDOW'")
    md.append("```")
    md.append("")

    md.append("#### 情况 4：停牌导致窗口内无价格")
    md.append("")
    md.append("例如构建 return_5d，但该股票在第 3-5 天停牌，第 5 天没有收盘价。")
    md.append("")
    md.append("```python")
    md.append("# 解决方案 A：标记为无效")
    md.append("# 如果 5 日后的 market.close 为 NaN → label_valid = False")
    md.append("")
    md.append("# 解决方案 B：用复牌后第一个有效价格替代（更宽松）")
    md.append("# 从第 5 天开始向后搜索第一个非 NaN 的 close")
    md.append("# 这种方式需要在 label_reason 中注明 'FORWARD_FILLED_DUE_TO_SUSPENSION'")
    md.append("```")
    md.append("")

    md.append("#### 情况 5：当天停牌，无法作为特征输入")
    md.append("")
    md.append("```python")
    md.append("# 直接用 status 列判断")
    md.append("panel['status.is_suspended']          # True = 停牌")
    md.append("panel['status.sample_usable_for_feature']  # 因子是否可用")
    md.append("")
    md.append("# 注意：停牌日通常仍有因子数据（因子表的行在停牌日可能存在）")
    md.append("# 所以 is_suspended=True 且 sample_usable_for_feature=True 是完全可能的")
    md.append("```")
    md.append("")

    md.append("#### 情况 6：源数据完全缺失")
    md.append("")
    md.append("```python")
    md.append("# 某些 (date, sid) 在 OHLCV 和因子表中都不存在")
    md.append("# 此时 sample_state = 'NO_SOURCE_RECORD'")
    md.append("# 这些样本的 market.* 和 feature.* 全部为 NaN")
    md.append("# 它们不应该参与任何标签计算或训练")
    md.append("no_data = panel['status.sample_state'] == 'NO_SOURCE_RECORD'")
    md.append("```")
    md.append("")

    md.append("### 7.5 推荐的标签有效性判定模板")
    md.append("")
    md.append("```python")
    md.append("def compute_label_validity(panel, label_values, target_name, forward_days=5):")
    md.append("    \"\"\"")
    md.append("    通用标签有效性判定。返回 label_valid 和 label_reason。")
    md.append("    \"\"\"")
    md.append("    n = len(panel)")
    md.append("    label_valid = pd.Series(True, index=panel.index)")
    md.append("    label_reason = pd.Series('', index=panel.index, dtype='object')")
    md.append("")
    md.append("    # 规则 1：标签值本身为 NaN")
    md.append("    is_nan = label_values.isna()")
    md.append("    label_valid[is_nan] = False")
    md.append("    label_reason[is_nan] = 'LABEL_IS_NAN'")
    md.append("")
    md.append("    # 规则 2：停牌（取决于你的策略 —— 有些模型允许停牌样本）")
    md.append("    # is_susp = panel['status.is_suspended']")
    md.append("    # label_valid[is_susp] = False")
    md.append("    # label_reason[is_susp] = 'SUSPENDED'")
    md.append("")
    md.append("    # 规则 3：没有足够的前向窗口")
    md.append("    # （由调用方在计算 label_values 之前处理）")
    md.append("")
    md.append("    return label_valid, label_reason")
    md.append("```")
    md.append("")

    md.append("### 7.6 训练阶段的 mask 组合")
    md.append("")
    md.append("panel_base 中只有一个可用性标志：`status.sample_usable_for_feature`。")
    md.append("训练可用性需要在训练阶段动态生成：")
    md.append("")
    md.append("```python")
    md.append("# 典型 train_mask 构建方式")
    md.append("train_mask = (")
    md.append("    panel['status.sample_usable_for_feature']   # 因子可用")
    md.append("    & label_valid['return_5d']                  # 标签有效")
    md.append("    & ~panel['status.is_suspended']             # 非停牌（可选）")
    md.append(")")
    md.append("")
    md.append("X_train = panel.loc[train_mask, feature_cols]")
    md.append("y_train = labels.loc[train_mask, 'label.return_5d']")
    md.append("```")
    md.append("")

    # ========== 8. FAQ ==========
    md.append("---")
    md.append("## 8. 常见问题")
    md.append("")
    md.append("**Q：为什么 sample_state 几乎全是 PARTIAL_FACTOR_MISSING？**")
    md.append("")
    md.append(f"A：因为当前因子表有 {len(feature_cols)} 个因子，包括大量基本面因子。")
    md.append("某些基本面因子对特定股票天然为 NaN（如未分红公司的 dividend_yield）。")
    md.append("只要有 1 个因子缺失，就会被标记为 PARTIAL_FACTOR_MISSING。这是正常的。")
    md.append("真正影响可用性的是 `status.sample_usable_for_feature`（只要不是全缺失就可用）。")
    md.append("")
    md.append("**Q：panel_base 生成后，daily_bars 和 factor_values 还需要保留吗？**")
    md.append("")
    md.append("A：不需要。它们的数据已经完全被 panel_base 吸收（left join 进去了）。")
    md.append("可以删除释放磁盘空间。但 `trading_calendar` 和 `instrument_master` 要保留，后续阶段需要用。")
    md.append("")
    md.append("**Q：如何只读取特定年份或特定股票的数据？**")
    md.append("")
    md.append("```python")
    md.append("# 方法 1：用 pyarrow 过滤（不会加载全量数据到内存）")
    md.append("import pyarrow.parquet as pq")
    md.append("table = pq.read_table(")
    md.append("    'data/processed/panel/panel_base.parquet',")
    md.append("    filters=[('date', '>=', '2023-01-01'), ('date', '<', '2024-01-01')]")
    md.append(")")
    md.append("panel_2023 = table.to_pandas()")
    md.append("")
    md.append("# 方法 2：先读再过滤（文件全部加载）")
    md.append("panel = pd.read_parquet('data/processed/panel/panel_base.parquet')")
    md.append("panel_2023 = panel.loc['2023']")
    md.append("stock = panel.loc[(slice(None), '600519.SH'), :]")
    md.append("```")
    md.append("")
    md.append("**Q：panel_base 的行顺序是什么？**")
    md.append("")
    md.append("A：按 `[date, sid]` 排序。同一天的所有股票按 sid 字母序排列。")
    md.append("")

    content = "\n".join(md)
    guide_path = panel_dir / "PANEL_BASE_GUIDE.md"
    guide_path.write_text(content, encoding="utf-8")
    logger.info("Saved guide to %s", guide_path)
    return str(guide_path)


def generate_column_reference(
    panel_dir: Path,
    assets_dir: Path,
) -> str:
    """Generate COLUMN_REFERENCE.md listing every column in every output file."""
    md = []
    md.append("# 列名参考手册")
    md.append("")
    md.append(f"> 自动生成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append("")

    files = [
        ("instrument_master", assets_dir / "instrument_master.parquet"),
        ("trading_calendar", assets_dir / "trading_calendar.parquet"),
        ("active_session_index", panel_dir / "active_session_index.parquet"),
        ("panel_base", panel_dir / "panel_base.parquet"),
    ]

    for name, path in files:
        info = _parquet_info(path)
        if not info.get("exists"):
            continue

        md.append(f"## {name}")
        md.append("")
        md.append(f"- 行数：{info['rows']:,}")
        md.append(f"- 文件大小：{info['size_mb']:.1f} MB")
        md.append("")
        md.append("| # | 列名 | 类型 |")
        md.append("|---|---|---|")
        for i, col in enumerate(info["columns"], 1):
            md.append(f"| {i} | `{col['name']}` | {col['type']} |")
        md.append("")

    content = "\n".join(md)
    ref_path = panel_dir / "COLUMN_REFERENCE.md"
    ref_path.write_text(content, encoding="utf-8")
    logger.info("Saved column reference to %s", ref_path)
    return str(ref_path)
