"""
Target report generator: quality report + column reference for target_block.

Mirrors the naming convention of engine/panel/report_generator.py.

Produces:
  - TARGET_REPORT_<name>.md   — quality report (stats, distribution, checks)
  - COLUMN_REFERENCE_<name>.md — column listing with types
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from engine.targets.registry import get_recipe, get_spec

logger = logging.getLogger(__name__)


def generate_target_report(
    target_block: pd.DataFrame,
    target_name: str,
    output_dir: Path,
) -> Path:
    """
    Generate a markdown quality report for a target_block.

    Parameters
    ----------
    target_block : DataFrame with index=[date, sid].
    target_name : e.g. "return_1d", "momentum_cls".
    output_dir : Directory to write the report file.

    Returns
    -------
    Path to the generated report file.
    """
    spec = get_spec(target_name)
    label_col = f"label.{target_name}"
    valid_col = f"label_valid.{target_name}"
    reason_col = f"label_reason.{target_name}"

    n = len(target_block)
    n_valid = target_block[valid_col].sum()
    n_invalid = n - n_valid

    lines = []
    _a = lines.append

    _a(f"# Target 质量报告: {target_name}")
    _a("")
    _a(f"> 自动生成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    _a(f">")
    _a(f"> 本文件由 `python scripts/build_target.py` 构建完成后自动输出。")
    _a("")

    # --- 1. Overview ---
    _a("## 1. 基本信息")
    _a("")
    _a("| 指标 | 值 |")
    _a("|---|---|")
    _a(f"| 目标名称 | `{target_name}` |")
    _a(f"| 任务类型 | {spec.task_type} |")
    _a(f"| 族 (family) | {spec.family or '—'} |")
    _a(f"| 前向窗口 | {spec.horizon or '—'} 交易日 |")
    _a(f"| 数据类型 | {spec.dtype} |")
    _a(f"| 总行数 | {n:,} |")
    _a(f"| 有效标签 | {n_valid:,} ({n_valid / n * 100:.2f}%) |")
    _a(f"| 无效标签 | {n_invalid:,} ({n_invalid / n * 100:.2f}%) |")
    _a("")

    # --- 2. Invalid reasons ---
    _a("## 2. 无效原因分布")
    _a("")
    if n_invalid > 0:
        reason_counts = (
            target_block.loc[~target_block[valid_col], reason_col]
            .value_counts()
        )
        _a("| 原因 | 数量 | 占无效样本 | 占总样本 |")
        _a("|---|---|---|---|")
        for reason, cnt in reason_counts.items():
            _a(f"| {reason} | {cnt:,} | {cnt / n_invalid * 100:.2f}% | {cnt / n * 100:.2f}% |")
    else:
        _a("无无效样本。")
    _a("")

    # --- 3. Label distribution ---
    valid_labels = target_block.loc[target_block[valid_col], label_col]

    if spec.task_type == "classification":
        _a("## 3. 分类标签分布")
        _a("")
        cls_counts = valid_labels.value_counts().sort_index()

        # Get class names from the recipe itself (no hardcoding)
        recipe = get_recipe(target_name, name=target_name)
        cls_names = recipe.class_names()

        _a("| 类别 | 名称 | 数量 | 占比 |")
        _a("|---|---|---|---|")
        n_valid_total = len(valid_labels)
        for cls_val, cnt in cls_counts.items():
            cls_int = int(cls_val)
            name = cls_names.get(cls_int, "—")
            _a(f"| {cls_int} | {name} | {cnt:,} | {cnt / n_valid_total * 100:.2f}% |")
        _a("")

    elif spec.task_type == "regression":
        _a("## 3. 回归标签分布")
        _a("")
        _a("| 统计量 | 值 |")
        _a("|---|---|")
        _a(f"| count | {len(valid_labels):,} |")
        _a(f"| mean | {valid_labels.mean():.6f} |")
        _a(f"| std | {valid_labels.std():.6f} |")
        _a(f"| min | {valid_labels.min():.6f} |")
        _a(f"| 1% | {valid_labels.quantile(0.01):.6f} |")
        _a(f"| 25% | {valid_labels.quantile(0.25):.6f} |")
        _a(f"| 50% (median) | {valid_labels.quantile(0.50):.6f} |")
        _a(f"| 75% | {valid_labels.quantile(0.75):.6f} |")
        _a(f"| 99% | {valid_labels.quantile(0.99):.6f} |")
        _a(f"| max | {valid_labels.max():.6f} |")
        _a(f"| NaN 数量 | {valid_labels.isna().sum():,} |")

        # Extreme returns
        n_gt_10pct = (valid_labels.abs() > 0.10).sum()
        n_gt_20pct = (valid_labels.abs() > 0.20).sum()
        _a(f"| |return| > 10% | {n_gt_10pct:,} ({n_gt_10pct / len(valid_labels) * 100:.2f}%) |")
        _a(f"| |return| > 20% | {n_gt_20pct:,} ({n_gt_20pct / len(valid_labels) * 100:.2f}%) |")
        _a("")

    # --- 4. Per-year breakdown ---
    _a("## 4. 按年分布")
    _a("")

    dates = target_block.index.get_level_values("date")
    years = dates.year
    target_block_with_year = target_block.copy()
    target_block_with_year["_year"] = years.values

    if spec.task_type == "classification":
        all_classes = sorted(valid_labels.dropna().unique())
        header = "| 年份 | 总行数 | 有效数 | 有效率 |"
        for c in all_classes:
            header += f" {int(c)} |"
        _a(header)
        _a("|---|---|---|---|" + "---|" * len(all_classes))

        for yr, grp in target_block_with_year.groupby("_year"):
            yr_n = len(grp)
            yr_valid = grp[valid_col].sum()
            yr_labels = grp.loc[grp[valid_col], label_col]
            yr_cls = yr_labels.value_counts()
            row = f"| {yr} | {yr_n:,} | {yr_valid:,} | {yr_valid / yr_n * 100:.1f}% |"
            for c in all_classes:
                cnt = yr_cls.get(c, 0)
                pct = cnt / yr_valid * 100 if yr_valid > 0 else 0
                row += f" {cnt:,} ({pct:.1f}%) |"
            _a(row)
    else:
        _a("| 年份 | 总行数 | 有效数 | 有效率 | mean | std | median |")
        _a("|---|---|---|---|---|---|---|")
        for yr, grp in target_block_with_year.groupby("_year"):
            yr_n = len(grp)
            yr_valid = grp[valid_col].sum()
            yr_labels = grp.loc[grp[valid_col], label_col]
            if len(yr_labels) > 0:
                _a(f"| {yr} | {yr_n:,} | {yr_valid:,} | {yr_valid / yr_n * 100:.1f}% "
                   f"| {yr_labels.mean():.6f} | {yr_labels.std():.6f} | {yr_labels.median():.6f} |")
            else:
                _a(f"| {yr} | {yr_n:,} | 0 | 0.0% | — | — | — |")

    _a("")

    # --- 5. Quality checks ---
    _a("## 5. 质量检查")
    _a("")

    checks = []

    # Check 1: valid rate not too low
    valid_rate = n_valid / n
    if valid_rate < 0.5:
        checks.append(("WARNING", f"有效率过低: {valid_rate:.2%} < 50%"))
    else:
        checks.append(("OK", f"有效率: {valid_rate:.2%}"))

    # Check 2: no NaN in valid labels
    nan_in_valid = target_block.loc[target_block[valid_col], label_col].isna().sum()
    if nan_in_valid > 0:
        checks.append(("ERROR", f"有效标签中存在 NaN: {nan_in_valid:,} 行"))
    else:
        checks.append(("OK", "有效标签中无 NaN"))

    # Check 3: classification class balance
    if spec.task_type == "classification" and len(valid_labels) > 0:
        cls_counts = valid_labels.value_counts(normalize=True)
        min_pct = cls_counts.min()
        max_pct = cls_counts.max()
        if max_pct / min_pct > 10:
            checks.append(("WARNING", f"类别不平衡: 最大类 {max_pct:.2%} / 最小类 {min_pct:.2%} = {max_pct / min_pct:.1f}x"))
        else:
            checks.append(("OK", f"类别平衡度: 最大/最小 = {max_pct / min_pct:.1f}x"))

    # Check 4: regression extreme values
    if spec.task_type == "regression" and len(valid_labels) > 0:
        extreme_rate = (valid_labels.abs() > 1.0).sum() / len(valid_labels)
        if extreme_rate > 0.01:
            checks.append(("WARNING", f"极端值 (|label|>100%) 占比: {extreme_rate:.2%}"))
        else:
            checks.append(("OK", f"极端值 (|label|>100%) 占比: {extreme_rate:.4%}"))

    all_ok = all(c[0] == "OK" for c in checks)
    for status, msg in checks:
        icon = {"OK": "+", "WARNING": "!", "ERROR": "-"}[status]
        _a(f"- [{icon}] {msg}")

    _a("")
    if all_ok:
        _a("**全部通过** ✓")
    else:
        _a("**存在警告或错误，请检查上方详情。**")
    _a("")

    # Write file
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"TARGET_REPORT_{target_name}.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    logger.info("Quality report saved → %s", report_path)
    return report_path


# ======================================================================
# Column reference
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


def generate_column_reference(
    target_block_path: Path,
    target_name: str,
    output_dir: Path,
) -> Path:
    """
    Generate COLUMN_REFERENCE_<name>.md listing every column in the target_block.
    Reads only parquet metadata — does NOT load data into memory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_block_path = Path(target_block_path)

    md = []
    md.append(f"# 列名参考: target_block — {target_name}")
    md.append("")
    md.append(f"> 自动生成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append("")

    if target_block_path.exists():
        pf = pq.ParquetFile(str(target_block_path))
        schema = pf.schema_arrow
        num_rows = pf.metadata.num_rows
        size_mb = os.path.getsize(target_block_path) / 1024 / 1024

        md.append(f"## target_block: {target_name}")
        md.append("")
        md.append(f"- 文件: `{target_block_path}`")
        md.append(f"- 行数: {num_rows:,}")
        md.append(f"- 文件大小: {size_mb:.1f} MB")
        md.append("")
        md.append("| # | 列名 | 类型 | 说明 |")
        md.append("|---|---|---|---|")

        for i in range(len(schema)):
            f = schema.field(i)
            name = f.name
            dtype = _arrow_type_str(f.type)
            # Auto-describe based on naming convention
            if name == "date":
                desc = "交易日（index）"
            elif name == "sid":
                desc = "股票标识（index）"
            elif name.startswith("label."):
                desc = "标签值"
            elif name.startswith("label_valid."):
                desc = "标签是否有效"
            elif name.startswith("label_reason."):
                desc = "无效原因（有效时为空字符串）"
            else:
                desc = "—"
            md.append(f"| {i + 1} | `{name}` | {dtype} | {desc} |")
        md.append("")
    else:
        md.append(f"文件不存在: `{target_block_path}`")

    ref_path = output_dir / f"COLUMN_REFERENCE_{target_name}.md"
    ref_path.write_text("\n".join(md), encoding="utf-8")
    logger.info("Column reference saved → %s", ref_path)
    return ref_path
