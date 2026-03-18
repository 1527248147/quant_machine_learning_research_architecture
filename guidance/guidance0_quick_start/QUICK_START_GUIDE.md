# Quick Start Guide — 从零搭建新模型并完成训练

本指南面向第一次使用本框架的用户，目标是：**接入自己的数据、搭建新模型、跑通训练、拿到结果。**

如果你只是想使用已有的 LSTM 多任务模型训练，直接跳到 [第二部分：使用已有模型跑通训练](#第二部分使用已有模型跑通训练)。

如果你要搭建全新的模型，请阅读完整指南。

---

## 目录

- [第一部分：概览与核心概念](#第一部分概览与核心概念)
  - [1.1 数据流总览](#11-数据流总览)
  - [1.2 目录结构速查](#12-目录结构速查)
  - [1.3 核心概念速查](#13-核心概念速查)
- [第二部分：使用已有模型跑通训练](#第二部分使用已有模型跑通训练)
  - [2.1 准备数据](#21-准备数据)
  - [2.2 配置 config](#22-配置-config)
  - [2.3 一键运行 pipeline](#23-一键运行-pipeline)
  - [2.4 验证每一步的产出](#24-验证每一步的产出)
  - [2.5 查看训练结果](#25-查看训练结果)
- [第三部分：搭建全新模型](#第三部分搭建全新模型)
  - [3.1 Model Triple 架构](#31-model-triple-架构)
  - [3.2 Step 1: 实现 Model](#32-step-1-实现-model)
  - [3.3 Step 2: 实现 ViewBuilder](#33-step-2-实现-viewbuilder)
  - [3.4 Step 3: 实现 Trainer](#34-step-3-实现-trainer)
  - [3.5 Step 4: 注册三元组](#35-step-4-注册三元组)
  - [3.6 Step 5: 配置并运行](#36-step-5-配置并运行)
- [第四部分：关键契约参考](#第四部分关键契约参考)
  - [4.1 BaseModel 抽象方法](#41-basemodel-抽象方法)
  - [4.2 BaseViewBuilder 抽象方法](#42-baseviewbuilder-抽象方法)
  - [4.3 BaseTrainer 抽象方法与共享工具](#43-basetrainer-抽象方法与共享工具)
  - [4.4 meta.json 契约](#44-metajson-契约)
  - [4.5 SplitBundle 字段](#45-splitbundle-字段)
  - [4.6 TrainingRunResult 字段](#46-trainingrunresult-字段)
  - [4.7 ModelLabelContract 五种模式](#47-modellabelcontract-五种模式)
  - [4.8 PathManager 常用路径](#48-pathmanager-常用路径)
  - [4.9 EarlyStopping 回调](#49-earlystopping-回调)
- [第五部分：常见问题与排错](#第五部分常见问题与排错)
- [附录：评估指标说明](#附录评估指标说明)

---

## 第一部分：概览与核心概念

### 1.1 数据流总览

```
原始 OHLCV + 因子 parquet 文件
    ↓ build_assets
标准资产层 (instrument_master, trading_calendar, daily_bars, factor_values)
    ↓ build_panel_base
统一底板 panel_base.parquet  (7.7M 行 × 496 列)
    ↓ build_target
标签块 target_block(s)  (return_c0c1, momentum_cls, ...)
    ↓ build_view
模型视图 (如 LSTM 的 3D memmap)
    ↓ train
训练结果 (best.pt, log.csv, config.json)
```

### 1.2 目录结构速查

```
framework/
├── configs/default.yaml          ← 实验配置（修改这个）
├── scripts/
│   ├── run_pipeline.py           ← 一键全流程（推荐）
│   ├── build_assets.py           ← 单步：构建资产
│   ├── build_panel_base.py       ← 单步：构建底板
│   ├── build_target.py           ← 单步：构建标签
│   ├── build_view.py             ← 单步：构建视图
│   └── train.py                  ← 单步：训练
├── engine/                       ← 核心代码包
│   ├── models/impl/<name>/       ← 模型实现
│   ├── views/impl/<name>/        ← 视图实现
│   └── training/impl/<name>/     ← 训练器实现
└── data/
    ├── processed/                ← 中间产物
    │   ├── assets/               ← instrument_master 等
    │   ├── panel/                ← panel_base.parquet
    │   ├── targets/              ← target_block(s)
    │   └── views/<model>/memmap/ ← memmap 文件 + meta.json
    └── training_result/<exp>/    ← 训练输出
        ├── checkpoints/best.pt   ← 最佳模型
        ├── log.csv               ← 训练日志
        └── config.json           ← 配置快照
```

### 1.3 核心概念速查

| 概念 | 是什么 | 在哪 |
|------|--------|------|
| panel_base | 统一底板（所有股票×交易日的特征+状态），不含标签 | `data/processed/panel/panel_base.parquet` |
| target_block | 某一个标签的计算结果（label + valid + reason） | `data/processed/targets/<name>/` |
| meta.json | ViewBuilder 产出的元数据，Trainer 读取它来了解视图结构 | `data/processed/views/<model>/memmap/meta.json` |
| Model Triple | 每个模型由 Model + ViewBuilder + Trainer 三个类组成 | `engine/{models,views,training}/impl/<name>/` |
| ModelLabelContract | 模型声明它需要什么标签（角色、数量、类型） | `engine/schema/contracts.py` |

---

## 第二部分：使用已有模型跑通训练

### 2.1 准备数据

框架需要两类日频 parquet 文件：

| 文件 | 必需列 | 说明 |
|------|--------|------|
| OHLCV 行情 | date, symbol, open, high, low, close, volume | 按年分文件，如 `2019.parquet` |
| 因子 | datetime/date, instrument/order_book_id, feature.* 列 | 按年分文件 |

把数据放到固定目录，然后在 config 中配置路径即可。

### 2.2 配置 config

编辑 `configs/default.yaml`，重点修改：

```yaml
# 数据路径 — 改成你的数据所在目录
data:
  raw_ohlcv_dir:   "你的OHLCV数据目录"
  raw_factor_dir:  "你的因子数据目录"
  processed_dir:   "data/processed"

# 时间范围 — 改成你的数据覆盖范围
panel:
  start_year:  2019
  end_year:    2025
  start_date:  "2019-01-01"
  end_date:    "2025-12-20"

# 标签 — 只需要写名称，registry 已有默认参数
targets:
  - name: "return_c0c1"        # 今日收盘 → 明日收盘
  - name: "momentum_cls"       # 5 类动量分类

# 模型 — 使用已有的 lstm_mtl
model:
  name: "lstm_mtl"
  params:
    embed_dim: 128
    hidden_size: 128
    num_layers: 2
    dropout: 0.2
    num_classes: 5
  label_roles:
    regression: "return_c0c1"
    classification: "momentum_cls"
  view:
    lookback: 60              # LSTM 回看窗口（交易日）
    k: 512                    # 每日采样股票数

# 训练切分和超参数
training:
  seed: 42
  split:
    train_start: "2019-01-01"
    train_end:   "2022-01-01"
    valid_start: "2022-01-01"
    valid_end:   "2024-01-01"
    test_start:  "2024-01-01"
    test_end:    "2025-12-15"
  epochs: 100
  lr: 2.0e-4
  batch_size: 4
  patience: 10
  amp: true

evaluation:
  run_test: true
  refit_before_test: false
```

### 2.3 一键运行 pipeline

```bash
# 推荐：一键全流程（带断点续跑）
python scripts/run_pipeline.py --config configs/default.yaml

# 失败后再次运行，会自动从上次失败的步骤继续
python scripts/run_pipeline.py --config configs/default.yaml

# 强制从头重跑
python scripts/run_pipeline.py --config configs/default.yaml --restart

# 只想从某一步开始
python scripts/run_pipeline.py --config configs/default.yaml --start view

# 查看会执行哪些步骤（不实际运行）
python scripts/run_pipeline.py --config configs/default.yaml --dry-run
```

也可以分步运行：

```bash
python scripts/build_assets.py --config configs/default.yaml
python scripts/build_panel_base.py --config configs/default.yaml
python scripts/build_target.py --config configs/default.yaml
python scripts/build_view.py --config configs/default.yaml
python scripts/train.py --config configs/default.yaml
```

### 2.4 验证每一步的产出

| 步骤 | 预期产出 | 验证方法 |
|------|---------|---------|
| build_assets | `data/processed/assets/` 下有 4 个 parquet 文件 | `ls data/processed/assets/` |
| build_panel_base | `data/processed/panel/panel_base.parquet` (~14 GB) | `python -c "import pyarrow.parquet as pq; print(pq.read_metadata('data/processed/panel/panel_base.parquet').num_rows)"` |
| build_target | `data/processed/targets/return_c0c1/` 和 `momentum_cls/` | 检查 target_block.parquet 存在且有 label.*/label_valid.*/label_reason.* 列 |
| build_view | `data/processed/views/lstm_mtl/memmap/meta.json` | 打开 meta.json 查看 T、N、D 是否合理 |
| train | `data/training_result/<exp_name>/checkpoints/best.pt` | 查看 log.csv 中 val_ic 是否在训练过程中上升 |

### 2.5 查看训练结果

训练完成后，结果在 `data/training_result/<exp_name>/` 下：

```
<exp_name>/
├── checkpoints/
│   ├── best.pt           ← 最佳模型（valid loss 最低的 epoch）
│   └── refit.pt          ← refit 模型（如果开启了 refit_before_test）
├── log.csv               ← 逐 epoch 训练日志
└── config.json           ← 本次训练的完整配置快照
```

**log.csv 字段说明：**

| 字段 | 含义 |
|------|------|
| train_loss | 训练集总损失 |
| train_ret | 回归损失（MSE 或 LambdaRank） |
| train_ce | 分类交叉熵损失 |
| val_loss | 验证集总损失（用于 early stopping） |
| val_ic | 验证集 IC（预测收益 vs 真实收益的 Pearson 相关） |
| val_rankic | 验证集 Rank IC（Spearman 相关） |
| val_acc | 验证集动量分类准确率 |

训练终端输出也会显示 test 集的最终指标（如果 `run_test: true`）。

---

## 第三部分：搭建全新模型

### 3.1 Model Triple 架构

每个模型由三个组件组成，分别放在三个 `impl/<name>/` 目录下：

```
engine/models/impl/my_model/model.py       ← Model: 网络结构
engine/views/impl/my_model/view.py         ← ViewBuilder: 数据格式转换
engine/training/impl/my_model/trainer.py   ← Trainer: 训练循环
```

三者通过 `meta.json` 和 `config` 传递信息：

```
ViewBuilder.build() → 产出 meta.json + 视图文件（memmap、numpy 等）
                ↓
Trainer.run() → 读 meta.json，调 ViewBuilder.get_dataset() 和 build_dataloader()
                ↓
              → 调 Model.build_model(input_dim, config) 构建网络
                ↓
              → 自己实现训练循环
                ↓
              → 返回 TrainingRunResult
```

**关键设计原则：** ViewBuilder 和 Trainer 之间的数据契约由 **你自己定义**。框架不强制 Dataset 的具体格式——你的 ViewBuilder 返回什么，你的 Trainer 就消费什么。只要最终 Trainer 返回 `TrainingRunResult` 即可。

### 3.2 Step 1: 实现 Model

创建 `engine/models/impl/my_model/__init__.py` 和 `engine/models/impl/my_model/model.py`：

```python
# engine/models/impl/my_model/__init__.py
from .model import MyModel

# engine/models/impl/my_model/model.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional

from engine.models.base import BaseModel, ModelLabelContract


class MyModel(BaseModel):
    """示例模型 — 以 LightGBM 为例。"""

    # ---- 必须声明 ----
    name = "my_model"
    contract = ModelLabelContract(
        mode="ANY_SINGLE",   # 接受任意一个标签
        min_labels=1,
        max_labels=1,
    )

    def __init__(self):
        self.model = None

    # ---- 6 个抽象方法 ----

    def build_model(self, input_dim: int, config: dict) -> None:
        """
        根据 input_dim 和 config 构建模型。

        input_dim 的含义取决于你的 ViewBuilder 输出的特征维度。
        比如 LSTM 是 2F+1（features + isna + row_present），
        LGB 可能就是 F（纯特征数）。

        config 结构：config["model"]["params"] 下读你的模型参数。
        """
        params = config.get("model", {}).get("params", {})
        import lightgbm as lgb
        self.model = lgb.LGBMRegressor(
            n_estimators=params.get("n_estimators", 1000),
            learning_rate=params.get("learning_rate", 0.05),
            num_leaves=params.get("num_leaves", 63),
            # ... 其他参数
        )

    def fit(self, train_loader, valid_loader, config, callbacks=None) -> dict:
        """
        训练模型。

        train_loader / valid_loader 的格式取决于你的 ViewBuilder。
        比如 LGB 可以传 dict {"X": ndarray, "y": ndarray}，
        LSTM 传 PyTorch DataLoader。

        返回值：dict，建议包含 best_iteration 等信息。
        """
        X_train, y_train = train_loader["X"], train_loader["y"]
        X_valid, y_valid = valid_loader["X"], valid_loader["y"]
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            callbacks=[lgb.early_stopping(50)],
        )
        return {"best_iteration": self.model.best_iteration_}

    def predict(self, loader) -> dict:
        """返回预测结果 dict。"""
        return {"pred": self.model.predict(loader["X"])}

    def save(self, path: Path) -> None:
        import joblib
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, str(path))

    def load(self, path: Path) -> None:
        import joblib
        self.model = joblib.load(str(path))

    def to_device(self, device) -> None:
        pass  # LGB 不需要 GPU
```

### 3.3 Step 2: 实现 ViewBuilder

创建 `engine/views/impl/my_model/__init__.py` 和 `engine/views/impl/my_model/view.py`：

```python
# engine/views/impl/my_model/__init__.py
from .view import MyViewBuilder

# engine/views/impl/my_model/view.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, List

import numpy as np

from engine.io.paths import PathManager
from engine.io.panel_reader import (
    read_feature_columns,
    read_panel_index,
    read_feature_chunks,
    read_target_labels,
)
from engine.views.base import BaseViewBuilder


class MyViewBuilder(BaseViewBuilder):
    """LightGBM 视图构建器 — 2D numpy array。"""

    name = "my_model"  # 必须和模型名称一致

    def required_columns(self) -> List[str]:
        return []  # 动态决定

    def build(self, paths: PathManager, config: dict) -> Path:
        """
        从 panel_base + target_block 构建模型视图。

        重点：不要加载完整 panel_base（14 GB），使用 panel_reader
        的工具函数按列、按块读取。

        必须产出 meta.json — Trainer 依赖它。
        """
        out_dir = paths.memmap_dir(self.name)

        # 已经构建过就跳过
        meta_path = out_dir / "meta.json"
        if meta_path.exists():
            return out_dir

        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. 发现特征列
        feat_cols = read_feature_columns(paths.panel_base_path)
        F = len(feat_cols)

        # 2. 读取 panel 索引（轻量，只有 date + sid + status）
        pidx = read_panel_index(paths, extra_columns=["status.sample_usable_for_feature"])
        usable = pidx.extra_columns["status.sample_usable_for_feature"]

        # 3. 按块读取特征（每次 50 列，避免 OOM）
        X = np.zeros((pidx.num_rows, F), dtype=np.float32)
        for chunk_start, chunk_end, chunk_col_names, chunk_vals in read_feature_chunks(
            paths, chunk_size=50, feat_columns=feat_cols,
        ):
            X[:, chunk_start:chunk_end] = np.nan_to_num(chunk_vals, nan=0.0)

        # 4. 读取标签
        target_name = config.get("targets", [{}])[0].get("name", "return_c0c1")
        labels = read_target_labels(paths, target_name)

        # 5. 保存
        np.save(out_dir / "X.npy", X)
        np.save(out_dir / "y.npy", labels.label_values)
        np.save(out_dir / "valid_mask.npy", labels.valid_mask)
        np.save(out_dir / "usable.npy", usable)

        # 6. 保存 meta.json（必须！Trainer 要读这个）
        meta = {
            "num_rows": pidx.num_rows,
            "F": F,
            "D": F,                    # 对 LGB 来说 input_dim = F
            "dates": pidx.date_strings,
            "instruments": pidx.sid_strings,
            "feat_cols": feat_cols,
            "target_name": target_name,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        return out_dir

    def get_dataset(self, view_dir: Path, day_start: int, day_end: int, config: dict) -> Any:
        """
        返回指定日期范围的数据。

        day_start / day_end 是 meta["dates"] 列表的索引（inclusive）。
        用它来切片你的数据。

        返回什么格式由你决定 — 只要 build_dataloader() 能消费即可。
        """
        meta_path = view_dir / "meta.json"
        with open(meta_path) as f:
            meta = json.load(f)

        dates = meta["dates"]
        X = np.load(view_dir / "X.npy", mmap_mode="r")
        y = np.load(view_dir / "y.npy", mmap_mode="r")
        valid = np.load(view_dir / "valid_mask.npy", mmap_mode="r")
        usable = np.load(view_dir / "usable.npy", mmap_mode="r")

        # 这里需要根据 day_start/day_end 选出对应日期的行
        # 具体逻辑取决于你的数据组织方式
        # 示例：假设数据按 date 排序
        # ...

        return {"X": X_slice, "y": y_slice}

    def build_dataloader(self, dataset: Any, config: dict, shuffle: bool = True) -> Any:
        """
        把 dataset 包装成 dataloader。

        对 LGB 来说可以直接返回 dataset dict（不需要 DataLoader）。
        对 PyTorch 模型返回 torch.utils.data.DataLoader。
        """
        return dataset  # LGB 直接用 dict
```

### 3.4 Step 3: 实现 Trainer

创建 `engine/training/impl/my_model/__init__.py` 和 `engine/training/impl/my_model/trainer.py`：

```python
# engine/training/impl/my_model/__init__.py
# (空文件)

# engine/training/impl/my_model/trainer.py
from __future__ import annotations
import logging
from pathlib import Path

from engine.io.paths import PathManager
from engine.training.base import BaseTrainer
from engine.training.results import (
    SelectionResult,
    TestResult,
    TrainingRunResult,
)

logger = logging.getLogger(__name__)


class MyTrainer(BaseTrainer):
    """LightGBM 训练器。"""

    name = "my_model"  # 必须和模型名称一致

    def run(self, config: dict, paths: PathManager) -> TrainingRunResult:
        """
        完整训练流程。这是唯一需要实现的抽象方法。

        典型流程：
        1. set_seed() — 固定随机种子
        2. setup_experiment() — 创建输出目录
        3. run_preflight() — 验证 label contract
        4. load_view_meta() — 读 meta.json
        5. build_split() — 时间切分
        6. 构建模型、训练、评估
        7. 返回 TrainingRunResult
        """
        train_cfg = config.get("training", {})
        eval_cfg = config.get("evaluation", {})
        model_cfg = config.get("model", {})
        model_name = model_cfg.get("name", self.name)

        # ---- 1. 基础设施（使用 BaseTrainer 共享工具） ----
        seed = train_cfg.get("seed", 42)
        self.set_seed(seed)
        device = self.resolve_device()

        exp_name, run_dir, ckpt_dir, log_path = self.setup_experiment(
            config, paths, model_name,
        )

        # ---- 2. Contract 校验 ----
        from engine.models.registry import get_model_class, get_view_class
        ModelClass = get_model_class(model_name)
        ViewClass = get_view_class(model_name)

        model_wrapper = ModelClass()
        view_builder = ViewClass()
        self.run_preflight(model_name, model_wrapper.contract, config)

        # ---- 3. 加载 view 元数据 ----
        view_dir = paths.memmap_dir(view_builder.name)
        meta = self.load_view_meta(view_dir)
        logger.info("View at %s, D=%d", view_dir, meta["D"])

        # ---- 4. 时间切分 ----
        lookback = model_cfg.get("view", {}).get("lookback", 0)
        split = self.build_split(meta, config, lookback=lookback)
        # split 字段: train_start_idx, train_end_idx, valid_start_idx, ...
        # 这些是 meta["dates"] 列表的索引

        # ---- 5. 构建模型 ----
        D = meta["D"]
        model_wrapper.build_model(D, config)

        # ---- 6. 训练（Selection 阶段） ----
        train_data = view_builder.get_dataset(
            view_dir, split.train_start_idx, split.train_end_idx, config,
        )
        valid_data = view_builder.get_dataset(
            view_dir, split.valid_start_idx, split.valid_end_idx, config,
        )
        train_dl = view_builder.build_dataloader(train_data, config, shuffle=True)
        valid_dl = view_builder.build_dataloader(valid_data, config, shuffle=False)

        fit_info = model_wrapper.fit(train_dl, valid_dl, config)

        # 保存最佳模型
        best_path = ckpt_dir / "best.pt"
        model_wrapper.save(best_path)

        selection = SelectionResult(
            best_epoch=fit_info.get("best_iteration", 0),
            best_iteration=fit_info.get("best_iteration"),
            valid_metrics=fit_info.get("valid_metrics", {}),
            model_path=best_path,
        )
        logger.info("Selection: best_iteration=%s", selection.best_iteration)

        # ---- 7. Test 评估（可选） ----
        test_result = None
        if eval_cfg.get("run_test", True):
            test_data = view_builder.get_dataset(
                view_dir, split.test_start_idx, split.test_end_idx, config,
            )
            test_dl = view_builder.build_dataloader(test_data, config, shuffle=False)
            preds = model_wrapper.predict(test_dl)

            # 计算你的评估指标
            test_metrics = {}  # {"ic": ..., "mse": ..., ...}

            test_result = TestResult(
                model_source="selection",
                test_metrics=test_metrics,
                model_path=best_path,
            )

        # ---- 8. 返回结果 ----
        return TrainingRunResult(
            selection=selection,
            test=test_result,
            config=config,
            exp_name=exp_name,
            run_dir=run_dir,
        )
```

### 3.5 Step 4: 注册三元组

编辑 `engine/models/registry.py`，在 `_register_builtins()` 中添加：

```python
def _register_builtins() -> None:
    from engine.models.impl.lstm_mtl import LSTMMTLModel
    from engine.training.impl.lstm_mtl.trainer import LSTMMTLTrainer
    from engine.views.impl.lstm_mtl import LSTMViewBuilder

    register_model("lstm_mtl", LSTMMTLModel, LSTMViewBuilder, LSTMMTLTrainer)

    # ---- 新增你的模型 ----
    from engine.models.impl.my_model import MyModel
    from engine.training.impl.my_model.trainer import MyTrainer
    from engine.views.impl.my_model import MyViewBuilder

    register_model("my_model", MyModel, MyViewBuilder, MyTrainer)
```

### 3.6 Step 5: 配置并运行

在 config 中把 model.name 改成你的模型名称：

```yaml
model:
  name: "my_model"
  params:
    n_estimators: 1000
    learning_rate: 0.05
```

然后运行：

```bash
python scripts/run_pipeline.py --config configs/default.yaml
```

---

## 第四部分：关键契约参考

### 4.1 BaseModel 抽象方法

文件: `engine/models/base.py`

| 方法 | 签名 | 说明 |
|------|------|------|
| `build_model` | `(input_dim: int, config: dict) -> None` | 根据维度和配置构建模型。`input_dim` 来自 `meta["D"]`，含义由 ViewBuilder 决定 |
| `fit` | `(train_loader, valid_loader, config, callbacks?) -> dict` | 训练。loader 格式由 ViewBuilder 决定。返回 dict（建议含 `best_iteration`） |
| `predict` | `(loader) -> dict` | 推理。返回 dict of numpy arrays |
| `save` | `(path: Path) -> None` | 保存 checkpoint |
| `load` | `(path: Path) -> None` | 加载 checkpoint |
| `to_device` | `(device) -> None` | 移动到 GPU/CPU |

还需声明两个类属性：

```python
name: str                    # 模型名称，必须唯一
contract: ModelLabelContract  # 标签契约
```

### 4.2 BaseViewBuilder 抽象方法

文件: `engine/views/base.py`

| 方法 | 签名 | 说明 |
|------|------|------|
| `required_columns` | `() -> List[str]` | 需要的 panel_base 列名（可返回空，动态决定） |
| `build` | `(paths: PathManager, config: dict) -> Path` | 构建视图产物。**必须生成 meta.json**。返回输出目录 |
| `get_dataset` | `(view_dir: Path, day_start: int, day_end: int, config: dict) -> Any` | 返回指定日期范围的数据集。day_start/day_end 是 `meta["dates"]` 的索引 |
| `build_dataloader` | `(dataset, config: dict, shuffle: bool) -> Any` | 把 dataset 包装成 dataloader |

还需声明：

```python
name: str  # 必须和对应模型名称一致
```

### 4.3 BaseTrainer 抽象方法与共享工具

文件: `engine/training/base.py`

**需要实现的抽象方法（只有一个）：**

```python
def run(self, config: dict, paths: PathManager) -> TrainingRunResult
```

**共享工具方法（直接调用，不需要覆盖）：**

| 方法 | 签名 | 作用 |
|------|------|------|
| `set_seed` | `(seed: int) -> None` | 固定 Python / numpy / torch 随机种子 |
| `resolve_device` | `() -> torch.device` | 检测 GPU，返回 device |
| `setup_experiment` | `(config, paths, model_name) -> (exp_name, run_dir, ckpt_dir, log_path)` | 创建输出目录，保存 config 快照 |
| `run_preflight` | `(model_name, contract, config) -> None` | 验证 label contract，失败抛异常 |
| `load_view_meta` | `(view_dir: Path) -> dict` | 读 `meta.json`，不存在时报错并提示先运行 build_view |
| `build_split` | `(meta, config, lookback) -> SplitBundle` | 根据 config 日期范围和 meta["dates"] 做时间切分 |

### 4.4 meta.json 契约

meta.json 是 ViewBuilder 产出、Trainer 消费的元数据。框架要求的**最低字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `dates` | `List[str]` | **必须**。日期字符串列表（YYYY-MM-DD），`build_split()` 依赖它做时间切分 |
| `D` | `int` | **推荐**。输入维度，传给 `Model.build_model(input_dim)` |

其余字段由你自定义，Trainer 自己读取。

LSTM 示例的完整 meta.json 字段：

```json
{
  "T": 1460,          // 交易日数
  "N": 5300,          // 股票数
  "D": 947,           // 输入维度 = 2F+1
  "F": 473,           // 原始特征数
  "dates": ["2019-01-02", "2019-01-03", ...],
  "instruments": ["000001.SZ", "000002.SZ", ...],
  "feat_cols": ["feature.alpha158__KLEN", ...],
  "isna_cols": ["feature.alpha158__KLEN__isna", ...],
  "regression_label": "return_c0c1",
  "classification_label": "momentum_cls"
}
```

### 4.5 SplitBundle 字段

文件: `engine/training/splitter.py`

```python
@dataclass
class SplitBundle:
    train_start_idx: int   # meta["dates"] 中 train 起始索引
    train_end_idx: int     # meta["dates"] 中 train 结束索引（inclusive）
    valid_start_idx: int
    valid_end_idx: int
    test_start_idx: int
    test_end_idx: int
    dates: List[str]       # 完整日期列表

    # 属性
    train_days -> int
    valid_days -> int
    test_days -> int
    train_plus_valid_start_idx -> int   # = train_start_idx
    train_plus_valid_end_idx -> int     # = valid_end_idx
```

这些索引直接传给 `ViewBuilder.get_dataset(view_dir, day_start, day_end, config)`。

### 4.6 TrainingRunResult 字段

文件: `engine/training/results.py`

```python
@dataclass
class SelectionResult:
    best_epoch: int                       # 最佳 epoch
    best_iteration: Optional[int]         # 最佳迭代次数（LGB 用）
    best_params: Dict = {}                # 最佳超参数
    valid_metrics: Dict = {}              # 验证集指标
    model_path: Optional[Path] = None     # 最佳模型路径

@dataclass
class TestResult:
    model_source: str     # "selection" 或 "refit"
    test_metrics: Dict    # 测试集指标
    model_path: Optional[Path] = None

@dataclass
class TrainingRunResult:
    selection: SelectionResult            # 必须有
    test: Optional[TestResult] = None     # run_test=false 时为 None
    config: Dict = {}
    exp_name: str = ""
    run_dir: Optional[Path] = None
```

### 4.7 ModelLabelContract 五种模式

文件: `engine/schema/contracts.py`

```python
@dataclass
class ModelLabelContract:
    mode: str           # "ANY_SINGLE" | "ANY_MULTI" | "EXACT" | "ROLE_BASED" | "CUSTOM"
    min_labels: int = 1
    max_labels: Optional[int] = 1
    exact_labels: List[str] = []
    required_roles: Dict[str, str] = {}
    custom_validator: Optional[Callable] = None
```

| 模式 | 适用场景 | 示例 |
|------|---------|------|
| ANY_SINGLE | 通用单任务（如 LGB） | `ModelLabelContract(mode="ANY_SINGLE")` |
| ANY_MULTI | 通用多任务 | `ModelLabelContract(mode="ANY_MULTI", min_labels=2, max_labels=None)` |
| ROLE_BASED | 推荐的多任务方式（如 LSTM） | `ModelLabelContract(mode="ROLE_BASED", min_labels=2, max_labels=2, required_roles={"regression": "regression", "classification": "classification"})` |
| EXACT | 写死标签名 | `ModelLabelContract(mode="EXACT", exact_labels=["return_c0c1", "momentum_cls"])` |
| CUSTOM | 自定义验证器 | `ModelLabelContract(mode="CUSTOM", custom_validator=my_fn)` |

### 4.8 PathManager 常用路径

文件: `engine/io/paths.py`

```python
paths = PathManager(processed_dir="data/processed", training_result_dir="data/training_result")

paths.panel_base_path          # data/processed/panel/panel_base.parquet
paths.target_block_path("x")   # data/processed/targets/x/target_block.parquet
paths.memmap_dir("lstm_mtl")   # data/processed/views/lstm_mtl/memmap/
paths.run_dir("exp1")          # data/training_result/exp1/
paths.run_checkpoint_dir("x")  # data/training_result/x/checkpoints/
paths.run_log_path("x")        # data/training_result/x/log.csv
paths.run_config_path("x")     # data/training_result/x/config.json
```

### 4.9 EarlyStopping 回调

文件: `engine/training/callbacks.py`

```python
from engine.training.callbacks import EarlyStopping, gate_lambda_schedule

es = EarlyStopping(patience=10)

for epoch in range(100):
    val_loss = ...
    state = es.step(val_loss, epoch)

    if state.best_epoch == epoch:
        # 保存最佳模型
        ...

    if state.should_stop:
        break

# state 字段:
#   best_val: float    — 最佳 val_loss
#   best_epoch: int    — 最佳 epoch
#   bad_epochs: int    — 连续无改善的 epoch 数
#   should_stop: bool  — 是否应该停止
```

---

## 第五部分：常见问题与排错

### Q1: `FileNotFoundError: Pre-built view not found`

**原因：** 你跳过了 `build_view` 步骤直接运行 `train`。

**解决：** 先运行 `python scripts/build_view.py --config configs/default.yaml`，或使用 `run_pipeline.py` 自动按顺序执行。

### Q2: `KeyError: Model 'xxx' not registered`

**原因：** 模型名称没有在 `engine/models/registry.py` 中注册。

**解决：** 在 `_register_builtins()` 中添加 `register_model(...)` 调用。

### Q3: 训练时 `MemoryError` 或 OOM

**可能原因：**
- `build_view` 时加载了完整 panel_base。使用 `read_feature_chunks()` 按块读取。
- 训练时 `batch_size` 太大或 `k`（每日采样股票数）太大。减小这些参数。
- LSTM `lookback` 太长，减小到 30 或 20 试试。

### Q4: pipeline 重跑时卡在已完成的步骤

**原因：** checkpoint 文件损坏或 config 改过。

**解决：**
```bash
# 强制从头
python scripts/run_pipeline.py --config configs/default.yaml --restart

# 或手动删除 checkpoint
rm .pipeline_checkpoint.json
```

### Q5: preflight 校验失败

**原因：** config 中的 targets/label_roles 和模型的 contract 不匹配。

**解决：** 检查模型的 `contract` 声明需要什么模式（ANY_SINGLE 需要 1 个标签，ROLE_BASED 需要指定角色），确保 config 正确对应。错误信息会指出具体缺少什么。

### Q6: val_ic 一直为 0 或 NaN

**可能原因：**
- 标签全部无效（`label_valid` 全 False）。检查 target_block 构建是否正确。
- 学习率太大导致模型发散。减小 `lr`。
- 特征全是 NaN。检查因子数据是否正确加载。

### Q7: 如何只重建 view 不重建 panel_base？

```bash
python scripts/run_pipeline.py --config configs/default.yaml --start view
```

### Q8: 如何使用 LambdaRank 替代 MSE？

在 config 的 `training:` 下设置：

```yaml
training:
  use_lambdarank: true
  lambdarank_k: 50
  lambdarank_sigma: 1.0
  lambdarank_bins: 5
```

### Q9: 如何只用部分因子训练？

两种方式：

**构建时过滤**（影响 memmap，改了需重建 view）：
```yaml
model:
  view:
    include_pattern: "^feature\\.alpha158__"   # 只保留匹配的
    exclude_pattern: "^feature\\.fund__"       # 排除匹配的
```

**训练时过滤**（不影响 memmap，动态生效）：
```yaml
training:
  include_pattern: "^feature\\.alpha158__"
  exclude_pattern: "^feature\\.fund__"
```

---

## 附录：评估指标说明

| 指标 | 全称 | 含义 | 好的范围 |
|------|------|------|---------|
| IC | Information Coefficient | 预测收益与真实收益的 Pearson 相关系数（按日计算后取平均） | > 0.03 算可用，> 0.05 算好 |
| RankIC | Rank IC | 预测收益与真实收益的 Spearman 秩相关（对异常值更稳健） | > 0.04 算可用，> 0.06 算好 |
| cls_acc | Classification Accuracy | 动量分类准确率（5 类随机基线 = 20%） | > 25% 算有区分度 |
| ret_loss | Regression Loss | 回归损失（MSE 或 LambdaRank） | 越小越好，关注下降趋势 |
| ce_loss | Cross-Entropy Loss | 分类交叉熵损失 | 越小越好 |
| val_loss | Validation Total Loss | ret_loss + ce_loss | 用于 early stopping 判断 |

> **注意：** IC/RankIC 是量化策略最核心的指标。一个 IC=0.05 的因子在 A 股已经有实际价值。不要追求过高的 IC（可能过拟合），关注 train 和 valid 的 IC 差距。
