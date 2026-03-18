# Model & Training 使用指南

> 本文件是 Phase 3 中 model 和 training 部分的完整教程。
> 它教你：如何运行训练、如何写一个新模型和它的 trainer、contract 是什么、config 怎么配、训练流程怎么运转。
>
> 前置阅读：`guidance/guidance3_view_building/VIEW_BUILDING_GUIDE.md`

---

## 1. 整体概览

### 1.1 训练在 pipeline 中的位置

```
Phase 1:  raw data → panel_base
Phase 2:  panel_base → target_blocks
Phase 3:  panel_base + target_blocks → build_view → view (memmap)
                                                        ↓
          view → train.py → selection → (refit) → test evaluation
                                                        ↓
          data/training_result/<exp_name>/
              ├── checkpoints/best.pt    # 最佳模型
              ├── log.csv                # 训练日志
              └── config.json            # 配置快照
```

### 1.2 核心设计：每个模型三件套

框架为每个模型维护三个组件，通过 registry 关联：

| 组件 | 基类 | 职责 | 存放位置 |
|------|------|------|---------|
| **Model** | `BaseModel` | 模型定义、参数、保存/加载 | `engine/models/impl/<name>/model.py` |
| **ViewBuilder** | `BaseViewBuilder` | 从源数据构建模型所需的输入格式 | `engine/views/impl/<name>/view.py` |
| **Trainer** | `BaseTrainer` | 完整的训练循环（损失、评估、调度） | `engine/training/impl/<name>/trainer.py` |

**为什么 trainer 也是模型专属的？** 因为不同模型的训练逻辑差异巨大：
- LSTM 需要 multi-task loss、gate L1 schedule、AMP
- LightGBM 直接调 `.fit()`，没有 epoch 概念
- 因子选择等功能也依赖于具体的 view 结构

通用的样板代码（seed 设置、设备选择、实验目录创建、split 构建等）放在 `BaseTrainer` 中，子类直接继承使用。

### 1.3 运行

```bash
# 基本用法
python scripts/train.py --config configs/default.yaml

# 指定实验名（决定输出目录）
python scripts/train.py --config configs/default.yaml --exp_name my_exp_v1

# 一键跑完整 pipeline（从 view 到训练）
python scripts/run_pipeline.py --config configs/default.yaml --start view
```

### 1.4 前置条件

| 文件 | 来源 |
|------|------|
| `data/processed/views/<model_name>/memmap/meta.json` | `scripts/build_view.py` |

train.py **只消费 memmap view**，不知道 panel_base 或 target_block 的存在。

---

## 2. Config 配置详解

### 2.1 config 结构

```yaml
model:
  name: "lstm_mtl"           # 模型名 → 决定用哪个 Model + ViewBuilder + Trainer

  params:                    # 模型架构参数（怎么建模型）
    embed_dim: 128
    hidden_size: 128
    num_layers: 2
    ...

  label_roles:               # 标签角色映射（contract 校验用）
    regression: "return_c0c1"
    classification: "momentum_cls"

  view:                      # view 构建 + 数据采样参数
    lookback: 60
    k: 512
    # include_pattern: ...   # build_view 时的因子筛选（改了需要重建 view）
    # exclude_pattern: ...

training:                    # 所有训练参数（通用 + 模型专属）
  seed: 42
  split:
    train_start: "2019-01-01"
    train_end:   "2022-01-01"
    valid_start: "2022-01-01"
    valid_end:   "2024-01-01"
    test_start:  "2024-01-01"
    test_end:    "2025-12-15"

  # 通用训练超参数
  epochs: 100
  lr: 2.0e-4
  weight_decay: 1.0e-3
  batch_size: 4
  grad_clip: 1.0
  patience: 10
  amp: true
  num_workers: 0
  shuffle: true
  log_interval: 200

  # --- lstm_mtl 专属 (换模型时替换此区域) ---
  # 回归损失选择: MSE (默认) 或 LambdaRank NDCG
  use_lambdarank: false      # true → 用 LambdaRank 排序损失; false → 用 MSE 回归损失
  lambdarank_k: 50           # LambdaRank: NDCG@k 的 k
  lambdarank_sigma: 1.0      # LambdaRank: pairwise 差异缩放因子
  lambdarank_bins: 5         # LambdaRank: 收益离散化分箱数
  # 多任务损失权重
  ret_weight: 1.0           # 回归损失权重（MSE 或 LambdaRank）
  ce_weight: 1.0            # CE 分类损失权重
  gate_l1_max: 5.0e-4       # Gating L1 正则化上限
  gate_warmup_epochs: 5     # gate 正则化预热 epoch
  gate_ramp_epochs: 20      # gate 正则化线性增长 epoch
  # include_pattern: ...    # 训练时因子筛选（不影响 memmap，无需重建 view）
  # exclude_pattern: ...

evaluation:                  # 通用评估配置
  run_test: true
  refit_before_test: false
  refit_train_scope: "train+valid"
  refit_iteration_source: "selection"
```

### 2.2 参数分区

| 位置 | 内容 | 换模型时怎么处理 |
|------|------|----------------|
| `model.params` | 模型架构（embed_dim, num_layers 等） | 整个替换 |
| `model.label_roles` | 标签角色映射 | 按新模型需求修改 |
| `model.view` | view 构建 + 数据采样 + build-time 因子筛选 | 整个替换 |
| `training:` 通用部分 | epochs, lr, batch_size 等 | 不用改 |
| `training:` 专属部分 | 损失权重、正则化、训练时因子筛选 | 删掉旧的，加上新的 |
| `evaluation.*` | run_test, refit 等 | 不用改 |

**换模型时的操作**：替换 `model:` 块 + 替换 `training:` 中的专属区域。废参数留着也不报错（代码全用 `.get(key, default)` 防御性读取），但建议删掉保持整洁。

### 2.3 参数读取约定

所有训练参数（通用 + 模型专属）统一从 `config["training"]` 读取：

```python
train_cfg = config.get("training", {})

# 通用参数
epochs = train_cfg.get("epochs", 100)
lr = train_cfg.get("lr", 2e-4)

# 模型专属参数（同一个 dict，只是逻辑上分区）
ret_w = train_cfg.get("ret_weight", 1.0)       # 不存在就用默认值，不报错
gate_l1_max = train_cfg.get("gate_l1_max", 5e-4)

# 模型架构参数
model_params = config.get("model", {}).get("params", {})
embed_dim = model_params.get("embed_dim", 128)

# 标签角色
label_roles = config.get("model", {}).get("label_roles", {})

# View 参数
view_cfg = config.get("model", {}).get("view", {})
lookback = view_cfg.get("lookback", 60)
```

**为什么不会报错？** `.get(key, default)` 如果 key 不存在，返回 default。所以：
- 换成 LGB 后，config 里没有 `ret_weight` → LSTM trainer 不会被调用，LGB trainer 不会读这个 key
- config 里残留 `ret_weight` → LGB trainer 不读它，它安静地躺在 dict 里

---

## 3. 从零写一个新模型（教程）

假设你要添加一个 LightGBM 模型。完整流程分 5 步。

### 3.1 第 1 步：创建文件结构

```
engine/
├── models/
│   ├── base.py                      # 抽象基类（不要修改）
│   ├── registry.py                  # 注册表（需要添加注册）
│   └── impl/
│       ├── lstm_mtl/                # 已有
│       │   ├── __init__.py
│       │   └── model.py
│       └── lgb/                     # 你要新建的 ↓
│           ├── __init__.py
│           └── model.py
├── views/
│   └── impl/
│       └── lgb/
│           ├── __init__.py
│           └── view.py
└── training/
    └── impl/
        └── lgb/
            ├── __init__.py
            ├── trainer.py           # ← 模型专属训练循环
            └── evaluator.py         # ← 模型专属评估（可选）
```

**三个包的组织方式一致**：`impl/` 作为分组层，模型名作为子目录。

### 3.2 第 2 步：写 model.py

#### 3.2.1 继承 `BaseModel`

你的模型**必须**继承 `BaseModel` 并实现 6 个方法：

```python
from engine.models.base import BaseModel, ModelLabelContract

class LGBModel(BaseModel):
    name = "lgb"
    contract = ModelLabelContract(...)  # ← 必须声明

    def build_model(self, input_dim: int, config: dict) -> None: ...
    def fit(self, train_loader, valid_loader, config, callbacks=None) -> dict: ...
    def predict(self, loader) -> dict: ...
    def save(self, path: Path) -> None: ...
    def load(self, path: Path) -> None: ...
    def to_device(self, device) -> None: ...
```

#### 3.2.2 声明 Contract（必须！）

**每个模型必须声明一个 `ModelLabelContract`**，告诉框架"我需要什么标签"。
框架在训练开始前会用 preflight 校验 config 中的 label_roles 是否满足 contract。

5 种 contract 模式：

| 模式 | 含义 | 适用场景 |
|------|------|---------|
| `ANY_SINGLE` | 只需要 1 个标签，名字随意 | 单任务回归/分类 |
| `ANY_MULTI` | 需要多个标签，名字随意 | 多任务，但不限定具体标签 |
| `EXACT` | 需要一组**特定名字**的标签 | 硬编码依赖某些标签 |
| `ROLE_BASED` | 需要一组**角色**，具体标签名由 config 映射 | **推荐**，最灵活 |
| `CUSTOM` | 自定义校验函数 | 复杂场景 |

**推荐使用 ROLE_BASED**，因为它将标签名和模型解耦：

```python
# LSTM 多任务模型：需要一个回归标签 + 一个分类标签
contract = ModelLabelContract(
    mode="ROLE_BASED",
    min_labels=2,
    max_labels=2,
    required_roles={"regression": "regression", "classification": "classification"},
)

# 单任务回归模型：只需要一个回归标签
contract = ModelLabelContract(
    mode="ROLE_BASED",
    min_labels=1,
    max_labels=1,
    required_roles={"regression": "regression"},
)

# 最简单的：只要 1 个标签就行
contract = ModelLabelContract(mode="ANY_SINGLE")
```

然后在 config 中配置哪个具体标签扮演哪个角色：

```yaml
model:
  label_roles:
    regression: "return_c0c1"       # return_c0c1 扮演 regression 角色
    classification: "momentum_cls" # momentum_cls 扮演 classification 角色
```

Preflight 校验流程：
```
model 声明: 我需要 "regression" 和 "classification" 两个角色
config 声明: regression → return_c0c1, classification → momentum_cls
preflight 检查:
  ✅ regression 角色存在 → 映射到 return_c0c1
  ✅ classification 角色存在 → 映射到 momentum_cls
  ✅ return_c0c1 在 targets 列表中
  ✅ momentum_cls 在 targets 列表中
  → 通过！
```

#### 3.2.3 读取超参数

模型架构参数从 `config["model"]["params"]` 读取，**不要硬编码**：

```python
def build_model(self, input_dim: int, config: dict) -> None:
    model_params = config.get("model", {}).get("params", {})
    self.model = lgb.LGBMRegressor(
        num_leaves=model_params.get("num_leaves", 128),
        n_estimators=model_params.get("n_estimators", 500),
        learning_rate=model_params.get("learning_rate", 0.05),
    )
```

#### 3.2.4 完整 model.py 示例

```python
"""LightGBM model for single-task regression."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Optional

from engine.models.base import BaseModel, ModelLabelContract


class LGBModel(BaseModel):
    """LightGBM regressor, wrapped for the framework."""

    name = "lgb"

    # 只需要 1 个回归标签
    contract = ModelLabelContract(
        mode="ROLE_BASED",
        min_labels=1,
        max_labels=1,
        required_roles={"regression": "regression"},
    )

    def __init__(self):
        self.model = None

    def build_model(self, input_dim: int, config: dict) -> None:
        import lightgbm as lgb
        model_params = config.get("model", {}).get("params", {})
        self.model = lgb.LGBMRegressor(
            num_leaves=model_params.get("num_leaves", 128),
            n_estimators=model_params.get("n_estimators", 500),
            learning_rate=model_params.get("learning_rate", 0.05),
        )

    def fit(self, train_loader, valid_loader, config, callbacks=None) -> dict:
        X_train, y_train = train_loader["X"], train_loader["y"]
        X_valid, y_valid = valid_loader["X"], valid_loader["y"]
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            callbacks=[lgb.early_stopping(50)],
        )
        return {"best_iteration": self.model.best_iteration_}

    def predict(self, loader) -> dict:
        return {"pred_ret": self.model.predict(loader["X"])}

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path: Path) -> None:
        with open(path, "rb") as f:
            self.model = pickle.load(f)

    def to_device(self, device) -> None:
        pass  # LightGBM 不需要 GPU
```

### 3.3 第 3 步：写 trainer.py

**这是新架构的关键：每个模型必须有自己的 Trainer。**

#### 3.3.1 继承 `BaseTrainer`

```python
from engine.training.base import BaseTrainer

class LGBTrainer(BaseTrainer):
    name = "lgb"

    def run(self, config, paths) -> TrainingRunResult:
        # 你的完整训练逻辑
        ...
```

#### 3.3.2 BaseTrainer 提供的共享工具

你不需要从零开始写样板代码。`BaseTrainer` 提供了以下工具方法，直接 `self.xxx()` 调用：

| 方法 | 作用 |
|------|------|
| `self.set_seed(seed)` | 设置 random / numpy / torch 的全局种子 |
| `self.resolve_device()` | 自动检测 CUDA，返回 `torch.device` |
| `self.setup_experiment(config, paths, model_name)` | 创建实验目录、保存 config 快照，返回 `(exp_name, run_dir, ckpt_dir, log_path)` |
| `self.run_preflight(model_name, contract, config)` | 校验 contract 是否满足 |
| `self.load_view_meta(view_dir)` | 加载 `meta.json`，如果 view 不存在则报错 |
| `self.build_split(meta, config, lookback)` | 基于日期构建 train/valid/test 的 memmap 索引 |

#### 3.3.3 读取训练参数

所有训练参数（通用 + 模型专属）都在 `config["training"]` 中。你的 Trainer 从同一个 dict 读取：

```python
train_cfg = config.get("training", {})

# 通用参数（每个 Trainer 都可以用）
epochs = train_cfg.get("epochs", 100)
lr = train_cfg.get("lr", 2e-4)

# 你的模型专属参数（只有你的 Trainer 会读）
n_estimators = train_cfg.get("n_estimators", 500)
early_stopping_rounds = train_cfg.get("early_stopping_rounds", 50)
```

不存在的 key → `.get()` 返回默认值，**不报错**。所以换模型后旧参数残留也无害。

#### 3.3.4 完整 trainer.py 示例（LightGBM）

```python
"""LightGBM Trainer — 与 LSTM Trainer 完全不同的训练逻辑。"""
from __future__ import annotations

import logging
from pathlib import Path

from engine.io.paths import PathManager
from engine.training.base import BaseTrainer
from engine.training.results import (
    SelectionResult, TestResult, TrainingRunResult,
)

logger = logging.getLogger(__name__)


class LGBTrainer(BaseTrainer):
    """Trainer for LightGBM — no epochs, no GPU, no AMP."""

    name = "lgb"

    def run(self, config: dict, paths: PathManager) -> TrainingRunResult:
        model_cfg = config.get("model", {})
        train_cfg = config.get("training", {})
        model_name = model_cfg.get("name", "lgb")
        seed = train_cfg.get("seed", 42)

        # ---- 共享工具：直接继承使用 ----
        self.set_seed(seed)
        device = self.resolve_device()
        exp_name, run_dir, ckpt_dir, log_path = self.setup_experiment(
            config, paths, model_name,
        )

        # 解析 model + view
        from engine.models.registry import get_model_class, get_view_class
        ModelClass = get_model_class(model_name)
        ViewClass = get_view_class(model_name)

        model_wrapper = ModelClass()
        view_builder = ViewClass()

        self.run_preflight(model_name, model_wrapper.contract, config)

        # 加载 view
        view_dir = paths.memmap_dir(view_builder.name)
        meta = self.load_view_meta(view_dir)

        # Split
        split = self.build_split(meta, config, lookback=0)

        # ---- 以下是 LGB 专属逻辑 ----
        D = meta["D"]
        model_wrapper.build_model(D, config)

        train_data = view_builder.get_dataset(view_dir, split, "train", config)
        valid_data = view_builder.get_dataset(view_dir, split, "valid", config)

        result = model_wrapper.fit(train_data, valid_data, config)

        model_path = ckpt_dir / "best.pkl"
        model_wrapper.save(model_path)

        selection = SelectionResult(
            best_epoch=1,
            best_iteration=result.get("best_iteration", 0),
            valid_metrics={},
            model_path=model_path,
        )

        return TrainingRunResult(
            selection=selection,
            test=None,
            config=config,
            exp_name=exp_name,
            run_dir=run_dir,
        )
```

对比 LSTM 的 trainer，你可以看到：
- **共享部分**（`set_seed`, `setup_experiment`, `run_preflight`, `load_view_meta`, `build_split`）完全一样
- **专属部分**完全不同：LGB 没有 epoch 循环、没有 AMP、没有 gate schedule、没有 early stopping 回调

### 3.4 第 4 步：注册

编辑 `engine/models/registry.py`：

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

Registry 是一个内存字典，不会生成任何文件。`register_model()` 将 `(Model, ViewBuilder, Trainer)` 三元组绑定到模型名上。之后框架通过 `get_model_class()` / `get_view_class()` / `get_trainer_class()` 查找。

### 3.5 第 5 步：写 config

```yaml
model:
  name: "lgb"
  params:
    num_leaves: 128
  label_roles:
    regression: "return_c0c1"
  view: {}

training:
  # 通用部分不变...
  epochs: 100
  lr: 2.0e-4
  ...

  # --- lgb 专属 (替换掉原来的 lstm_mtl 专属区域) ---
  n_estimators: 500
  early_stopping_rounds: 50
```

---

## 4. 规则清单（必须遵守）

### 4.1 Model 的规则

| 规则 | 说明 |
|------|------|
| **继承 `BaseModel`** | 位于 `engine/models/base.py`，必须实现所有 6 个抽象方法 |
| **声明 `name`** | 必须与 config 中的 `model.name`、registry 注册名一致 |
| **声明 `contract`** | 必须声明 `ModelLabelContract`，推荐 `ROLE_BASED` 模式 |
| **从 `model.params` 读架构参数** | `config.get("model", {}).get("params", {})`，不要硬编码 |
| **不要自己读数据** | model 不知道 panel_base 的存在。数据通过 DataLoader 传入 |
| **放在 `impl/` 子目录中** | `engine/models/impl/<model_name>/model.py` |

### 4.2 Trainer 的规则

| 规则 | 说明 |
|------|------|
| **继承 `BaseTrainer`** | 位于 `engine/training/base.py`，必须实现 `run()` 方法 |
| **声明 `name`** | 必须与模型名一致 |
| **放在 `impl/` 子目录中** | `engine/training/impl/<model_name>/trainer.py` |
| **用基类的共享工具** | `set_seed`, `resolve_device`, `setup_experiment`, `run_preflight`, `load_view_meta`, `build_split` |
| **从 `training:` 读取所有训练参数** | 通用和专属参数统一从 `config["training"]` 读取 |
| **用 `.get(key, default)` 读参数** | 保证换模型后旧参数残留不报错 |
| **评估器放同目录** | 模型专属的评估逻辑放在 `training/impl/<name>/evaluator.py`（可选） |

### 4.3 Contract 的规则

| 规则 | 说明 |
|------|------|
| **每个 model 必须有 contract** | 没有 contract，preflight 无法校验 |
| **推荐 ROLE_BASED** | 最灵活，标签名不写死在模型代码里 |
| **角色名在 config 里映射** | `model.label_roles` 将角色映射到具体 target 名 |
| **preflight 在训练前自动运行** | 如果 contract 不满足，训练不会开始，直接报错 |

### 4.4 Config 的规则

| 规则 | 说明 |
|------|------|
| **view 必须预先构建** | trainer 不会自动 build view。如果 memmap 不存在，直接报错 |
| **时间切分互不重叠** | `[train_start, train_end)` `[valid_start, valid_end)` `[test_start, test_end)` |
| **模型架构参数放 `model.params`** | embed_dim, hidden_size 等 |
| **所有训练参数放 `training:`** | 通用的（epochs, lr）和模型专属的（ret_weight, gate_l1_max）都放这里 |
| **模型专属参数用注释分隔** | `# --- lstm_mtl 专属 ---`，换模型时替换这个区域 |

### 4.5 命名规则

```
model name = "lstm_mtl"

→ engine/models/impl/lstm_mtl/model.py      # LSTMMTLModel(BaseModel)
→ engine/views/impl/lstm_mtl/view.py         # LSTMViewBuilder(BaseViewBuilder)
→ engine/training/impl/lstm_mtl/trainer.py   # LSTMMTLTrainer(BaseTrainer)
→ engine/training/impl/lstm_mtl/evaluator.py # 评估工具（可选）
→ registry: register_model("lstm_mtl", Model, View, Trainer)
→ config: model.name = "lstm_mtl"
→ view 输出: data/processed/views/lstm_mtl/memmap/
→ 实验输出: data/training_result/<exp_name>/
```

**六处名字必须一致**：model.name 属性、view.name 属性、trainer.name 属性、registry 注册名、config 中的 model.name、目录名。

---

## 5. 训练架构详解

### 5.1 调度流程

```
train.py
  └─ run_training(config, paths)                    # training/trainer.py
       └─ registry.get_trainer_class("lstm_mtl")     # 查注册表
            └─ LSTMMTLTrainer.run(config, paths)     # training/impl/lstm_mtl/trainer.py
                 ├─ self.set_seed()                  # ← BaseTrainer
                 ├─ self.resolve_device()             # ← BaseTrainer
                 ├─ self.setup_experiment()            # ← BaseTrainer
                 ├─ self.run_preflight()               # ← BaseTrainer
                 ├─ self.load_view_meta()               # ← BaseTrainer
                 ├─ self.build_split()                  # ← BaseTrainer
                 ├─ self._run_selection()               # ← LSTM 专属
                 ├─ self._run_refit()                   # ← LSTM 专属
                 └─ test evaluation                     # ← LSTM 专属
```

`training/trainer.py` 只是一个薄调度层——读 config 中的 `model.name`，查 registry 找到对应的 Trainer 子类，调用 `run()`。所有实际逻辑在子类中。

### 5.2 LSTM-MTL 的 Selection 阶段

```
for epoch in range(1, epochs+1):
    1. 创建 train/valid DataLoader（每 epoch 重新采样）
    2. 计算 gate_lambda（warmup → linear ramp）
    3. train_one_epoch:
       - forward: X → model → (pred_ret, mom_logits)
       - regression_loss = MSE 或 LambdaRank NDCG（由 use_lambdarank 控制）
       - loss = ret_weight * regression_loss + ce_weight * CE + gate_lambda * gate_L1
       - backward + gradient clipping + optimizer.step
    4. eval_one_epoch (valid):
       - 计算 val_loss, IC, RankIC, accuracy
    5. early stopping:
       - val_loss 改善 → 保存 best.pt
       - 连续 patience 个 epoch 无改善 → 停止
    6. 写 log.csv
```

### 5.3 LSTM-MTL 的 Refit 阶段（可选）

```
1. 重新初始化模型权重（从零开始）
2. 用 train+valid 合并数据训练
3. 训练 epoch 数 = selection 阶段的 best_epoch（冻结）
4. 保存 refit.pt
```

### 5.4 Test 阶段

```
1. 加载 best.pt（selection）或 refit.pt（refit）
2. 在 test 集上跑 eval_one_epoch
3. 报告 loss, IC, RankIC, accuracy
```

### 5.5 回归损失选择：MSE vs LambdaRank

LSTM-MTL 的回归任务支持两种损失函数，通过 `training.use_lambdarank` 切换：

| 损失 | 配置值 | 特点 | 适用场景 |
|------|--------|------|---------|
| **MSE** | `use_lambdarank: false` | 直接优化预测精度 | 默认，通用回归 |
| **LambdaRank NDCG** | `use_lambdarank: true` | 优化排序质量（ΔNDCG@k 加权 pairwise 损失） | 更关注截面排序（选股排名） |

**LambdaRank 工作流程**：
1. `returns_to_relevance()` 将连续收益率按每日排名分箱为离散相关性等级（0..n_bins-1）
2. `lambdarank_ndcg_loss()` 计算 ΔNDCG@k 加权的 pairwise logistic 损失
3. 只对 rel_i > rel_j 的有效配对计算，权重为 |ΔGain| × |ΔDiscount| / IDCG

相关参数（在 `training:` lstm_mtl 专属区域）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `use_lambdarank` | `false` | 是否启用 LambdaRank |
| `lambdarank_k` | `50` | NDCG@k 的 k 值 |
| `lambdarank_sigma` | `1.0` | pairwise 差异缩放因子 |
| `lambdarank_bins` | `5` | 收益离散化分箱数 |

> 注意：LambdaRank 在 selection、refit、test 三个阶段都会生效（保持一致性）。

### 5.6 LSTM-MTL 的评估指标

| 指标 | 含义 | 计算方式 |
|------|------|---------|
| **IC** | 信息系数 | 每天的 Pearson(predicted_return, actual_return)，取平均 |
| **RankIC** | 秩信息系数 | 每天的 Spearman(predicted_return, actual_return)，取平均 |
| **cls_acc** | 分类准确率 | 有效标签中预测正确的比例 |
| **ret_loss** | 回归损失 | MSE 或 LambdaRank NDCG（取决于 `use_lambdarank`） |
| **ce_loss** | 分类损失 | CE with ignore_index=-1（只在 valid 标签上计算） |

> 其他模型的评估指标由其自己的 evaluator 定义，不受此限制。

### 5.7 因子选择（Feature Filtering）

LSTM-MTL 支持**两个层级**的因子筛选：

| 层级 | 配置位置 | 何时生效 | 改了需要重建 view 吗 |
|------|---------|---------|-------------------|
| **Build-time** | `model.view` | `build_view.py` 构建 memmap 时 | **需要**（`--force` 重建） |
| **Training-time** | `training` | 训练时 Dataset 读取数据时 | **不需要** |

#### Build-time 筛选（缩小 memmap 体积）

```yaml
model:
  view:
    include_pattern: "^feature\\.alpha158__"   # 只保留 alpha158 因子
    exclude_pattern: "^feature\\.fund__"        # 排除 fund 因子
```

memmap 构建时直接跳过不需要的因子列，减少磁盘占用和 I/O。适合你确定某些因子永远不用的场景。

#### Training-time 筛选（灵活实验，不重建 view）

```yaml
training:
  # --- lstm_mtl 专属 ---
  include_pattern: "^feature\\.alpha158__"   # 只包含 alpha158 因子
  exclude_pattern: "^feature\\.fund__"        # 排除 fund 因子
```

memmap 保留全部因子，但 Dataset 在 `__getitem__()` 时按索引切片，只返回匹配的因子。适合快速实验不同因子组合。

#### 工作原理（Training-time）

1. memmap 包含全部 F 个因子（通用）
2. `LSTMViewBuilder.resolve_feature_filter()` 读取 `training` 中的正则模式
3. `compute_feature_indices()` 计算要保留的维度索引（feature + 对应 isna_flag + row_present）
4. `MemmapDayWindowDataset.__getitem__()` 在返回数据时按索引切片
5. Trainer 通过 `view_builder.get_effective_input_dim()` 获取过滤后的 D 来构建模型

**因子选择是模型专属功能**——它依赖于具体 view 的数据布局。其他模型需要根据自己的 view 结构实现类似功能。

---

## 6. 框架提供的工具

### 6.1 训练模块 — `engine/training/`

| 文件 | 作用 |
|------|------|
| `base.py` | `BaseTrainer` 抽象基类，提供共享工具方法 |
| `trainer.py` | 薄调度层：查 registry → 调用对应 Trainer 的 `run()` |
| `splitter.py` | `build_split_bundle()`：日期 → memmap 索引的转换 |
| `callbacks.py` | `EarlyStopping`、`gate_lambda_schedule()` |
| `preflight.py` | `validate_contract()`：训练前校验 model-label contract |
| `results.py` | `SelectionResult`、`TestResult`、`TrainingRunResult` 数据类 |
| `impl/lstm_mtl/trainer.py` | LSTM-MTL 专属训练循环 |
| `impl/lstm_mtl/evaluator.py` | LSTM-MTL 专属评估（IC, RankIC, compute_losses） |

### 6.2 模型模块 — `engine/models/`

| 文件 | 作用 |
|------|------|
| `base.py` | `BaseModel` 抽象基类，re-export `ModelLabelContract` |
| `registry.py` | `register_model()`、`get_model_class()`、`get_view_class()`、`get_trainer_class()` |
| `impl/lstm_mtl/model.py` | LSTM 多任务模型实现（参考实现） |

### 6.3 Contract 模块 — `engine/schema/contracts.py`

```python
@dataclass
class ModelLabelContract:
    mode: str          # "ANY_SINGLE" | "ANY_MULTI" | "EXACT" | "ROLE_BASED" | "CUSTOM"
    min_labels: int = 1
    max_labels: int = 1
    exact_labels: List[str] = []       # EXACT 模式用
    required_roles: Dict[str, str] = {} # ROLE_BASED 模式用
    custom_validator: Callable = None    # CUSTOM 模式用
```

---

## 7. 训练输出说明

### 7.1 目录结构

```
data/training_result/<exp_name>/
├── checkpoints/
│   ├── best.pt              # selection 阶段的最佳模型
│   └── refit.pt             # refit 后的模型（可选）
├── log.csv                  # 每 epoch 的训练/验证指标
└── config.json              # 本次实验的完整配置快照
```

### 7.2 log.csv 格式（LSTM-MTL）

```csv
epoch,train_loss,train_ret,train_ce,gate_lam,train_gate,val_loss,val_ret,val_ce,val_ic,val_rankic,val_acc,epoch_time_s
1,0.01234567,0.00567890,0.00666677,0.00e+00,0.00000000,0.01345678,0.00612345,0.00733333,0.0123,0.0098,0.2345,45.67
```

> 其他模型的 log.csv 格式由其 Trainer 自行决定。

### 7.3 best.pt 内容（LSTM-MTL）

```python
{
    "epoch": 42,
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "best_val_loss": 0.01234,
}
```

---

## 8. 已有的模型

### 8.1 lstm_mtl（LSTM 多任务）

| 项目 | 内容 |
|------|------|
| Model | `engine/models/impl/lstm_mtl/model.py` |
| View | `engine/views/impl/lstm_mtl/view.py` |
| Trainer | `engine/training/impl/lstm_mtl/trainer.py` |
| Evaluator | `engine/training/impl/lstm_mtl/evaluator.py` |
| Contract | `ROLE_BASED`，需要 regression + classification 两个角色 |
| 架构 | InputFeatureGating → Linear → LSTM → regression head + classification head |
| 架构参数 | `embed_dim`, `hidden_size`, `num_layers`, `dropout` 等（在 `model.params` 中） |
| 训练专属参数 | `use_lambdarank`, `lambdarank_k/sigma/bins`, `ret_weight`, `ce_weight`, `gate_l1_max`, `gate_warmup/ramp_epochs`（在 `training:` 中） |
| View 参数 | `lookback`, `k`（在 `model.view` 中） |
| 因子筛选 | build-time: `model.view.include/exclude_pattern`; training-time: `training.include/exclude_pattern` |

---

## 9. 常见问题

**Q：跑 train.py 报 "Pre-built memmap view not found"？**

A：先跑 `python scripts/build_view.py --config configs/default.yaml`。

**Q：跑 train.py 报 ContractViolation？**

A：检查 config 中的 `model.label_roles` 是否满足模型的 contract。
比如 lstm_mtl 要求 regression + classification 两个角色，但你只配了一个。

**Q：我的模型不用 PyTorch，还能用这个框架吗？**

A：可以。写自己的 Trainer 子类即可。`training:` 下的通用参数（epochs, lr 等）你的 Trainer 可以直接忽略不读，不会报错。`to_device()` 可以空实现。

**Q：如何跳过 test evaluation？**

A：config 中设置 `evaluation.run_test: false`。

**Q：refit 是什么？什么时候该用？**

A：refit 是用 train+valid 合并数据重新训练模型（训练 epoch 数固定为 selection 阶段的最佳 epoch）。
目的是利用更多数据获得更好的最终模型。设置 `evaluation.refit_before_test: true` 启用。

**Q：如何恢复一个已训练好的模型？**

A：
```python
from engine.models.impl.lstm_mtl.model import LSTMMTLModel

model = LSTMMTLModel()
model.build_model(input_dim=947, config=cfg)
model.load(Path("data/training_result/my_exp/checkpoints/best.pt"))
```

**Q：换模型后 config 里的旧参数会报错吗？**

A：不会。所有代码用 `.get(key, default)` 读参数，找不到 key 就用默认值。旧参数留着不报错，但建议删掉保持整洁。换模型时只需要：
1. 替换 `model:` 块（name, params, label_roles, view）
2. 替换 `training:` 中 `--- xxx 专属 ---` 注释之间的区域

**Q：因子选择功能所有模型都能用吗？**

A：不是。因子选择（`include_pattern` / `exclude_pattern`）是 LSTM-MTL 的 ViewBuilder 实现的功能，因为它依赖 memmap 的 `[features, isna_flags, row_present]` 布局。其他模型需要根据自己的 view 结构实现类似功能。
