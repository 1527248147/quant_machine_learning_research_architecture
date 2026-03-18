"""
LSTM Multi-Task Trainer.

Contains the full training loop specific to the LSTM-MTL model:
    - Multi-task loss (MSE regression + CE classification + gate L1)
    - Gate lambda warmup/ramp schedule
    - IC / RankIC / accuracy evaluation
    - Selection (train on train, select on valid)
    - Refit (retrain on train+valid with frozen epoch count)
    - Test evaluation

Other models (e.g., LightGBM) would implement their own Trainer subclass
with completely different training logic.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import torch
from torch.optim import AdamW

from engine.io.paths import PathManager
from engine.training.base import BaseTrainer
from engine.training.callbacks import EarlyStopping, gate_lambda_schedule
from engine.training.results import (
    SelectionResult,
    TestResult,
    TrainingRunResult,
)

from engine.training.impl.lstm_mtl.evaluator import EvalMetrics, compute_losses, eval_one_epoch
from engine.models.impl.lstm_mtl.model import LSTMMultiTask

logger = logging.getLogger(__name__)


# ======================================================================
# Single training epoch (LSTM-specific)
# ======================================================================

def train_one_epoch(
    model: torch.nn.Module,
    dl,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device,
    use_amp: bool,
    ret_w: float,
    ce_w: float,
    gate_lam: float,
    grad_clip: float,
    log_interval: int = 200,
    max_batches: Optional[int] = None,
    use_lambdarank: bool = False,
    lambdarank_k: int = 50,
    lambdarank_sigma: float = 1.0,
    lambdarank_bins: int = 5,
) -> dict:
    """Run one LSTM-MTL training epoch. Returns dict with loss statistics."""
    model.train()

    total_loss = 0.0
    total_ret = 0.0
    total_ce = 0.0
    total_gate = 0.0
    total_batches = 0

    amp_ctx = torch.amp.autocast(
        device_type="cuda", dtype=torch.float16,
        enabled=(use_amp and device.type == "cuda"),
    )
    optimizer.zero_grad(set_to_none=True)

    t_epoch = time.time()

    for bi, batch in enumerate(dl):
        if max_batches is not None and bi >= max_batches:
            break

        X = batch["X"].to(device, non_blocking=True)
        y_ret = batch["y_ret"].to(device, non_blocking=True)
        y_mom = batch["y_mom"].to(device, non_blocking=True)
        ret_mask = batch["ret_mask"].to(device, non_blocking=True)
        mom_mask = batch["mom_mask"].to(device, non_blocking=True)

        B, K, L, D = X.shape
        x_flat = X.reshape(B * K, L, D)

        with amp_ctx:
            pred_ret_flat, mom_logits_flat = model(x_flat)
            pred_ret = pred_ret_flat.reshape(B, K)
            mom_logits = mom_logits_flat.reshape(B, K, -1)

            ret_loss, ce_loss, _ = compute_losses(
                pred_ret, mom_logits, y_ret, y_mom, ret_mask, mom_mask,
                use_lambdarank=use_lambdarank,
                lambdarank_k=lambdarank_k,
                lambdarank_sigma=lambdarank_sigma,
                lambdarank_bins=lambdarank_bins,
            )

            gate_loss = (
                model.gate_l1(reduction="mean")
                if hasattr(model, "gate_l1")
                else torch.tensor(0.0, device=device)
            )

            loss = ret_w * ret_loss + ce_w * ce_loss + gate_lam * gate_loss

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)

        total_loss += float(loss.item())
        total_ret += float(ret_loss.item())
        total_ce += float(ce_loss.item())
        total_gate += float(gate_loss.item())
        total_batches += 1

        if log_interval > 0 and (bi + 1) % log_interval == 0:
            n = total_batches
            logger.info(
                "  [TRAIN] step=%d  loss=%.6f (ret=%.6f ce=%.6f gate=%.6f)",
                bi + 1,
                total_loss / n, total_ret / n, total_ce / n, total_gate / n,
            )

    dt = time.time() - t_epoch
    n = max(total_batches, 1)

    return {
        "loss": total_loss / n,
        "ret_loss": total_ret / n,
        "ce_loss": total_ce / n,
        "gate_l1": total_gate / n,
        "epoch_time_s": dt,
    }


# ======================================================================
# LSTM-MTL Trainer
# ======================================================================

class LSTMMTLTrainer(BaseTrainer):
    """Trainer for the LSTM multi-task model."""

    name = "lstm_mtl"

    # ------------------------------------------------------------------
    # Selection stage
    # ------------------------------------------------------------------

    def _run_selection(
        self,
        model: torch.nn.Module,
        view_builder,
        view_dir: Path,
        split,
        config: dict,
        device: torch.device,
        ckpt_dir: Path,
        log_path: Path,
    ) -> SelectionResult:
        """Train on train set, select best model on valid set."""
        train_cfg = config.get("training", {})
        epochs = train_cfg.get("epochs", 100)
        lr = train_cfg.get("lr", 2e-4)
        weight_decay = train_cfg.get("weight_decay", 1e-3)
        grad_clip = train_cfg.get("grad_clip", 1.0)
        patience = train_cfg.get("patience", 10)
        ret_w = train_cfg.get("ret_weight", 1.0)
        ce_w = train_cfg.get("ce_weight", 1.0)
        gate_l1_max = train_cfg.get("gate_l1_max", 5e-4)
        gate_warmup = train_cfg.get("gate_warmup_epochs", 5)
        gate_ramp = train_cfg.get("gate_ramp_epochs", 20)
        log_interval = train_cfg.get("log_interval", 200)
        use_amp = train_cfg.get("amp", False) and device.type == "cuda"
        shuffle = train_cfg.get("shuffle", True)
        use_lambdarank = train_cfg.get("use_lambdarank", False)
        lambdarank_k = train_cfg.get("lambdarank_k", 50)
        lambdarank_sigma = train_cfg.get("lambdarank_sigma", 1.0)
        lambdarank_bins = train_cfg.get("lambdarank_bins", 5)

        if use_lambdarank:
            logger.info("Regression loss: LambdaRank NDCG@%d (sigma=%.1f, bins=%d)",
                        lambdarank_k, lambdarank_sigma, lambdarank_bins)
        else:
            logger.info("Regression loss: MSE")

        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if device.type == "cuda" else None
        es = EarlyStopping(patience=patience)

        # CSV log header
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(
                "epoch,train_loss,train_ret,train_ce,gate_lam,train_gate,"
                "val_loss,val_ret,val_ce,val_ic,val_rankic,val_acc,epoch_time_s\n"
            )

        best_model_path = ckpt_dir / "best.pt"
        va = None

        for epoch in range(1, epochs + 1):
            t_epoch = time.time()

            train_ds = view_builder.get_dataset(
                view_dir, split.train_start_idx, split.train_end_idx, config,
            )
            valid_ds = view_builder.get_dataset(
                view_dir, split.valid_start_idx, split.valid_end_idx, config,
            )
            train_dl = view_builder.build_dataloader(train_ds, config, shuffle=shuffle)
            valid_dl = view_builder.build_dataloader(valid_ds, config, shuffle=False)

            lam = gate_lambda_schedule(epoch, gate_warmup, gate_ramp, gate_l1_max)

            logger.info("=" * 60)
            logger.info("EPOCH %d / %d  (gate_lam=%.2e)", epoch, epochs, lam)

            tr = train_one_epoch(
                model=model,
                dl=train_dl,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                use_amp=use_amp,
                ret_w=ret_w,
                ce_w=ce_w,
                gate_lam=lam,
                grad_clip=grad_clip,
                log_interval=log_interval,
                use_lambdarank=use_lambdarank,
                lambdarank_k=lambdarank_k,
                lambdarank_sigma=lambdarank_sigma,
                lambdarank_bins=lambdarank_bins,
            )

            va = eval_one_epoch(
                model, valid_dl, device, use_amp,
                use_lambdarank=use_lambdarank,
                lambdarank_k=lambdarank_k,
                lambdarank_sigma=lambdarank_sigma,
                lambdarank_bins=lambdarank_bins,
            )

            dt_epoch = time.time() - t_epoch

            logger.info(
                "  Train: loss=%.6f (ret=%.6f ce=%.6f gate=%.6f) %.1fs",
                tr["loss"], tr["ret_loss"], tr["ce_loss"], tr["gate_l1"],
                tr["epoch_time_s"],
            )
            logger.info(
                "  Valid: loss=%.6f (ret=%.6f ce=%.6f) IC=%.4f RankIC=%.4f Acc=%.4f %.1fs",
                va.total_loss, va.ret_loss, va.ce_loss,
                va.ic, va.rankic, va.cls_acc, va.eval_time_s,
            )

            with open(log_path, "a", encoding="utf-8") as f:
                f.write(
                    f"{epoch},{tr['loss']:.8f},{tr['ret_loss']:.8f},{tr['ce_loss']:.8f},"
                    f"{lam:.6e},{tr['gate_l1']:.8f},"
                    f"{va.total_loss:.8f},{va.ret_loss:.8f},{va.ce_loss:.8f},"
                    f"{va.ic:.8f},{va.rankic:.8f},{va.cls_acc:.8f},"
                    f"{dt_epoch:.2f}\n"
                )

            state = es.step(va.total_loss, epoch)
            if state.best_epoch == epoch:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "best_val_loss": state.best_val,
                    },
                    str(best_model_path),
                )
                logger.info("  ** NEW BEST: epoch=%d val_loss=%.6f → saved", epoch, state.best_val)
            else:
                logger.info(
                    "  patience %d/%d (best: epoch %d, val_loss=%.6f)",
                    state.bad_epochs, patience, state.best_epoch, state.best_val,
                )

            if state.should_stop:
                logger.info("Early stopping at epoch %d", epoch)
                break

        # Restore best model
        if best_model_path.exists():
            ckpt = torch.load(str(best_model_path), map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            logger.info("Restored best model from epoch %d", ckpt["epoch"])

        return SelectionResult(
            best_epoch=es.state.best_epoch,
            best_iteration=es.state.best_epoch,
            valid_metrics=va.as_dict() if va else {},
            model_path=best_model_path,
        )

    # ------------------------------------------------------------------
    # Refit stage
    # ------------------------------------------------------------------

    def _run_refit(
        self,
        model: torch.nn.Module,
        view_builder,
        view_dir: Path,
        split,
        selection: SelectionResult,
        config: dict,
        device: torch.device,
        ckpt_dir: Path,
    ) -> tuple:
        """Retrain on train+valid with frozen epoch count. Returns (refit_model, refit_path)."""
        train_cfg = config.get("training", {})
        lr = train_cfg.get("lr", 2e-4)
        weight_decay = train_cfg.get("weight_decay", 1e-3)
        grad_clip = train_cfg.get("grad_clip", 1.0)
        ret_w = train_cfg.get("ret_weight", 1.0)
        ce_w = train_cfg.get("ce_weight", 1.0)
        gate_l1_max = train_cfg.get("gate_l1_max", 5e-4)
        gate_warmup = train_cfg.get("gate_warmup_epochs", 5)
        gate_ramp = train_cfg.get("gate_ramp_epochs", 20)
        log_interval = train_cfg.get("log_interval", 200)
        use_amp = train_cfg.get("amp", False) and device.type == "cuda"
        shuffle = train_cfg.get("shuffle", True)
        use_lambdarank = train_cfg.get("use_lambdarank", False)
        lambdarank_k = train_cfg.get("lambdarank_k", 50)
        lambdarank_sigma = train_cfg.get("lambdarank_sigma", 1.0)
        lambdarank_bins = train_cfg.get("lambdarank_bins", 5)

        refit_epochs = selection.best_epoch
        logger.info("=" * 60)
        logger.info("REFIT: training on train+valid for %d epochs (frozen from selection)", refit_epochs)

        # Re-initialise model weights
        cfg_attr = getattr(model, "cfg", None)
        if cfg_attr is not None:
            model = LSTMMultiTask(cfg_attr).to(device)
        else:
            raise RuntimeError("Cannot re-initialise model for refit — cfg not found")

        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if device.type == "cuda" else None

        for epoch in range(1, refit_epochs + 1):
            ds = view_builder.get_dataset(
                view_dir,
                split.train_plus_valid_start_idx,
                split.train_plus_valid_end_idx,
                config,
            )
            dl = view_builder.build_dataloader(ds, config, shuffle=shuffle)
            lam = gate_lambda_schedule(epoch, gate_warmup, gate_ramp, gate_l1_max)

            tr = train_one_epoch(
                model=model,
                dl=dl,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                use_amp=use_amp,
                ret_w=ret_w,
                ce_w=ce_w,
                gate_lam=lam,
                grad_clip=grad_clip,
                log_interval=log_interval,
                use_lambdarank=use_lambdarank,
                lambdarank_k=lambdarank_k,
                lambdarank_sigma=lambdarank_sigma,
                lambdarank_bins=lambdarank_bins,
            )
            logger.info(
                "  REFIT epoch %d/%d  loss=%.6f (ret=%.6f ce=%.6f) %.1fs",
                epoch, refit_epochs, tr["loss"], tr["ret_loss"], tr["ce_loss"],
                tr["epoch_time_s"],
            )

        refit_path = ckpt_dir / "refit.pt"
        torch.save({"model_state": model.state_dict(), "refit_epochs": refit_epochs}, str(refit_path))
        logger.info("Refit model saved → %s", refit_path)
        return model, refit_path

    # ------------------------------------------------------------------
    # Full pipeline (implements BaseTrainer.run)
    # ------------------------------------------------------------------

    def run(
        self,
        config: dict,
        paths: PathManager,
    ) -> TrainingRunResult:
        """
        Full LSTM-MTL training pipeline.

        1. Preflight: validate contract
        2. Locate pre-built memmap view
        3. Split: date-based train/valid/test
        4. Selection: train on train, select on valid (early stopping)
        5. Refit (optional): retrain on train+valid
        6. Test (optional): evaluate on test
        """
        train_cfg = config.get("training", {})
        eval_cfg = config.get("evaluation", {})
        model_cfg = config.get("model", {})

        model_name = model_cfg.get("name", "lstm_mtl")
        seed = train_cfg.get("seed", 42)
        run_test = eval_cfg.get("run_test", True)
        refit_before_test = eval_cfg.get("refit_before_test", False)

        self.set_seed(seed)
        device = self.resolve_device()

        # --- Experiment dir ---
        exp_name, run_dir, ckpt_dir, log_path = self.setup_experiment(
            config, paths, model_name,
        )

        # --- Resolve model + view classes (lazy import to avoid circular) ---
        from engine.models.registry import get_model_class, get_view_class

        ModelClass = get_model_class(model_name)
        ViewClass = get_view_class(model_name)

        model_wrapper = ModelClass()
        view_builder = ViewClass()

        # --- Preflight: contract check ---
        self.run_preflight(model_name, model_wrapper.contract, config)

        # --- Locate pre-built memmap view ---
        view_dir = paths.memmap_dir(view_builder.name)
        meta = self.load_view_meta(view_dir)
        logger.info("Using pre-built memmap view at %s", view_dir)

        # --- Split ---
        lookback = model_cfg.get("view", {}).get("lookback", 60)
        split = self.build_split(meta, config, lookback=lookback)

        # --- Build model (with feature filtering) ---
        if hasattr(view_builder, "get_effective_input_dim"):
            D = view_builder.get_effective_input_dim(meta, config)
        else:
            D = meta["D"]

        model_wrapper.build_model(D, config)
        model_wrapper.to_device(device)
        model = model_wrapper.model

        param_count = sum(p.numel() for p in model.parameters())
        effective_F = (D - 1) // 2
        logger.info(
            "Model built: %s  params=%d  D=%d (effective F=%d, original F=%d)",
            model_name, param_count, D, effective_F, meta["F"],
        )

        # --- Selection stage ---
        selection = self._run_selection(
            model=model,
            view_builder=view_builder,
            view_dir=view_dir,
            split=split,
            config=config,
            device=device,
            ckpt_dir=ckpt_dir,
            log_path=log_path,
        )
        logger.info(
            "Selection complete: best_epoch=%d val_loss=%.6f",
            selection.best_epoch,
            selection.valid_metrics.get("loss", float("inf")),
        )

        # --- Refit (optional) + Test ---
        test_result = None
        if run_test:
            test_model_path = selection.model_path
            model_source = "selection"

            if refit_before_test:
                model, refit_path = self._run_refit(
                    model=model,
                    view_builder=view_builder,
                    view_dir=view_dir,
                    split=split,
                    selection=selection,
                    config=config,
                    device=device,
                    ckpt_dir=ckpt_dir,
                )
                test_model_path = refit_path
                model_source = "refit"

            # --- Test evaluation ---
            logger.info("=" * 60)
            logger.info("TEST EVALUATION (model_source=%s)", model_source)

            use_amp = train_cfg.get("amp", False) and device.type == "cuda"

            test_ds = view_builder.get_dataset(
                view_dir, split.test_start_idx, split.test_end_idx, config,
            )
            test_dl = view_builder.build_dataloader(test_ds, config, shuffle=False)

            te = eval_one_epoch(
                model, test_dl, device, use_amp,
                use_lambdarank=train_cfg.get("use_lambdarank", False),
                lambdarank_k=train_cfg.get("lambdarank_k", 50),
                lambdarank_sigma=train_cfg.get("lambdarank_sigma", 1.0),
                lambdarank_bins=train_cfg.get("lambdarank_bins", 5),
            )

            logger.info(
                "  Test: loss=%.6f IC=%.4f RankIC=%.4f Acc=%.4f",
                te.total_loss, te.ic, te.rankic, te.cls_acc,
            )

            test_result = TestResult(
                model_source=model_source,
                test_metrics=te.as_dict(),
                model_path=test_model_path,
            )

        result = TrainingRunResult(
            selection=selection,
            test=test_result,
            config=config,
            exp_name=exp_name,
            run_dir=run_dir,
        )

        logger.info("=" * 60)
        logger.info("Training run complete → %s", run_dir)
        return result
