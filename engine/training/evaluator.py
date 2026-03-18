"""
Backward-compatible re-exports.

Evaluation logic has moved to model-specific packages.
For LSTM-MTL, see ``engine.training.impl.lstm_mtl.evaluator``.
"""
from engine.training.impl.lstm_mtl.evaluator import (  # noqa: F401
    EvalMetrics,
    batch_ic_rankic,
    compute_losses,
    eval_one_epoch,
    lambdarank_ndcg_loss,
    pearson_corr,
    returns_to_relevance,
    sanitize_mom_labels,
    spearman_corr,
)
