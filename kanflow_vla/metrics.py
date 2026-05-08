"""
Reusable metric helpers for KANFlow-VLA / RCAR-VLA training and evaluation.

Keeps ``train.py`` and ``eval_rcar.py`` focused on control flow while this
module owns the math.
"""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as F


# ── Classification metrics ──────────────────────────────────────────────────

def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Top-1 accuracy for a batch of logits vs integer labels.

    Args:
        logits: ``(B, C)`` raw class logits.
        labels: ``(B,)`` ground-truth class indices.

    Returns:
        Accuracy as a Python float in [0, 1].
    """
    if logits.numel() == 0 or labels.numel() == 0:
        return 0.0
    preds = logits.argmax(dim=-1)
    return float((preds == labels).float().mean().item())


def per_class_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> dict[int, float]:
    """Per-class accuracy breakdown.

    Returns a dict mapping ``class_id -> accuracy`` (0.0 if no samples).
    """
    preds = logits.argmax(dim=-1)
    result: dict[int, float] = {}
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() == 0:
            result[c] = 0.0
        else:
            result[c] = float((preds[mask] == c).float().mean().item())
    return result


# ── Regression metrics ──────────────────────────────────────────────────────

def mae(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Mean Absolute Error.

    Args:
        predictions: ``(B,)`` predicted values.
        targets:     ``(B,)`` ground-truth values.

    Returns:
        MAE as a Python float.
    """
    if predictions.numel() == 0:
        return 0.0
    return float((predictions - targets).abs().mean().item())


def mse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Mean Squared Error."""
    if predictions.numel() == 0:
        return 0.0
    return float((predictions - targets).pow(2).mean().item())


# ── Calibration / entropy ───────────────────────────────────────────────────

def prediction_entropy(logits: torch.Tensor) -> float:
    """Mean entropy of a batch of categorical logits.

    Higher entropy → more uncertain predictions.  Useful for monitoring
    whether ``mode_head`` is confidently separating act/ask/abstain/recover.
    """
    if logits.numel() == 0:
        return 0.0
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return float(entropy.mean().item())


# ── RCAR-specific aggregate metrics ────────────────────────────────────────

def rcar_score(
    task_success: float,
    language_fidelity: float,
    recovery_success: float,
    clarification_success: float,
    unsafe_action_rate: float,
) -> float:
    """Compute the composite RCAR score from the implementation plan.

    rcar = 0.30·task_success + 0.20·lang_fidelity + 0.20·recovery_success
         + 0.15·clarification_success + 0.15·(1 − unsafe_action_rate)
    """
    return (
        0.30 * task_success
        + 0.20 * language_fidelity
        + 0.20 * recovery_success
        + 0.15 * clarification_success
        + 0.15 * (1.0 - unsafe_action_rate)
    )


def correction_latency(step_counts: list[int]) -> dict[str, float]:
    """Summarise correction-to-success latency across episodes.

    Args:
        step_counts: list of step counts from correction point to task success
                     (only for episodes that eventually succeeded).

    Returns:
        Dict with ``mean``, ``median``, ``max``, and ``count``.
    """
    if not step_counts:
        return {"mean": 0.0, "median": 0.0, "max": 0.0, "count": 0}
    import statistics
    return {
        "mean": statistics.mean(step_counts),
        "median": statistics.median(step_counts),
        "max": float(max(step_counts)),
        "count": len(step_counts),
    }


# ── Accumulator for epoch-level metrics ─────────────────────────────────────

class MetricAccumulator:
    """Running mean accumulator for logging.

    Usage::

        acc = MetricAccumulator()
        for batch in loader:
            acc.update({"loss": loss_val, "mode_acc": mode_acc})
        print(acc.compute())  # {"loss": ..., "mode_acc": ...}
    """

    def __init__(self):
        self._sums: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)

    def update(self, metrics: dict[str, float | torch.Tensor]) -> None:
        for k, v in metrics.items():
            val = float(v) if isinstance(v, torch.Tensor) else v
            self._sums[k] += val
            self._counts[k] += 1

    def compute(self) -> dict[str, float]:
        return {
            k: self._sums[k] / max(self._counts[k], 1)
            for k in self._sums
        }

    def reset(self) -> None:
        self._sums.clear()
        self._counts.clear()
