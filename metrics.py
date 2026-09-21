"""Classification metrics used for validation and testing."""
from __future__ import annotations

import torch


def update_confusion_matrix(
    matrix: torch.Tensor, targets: torch.Tensor, predictions: torch.Tensor
) -> None:
    classes = matrix.shape[0]
    indices = targets.to(torch.int64).cpu() * classes + predictions.to(torch.int64).cpu()
    matrix += torch.bincount(indices, minlength=classes * classes).reshape(classes, classes)


def metrics_from_confusion_matrix(matrix: torch.Tensor) -> dict[str, float]:
    matrix = matrix.to(torch.float64)
    true_positive = matrix.diag()
    support = matrix.sum(dim=1)
    predicted = matrix.sum(dim=0)
    recall = true_positive / support.clamp_min(1.0)
    precision = true_positive / predicted.clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-12)
    valid = support > 0
    accuracy = true_positive.sum() / matrix.sum().clamp_min(1.0)
    return {
        "accuracy": float(accuracy),
        "macro_precision": float(precision[valid].mean()),
        "macro_recall": float(recall[valid].mean()),
        "macro_f1": float(f1[valid].mean()),
        "balanced_accuracy": float(recall[valid].mean()),
    }
