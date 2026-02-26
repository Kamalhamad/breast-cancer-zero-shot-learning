"""Evaluation helpers and artifact writers."""

from __future__ import annotations

import json
from pathlib import Path

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)


def evaluate_model(y_true, y_pred):
    """Evaluate classification performance."""
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    return accuracy, cm, report


def save_metrics_json(path: Path, payload: dict) -> None:
    """Save metrics payload as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_report_text(path: Path, report_text: str) -> None:
    """Save classification report text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_text, encoding="utf-8")


def save_confusion_matrix_figure(path: Path, cm, labels=None) -> None:
    """Save confusion matrix figure as PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt

        display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        display.plot(cmap="Blues", values_format="d")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
    except ModuleNotFoundError:
        fallback = path.with_suffix(".txt")
        fallback.write_text(str(cm), encoding="utf-8")
