"""Training entrypoint for seen-class baseline."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from .config import ensure_output_dirs, load_config
from .evaluation import (
    evaluate_model,
    save_confusion_matrix_figure,
    save_metrics_json,
    save_report_text,
)


def set_global_seed(seed: int) -> None:
    """Set global seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except Exception:
        pass


def train_classifier(
    features: np.ndarray, labels, max_iter: int = 1000, random_state: int = 42
):
    """Train logistic regression classifier on seen classes."""
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)

    classifier = LogisticRegression(max_iter=max_iter, random_state=random_state)
    classifier.fit(features, encoded_labels)
    return classifier, encoder


def _load_or_generate_training_data(train_cfg: dict, seed: int, synthetic: bool):
    features_path = Path(train_cfg.get("features_path", ""))
    labels_path = Path(train_cfg.get("labels_path", ""))

    if features_path.exists() and labels_path.exists():
        return np.load(features_path), np.load(labels_path)

    if not synthetic:
        raise FileNotFoundError(
            "Missing training arrays: "
            f"{features_path} and/or {labels_path}. "
            "Use --synthetic for scaffolding run."
        )

    rng = np.random.default_rng(seed)
    features = rng.normal(size=(160, 32))
    labels = np.array(["benign"] * 80 + ["malignant"] * 80)
    return features, labels


def run_training(config_path: str, synthetic: bool = False) -> dict:
    """Run training and persist baseline artifacts."""
    config = load_config(config_path)
    seed = int(config.get("seed", 42))
    set_global_seed(seed)

    metrics_dir, figures_dir = ensure_output_dirs(config)
    train_cfg = config.get("train", {})
    out_cfg = config.get("outputs", {})

    features, labels = _load_or_generate_training_data(train_cfg, seed, synthetic)
    classifier, encoder = train_classifier(
        features,
        labels,
        max_iter=int(train_cfg.get("max_iter", 1000)),
        random_state=seed,
    )

    y_true = encoder.transform(labels)
    y_pred = classifier.predict(features)
    accuracy, cm, report = evaluate_model(y_true, y_pred)

    model_path = Path(train_cfg.get("model_path", "models/logreg_model.joblib"))
    encoder_path = Path(train_cfg.get("encoder_path", "models/label_encoder.joblib"))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    encoder_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, model_path)
    joblib.dump(encoder, encoder_path)

    metrics_file = metrics_dir / out_cfg.get("train_metrics_file", "train_metrics.json")
    report_file = metrics_dir / out_cfg.get(
        "classification_report_file", "classification_report.txt"
    )
    cm_file = figures_dir / out_cfg.get("confusion_matrix_file", "confusion_matrix.png")

    payload = {
        "seed": seed,
        "synthetic_mode": synthetic,
        "n_samples": int(features.shape[0]),
        "n_features": int(features.shape[1]),
        "train_accuracy": float(accuracy),
        "model_path": str(model_path),
        "encoder_path": str(encoder_path),
    }
    save_metrics_json(metrics_file, payload)
    save_report_text(report_file, report)
    save_confusion_matrix_figure(cm_file, cm, labels=encoder.classes_)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train baseline classifier")
    parser.add_argument(
        "--config", default="configs/baseline.yaml", help="Path to config YAML"
    )
    parser.add_argument(
        "--synthetic", action="store_true", help="Run with synthetic scaffolding data"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_training(args.config, synthetic=args.synthetic)
    print(f"Training complete. Accuracy={result['train_accuracy']:.4f}")


if __name__ == "__main__":
    main()
