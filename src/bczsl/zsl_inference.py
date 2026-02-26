"""Inference entrypoint for baseline unseen evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np

from .config import ensure_output_dirs, load_config
from .embeddings import cosine_similarity
from .evaluation import evaluate_model, save_metrics_json


def predict_unseen(feature_vector, classifier, encoder, class_embeddings):
    """Predict unseen class using embedding similarity."""
    predicted_label_index = classifier.predict([feature_vector])[0]
    _ = encoder.inverse_transform([predicted_label_index])[0]

    best_match = None
    best_score = -1.0

    for class_name, embedding in class_embeddings.items():
        score = cosine_similarity(feature_vector, embedding)
        if score > best_score:
            best_score = score
            best_match = class_name

    return best_match


def _load_or_generate_inference_data(infer_cfg: dict, seed: int, synthetic: bool):
    features_path = Path(infer_cfg.get("features_path", ""))
    labels_path = Path(infer_cfg.get("labels_path", ""))

    if features_path.exists() and labels_path.exists():
        return np.load(features_path), np.load(labels_path)

    if not synthetic:
        raise FileNotFoundError(
            "Missing inference arrays: "
            f"{features_path} and/or {labels_path}. "
            "Use --synthetic for scaffolding run."
        )

    rng = np.random.default_rng(seed + 1)
    features = rng.normal(size=(40, 32))
    labels = np.array(["benign"] * 20 + ["malignant"] * 20)
    return features, labels


def run_inference(config_path: str, synthetic: bool = False) -> dict:
    """Run inference and save metrics JSON."""
    config = load_config(config_path)
    seed = int(config.get("seed", 42))

    ensure_output_dirs(config)
    train_cfg = config.get("train", {})
    infer_cfg = config.get("inference", {})
    out_cfg = config.get("outputs", {})

    model_path = Path(train_cfg.get("model_path", "models/logreg_model.joblib"))
    encoder_path = Path(train_cfg.get("encoder_path", "models/label_encoder.joblib"))
    if not model_path.exists() or not encoder_path.exists():
        raise FileNotFoundError("Missing model artifacts. Run training first.")

    classifier = joblib.load(model_path)
    encoder = joblib.load(encoder_path)

    features, labels = _load_or_generate_inference_data(infer_cfg, seed, synthetic)

    y_true = encoder.transform(labels)
    y_pred = classifier.predict(features)
    accuracy, cm, _ = evaluate_model(y_true, y_pred)

    metrics_dir = Path(out_cfg.get("metrics_dir", "reports/metrics"))
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = metrics_dir / out_cfg.get(
        "inference_metrics_file", "inference_metrics.json"
    )

    payload = {
        "seed": seed,
        "synthetic_mode": synthetic,
        "n_samples": int(features.shape[0]),
        "n_features": int(features.shape[1]),
        "inference_accuracy": float(accuracy),
        "confusion_matrix": cm.tolist(),
    }
    save_metrics_json(metrics_file, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run baseline inference")
    parser.add_argument(
        "--config", default="configs/baseline.yaml", help="Path to config YAML"
    )
    parser.add_argument(
        "--synthetic", action="store_true", help="Run with synthetic scaffolding data"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_inference(args.config, synthetic=args.synthetic)
    print(f"Inference complete. Accuracy={result['inference_accuracy']:.4f}")


if __name__ == "__main__":
    main()
