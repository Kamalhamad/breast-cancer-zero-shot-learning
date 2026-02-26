# Case Study — Breast Cancer Zero-Shot Learning

## Problem
Breast cancer histopathology classification often lacks labeled examples for all clinically relevant subtypes.
This project explores whether class semantics can help generalize to unseen classes.

## Approach
1. Extract visual features from images using ResNet50.
2. Learn class semantics with Word2Vec embeddings.
3. Train a seen-class classifier.
4. Map outputs toward unseen classes via embedding similarity.

## Engineering upgrades completed
- Standardized Python package structure under `src/bczsl`.
- Added config-driven execution (`configs/baseline.yaml`).
- Added deterministic seed handling.
- Added artifact scaffolding (metrics JSON/text + confusion matrix figure).
- Added tests and CI quality checks.

## What I would do next
- Add experiment tracking (MLflow or structured CSV logs).
- Add ablation studies across encoders and similarity metrics.
- Add calibration analysis and thresholding strategy.
- Add a lightweight demo UI.
