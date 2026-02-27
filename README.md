# Breast Cancer Zero-Shot Learning

Portfolio-ready implementation of an **inductive zero-shot learning (ZSL)** pipeline for breast cancer histopathology classification.

## Milestone Status
- ✅ **Milestone A:** Package restructure to `src/bczsl` + import hygiene.
- ✅ **Milestone B:** Config-driven CLI, deterministic seeds, and metrics artifact scaffolding.
- ✅ **Milestone C:** Unit tests + lint/format checks + GitHub Actions CI workflow.
- ✅ **Milestone D (docs subset):** Added model card, case study, and open-source license.

## Project Structure

```text
.
├── configs/
│   └── baseline.yaml
├── docs/
│   ├── PORTFOLIO_UPGRADE_PLAN.md
│   ├── case-study.md
│   └── model-card.md
├── reports/
│   ├── figures/
│   └── metrics/
├── src/
│   └── bczsl/
│       ├── __init__.py
│       ├── config.py
│       ├── data_processing.py
│       ├── embeddings.py
│       ├── evaluation.py
│       ├── feature_extraction.py
│       ├── train.py
│       └── zsl_inference.py
├── tests/
├── .github/workflows/ci.yml
├── pyproject.toml
├── requirements.txt
└── LICENSE
```

## Setup

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pytest ruff black
```

## Run Baseline (Config + CLI)

```bash
# Windows PowerShell
$env:PYTHONPATH="src"
python -m bczsl.train --config configs/baseline.yaml --synthetic
python -m bczsl.zsl_inference --config configs/baseline.yaml --synthetic
```

Artifacts are written to:
- `reports/metrics/train_metrics.json`
- `reports/metrics/inference_metrics.json`
- `reports/metrics/classification_report.txt`
- `reports/figures/confusion_matrix.png`

## Quality Checks

```bash
python -m ruff check .
python -m black --check .
PYTHONPATH=src python -m pytest -q
```

## PowerShell file-path tip
Typing a path alone (for example `reports\metrics\classification_report.txt`) is treated as a command.
Use `Get-Content <path>` or `ii <path>` instead.


## Community & Governance
- Contribution guide: `CONTRIBUTING.md`
- Security policy: `SECURITY.md`
- PR template and issue templates: `.github/`
