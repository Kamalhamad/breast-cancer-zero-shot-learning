# Breast Cancer Zero-Shot Learning: Portfolio Upgrade Plan

## 1) Repository overview (current state)

### What is working well
- Clear project objective: inductive zero-shot classification for breast cancer histopathology.
- README already explains dataset, methodology, and high-level results.
- Core pipeline modules exist for data processing, feature extraction, semantic embeddings, training, inference, and evaluation.

### Gaps that reduce portfolio readiness
1. **Package/file layout is inconsistent**
   - README shows `src/train.py`, `src/zsl_inference.py`, and `src/evaluation.py`, but files currently live in nested folders (`src/src/...`).
2. **Reproducibility is weak**
   - No pinned dependency list, no environment file, no experiment config file.
3. **Limited execution UX**
   - No CLI entry points or one-command training/evaluation workflow.
4. **No tests and little validation scaffolding**
   - Missing unit tests and integration checks for core functions.
5. **No artifacts for credibility**
   - No saved confusion matrices, logs, model cards, or sample outputs in repo.
6. **No software engineering hygiene**
   - Missing CI, linting/formatting setup, contribution guidance, and license.

---

## 2) Recommended target architecture

Use a clean Python package structure:

```text
breast-cancer-zero-shot-learning/
├── README.md
├── requirements.txt
├── pyproject.toml
├── LICENSE
├── .gitignore
├── .github/workflows/ci.yml
├── configs/
│   └── baseline.yaml
├── data/
│   ├── raw/            # excluded from git
│   ├── processed/      # excluded from git
│   └── README.md
├── models/             # excluded from git
├── reports/
│   ├── figures/
│   └── metrics/
├── src/
│   └── bczsl/
│       ├── __init__.py
│       ├── data_processing.py
│       ├── feature_extraction.py
│       ├── embeddings.py
│       ├── train.py
│       ├── zsl_inference.py
│       └── evaluation.py
├── scripts/
│   ├── run_train.sh
│   └── run_eval.sh
└── tests/
    ├── test_embeddings.py
    ├── test_data_processing.py
    └── test_inference.py
```

---

## 3) Step-by-step upgrade roadmap

## Phase 1 — Quick professionalism wins (1 day)

1. **Normalize structure**
   - Move nested `src/src/...` modules into one package path (`src/bczsl/`).
   - Update imports to absolute package imports.
2. **Strengthen README for recruiters**
   - Add “Problem → Approach → Result” one-screen summary.
   - Add architecture diagram image and confusion matrix image.
   - Add “How to reproduce baseline in <10 minutes”.
3. **Pin environment**
   - Keep `requirements.txt` curated and add `pyproject.toml` for tooling.

## Phase 2 — Reproducibility and usability (1–2 days)

4. **Add deterministic behavior**
   - Seed NumPy/TensorFlow/sklearn and document seed strategy.
5. **Add config-driven experiments**
   - Create `configs/baseline.yaml` with dataset paths, split ratios, model parameters.
   - Load config in `train.py` and `zsl_inference.py`.
6. **Create simple CLI commands**
   - Example:
     - `python -m bczsl.train --config configs/baseline.yaml`
     - `python -m bczsl.zsl_inference --config configs/baseline.yaml`

## Phase 3 — Portfolio-grade evidence (2–3 days)

7. **Add metrics and figure generation**
   - Save confusion matrices and classification reports to `reports/metrics/`.
   - Save publication-style figures to `reports/figures/`.
8. **Add experiment tracking (lightweight)**
   - Start with CSV or JSON logs (timestamp, config hash, metrics).
   - Optional: upgrade to MLflow later.
9. **Write a concise model card**
   - Include intended use, limitations, fairness risks, and failure modes.

## Phase 4 — Engineering quality bar (1–2 days)

10. **Add tests**
    - Unit tests for cosine similarity, split behavior, and inference mapping.
    - Integration smoke test using synthetic arrays.
11. **Add code quality automation**
    - `black`, `ruff`, `pytest`, and pre-commit hooks.
12. **Set up GitHub Actions CI**
    - Lint + tests on each PR/push.

## Phase 5 — Showcase polish (1 day)

13. **Portfolio narrative assets**
    - Add `docs/case-study.md` with:
      - project motivation,
      - design choices,
      - ablations,
      - what you’d do next.
14. **Add demo surface**
    - Minimum: notebook walkthrough.
    - Better: tiny Streamlit app with sample images and predictions.
15. **Add licensing and attribution**
    - MIT license and explicit dataset citation section.

---

## 4) Prioritized enhancement backlog

### Highest impact first
1. Fix structure + import paths.
2. Add reproducible run commands/config.
3. Generate and commit baseline artifacts (metrics + confusion matrix).
4. Add tests + CI.
5. Add model card + case study narrative.

---

## 5) “Portfolio-ready” definition of done

Use this checklist before showcasing:
- [ ] One-command reproducible training + inference.
- [ ] Clear README with visual results and architecture.
- [ ] Baseline metrics artifact committed.
- [ ] Tests passing in CI.
- [ ] Model card and ethics/limitations documented.
- [ ] Clean package structure and import hygiene.
- [ ] License + contribution guidelines present.

---

## 6) Suggested milestones (practical sequence)

- **Milestone A (today):** repository structure cleanup + README rewrite.
- **Milestone B:** config + deterministic baseline run + saved metrics.
- **Milestone C:** tests + lint + CI.
- **Milestone D:** case study + demo app + final polish.

