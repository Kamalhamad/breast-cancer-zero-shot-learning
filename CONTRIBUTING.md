# Contributing

Thanks for your interest in contributing to this project.

## Development setup

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pytest ruff black
```

## Local checks

```bash
python -m ruff check .
python -m black --check .
PYTHONPATH=src python -m pytest -q
```

## Run baseline pipeline

```bash
PYTHONPATH=src python -m bczsl.train --config configs/baseline.yaml --synthetic
PYTHONPATH=src python -m bczsl.zsl_inference --config configs/baseline.yaml --synthetic
```

## Pull request checklist
- [ ] Tests pass locally.
- [ ] Lint and format checks pass.
- [ ] README/docs are updated when behavior changes.
- [ ] New files follow project structure and naming conventions.
