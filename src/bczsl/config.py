"""Configuration helpers."""

from __future__ import annotations

from pathlib import Path


def _parse_scalar(raw: str):
    raw = raw.strip()
    if raw.isdigit():
        return int(raw)
    try:
        return float(raw)
    except ValueError:
        return raw


def _minimal_yaml_parse(text: str) -> dict:
    """Parse simple two-level YAML used by configs/baseline.yaml."""
    result: dict = {}
    current_section = None

    for line in text.splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue

        indent = len(line) - len(line.lstrip(" "))
        key, value = [part.strip() for part in line.split(":", 1)]

        if indent == 0:
            if value == "":
                result[key] = {}
                current_section = key
            else:
                result[key] = _parse_scalar(value)
                current_section = None
        else:
            if current_section is None:
                continue
            result[current_section][key] = _parse_scalar(value)

    return result


def load_config(path: str) -> dict:
    """Load YAML config, with lightweight fallback parser if PyYAML is unavailable."""
    config_text = Path(path).read_text(encoding="utf-8")

    try:
        import yaml

        return yaml.safe_load(config_text)
    except ModuleNotFoundError:
        return _minimal_yaml_parse(config_text)


def ensure_output_dirs(config: dict) -> tuple[Path, Path]:
    """Ensure metrics and figures output directories exist."""
    outputs = config.get("outputs", {})
    metrics_dir = Path(outputs.get("metrics_dir", "reports/metrics"))
    figures_dir = Path(outputs.get("figures_dir", "reports/figures"))
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return metrics_dir, figures_dir
