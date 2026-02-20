"""Configuration loading and normalization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge `override` into `base` and return `base`."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge_dicts(base[key], value)
        else:
            base[key] = value
    return base


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML config and normalize key directories."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    project_root = path.resolve().parent.parent
    config.setdefault("_meta", {})
    config["_meta"]["project_root"] = str(project_root)
    config["_meta"]["config_path"] = str(path.resolve())
    return config


def apply_overrides(config: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    """Apply nested dictionary overrides to config."""
    if not overrides:
        return config
    return _merge_dicts(config, overrides)


def ensure_dir(path: str | Path) -> Path:
    """Create directory if needed and return the Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory

