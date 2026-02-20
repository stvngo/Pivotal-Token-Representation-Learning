"""Compatibility wrappers around the new probe training pipeline."""

from __future__ import annotations

from typing import Any

from probe_pipeline.config import load_yaml_config
from probe_pipeline.logging_utils import build_logger
from probe_pipeline.train_pytorch import run_pytorch_training
from probe_pipeline.train_sklearn import run_sklearn_training


def train_pytorch_from_config(
    config_path: str = "configs/pipeline.yaml",
    layers: list[int] | None = None,
    num_layers: int | None = None,
    epochs: int | None = None,
) -> dict[str, Any]:
    """Train PyTorch probes using YAML config."""
    config = load_yaml_config(config_path)
    logger = build_logger(
        "models.train.pytorch",
        config["paths"]["outputs"]["logs_dir"] + "/models_train_pytorch.log",
    )
    return run_pytorch_training(
        config=config,
        logger=logger,
        layers_override=layers,
        num_layers_override=num_layers,
        epochs_override=epochs,
    )


def train_sklearn_from_config(
    config_path: str = "configs/pipeline.yaml",
    layers: list[int] | None = None,
    num_layers: int | None = None,
) -> dict[str, Any]:
    """Train sklearn probes using YAML config."""
    config = load_yaml_config(config_path)
    logger = build_logger(
        "models.train.sklearn",
        config["paths"]["outputs"]["logs_dir"] + "/models_train_sklearn.log",
    )
    return run_sklearn_training(
        config=config,
        logger=logger,
        layers_override=layers,
        num_layers_override=num_layers,
    )


def train_model(config_path: str = "configs/pipeline.yaml") -> dict[str, Any]:
    """Backwards-compatible alias: trains PyTorch probes."""
    return train_pytorch_from_config(config_path=config_path)