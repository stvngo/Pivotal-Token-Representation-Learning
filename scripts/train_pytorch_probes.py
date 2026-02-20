"""Standalone script for PyTorch layer-wise probe training."""

from __future__ import annotations

import argparse
import json

from probe_pipeline.config import load_yaml_config
from probe_pipeline.logging_utils import build_logger
from probe_pipeline.train_pytorch import run_pytorch_training


def _parse_layers(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PyTorch linear probes by layer.")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--layers", default=None, help="Comma-separated layer list.")
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    logger = build_logger(
        "scripts.train_pytorch",
        config["paths"]["outputs"]["logs_dir"] + "/scripts_train_pytorch.log",
        level=args.log_level,
    )
    summary = run_pytorch_training(
        config=config,
        logger=logger,
        layers_override=_parse_layers(args.layers),
        num_layers_override=args.num_layers,
        epochs_override=args.epochs,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

