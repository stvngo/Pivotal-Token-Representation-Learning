"""Standalone script for re-evaluating saved probes."""

from __future__ import annotations

import argparse
import json

from probe_pipeline.config import load_yaml_config
from probe_pipeline.evaluate import evaluate_saved_probes
from probe_pipeline.logging_utils import build_logger


def _parse_layers(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved probe artifacts.")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--backend", choices=["pytorch", "sklearn"], default="pytorch")
    parser.add_argument("--layers", default=None, help="Comma-separated layer list.")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    logger = build_logger(
        "scripts.evaluate",
        config["paths"]["outputs"]["logs_dir"] + "/scripts_evaluate.log",
        level=args.log_level,
    )
    results = evaluate_saved_probes(
        config=config,
        logger=logger,
        backend=args.backend,
        threshold_override=args.threshold,
        layers_override=_parse_layers(args.layers),
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

