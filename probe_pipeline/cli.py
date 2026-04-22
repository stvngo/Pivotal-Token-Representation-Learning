"""CLI entrypoint for end-to-end probe experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import apply_overrides, load_yaml_config
from .logging_utils import build_logger


def _parse_layers(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pivotal-token probe pipeline")
    parser.add_argument(
        "--config",
        default="configs/pipeline.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    train_pt = subparsers.add_parser("train-pytorch", help="Train PyTorch linear probes.")
    train_pt.add_argument("--layers", default=None, help="Comma-separated layer list.")
    train_pt.add_argument("--num-layers", type=int, default=None, help="Use first N layers.")
    train_pt.add_argument("--epochs", type=int, default=None, help="Override epoch count.")

    train_sk = subparsers.add_parser("train-sklearn", help="Train sklearn logistic probes.")
    train_sk.add_argument("--layers", default=None, help="Comma-separated layer list.")
    train_sk.add_argument("--num-layers", type=int, default=None, help="Use first N layers.")

    extract = subparsers.add_parser("extract-activations", help="Extract and cache train/test activations.")

    evaluate = subparsers.add_parser("evaluate", help="Evaluate saved probe artifacts.")
    evaluate.add_argument("--backend", choices=["pytorch", "sklearn"], default="pytorch")
    evaluate.add_argument("--layers", default=None, help="Comma-separated layer list.")
    evaluate.add_argument("--threshold", type=float, default=None, help="Override decision threshold.")

    plot_acc = subparsers.add_parser("plot-accuracy", help="Plot accuracy curves across layers.")

    plot_embed = subparsers.add_parser("plot-embeddings", help="Plot PCA+t-SNE for selected layers.")
    plot_embed.add_argument("--backend", choices=["pytorch", "sklearn"], default="pytorch")
    plot_embed.add_argument("--layers", default=None, help="Comma-separated layer list.")
    plot_embed.add_argument("--num-layers", type=int, default=None, help="Top-N layers by accuracy.")

    plot_cm = subparsers.add_parser("plot-confusion", help="Plot confusion matrices from evaluation JSON.")
    plot_cm.add_argument("--backend", choices=["pytorch", "sklearn"], default="pytorch")
    plot_cm.add_argument("--layers", default=None, help="Comma-separated layer list.")

    plot_heatmap = subparsers.add_parser("plot-heatmap", help="Plot activation heatmaps for selected layers.")
    plot_heatmap.add_argument("--backend", choices=["pytorch", "sklearn"], default="sklearn")
    plot_heatmap.add_argument("--layers", default=None, help="Comma-separated layer list (e.g. 14,16,20).")

    plot_analysis = subparsers.add_parser(
        "plot-probe-analysis",
        help="Plot probe weights, sample correlation, and centroid difference.",
    )
    plot_analysis.add_argument("--backend", choices=["pytorch", "sklearn"], default="sklearn")
    plot_analysis.add_argument("--layers", default=None, help="Comma-separated layer list.")
    plot_analysis.add_argument("--top-k", type=int, default=64, help="Top-k dimensions for weight/diff plots.")

    double_neg = subparsers.add_parser(
        "double-negatives",
        help="Create dataset variant with additional negative labels.",
    )
    double_neg.add_argument("--input-dataset", required=True, help="Path to HF dataset directory.")
    double_neg.add_argument("--output-dataset", required=True, help="Output path for transformed dataset.")

    smoke = subparsers.add_parser(
        "smoke-train",
        help="Quick smoke run: 1 layer, 1 epoch PyTorch + 1 layer sklearn.",
    )
    smoke.add_argument("--layer", type=int, default=0, help="Single layer to train.")

    create_steering = subparsers.add_parser(
        "create-steering-config",
        help="Create steering configs from probe analysis for specified layers.",
    )
    create_steering.add_argument("--backend", choices=["pytorch", "sklearn"], default="sklearn")
    create_steering.add_argument("--layers", default=None, help="Comma-separated layer list (e.g. 14,16,20).")
    create_steering.add_argument("--factors", default=None, help="Comma-separated factors (e.g. 1.2,1.4,2).")
    create_steering.add_argument("--vector-type", choices=["centroid_diff", "probe_weights"], default="centroid_diff")

    eval_gsm8k = subparsers.add_parser("eval-gsm8k", help="Evaluate base and steered models on GSM8K subset.")
    eval_gsm8k.add_argument("--max-examples", type=int, default=None)
    eval_gsm8k.add_argument("--layers", default=None, help="Layer for steering config (uses first if multiple).")
    eval_gsm8k.add_argument("--factors", default=None, help="Comma-separated factors to evaluate.")

    error_analysis = subparsers.add_parser(
        "error-analysis",
        help="Run error analysis and visualizations on GSM8K evaluation results.",
    )
    error_analysis.add_argument("--results-path", default=None, help="Path to results.json from eval-gsm8k.")

    return parser


def _create_logger(config: dict[str, Any], command: str, log_level: str):
    logs_dir = Path(config["paths"]["outputs"]["logs_dir"])
    logs_dir.mkdir(parents=True, exist_ok=True)
    return build_logger(
        name=f"probe_pipeline.{command}",
        log_path=logs_dir / f"{command}.log",
        level=log_level,
    )


def _run_double_negatives(config: dict[str, Any], args: argparse.Namespace, logger: Any) -> dict[str, Any]:
    from datasets import load_from_disk
    from .modeling import load_model_and_tokenizer
    from .preprocess import create_doubled_negatives_dataset, save_probe_dataset

    dataset = load_from_disk(args.input_dataset)
    model_cfg = config["model"]
    _, tokenizer, _ = load_model_and_tokenizer(
        model_name=model_cfg["name"],
        device_name=config.get("device", "cpu"),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
    )
    new_rows = create_doubled_negatives_dataset(dataset, tokenizer, seed=int(config.get("seed", 42)))
    save_probe_dataset(new_rows, args.output_dataset)
    logger.info("Saved doubled-negatives dataset to %s", args.output_dataset)
    return {"output_dataset": args.output_dataset, "rows": len(new_rows)}


def _run_smoke_train(config: dict[str, Any], args: argparse.Namespace, logger: Any) -> dict[str, Any]:
    from .train_pytorch import run_pytorch_training
    from .train_sklearn import run_sklearn_training

    overrides = {
        "training": {
            "max_train_samples_per_class": 64,
            "max_val_samples_per_class": 64,
            "batch_size": 16,
        }
    }
    cfg = apply_overrides(config, overrides)
    layer = args.layer
    pt = run_pytorch_training(
        config=cfg,
        logger=logger,
        layers_override=[layer],
        num_layers_override=1,
        epochs_override=1,
    )
    sk = run_sklearn_training(
        config=cfg,
        logger=logger,
        layers_override=[layer],
        num_layers_override=1,
    )
    return {"pytorch": pt, "sklearn": sk}


def _run_create_steering_config(config: dict[str, Any], args: argparse.Namespace, logger: Any) -> dict[str, Any]:
    from .steering import create_steering_config

    steer_cfg = config.get("steering", {})
    layers = _parse_layers(args.layers) or steer_cfg.get("layers", [14, 16, 20])
    factors = steer_cfg.get("factors", [1.2, 1.4, 2.0])
    if args.factors:
        factors = [float(x.strip()) for x in args.factors.split(",")]
    vector_type = getattr(args, "vector_type", "centroid_diff")

    if not layers:
        raise ValueError("No layers specified. Use --layers or config.steering.layers.")

    backend_root = Path(
        config["paths"]["outputs"]["pytorch_dir"]
        if args.backend == "pytorch"
        else config["paths"]["outputs"]["sklearn_dir"]
    )
    steer_dir = backend_root / "steering_configs"
    created = []
    for layer in layers:
        cfg = create_steering_config(
            config=config,
            backend=args.backend,
            layer=layer,
            factors=factors,
            vector_type=vector_type,
            output_dir=steer_dir,
        )
        config_path = steer_dir / f"steering_layer{layer}.json"
        created.append({"layer": layer, "config_path": str(config_path)})
        logger.info("Created steering config for layer %s at %s", layer, config_path)

    return {"created": created, "factors": factors}


def _run_eval_gsm8k(config: dict[str, Any], args: argparse.Namespace, logger: Any) -> dict[str, Any]:
    from pathlib import Path

    from .gsm8k_eval import run_gsm8k_comparison

    eval_cfg = config.get("evaluation", {})
    max_examples = args.max_examples if getattr(args, "max_examples", None) is not None else eval_cfg.get("gsm8k_max_examples", 200)
    layers = _parse_layers(getattr(args, "layers", None))
    steer_cfg = config.get("steering", {})
    factors = steer_cfg.get("factors", [1.2, 1.4, 2.0])
    if getattr(args, "factors", None):
        factors = [float(x.strip()) for x in args.factors.split(",")]

    backend_root = Path(config["paths"]["outputs"]["sklearn_dir"])
    steer_dir = backend_root / "steering_configs"
    layer = (layers or steer_cfg.get("layers", [14]))[0]
    steering_config_path = steer_dir / f"steering_layer{layer}.json"
    if not steering_config_path.exists():
        logger.warning("Steering config not found at %s. Run create-steering-config first.", steering_config_path)
        steering_config_path = None

    output_dir = Path(config["paths"]["outputs"].get("gsm8k_eval_dir", str(backend_root.parent / "gsm8k_eval")))

    split = eval_cfg.get("gsm8k_split", "test")
    result = run_gsm8k_comparison(
        config=config,
        logger=logger,
        max_examples=max_examples,
        split=split,
        steering_config_path=steering_config_path,
        factors=factors,
        output_dir=output_dir,
    )
    return result


def _run_error_analysis(config: dict[str, Any], args: argparse.Namespace, logger: Any) -> dict[str, Any]:
    from pathlib import Path

    from .gsm8k_eval import run_error_analysis as run_err_analysis

    results_path = getattr(args, "results_path", None)
    if results_path is None:
        output_dir = Path(config["paths"]["outputs"].get("gsm8k_eval_dir", "artifacts/cached3/gsm8k_eval"))
        results_path = output_dir / "results.json"
    results_path = Path(results_path)
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}. Run eval-gsm8k first.")

    output_dir = results_path.parent / "error_analysis"
    return run_err_analysis(results_path=results_path, output_dir=output_dir, logger=logger)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    logger = _create_logger(config=config, command=args.command, log_level=args.log_level)

    layers = _parse_layers(getattr(args, "layers", None))

    if args.command == "train-pytorch":
        from .train_pytorch import run_pytorch_training

        result = run_pytorch_training(
            config=config,
            logger=logger,
            layers_override=layers,
            num_layers_override=args.num_layers,
            epochs_override=args.epochs,
        )
    elif args.command == "train-sklearn":
        from .train_sklearn import run_sklearn_training

        result = run_sklearn_training(
            config=config,
            logger=logger,
            layers_override=layers,
            num_layers_override=args.num_layers,
        )
    elif args.command == "extract-activations":
        from .extract import run_activation_extraction

        result = run_activation_extraction(config=config, logger=logger)
    elif args.command == "evaluate":
        from .evaluate import evaluate_saved_probes

        result = evaluate_saved_probes(
            config=config,
            logger=logger,
            backend=args.backend,
            threshold_override=args.threshold,
            layers_override=layers,
        )
    elif args.command == "plot-accuracy":
        from .visualize import plot_accuracy_across_layers

        path = plot_accuracy_across_layers(config=config, logger=logger)
        result = {"plot": str(path)}
    elif args.command == "plot-embeddings":
        from .visualize import plot_pca_tsne

        paths = plot_pca_tsne(
            config=config,
            logger=logger,
            backend=args.backend,
            layers_override=layers,
            num_layers_override=args.num_layers,
        )
        result = {"plots": [str(p) for p in paths]}
    elif args.command == "plot-confusion":
        from .visualize import plot_confusion_matrices

        paths = plot_confusion_matrices(
            config=config,
            logger=logger,
            backend=args.backend,
            layers_override=layers,
        )
        result = {"plots": [str(p) for p in paths]}
    elif args.command == "plot-heatmap":
        from .visualize import plot_activation_heatmap

        paths = plot_activation_heatmap(
            config=config,
            logger=logger,
            backend=args.backend,
            layers_override=layers,
        )
        result = {"plots": [str(p) for p in paths]}
    elif args.command == "plot-probe-analysis":
        from .visualize import plot_probe_analysis

        paths = plot_probe_analysis(
            config=config,
            logger=logger,
            backend=args.backend,
            layers_override=layers,
            top_k=getattr(args, "top_k", 64),
        )
        result = {"plots": [str(p) for p in paths]}
    elif args.command == "double-negatives":
        result = _run_double_negatives(config=config, args=args, logger=logger)
    elif args.command == "smoke-train":
        result = _run_smoke_train(config=config, args=args, logger=logger)
    elif args.command == "create-steering-config":
        result = _run_create_steering_config(config=config, args=args, logger=logger)
    elif args.command == "eval-gsm8k":
        result = _run_eval_gsm8k(config=config, args=args, logger=logger)
    elif args.command == "error-analysis":
        result = _run_error_analysis(config=config, args=args, logger=logger)
    else:
        raise ValueError(f"Unsupported command: {args.command}")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

