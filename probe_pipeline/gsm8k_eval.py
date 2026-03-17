"""GSM8K evaluation pipeline for base vs steered models."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from tqdm.auto import tqdm

from .modeling import load_model_and_tokenizer
from .steering import generate_with_steering, load_steering_config


def extract_gsm8k_answer(text: str) -> str | None:
    """Extract the final numeric answer from GSM8K-style output (after ####)."""
    match = re.search(r"####\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "").strip()
    # Fallback: last number in the text
    numbers = re.findall(r"[-+]?\d*\.?\d+", text)
    return numbers[-1] if numbers else None


def extract_gsm8k_ground_truth(answer_str: str) -> str | None:
    """Extract ground truth from GSM8K answer field (after ####)."""
    return extract_gsm8k_answer(answer_str)


def is_correct(predicted: str | None, ground_truth: str | None) -> bool:
    """Compare predicted and ground truth (normalize for comparison)."""
    if predicted is None or ground_truth is None:
        return False
    pred_norm = predicted.strip().replace(",", "")
    gt_norm = ground_truth.strip().replace(",", "")
    try:
        return float(pred_norm) == float(gt_norm)
    except ValueError:
        return pred_norm == gt_norm


def run_gsm8k_evaluation(
    config: dict[str, Any],
    logger: Any,
    max_examples: int = 200,
    split: str = "validation",
    seed: int | None = None,
    steering_config_path: Path | None = None,
    steering_factor: float | None = None,
    max_new_tokens: int = 256,
    temperature: float = 0.6,
) -> dict[str, Any]:
    """
    Run GSM8K evaluation. If steering_config_path and steering_factor are provided,
    evaluates the steered model; otherwise evaluates the base model.
    """
    seed = seed if seed is not None else int(config.get("seed", 42))
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.manual_seed(seed)

    model_cfg = config["model"]
    model, tokenizer, device = load_model_and_tokenizer(
        model_name=model_cfg["name"],
        device_name=config.get("device", "auto"),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
        output_hidden_states=False,
    )

    ds = load_dataset("openai/gsm8k", "main", split=split)
    indices = np.random.choice(len(ds), size=min(max_examples, len(ds)), replace=False)
    subset = ds.select(indices.tolist())

    use_steering = steering_config_path is not None and steering_factor is not None
    if use_steering:
        steer_cfg, steer_vector = load_steering_config(Path(steering_config_path))
        layer = steer_cfg["layer"]
        strength = steering_factor - 1.0  # factor=1.2 => add 0.2*v
    else:
        layer = None
        steer_vector = None
        strength = 0.0

    results: list[dict[str, Any]] = []
    correct = 0

    for i, example in enumerate(tqdm(subset, desc="Evaluating")):
        question = example["question"]
        ground_truth = extract_gsm8k_ground_truth(example["answer"])
        prompt = f"Question: {question}\n\nLet's think step by step.\n\n"

        if use_steering and steer_vector is not None:
            output = generate_with_steering(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                layer=layer,
                vector=steer_vector,
                strength=strength,
                device=device,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        else:
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        predicted = extract_gsm8k_answer(output)
        ok = is_correct(predicted, ground_truth)
        if ok:
            correct += 1

        results.append({
            "idx": i,
            "question": question,
            "ground_truth": ground_truth,
            "predicted": predicted,
            "correct": ok,
            "full_output": output[:500],
        })

    accuracy = correct / len(results) if results else 0.0
    summary = {
        "model": model_cfg["name"],
        "max_examples": len(results),
        "correct": correct,
        "accuracy": accuracy,
        "seed": seed,
        "steering": use_steering,
    }
    if use_steering:
        summary["steering_config"] = str(steering_config_path)
        summary["steering_factor"] = steering_factor

    return {"summary": summary, "results": results}


def run_gsm8k_comparison(
    config: dict[str, Any],
    logger: Any,
    max_examples: int = 200,
    split: str = "validation",
    steering_config_path: Path | None = None,
    factors: list[float] | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run base + steered models and save results for comparison."""
    factors = factors or [1.2, 1.4, 2.0]
    seed = int(config.get("seed", 42))

    if output_dir is None:
        outputs_cfg = config["paths"]["outputs"]
        backend_root = Path(outputs_cfg.get("sklearn_dir", outputs_cfg.get("pytorch_dir", "artifacts")))
        output_dir = backend_root.parent / "gsm8k_eval"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries: dict[str, dict[str, Any]] = {}
    all_results: dict[str, list] = {}

    # Base model
    logger.info("Evaluating base model...")
    base_out = run_gsm8k_evaluation(
        config=config,
        logger=logger,
        max_examples=max_examples,
        split=split,
        seed=seed,
    )
    all_summaries["base"] = base_out["summary"]
    all_results["base"] = base_out["results"]
    logger.info("Base accuracy: %.2f%%", base_out["summary"]["accuracy"] * 100)

    # Steered models
    if steering_config_path and Path(steering_config_path).exists():
        for factor in factors:
            name = f"steered_{factor}"
            logger.info("Evaluating steered model (factor=%.1f)...", factor)
            steer_out = run_gsm8k_evaluation(
                config=config,
                logger=logger,
                max_examples=max_examples,
                split=split,
                seed=seed,
                steering_config_path=steering_config_path,
                steering_factor=factor,
            )
            all_summaries[name] = steer_out["summary"]
            all_results[name] = steer_out["results"]
            logger.info("%s accuracy: %.2f%%", name, steer_out["summary"]["accuracy"] * 100)

    # Save
    with (output_dir / "summaries.json").open("w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)
    with (output_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    logger.info("Saved evaluation results to %s", output_dir)

    # Run error analysis
    if len(all_results) > 1:
        run_error_analysis(output_dir / "results.json", output_dir, logger)

    return {"summaries": all_summaries, "output_dir": str(output_dir)}


def run_error_analysis(
    results_path: Path,
    output_dir: Path,
    logger: Any,
) -> dict[str, Any]:
    """Analyze errors and produce visualizations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    with results_path.open("r", encoding="utf-8") as f:
        all_results = json.load(f)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Accuracy comparison bar chart
    names = list(all_results.keys())
    accuracies = []
    for name in names:
        res = all_results[name]
        correct = sum(1 for r in res if r.get("correct"))
        accuracies.append(correct / len(res) * 100 if res else 0)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=names, y=accuracies, ax=ax, palette="viridis")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("GSM8K Accuracy: Base vs Steered Models")
    ax.set_ylim(0, 100)
    for i, v in enumerate(accuracies):
        ax.text(i, v + 2, f"{v:.1f}%", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_comparison.png")
    plt.close(fig)
    logger.info("Saved accuracy comparison to %s", output_dir / "accuracy_comparison.png")

    # Error overlap: which examples does base get wrong that steered gets right (and vice versa)
    if "base" in all_results and len(all_results) > 1:
        base_res = {r["idx"]: r["correct"] for r in all_results["base"]}
        error_analysis = []
        for name in names:
            if name == "base":
                continue
            res = all_results[name]
            base_wrong_steer_right = sum(1 for r in res if not base_res.get(r["idx"], False) and r["correct"])
            base_right_steer_wrong = sum(1 for r in res if base_res.get(r["idx"], False) and not r["correct"])
            error_analysis.append({
                "model": name,
                "base_wrong_steer_right": base_wrong_steer_right,
                "base_right_steer_wrong": base_right_steer_wrong,
            })

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        models = [e["model"] for e in error_analysis]
        gained = [e["base_wrong_steer_right"] for e in error_analysis]
        lost = [e["base_right_steer_wrong"] for e in error_analysis]
        x = np.arange(len(models))
        w = 0.35
        ax2.bar(x - w/2, gained, w, label="Gained (base wrong → steered right)", color="green", alpha=0.7)
        ax2.bar(x + w/2, lost, w, label="Lost (base right → steered wrong)", color="red", alpha=0.7)
        ax2.set_xticks(x)
        ax2.set_xticklabels(models)
        ax2.set_ylabel("Count")
        ax2.set_title("Error Analysis: Gained vs Lost vs Base")
        ax2.legend()
        fig2.tight_layout()
        fig2.savefig(output_dir / "error_analysis.png")
        plt.close(fig2)
        logger.info("Saved error analysis to %s", output_dir / "error_analysis.png")

        with (output_dir / "error_analysis.json").open("w", encoding="utf-8") as f:
            json.dump(error_analysis, f, indent=2)

    return {"output_dir": str(output_dir)}
