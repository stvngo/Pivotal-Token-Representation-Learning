# Pivotal Token Representation Learning

End-to-end, config-driven pipeline for pivotal-token probing experiments:

- Layer-wise activation extraction/caching
- PyTorch linear probes
- Scikit-learn logistic probes
- Per-layer metrics (accuracy, F1, AUROC, confusion matrix, FP/FN ratio)
- PCA/t-SNE and confusion-matrix visualization
- CLI orchestration with YAML config and smoke-test mode

## Quick start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Set paths in `configs/pipeline.yaml`:

- `paths.activations.train`
- `paths.activations.test`
- output folders under `paths.outputs.*`

3. Run training and evaluation:

```bash
python main.py --config configs/pipeline.yaml train-pytorch
python main.py --config configs/pipeline.yaml train-sklearn
python main.py --config configs/pipeline.yaml evaluate --backend pytorch
python main.py --config configs/pipeline.yaml evaluate --backend sklearn
```

4. Plot results:

```bash
python main.py --config configs/pipeline.yaml plot-accuracy
python main.py --config configs/pipeline.yaml plot-embeddings --backend pytorch --num-layers 2
python main.py --config configs/pipeline.yaml plot-confusion --backend pytorch
```

## Smoke tests

Run a fast one-layer trial (small sampled subset):

```bash
python main.py --config configs/pipeline.yaml smoke-train --layer 0
```

Or run only a few layers:

```bash
python main.py --config configs/pipeline.yaml train-pytorch --num-layers 3 --epochs 1
python main.py --config configs/pipeline.yaml train-sklearn --num-layers 3
```

## Artifact layout

- `artifacts/pytorch/probe_states/`
- `artifacts/pytorch/analysis_data/layer_<n>/`
- `artifacts/pytorch/metrics/`
- `artifacts/sklearn/probe_states/`
- `artifacts/sklearn/analysis_data/layer_<n>/`
- `artifacts/sklearn/metrics/`
- `artifacts/plots/`
- `artifacts/logs/`

## Notes

- All paths are relative to the repository root.
- The pipeline expects cached activations saved as dictionaries:
  `layer -> {"pivotal": [tensor...], "non_pivotal": [tensor...]}`.
- `data-generation/` and `models/` now provide compatibility wrappers to the new `probe_pipeline/` modules.
