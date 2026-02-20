import numpy as np

from probe_pipeline.metrics import compute_binary_metrics


def test_compute_binary_metrics_shapes_and_counts():
    y_true = np.array([0, 0, 1, 1], dtype=np.int64)
    y_prob = np.array([0.1, 0.8, 0.7, 0.2], dtype=np.float32)
    metrics = compute_binary_metrics(y_true=y_true, y_prob=y_prob, threshold=0.5)

    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert metrics["TN"] == 1
    assert metrics["FP"] == 1
    assert metrics["FN"] == 1
    assert metrics["TP"] == 1
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1_score"] <= 1.0

