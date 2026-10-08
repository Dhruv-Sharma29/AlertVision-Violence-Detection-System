"""Export measured binary classification results without requiring TensorFlow."""
import hashlib
import json
import math
from pathlib import Path


def export_metrics(labels, probabilities, output, *, dataset, model_path, threshold=0.5, seed=42):
    labels = list(labels)
    if any(x not in (0, 1) for x in labels):
        raise ValueError("Labels must be binary: non-violence=0, violence=1")
    labels = [int(x) for x in labels]
    probabilities = [float(x) for x in probabilities]
    if not labels or len(labels) != len(probabilities):
        raise ValueError("Non-empty matching test labels and probabilities are required")
    if not 0 <= threshold <= 1 or any(not math.isfinite(x) or not 0 <= x <= 1 for x in probabilities):
        raise ValueError("Threshold and probabilities must lie in [0, 1]")
    model = Path(model_path)
    # Refuse to describe results without an actual model artifact.
    model_hash = hashlib.sha256(model.read_bytes()).hexdigest()
    predictions = [int(x >= threshold) for x in probabilities]
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    result = {
        "dataset": dataset, "split_seed": seed, "threshold": threshold,
        "model_sha256": model_hash, "test_samples": len(labels),
        "accuracy": (tp + tn) / len(labels), "precision": precision,
        "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
