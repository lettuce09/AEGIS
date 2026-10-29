"""Sample Dice and the paper's subgroup max-minus-min EA gap."""
from __future__ import annotations

import math
from collections import defaultdict

import torch


def binary_dice(prediction, target, threshold=0.5):
    """Return one Dice per sample from probabilities/binary masks (not logits).

    Supported shapes: HW, BHW, or B1HW. Both-empty masks score 1; a single
    empty mask scores 0. Values are fractions in [0,1].
    """
    prediction, target = torch.as_tensor(prediction), torch.as_tensor(target)
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target must have the same shape")
    if prediction.ndim == 2:
        prediction, target = prediction[None], target[None]
    if prediction.ndim not in (3, 4) or (prediction.ndim == 4 and prediction.shape[1] != 1):
        raise ValueError("Expected HW, BHW, or B1HW binary segmentation arrays")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("Metrics require finite arrays")
    if torch.any((prediction < 0) | (prediction > 1)):
        raise ValueError("Pass probabilities or binary masks, not logits")
    if torch.any((target != 0) & (target != 1)):
        raise ValueError("Targets must be binary")
    pred = (prediction > threshold).reshape(prediction.shape[0], -1)
    truth = target.bool().reshape(target.shape[0], -1)
    intersection = (pred & truth).sum(1).float()
    denominator = pred.sum(1) + truth.sum(1)
    return torch.where(denominator > 0, 2 * intersection / denominator.clamp_min(1),
                       torch.ones_like(intersection))


def summarize(rows, attributes=("gender", "age", "site")):
    """Summarize slice/sample records; Dice and gaps are returned in percent.

    Every input row contains a ``dice`` fraction and integer demographic labels.
    EA = highest subgroup mean - lowest subgroup mean, in percentage points.
    If fewer than two observed groups exist EA is unavailable (None), not zero.
    Missing labels remain in overall Dice and are counted per attribute.
    """
    rows = list(rows)
    if not rows:
        raise ValueError("Cannot summarize an empty evaluation")
    values = [float(row["dice"]) for row in rows]
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError("dice must be finite and in [0,1]")
    result = {"n_samples": len(rows), "overall_dice": 100 * sum(values) / len(values),
              "unit": "percent; ea_gap in percentage points", "aggregation": "sample_mean",
              "attributes": {}}
    for attr in attributes:
        groups = defaultdict(list)
        missing = 0
        for row, value in zip(rows, values):
            group = row.get(attr, -1)
            if group is None or group == "" or group == -1 or group == "-1":
                missing += 1
                continue
            text = str(group)
            if not text.isdecimal():
                raise ValueError(f"{attr} must contain nonnegative integer labels or -1")
            groups[int(group)].append(value)
        stats = {str(group): {"dice": 100 * sum(scores) / len(scores), "n": len(scores)}
                 for group, scores in sorted(groups.items())}
        means = [group["dice"] for group in stats.values()]
        result["attributes"][attr] = {
            "groups": stats, "n_labeled": len(rows) - missing, "missing": missing,
            "ea_gap": max(means) - min(means) if len(means) >= 2 else None,
            "worst_group_dice": min(means) if means else None,
        }
    return result
