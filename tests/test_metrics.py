import pytest
import torch

from aegis.metrics import binary_dice, summarize


def test_dice_empty_disjoint_and_partial_cases():
    pred = torch.tensor([[[0, 0]], [[1, 0]], [[1, 0]], [[1, 1]]])
    target = torch.tensor([[[0, 0]], [[0, 0]], [[0, 1]], [[1, 0]]])
    assert binary_dice(pred, target).tolist() == pytest.approx([1, 0, 0, 2 / 3])
    assert binary_dice(pred[0], target[0]).shape == (1,)


def test_dice_rejects_logits_and_nonbinary_targets():
    with pytest.raises(ValueError, match="not logits"):
        binary_dice(torch.full((2, 3), 2.), torch.zeros(2, 3))
    with pytest.raises(ValueError, match="binary"):
        binary_dice(torch.zeros(2, 3), torch.full((2, 3), .5))


def test_ea_is_group_max_minus_min_not_deviation_from_overall():
    rows = [{"dice": .9, "site": 0}] * 3 + [{"dice": .5, "site": 1}, {"dice": .2, "site": -1}]
    summary = summarize(rows, attributes=["site"])
    assert summary["overall_dice"] == pytest.approx(68)
    stats = summary["attributes"]["site"]
    assert stats["ea_gap"] == pytest.approx(40)
    assert stats["groups"]["0"]["n"] == 3
    assert stats["missing"] == 1 and stats["n_labeled"] == 4


def test_single_group_or_no_labels_do_not_claim_zero_gap():
    summary = summarize([{"dice": .7, "site": 0}])
    assert summary["attributes"]["site"]["ea_gap"] is None
    assert summary["attributes"]["gender"]["ea_gap"] is None
    assert summary["attributes"]["gender"]["missing"] == 1
    with pytest.raises(ValueError, match="empty"):
        summarize([])
