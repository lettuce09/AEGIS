import csv
import json
from pathlib import Path

import numpy as np
import pytest

from aegis.data import SegmentationDataset, read_manifest
from aegis.prepare import split_manifest, synthetic, volumes, write_manifest


def _rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def _rewrite(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def test_synthetic_dataset_shapes_and_independent_splits(tmp_path):
    manifest = synthetic(tmp_path / "demo", image_size=32)
    train = SegmentationDataset(manifest, "train", image_size=64, training=True)
    val = SegmentationDataset(manifest, "val", image_size=64, training=False)
    sample = train[0]
    assert sample["image"].shape == (3, 64, 64)
    assert sample["mask"].shape == (1, 64, 64)
    assert sample["organ"].shape == (4,)
    assert sample["box"].shape == (4,)
    assert 0 <= sample["image"].min() <= sample["image"].max() <= 255
    assert train.training and not val.training
    assert {x["patient_id"] for x in train.rows}.isdisjoint({x["patient_id"] for x in val.rows})


def test_patient_leakage_is_rejected(tmp_path):
    manifest = synthetic(tmp_path / "demo", image_size=16)
    rows = _rows(manifest)
    rows[4]["patient_id"] = rows[0]["patient_id"]
    _rewrite(manifest, rows)
    with pytest.raises(ValueError, match="multiple splits"):
        read_manifest(manifest)


def test_duplicate_samples_and_reused_images_are_rejected(tmp_path):
    manifest = synthetic(tmp_path / "demo", image_size=16)
    rows = _rows(manifest)
    rows[1]["sample_id"] = rows[0]["sample_id"]
    _rewrite(manifest, rows)
    with pytest.raises(ValueError, match="Duplicate sample_id"):
        read_manifest(manifest)
    rows[1]["sample_id"] = "distinct"
    rows[1]["image"] = rows[0]["image"]
    _rewrite(manifest, rows)
    with pytest.raises(ValueError, match="Duplicate image"):
        read_manifest(manifest)


def test_missing_active_labels_fail_training_but_allow_evaluation(tmp_path):
    manifest = synthetic(tmp_path / "demo", image_size=16)
    rows = _rows(manifest)
    rows[0]["site"] = ""
    rows[4]["site"] = "-1"
    _rewrite(manifest, rows)
    with pytest.raises(ValueError, match="Missing site"):
        SegmentationDataset(manifest, "train", training=True, require_groups=False)
    val = SegmentationDataset(manifest, "val", require_groups=False)
    assert val[0]["group"].item() == -1


def test_identifiers_preserve_leading_zeros(tmp_path):
    manifest = synthetic(tmp_path / "demo", image_size=16)
    rows = _rows(manifest)
    rows[0]["patient_id"], rows[0]["sample_id"] = "000001", "000002"
    _rewrite(manifest, rows)
    sample = SegmentationDataset(manifest, "train")[0]
    assert sample["patient_id"] == "000001" and sample["sample_id"] == "000002"


def test_provided_box_scales_and_unlabeled_inference(tmp_path):
    manifest = synthetic(tmp_path / "demo", image_size=16)
    rows = _rows(manifest)
    rows[0]["box"], rows[0]["mask"] = json.dumps([2, 3, 10, 12]), ""
    _rewrite(manifest, rows)
    data = SegmentationDataset(manifest, "train", image_size=32, prompt_mode="provided", require_masks=False)
    sample = data[0]
    assert sample["box"].tolist() == [4, 6, 20, 24]
    assert "mask" not in sample
    with pytest.raises(ValueError, match="Image-only"):
        SegmentationDataset(manifest, "train", require_masks=False)


def test_bad_masks_and_invalid_boxes_fail(tmp_path):
    manifest = synthetic(tmp_path / "demo", image_size=16)
    rows = _rows(manifest)
    mask_path = manifest.parent / rows[0]["mask"]
    np.save(mask_path, np.full((16, 16), 2, dtype=np.uint8))
    with pytest.raises(ValueError, match="binary"):
        SegmentationDataset(manifest, "train")[0]
    rows[0]["box"] = json.dumps([10, 3, 2, 12])
    _rewrite(manifest, rows)
    with pytest.raises(ValueError, match="positive area"):
        SegmentationDataset(manifest, "train", prompt_mode="provided", require_masks=False)[0]


def test_patient_split_is_deterministic_and_relative_paths_portable(tmp_path):
    manifest = synthetic(tmp_path / "demo", image_size=16)
    first, second = tmp_path / "a" / "splits.csv", tmp_path / "b" / "splits.csv"
    split_manifest(manifest, first, ratios=(.5, .25, .25), seed=7)
    split_manifest(manifest, second, ratios=(.5, .25, .25), seed=7)
    a, b = read_manifest(first), read_manifest(second)
    assert [(r["sample_id"], r["split"]) for r in a] == [(r["sample_id"], r["split"]) for r in b]
    assert all(Path(r["image"]).exists() for r in a)
    with pytest.raises(ValueError, match="sum to 1"):
        split_manifest(manifest, tmp_path / "bad.csv", ratios=(.8, .3, 0))


def test_volume_preprocessing_filters_and_tracks_slices(tmp_path):
    image = np.arange(3 * 12 * 16).reshape(3, 12, 16).astype(np.float32)
    mask = np.zeros_like(image, dtype=np.uint8)
    mask[1, 3:8, 5:11] = 1
    np.save(tmp_path / "volume.npy", image)
    np.save(tmp_path / "label.npy", mask)
    manifest = tmp_path / "volumes.csv"
    manifest.write_text("volume_id,patient_id,image,mask,split,gender,age,site\nv001,p001,volume.npy,label.npy,train,0,0,1\n")
    report = volumes(manifest, tmp_path / "converted", image_size=32)
    assert report["saved_slices"] == 1 and report["filtered_slices"] == 2
    data = SegmentationDataset(tmp_path / "converted" / "manifest.csv", "train", image_size=32)
    assert data[0]["mask"].sum() > 0


def test_non_square_input_resizes_boxes_with_independent_xy_scales(tmp_path):
    image = np.full((8, 16), 100, dtype=np.uint8)
    mask = np.zeros((8, 16), dtype=np.uint8)
    mask[2:6, 3:10] = 1
    np.save(tmp_path / "image.npy", image)
    np.save(tmp_path / "mask.npy", mask)
    manifest = tmp_path / "manifest.csv"
    write_manifest([{"sample_id": "a", "patient_id": "p", "image": str(tmp_path / "image.npy"),
                     "mask": str(tmp_path / "mask.npy"), "split": "test", "site": 0,
                     "box": [3, 2, 10, 6]}], manifest)
    sample = SegmentationDataset(manifest, "test", image_size=32, prompt_mode="provided")[0]
    assert sample["box"].tolist() == [6, 8, 20, 24]
    assert sample["mask"].shape == (1, 32, 32)


@pytest.mark.parametrize("stale_mask", ["missing-annotation.npy", "unavailable-annotation.nii.gz"])
def test_unlabeled_inference_ignores_stale_mask_column(tmp_path, stale_mask):
    manifest = synthetic(tmp_path / "demo", image_size=16)
    rows = _rows(manifest)
    # Repeat a missing, possibly non-NPY path: no mask checks may run here.
    for row in rows:
        row["mask"] = stale_mask
    _rewrite(manifest, rows)
    data = SegmentationDataset(manifest, "test", prompt_mode="provided", require_masks=False)
    assert len(data) == 4 and "mask" not in data[0]
    assert all(row["mask"] == "" for row in data.rows)
    with pytest.raises((FileNotFoundError, ValueError)):
        SegmentationDataset(manifest, "test", require_masks=True)
