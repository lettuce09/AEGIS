"""Explicit, patient-disjoint manifests for prompted binary segmentation."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

ATTRIBUTES = ("gender", "age", "site")
SPLITS = ("train", "val", "test")
MANIFEST_FIELDS = ("sample_id", "patient_id", "image", "mask", "split", *ATTRIBUTES, "box")


def _group(value: str | int | None, field: str) -> int:
    if value is None or str(value).strip() in ("", "-1"):
        return -1
    text = str(value).strip()
    if not text.isdecimal():
        raise ValueError(f"{field} must be a nonnegative integer or -1 for missing")
    return int(text)


def read_manifest(path: str | Path, *, require_masks: bool = True, check_files: bool = True,
                  allow_unassigned: bool = False) -> list[dict]:
    """Read a CSV, resolve paths relative to it, and reject patient leakage.

    Identifiers are opaque strings, never inferred from filenames. Empty group
    cells are represented by -1. A missing group is not a new demographic group.
    When require_masks=False the mask column is ignored, including stale paths.
    """
    path = Path(path).resolve()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "patient_id", "image", "split"}
        if require_masks:
            required.add("mask")
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("Manifest is empty")
    seen_ids, seen_images, seen_masks, patient_splits = set(), {}, {}, {}
    for number, row in enumerate(rows, 2):
        for field in ("sample_id", "patient_id", "image"):
            if not row.get(field, "").strip():
                raise ValueError(f"Manifest row {number}: {field} is empty")
        # Trim accidental whitespace, but keep leading zeros and opaque text.
        row["sample_id"] = row["sample_id"].strip()
        row["patient_id"] = row["patient_id"].strip()
        sample = row["sample_id"]
        if sample in seen_ids:
            raise ValueError(f"Duplicate sample_id at row {number}")
        seen_ids.add(sample)
        split = row.get("split", "").strip()
        if split not in SPLITS and not (allow_unassigned and split == ""):
            raise ValueError(f"Manifest row {number}: split must be train, val, or test")
        row["split"] = split
        patient = row["patient_id"]
        if patient in patient_splits and patient_splits[patient] != split:
            raise ValueError("Patient appears in multiple splits; all scans/timepoints must stay together")
        patient_splits[patient] = split
        for attr in ATTRIBUTES:
            row[attr] = _group(row.get(attr), attr)
        for field, seen in (("image", seen_images), ("mask", seen_masks)):
            if field == "mask" and not require_masks:
                # Image-only inference must not depend on any annotation path.
                row["mask"] = ""
                continue
            value = row.get(field, "").strip()
            if not value:
                if field == "image" or require_masks:
                    raise ValueError(f"Manifest row {number}: {field} is empty")
                row[field] = ""
                continue
            resolved = (path.parent / value).resolve()
            if resolved.suffix.lower() != ".npy":
                raise ValueError(f"Manifest {field} must refer to .npy arrays")
            if resolved in seen:
                raise ValueError(f"Duplicate {field} path in manifest (would double-count a sample)")
            seen[resolved] = split
            if check_files and not resolved.is_file():
                raise FileNotFoundError(f"Manifest row {number}: {field} file does not exist")
            row[field] = str(resolved)
        if row.get("mask") and row["image"] == row["mask"]:
            raise ValueError("Image and mask must be different files")
        raw_box = row.get("box", "")
        if raw_box:
            try:
                box = json.loads(raw_box)
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Manifest row {number}: box must be a JSON list of four numbers") from exc
            if (not isinstance(box, list) or len(box) != 4
                    or any(isinstance(x, bool) or not isinstance(x, (float, int)) for x in box)
                    or not np.isfinite(box).all()):
                raise ValueError("box must be [x_min, y_min, x_max, y_max] with finite numbers")
            row["box"] = box
        else:
            row["box"] = None
    return rows


class SegmentationDataset(Dataset):
    """A single manifest split with independent augmentation state.

    ``prompt_mode='mask'`` constructs boxes from labels for a prompted benchmark.
    For image-only inference use ``prompt_mode='provided', require_masks=False``.
    Images stay in [0,255], as expected by SAM's internal normalization.
    Missing active-attribute labels are rejected during training; evaluation may
    permit them with ``require_groups=False`` and reports their count separately.
    """

    def __init__(self, manifest, split, attribute="site", image_size=256,
                 modality="MRI", task="HNSCC", training=False, require_groups=True,
                 prompt_mode="mask", box_jitter=0, require_masks=True):
        if attribute not in ATTRIBUTES:
            raise ValueError(f"attribute must be one of {ATTRIBUTES}")
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        if prompt_mode not in ("mask", "provided"):
            raise ValueError("prompt_mode must be mask or provided")
        if not require_masks and prompt_mode != "provided":
            raise ValueError("Image-only inference requires provided boxes")
        if image_size <= 0 or box_jitter < 0:
            raise ValueError("image_size must be positive and box_jitter nonnegative")
        self.rows = [row for row in read_manifest(manifest, require_masks=require_masks)
                     if row["split"] == split]
        if not self.rows:
            raise ValueError(f"Manifest has no rows for split={split}")
        if (require_groups or training) and any(row[attribute] < 0 for row in self.rows):
            raise ValueError(f"Missing {attribute} labels for the selected split")
        if prompt_mode == "provided" and any(row["box"] is None for row in self.rows):
            raise ValueError("prompt_mode=provided requires a box for every selected sample")
        from .model import hierarchy_indices
        self.modal, self.organ = hierarchy_indices(modality, task)
        self.attribute, self.image_size = attribute, int(image_size)
        self.training, self.prompt_mode = training, prompt_mode
        self.box_jitter, self.require_masks = int(box_jitter), require_masks
        self.manifest = str(Path(manifest).resolve())

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = np.load(row["image"], allow_pickle=False)
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=-1)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Images must have shape (H,W) or (H,W,3)")
        if not np.issubdtype(image.dtype, np.number) or not np.isfinite(image).all():
            raise ValueError("Images must be finite numeric arrays")
        if image.size == 0 or image.min() < 0 or image.max() > 255:
            raise ValueError("Image values must be in [0,255]")
        height, width = image.shape[:2]
        mask = None
        if self.require_masks:
            mask = np.load(row["mask"], allow_pickle=False)
            if mask.shape != (height, width) or not np.isin(mask, (0, 1)).all():
                raise ValueError("Masks must be binary (H,W) arrays matching image dimensions")
        if self.prompt_mode == "mask":
            ys, xs = np.nonzero(mask)
            box = ([float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
                   if len(xs) else [0., 0., float(width), float(height)])
        else:
            box = list(row["box"])
        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError("Box must have positive area inside the source image")
        if self.training and self.box_jitter:
            jitter = torch.randint(self.box_jitter + 1, (4,)).tolist()
            box = [max(0, x0 - jitter[0]), max(0, y0 - jitter[1]),
                   min(width, x1 + jitter[2]), min(height, y1 + jitter[3])]
        tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()
        if (height, width) != (self.image_size, self.image_size):
            tensor = F.interpolate(tensor[None], size=(self.image_size, self.image_size),
                                   mode="bilinear", align_corners=False, antialias=True)[0]
        result = {
            "image": tensor,
            "box": torch.tensor(box, dtype=torch.float32) * torch.tensor(
                [self.image_size / width, self.image_size / height] * 2),
            "modal": torch.tensor(self.modal, dtype=torch.long),
            "organ": torch.tensor(self.organ, dtype=torch.long),
            "group": torch.tensor(row[self.attribute], dtype=torch.long),
            "sample_id": row["sample_id"], "patient_id": row["patient_id"],
            **{attr: row[attr] for attr in ATTRIBUTES},
        }
        if mask is not None:
            target = torch.from_numpy(np.ascontiguousarray(mask)).float()[None]
            if target.shape[-2:] != (self.image_size, self.image_size):
                target = F.interpolate(target[None], size=(self.image_size, self.image_size),
                                       mode="nearest")[0]
            result["mask"] = target
        return result
