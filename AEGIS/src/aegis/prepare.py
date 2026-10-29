"""Create synthetic data, patient splits, or normalized slices from local volumes."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import ATTRIBUTES, MANIFEST_FIELDS, SPLITS, _group, read_manifest


def write_manifest(rows, output):
    """Write relative paths and portable JSON boxes; never overwrite a manifest."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            clean = {key: row.get(key, "") for key in MANIFEST_FIELDS}
            for field in ("image", "mask"):
                if clean[field]:
                    clean[field] = os.path.relpath(Path(clean[field]).resolve(), output.parent)
            if isinstance(clean["box"], (list, tuple)):
                clean["box"] = json.dumps(clean["box"])
            if clean["box"] is None:
                clean["box"] = ""
            writer.writerow(clean)


def synthetic(output, image_size=64, seed=42):
    """Generate 12 artificial samples, with disjoint patients in three splits."""
    output = Path(output).resolve()
    if image_size < 16:
        raise ValueError("Synthetic image_size must be at least 16")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Synthetic output directory must be empty")
    (output / "images").mkdir(parents=True, exist_ok=True)
    (output / "masks").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:image_size, :image_size]
    rows = []
    for split in SPLITS:
        for group in range(2):
            for replicate in range(2):
                sample_id = f"synthetic_{split}_g{group}_{replicate}"
                center = rng.uniform(0.4, 0.6, 2) * image_size
                radius = rng.uniform(0.13, 0.22) * image_size
                mask = (((xx - center[0]) ** 2 + (yy - center[1]) ** 2) < radius**2).astype(np.uint8)
                image = np.clip(30 + 140 * mask + 25 * group + rng.normal(0, 8, mask.shape), 0, 255).astype(np.uint8)
                image = np.repeat(image[..., None], 3, axis=-1)
                image_path, mask_path = output / "images" / f"{sample_id}.npy", output / "masks" / f"{sample_id}.npy"
                np.save(image_path, image)
                np.save(mask_path, mask)
                ys, xs = np.nonzero(mask)
                rows.append({"sample_id": sample_id, "patient_id": sample_id,
                             "image": str(image_path), "mask": str(mask_path), "split": split,
                             "gender": group, "age": group, "site": group,
                             "box": [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]})
    write_manifest(rows, output / "manifest.csv")
    return output / "manifest.csv"


def split_manifest(manifest, output, ratios=(0.8, 0.2, 0.0), seed=42):
    """Deterministically assign sorted patient IDs; no slice-level splitting."""
    if len(ratios) != 3 or any(not math.isfinite(x) or x < 0 for x in ratios) or not math.isclose(sum(ratios), 1.0):
        raise ValueError("Three nonnegative ratios must sum to 1")
    rows = read_manifest(manifest, allow_unassigned=True)
    patients = sorted({row["patient_id"] for row in rows})
    random.Random(seed).shuffle(patients)
    counts = [int(len(patients) * value) for value in ratios]
    # Largest-remainder allocation makes all patients assigned exactly once.
    remainder_order = sorted(range(3), key=lambda i: (len(patients) * ratios[i] - counts[i], -i), reverse=True)
    for i in remainder_order[: len(patients) - sum(counts)]:
        counts[i] += 1
    if any(ratio > 0 and count == 0 for ratio, count in zip(ratios, counts)):
        raise ValueError("Too few patients for the requested nonempty splits")
    assignments, start = {}, 0
    for split, count in zip(SPLITS, counts):
        assignments.update({patient: split for patient in patients[start:start + count]})
        start += count
    for row in rows:
        row["split"] = assignments[row["patient_id"]]
    write_manifest(rows, output)
    return {split: count for split, count in zip(SPLITS, counts)}


def _read_volume_pair(image_path, mask_path):
    if image_path.suffix == ".npy" and mask_path.suffix == ".npy":
        image, mask = np.load(image_path, allow_pickle=False), np.load(mask_path, allow_pickle=False)
    else:
        if not all(str(path).endswith((".nii", ".nii.gz")) for path in (image_path, mask_path)):
            raise ValueError("Volume pairs must both be .npy or NIfTI (.nii/.nii.gz)")
        try:
            import SimpleITK as sitk
        except ImportError as exc:
            raise ImportError("Install the preprocessing extra: pip install '.[preprocess]'") from exc
        image_itk, mask_itk = sitk.ReadImage(str(image_path)), sitk.ReadImage(str(mask_path))
        if image_itk.GetDimension() != 3 or mask_itk.GetDimension() != 3:
            raise ValueError("NIfTI inputs must be 3D scalar volumes")
        if (image_itk.GetSize() != mask_itk.GetSize()
                or not np.allclose(image_itk.GetSpacing(), mask_itk.GetSpacing(), rtol=0, atol=1e-6)
                or not np.allclose(image_itk.GetOrigin(), mask_itk.GetOrigin(), rtol=0, atol=1e-6)
                or not np.allclose(image_itk.GetDirection(), mask_itk.GetDirection(), rtol=0, atol=1e-6)):
            raise ValueError("NIfTI image/mask physical geometry differs; align and review it before preprocessing")
        # Reorient the matched pair together to LPS, so NumPy axis 0 is axial.
        image_itk = sitk.DICOMOrient(image_itk, "LPS")
        mask_itk = sitk.DICOMOrient(mask_itk, "LPS")
        image, mask = sitk.GetArrayFromImage(image_itk), sitk.GetArrayFromImage(mask_itk)
    if image.ndim != 3 or image.shape != mask.shape or any(size == 0 for size in image.shape):
        raise ValueError("Volume arrays must be matching nonempty (depth,height,width)")
    if not np.isfinite(image).all() or not np.isfinite(mask).all():
        raise ValueError("Volumes must be finite")
    if not np.equal(mask, np.floor(mask)).all():
        raise ValueError("Volume masks must contain integer label IDs")
    return image, mask


def volumes(manifest, output, image_size=256, foreground_label=1, min_foreground_pixels=1,
            include_empty=False):
    """Convert explicit pairs; no clinical metadata or patient IDs are inferred.

    Input columns: volume_id,patient_id,image,mask,split,gender,age,site.
    NPY volumes must already be axial D,H,W. NIfTI geometry is checked and the
    pair reoriented together. This does not perform registration or repair.
    """
    if image_size <= 0 or min_foreground_pixels < 0:
        raise ValueError("image_size must be positive and min_foreground_pixels nonnegative")
    manifest, output = Path(manifest).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Preprocessing output directory must be empty")
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"volume_id", "patient_id", "image", "mask", "split"}
        if required - set(reader.fieldnames or []):
            raise ValueError(f"Volume manifest requires {sorted(required)}")
        sources = list(reader)
    if not sources:
        raise ValueError("Volume manifest is empty")
    seen, patient_split = set(), {}
    for row in sources:
        volume_id = row["volume_id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", volume_id) or volume_id in seen:
            raise ValueError("volume_id must be unique and contain only ASCII letters, numbers, _ or -")
        seen.add(volume_id)
        if not row["patient_id"] or row["split"] not in SPLITS:
            raise ValueError("A patient_id and train/val/test split are required")
        if patient_split.setdefault(row["patient_id"], row["split"]) != row["split"]:
            raise ValueError("Patient appears in multiple splits")
        for attr in ATTRIBUTES:
            row[attr] = _group(row.get(attr), attr)
    (output / "images").mkdir(parents=True, exist_ok=True)
    (output / "masks").mkdir(parents=True, exist_ok=True)
    rows, total, filtered, vanished = [], 0, 0, 0
    for source in sources:
        image, labels = _read_volume_pair(manifest.parent / source["image"], manifest.parent / source["mask"])
        for index, (image_slice, label_slice) in enumerate(zip(image, labels)):
            total += 1
            mask = (label_slice == foreground_label).astype(np.uint8)
            count = int(mask.sum())
            if (count == 0 and not include_empty) or (0 < count < min_foreground_pixels):
                filtered += 1
                continue
            image_slice = image_slice.astype(np.float32)
            low, high = float(image_slice.min()), float(image_slice.max())
            normalized = (255 * (image_slice - low) / (high - low)).astype(np.uint8) if high > low else np.zeros_like(mask)
            # Same operation order as the supplied scripts: normalize, resize, replicate.
            image_resized = F.interpolate(torch.from_numpy(normalized.copy())[None, None].float(),
                                          size=(image_size, image_size), mode="bilinear", align_corners=False)[0, 0]
            mask_resized = F.interpolate(torch.from_numpy(mask)[None, None].float(),
                                         size=(image_size, image_size), mode="nearest")[0, 0].numpy().astype(np.uint8)
            if count > 0 and not mask_resized.any():
                vanished += 1
                continue
            rgb = np.repeat(image_resized.round().clamp(0, 255).numpy().astype(np.uint8)[..., None], 3, axis=-1)
            sample_id = f"{source['volume_id']}_slice_{index:04d}"
            image_path, mask_path = output / "images" / f"{sample_id}.npy", output / "masks" / f"{sample_id}.npy"
            np.save(image_path, rgb)
            np.save(mask_path, mask_resized)
            rows.append({"sample_id": sample_id, "patient_id": source["patient_id"],
                         "image": str(image_path), "mask": str(mask_path), "split": source["split"],
                         **{attr: source[attr] for attr in ATTRIBUTES}})
    if not rows:
        raise ValueError("No slices survived preprocessing; check target label and filtering")
    write_manifest(rows, output / "manifest.csv")
    report = {"input_volumes": len(sources), "input_slices": total, "saved_slices": len(rows),
              "filtered_slices": filtered, "foreground_lost_after_resize": vanished,
              "image_size": image_size, "foreground_label": foreground_label,
              "min_foreground_pixels": min_foreground_pixels, "include_empty": include_empty,
              "normalization": "per-slice min-max to uint8 [0,255]",
              "interpolation": "PyTorch bilinear image / nearest mask",
              "scope": "Reconstructed preprocessing; exact original inclusion decisions unavailable"}
    (output / "preprocessing.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("synthetic", help="Create artificial data, never patient data")
    demo.add_argument("--output", required=True)
    demo.add_argument("--image-size", type=int, default=64)
    demo.add_argument("--seed", type=int, default=42)
    split = commands.add_parser("split", help="Reassign an explicit manifest by patient")
    split.add_argument("--manifest", required=True)
    split.add_argument("--output", required=True)
    split.add_argument("--ratios", nargs=3, type=float, default=(0.8, 0.2, 0.0), metavar=("TRAIN", "VAL", "TEST"))
    split.add_argument("--seed", type=int, default=42)
    volume = commands.add_parser("volumes", help="Convert explicit NPY/NIfTI volume pairs to 2D slices")
    volume.add_argument("--manifest", required=True)
    volume.add_argument("--output", required=True)
    volume.add_argument("--image-size", type=int, default=256)
    volume.add_argument("--foreground-label", type=int, default=1)
    volume.add_argument("--min-foreground-pixels", type=int, default=1)
    volume.add_argument("--include-empty", action="store_true")
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    result = {"synthetic": synthetic, "split": split_manifest, "volumes": volumes}[command](**args)
    print(json.dumps(result, indent=2) if isinstance(result, dict) else str(result))


if __name__ == "__main__":
    main()
