"""Predict masks using user-supplied boxes; no masks or demographics required."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import SegmentationDataset
from .model import build_model
from .runtime import load_checkpoint, model_input, resolve_device, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is not empty; choose a new --output")
    output.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint)
    config = checkpoint["config"]
    d = config["data"]
    data = SegmentationDataset(args.manifest, args.split, attribute=d["attribute"],
        image_size=config["model"]["image_size"], modality=d.get("modality", "MRI"),
        task=d.get("task", "HNSCC"), require_groups=False, require_masks=False,
        prompt_mode="provided", training=False)
    if not len(data):
        raise ValueError("Inference split is empty")
    model = build_model(config["model"], load_pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in DataLoader(data, batch_size=args.batch_size):
            logits, _ = model(model_input(batch, device))
            masks = (logits.sigmoid() > 0.5).cpu().numpy().astype(np.uint8)
            for name, mask in zip(batch["sample_id"], masks):
                # Numeric filenames prevent identifiers from becoming path components.
                filename = f"prediction_{len(predictions):06d}.npy"
                np.save(output / filename, mask[0], allow_pickle=False)
                predictions.append({"sample_id": name, "prediction": filename})
    write_json(output / "predictions.json", {"predictions": predictions,
        "spatial_grid": "resized model input; not restored to original volume",
        "image_size": config["model"]["image_size"],
        "smoke_only": bool(config["model"].get("smoke", False))})
    print(f"Wrote {len(predictions)} masks to {output}")


if __name__ == "__main__":
    main()
