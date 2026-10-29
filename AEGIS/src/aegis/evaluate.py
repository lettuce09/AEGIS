"""Evaluate a fixed checkpoint without selecting models on the test split."""
import argparse
import csv
from pathlib import Path

from torch.utils.data import DataLoader

from .model import build_model
from .runtime import load_checkpoint, resolve_device, write_json
from .train import dataset, evaluate_loader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--prompt-mode", choices=["mask", "provided"], default=None)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is not empty; choose a new --output")
    output.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint)
    config = checkpoint["config"]
    if args.prompt_mode:
        config["data"]["prompt_mode"] = args.prompt_mode
    data = dataset(config, args.manifest, args.split)
    model = build_model(config["model"], load_pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    rows, report = evaluate_loader(model, DataLoader(data, batch_size=args.batch_size), device)
    report.update({"split": args.split, "checkpoint_epoch": checkpoint["epoch"] + 1,
                   "prompt_mode": config["data"].get("prompt_mode", "mask"),
                   "smoke_only": bool(config["model"].get("smoke", False)),
                   "aggregation": "mean per 2D slice; dice in %, ea_gap in percentage points"})
    write_json(output / "summary.json", report)
    with (output / "per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Dice={report['overall_dice']:.3f}%; results: {output}")


if __name__ == "__main__":
    main()
