"""Single-device AEGIS fine-tuning with patient-separated validation."""
import argparse
import copy
import json
import platform
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .data import SegmentationDataset
from .losses import AttributeConditionalAlignment, HistoryAwareWeighting, per_sample_segmentation_loss
from .metrics import binary_dice, summarize
from .model import build_model
from .runtime import (atomic_save, load_checkpoint, manifest_sha256, model_input, read_config,
                      resolve_device, restore_rng, rng_state, seed_all, write_json)


def dataset(config, manifest, split, training=False):
    d = config["data"]
    return SegmentationDataset(
        manifest, split, attribute=d["attribute"], image_size=config["model"]["image_size"],
        modality=d.get("modality", "MRI"), task=d.get("task", "HNSCC"),
        training=training, require_groups=training, prompt_mode=d.get("prompt_mode", "mask"),
        box_jitter=d.get("box_jitter", 0) if training else 0,
    )


@torch.no_grad()
def evaluate_loader(model, loader, device):
    model.eval()
    rows = []
    for batch in loader:
        logits, _ = model(model_input(batch, device))
        dice = binary_dice(logits.sigmoid().cpu(), batch["mask"]).tolist()
        for i, value in enumerate(dice):
            row = {"sample_id": batch["sample_id"][i], "patient_id": batch["patient_id"][i], "dice": value}
            row.update({key: int(batch[key][i]) for key in ("gender", "age", "site")})
            rows.append(row)
    if not rows:
        raise ValueError("Evaluation split is empty")
    return rows, summarize(rows)


def validate_config(config):
    t, f, d = config["train"], config["fairness"], config["data"]
    if t["epochs"] < 1 or t["batch_size"] < 1 or t.get("num_workers", 0) < 0:
        raise ValueError("epochs/batch_size must be positive and num_workers nonnegative")
    if t["lr"] <= 0 or t.get("weight_decay", 0.01) < 0:
        raise ValueError("Invalid optimizer parameters")
    if f["num_groups"] < 2 or f["warmup_epochs"] < 0:
        raise ValueError("num_groups must be >=2; warmup_epochs must be nonnegative")
    if f["lambda_align"] < 0 or f["lambda_fair"] < 0:
        raise ValueError("Loss weights must be nonnegative")
    if d["attribute"] not in ("gender", "age", "site"):
        raise ValueError("attribute must be gender, age or site")


def resume_signature(config):
    value = copy.deepcopy(config)
    value["data"].pop("manifest", None)
    for name in ("epochs", "num_workers"):
        value["train"].pop(name, None)
    # Full checkpoints contain all model weights; original file locations are not needed.
    for name in ("sam_checkpoint", "medsegx_checkpoint"):
        value["model"].pop(name, None)
    return value


def run(config, manifest, output, device, resume=None):
    validate_config(config)
    output = Path(output)
    if output.exists() and any(output.iterdir()) and not resume:
        raise ValueError("Output is not empty; choose a new --output or explicitly --resume")
    if resume:
        resume_path = Path(resume).resolve()
        if resume_path != (output / "last.pt").resolve():
            raise ValueError("Resume must use last.pt in the same --output directory")
        if not (output / "best.pt").is_file() or not (output / "history.jsonl").is_file():
            raise ValueError("Resume requires the original best.pt and history.jsonl alongside last.pt")
    output.mkdir(parents=True, exist_ok=True)
    seed_all(config.get("seed", 2025))
    t, f = config["train"], config["fairness"]
    train_data, val_data = dataset(config, manifest, "train", True), dataset(config, manifest, "val")
    if not len(train_data) or not len(val_data):
        raise ValueError("Both train and val splits must be nonempty")
    # Check the entire training group vocabulary before taking any optimization step.
    groups = {row[config["data"]["attribute"]] for row in train_data.rows}
    if groups != set(range(f["num_groups"])):
        raise ValueError(f"Training groups {sorted(groups)} must equal 0..{f['num_groups'] - 1}")
    kwargs = {"batch_size": t["batch_size"], "num_workers": t.get("num_workers", 0),
              "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_data, shuffle=True, drop_last=False, **kwargs)
    val_loader = DataLoader(val_data, shuffle=False, **kwargs)
    checkpoint = load_checkpoint(resume) if resume else None
    digest = manifest_sha256(manifest)
    if checkpoint:
        history = [json.loads(line) for line in (output / "history.jsonl").read_text().splitlines() if line.strip()]
        if not history or [row["epoch"] for row in history] != list(range(1, checkpoint["epoch"] + 2)):
            raise ValueError("History does not match the latest checkpoint; restore a consistent run directory")
        if checkpoint["manifest_sha256"] != digest or resume_signature(checkpoint["config"]) != resume_signature(config):
            raise ValueError("Resume requires the same manifest and training configuration (epochs may increase)")
    model = build_model(config["model"], load_pretrained=checkpoint is None).to(device)
    weighting = HistoryAwareWeighting(f["num_groups"], beta=f["beta"], temperature=f["temperature"]).to(device)
    alignment = AttributeConditionalAlignment(
        background_weight=f["background_weight"], pixel_threshold=f["pixel_threshold"],
        pair_reduction=f.get("pair_reduction", "ordered_sum"), normalize=f.get("normalize", False),
    ).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                 lr=t["lr"], weight_decay=t.get("weight_decay", 0.01))
    use_amp = bool(t.get("amp", False) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start, best = 0, float("-inf")
    if checkpoint:
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        weighting.load_state_dict(checkpoint["weighting"], strict=True)
        scaler.load_state_dict(checkpoint["scaler"])
        start, best = checkpoint["epoch"] + 1, checkpoint["best_val_dice"]
        restore_rng(checkpoint["rng"])
    if start >= t["epochs"]:
        raise ValueError("Checkpoint already reached --epochs (which specifies total epochs)")
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    write_json(output / "environment.json", {"python": platform.python_version(), "torch": str(torch.__version__),
               "numpy": np.__version__, "device": str(device), "manifest_sha256": digest,
               "smoke_only": bool(config["model"].get("smoke", False))})
    print(f"train={len(train_data)} val={len(val_data)} device={device} smoke={config['model'].get('smoke', False)}", flush=True)
    for epoch in range(start, t["epochs"]):
        model.train()
        active = epoch >= f["warmup_epochs"]
        totals = {k: 0.0 for k in ("total", "task", "align", "fair")}
        count = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            target, group = batch["mask"].to(device), batch["group"].to(device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits, features = model(model_input(batch, device))
            # Compute statistics in float32 even when the network uses AMP.
            losses = per_sample_segmentation_loss(logits.float(), target.float())
            task_loss = losses.mean()
            align_loss, fair_loss = task_loss * 0, task_loss * 0
            if active:
                if f["lambda_align"]:
                    align_loss = alignment(features.float(), target.float(), group,
                                           is_style_attribute=config["data"]["attribute"] == "site")
                if f["lambda_fair"]:
                    fair_loss = weighting(losses, group)
            loss = task_loss + f["lambda_align"] * align_loss + f["lambda_fair"] * fair_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss; no checkpoint was written for this epoch")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            n = len(group)
            count += n
            for key, value in zip(totals, (loss, task_loss, align_loss, fair_loss)):
                totals[key] += float(value.detach()) * n
        _, report = evaluate_loader(model, val_loader, device)
        score = report["overall_dice"]
        improved = score > best
        best = max(best, score)
        record = {"epoch": epoch + 1, "fairness_active": active,
                  "loss": {k: v / count for k, v in totals.items()}, "validation": report}
        with (output / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        saved = {"format_version": 1, "epoch": epoch, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "weighting": weighting.state_dict(),
                 "scaler": scaler.state_dict(), "config": config, "best_val_dice": best,
                 "manifest_sha256": digest, "rng": rng_state()}
        atomic_save(saved, output / "last.pt")
        if improved:
            atomic_save(saved, output / "best.pt")
        print(f"epoch {epoch + 1}/{t['epochs']} loss={totals['total']/count:.5f} val_dice={score:.3f}%", flush=True)
    return output / "last.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", help="Override config data.manifest")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", help="Full AEGIS last.pt checkpoint")
    parser.add_argument("--epochs", type=int, help="Total epochs, including resumed epochs")
    parser.add_argument("--attribute", choices=["gender", "age", "site"])
    args = parser.parse_args()
    config = read_config(args.config)
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    if args.attribute:
        config["data"]["attribute"] = args.attribute
    manifest = args.manifest or config["data"]["manifest"]
    config["data"]["manifest"] = str(manifest)
    run(config, manifest, args.output, resolve_device(args.device), args.resume)


if __name__ == "__main__":
    main()
