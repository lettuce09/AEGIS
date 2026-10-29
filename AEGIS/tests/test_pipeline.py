"""A real CLI round trip plus exact CPU resume of the small software test model."""
import copy
import csv
import json
import subprocess
import sys
from pathlib import Path

import torch

from aegis.prepare import synthetic
from aegis.runtime import load_checkpoint, read_config
from aegis.train import run


def test_warmup_resume_evaluation_and_unlabeled_inference(tmp_path):
    torch.set_num_threads(1)
    manifest = synthetic(tmp_path / "data", image_size=32)
    config = read_config(Path(__file__).resolve().parents[1] / "configs/smoke.yaml")
    config["data"]["manifest"] = str(manifest)
    config["train"]["epochs"] = 1
    out = tmp_path / "resumed"
    run(config, manifest, out, torch.device("cpu"))
    warmup = load_checkpoint(out / "last.pt")
    assert torch.equal(warmup["weighting"]["history"], torch.zeros(2))
    config["train"]["epochs"] = 2
    run(config, manifest, out, torch.device("cpu"), out / "last.pt")
    resumed = load_checkpoint(out / "last.pt")
    assert torch.all(resumed["weighting"]["history"] > 0)
    continuous_out = tmp_path / "continuous"
    run(copy.deepcopy(config), manifest, continuous_out, torch.device("cpu"))
    continuous = load_checkpoint(continuous_out / "last.pt")
    for key in continuous["model"]:
        torch.testing.assert_close(resumed["model"][key], continuous["model"][key], rtol=0, atol=0)
    torch.testing.assert_close(resumed["weighting"]["history"], continuous["weighting"]["history"], rtol=0, atol=0)
    subprocess.run([sys.executable, "-m", "aegis.evaluate", "--checkpoint", str(out / "best.pt"),
                    "--manifest", str(manifest), "--output", str(tmp_path / "eval"), "--device", "cpu"], check=True)
    report = json.loads((tmp_path / "eval/summary.json").read_text())
    assert report["n_samples"] == 4
    assert report["smoke_only"] is True
    # Delete labels AND group columns from the inference input.
    with manifest.open() as handle:
        rows = list(csv.DictReader(handle))
    unlabeled = manifest.with_name("unlabeled.csv")
    fields = ["sample_id", "patient_id", "image", "split", "box"]
    with unlabeled.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    subprocess.run([sys.executable, "-m", "aegis.infer", "--checkpoint", str(out / "best.pt"),
                    "--manifest", str(unlabeled), "--output", str(tmp_path / "predictions"), "--device", "cpu"], check=True)
    assert len(list((tmp_path / "predictions").glob("*.npy"))) == 4
