# AEGIS

**Anatomy-Embedded Group-Invariant Segmentation for Fair Medical Foundation Models**

MICCAI 2026 · [中文说明](README.zh-CN.md)

Sen Wang, Zhaoyi Zhan, Zheng Xing, Minqing Zhang, Chloe Meng Jiang, Zhiliang Wang,
Ruimao Zhang, Xiaohui Duan, and Weibing Zhao.

AEGIS adapts MedSegX with attribute-conditional foreground/background prototype
alignment and EMA-based subgroup loss weighting. Sensitive labels are used during
training and subgroup evaluation; the segmentation network does not consume them.

This is a compact, cleaned implementation reconstructed from the paper and the
available experiment code. It includes data preparation, training, checkpoint
resume, inference, and subgroup evaluation. **It has not been validated to reproduce
the paper's numerical results.** Original split manifests, trained AEGIS weights,
and complete external-dataset preparation pipelines were not available in the
source snapshot. [Method and reconstruction notes](docs/METHOD.md) describe the
known differences, including the alignment-loss scale and HNSCC initialization.

## Installation

Use Python 3.10 or later in a fresh environment. Install a suitable PyTorch build
for your hardware, then install this package. For a CPU software check:

```bash
git clone https://github.com/lettuce09/AEGIS.git
cd AEGIS
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
```

For GPU training, use the appropriate CUDA-enabled PyTorch installation instead
of the CPU command. The release has a single-device training loop; set `--device
cuda:0`. Actual training memory depends on batch size and image resolution. The
paper used RTX 4090 GPUs and batch size 64; this release has not benchmarked that
memory requirement. Reduce batch size if necessary and record that protocol change.

## Quick software check

```bash
bash scripts/smoke.sh
pytest -q
```

The script creates artificial images, trains a tiny convolutional model for two
epochs, evaluates a separate synthetic test split, and writes predicted masks.
The tiny model is a test double, **not MedSegX or an AEGIS performance demonstration**.
Tests additionally check real SAM/ConMoAE components at reduced dimensions,
loss values and gradients, patient separation, and exact CPU checkpoint resume.
Use fresh `data/synthetic` and `runs/smoke` directories when rerunning the script.

## Prepare data and pretrained models

See [docs/DATA.md](docs/DATA.md) for the full manifest schema, preprocessing, group
coding and external-dataset limitations. Neither patient data nor demographic
index tables are distributed. Do not commit local data or experiment outputs.

A slice manifest contains explicit, globally consistent patient identifiers and
relative paths to NumPy arrays:

```csv
sample_id,patient_id,image,mask,split,gender,age,site,box
example_001,synthetic_patient_001,images/example_001.npy,masks/example_001.npy,train,0,0,0,"[20, 20, 80, 80]"
```

This row illustrates the schema; it is not a patient record or a bundled array.
`image` is `(H,W,3)` or `(H,W)` in `[0,255]`; `mask` is binary `(H,W)`.
`split` is `train`, `val`, or `test`. All slices and scans from one patient must
remain in one split. Missing demographics use `-1` and are rejected for the
selected training attribute; evaluation reports missing counts.

Obtain the original weights from their maintainers:

- [SAM ViT-B](https://github.com/facebookresearch/segment-anything#model-checkpoints): `sam_vit_b_01ec64.pth`.
- [MedSegX ViT-B](https://github.com/MedSegX/MedSegX-code#%EF%B8%8F-model): follow the checkpoint link in the upstream README.

Put them in `checkpoints/` or edit `configs/hnscc.yaml`. No weights are downloaded
automatically. Official MedSegX predates the local HNSCC task; the configuration
explicitly sets `task_init_from: PharynxCancer` to initialize the added task row.
This is a documented release choice, not a recovered paper setting. If you have
an original HNSCC-expanded MedSegX checkpoint with matching dimensions, its
existing task row is retained. Incompatible or incomplete checkpoints fail clearly.

## Train, evaluate, and infer

```bash
python -m aegis.train \
  --config configs/hnscc.yaml --manifest data/manifest.csv \
  --attribute site --output runs/hnscc-site --device cuda:0

# Resume the last completed epoch; --epochs means the total epoch count.
python -m aegis.train \
  --config configs/hnscc.yaml --manifest data/manifest.csv \
  --attribute site --output runs/hnscc-site --device cuda:0 \
  --resume runs/hnscc-site/last.pt --epochs 80

# Select best.pt using validation Dice; evaluate a held-out test split once.
python -m aegis.evaluate \
  --checkpoint runs/hnscc-site/best.pt --manifest data/manifest.csv \
  --split test --output runs/hnscc-site/test --device cuda:0

# User-provided bounding boxes; no segmentation masks or demographics needed.
python -m aegis.infer \
  --checkpoint runs/hnscc-site/best.pt --manifest data/inference.csv \
  --output runs/hnscc-site/predictions --device cuda:0
```

Training operates on **one selected attribute per run** (`gender`, `age`, or
`site`). The paper does not specify an unambiguous joint multi-attribute objective,
so this release does not invent one. Evaluate every available attribute on the
same fixed predictions. Change `fairness.num_groups` for attributes with more than
two categories; group IDs must be contiguous from zero and present in training.

The default training/evaluation prompt mode derives a box from the ground truth:
this is an explicitly **box-prompted benchmark**, not fully automatic segmentation.
Use `--prompt-mode provided` when evaluating externally supplied boxes. Inference
always requires provided boxes and never reads mask arrays. Predictions are saved
on the resized 2D model grid; this release does not reconstruct 3D clinical volumes.

`summary.json` reports slice-averaged Dice (%), subgroup Dice, subgroup counts,
missing metadata counts, worst-group Dice, and **EA Gap = max subgroup Dice − min
subgroup Dice**, in percentage points. EA Gap is unavailable when fewer than two
groups are observed. `per_sample.csv` is local research output, excluded from Git.

`last.pt` includes model, optimizer, AMP scaler, EMA history, configuration, RNG
state, and a manifest checksum. `best.pt` is chosen only by validation Dice. Resume uses the latest `last.pt` in its original run directory with matching
`history.jsonl` and `best.pt`; copied partial runs are rejected. Full
release checkpoints contain the backbone and do not require the original pretrained
files at inference time. Determinism is tested on the CPU smoke configuration;
GPU/library changes may change results.

## Paper configuration

| Parameter | Value |
| --- | --- |
| Backbone / input | MedSegX SAM ViT-B / 256 × 256 |
| Trainable parameters | ConMoAE adapters and mask decoder |
| Optimizer / learning rate / weight decay | AdamW / 5e-5 / 0.01 |
| Epochs / batch size / warm-up | 80 / 64 / 5 epochs |
| Alignment / fairness / background weights | 0.1 / 1.0 / 0.05 |
| EMA momentum / temperature | 0.9 / 0.1 |
| Foreground validity | more than 2 pixels on the feature grid |

For ablations, set `lambda_align: 0` and/or `lambda_fair: 0` in a copied config.
The objective-only ablation uses AEGIS history weighting; it is not a substitute
implementation of the separately reported GroupDRO baseline. The original paper's
8:2 split is not recoverable from this snapshot. Patient-level splitting in this
release is a reproducibility safeguard; it is not claimed to be that original split.

## Repository layout

```text
configs/             paper-oriented and synthetic smoke configurations
src/aegis/
  backbones/         minimal adapted SAM/MedSegX implementation and hierarchy
  losses.py          segmentation, conditional alignment, historical weighting
  data.py            checked patient-disjoint manifest loader
  prepare.py         synthetic data, patient splitting, local volume preprocessing
  train.py           single-device training and checkpoint resume
  evaluate.py        fixed-checkpoint Dice and subgroup evaluation
  infer.py           image + supplied box inference
  metrics.py         slice Dice and max-minus-min EA gap
  runtime.py         checkpoint and reproducibility utilities
docs/                data protocol, method mapping, release validation
scripts/smoke.sh      complete artificial-data software check
tests/               numerical, data, model, and pipeline checks
```

## Citation and acknowledgements

```bibtex
@inproceedings{wang2026aegis,
  title={AEGIS: Anatomy-Embedded Group-Invariant Segmentation for Fair Medical Foundation Models},
  author={Wang, Sen and Zhan, Zhaoyi and Xing, Zheng and Zhang, Minqing and Jiang, Chloe Meng and Wang, Zhiliang and Zhang, Ruimao and Duan, Xiaohui and Zhao, Weibing},
  booktitle={Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  year={2026}
}
```

Proceedings page numbers and a DOI are omitted until verified. Please also cite
[MedSegX](https://github.com/MedSegX/MedSegX-code) and
[SAM](https://github.com/facebookresearch/segment-anything) when using their code
or weights. This repository retains their Apache-2.0 notices; see [LICENSE](LICENSE)
and [NOTICE](NOTICE). Dataset and checkpoint terms remain those of their original
providers. This research implementation is not a validated clinical device.

The release organization follows the installation/data/checkpoint/training pattern
of [MedSegX](https://github.com/MedSegX/MedSegX-code) and
[MedSAM](https://github.com/bowang-lab/MedSAM), with a machine-readable citation
following [GitHub's citation-file guidance](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files).
