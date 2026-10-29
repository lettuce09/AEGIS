#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root after installation. Use fresh output directories.
python -m aegis.prepare synthetic --output data/synthetic --image-size 32
python -m aegis.train --config configs/smoke.yaml --output runs/smoke --device cpu
python -m aegis.evaluate --checkpoint runs/smoke/best.pt --manifest data/synthetic/manifest.csv --output runs/smoke/test --device cpu
python -m aegis.infer --checkpoint runs/smoke/best.pt --manifest data/synthetic/manifest.csv --output runs/smoke/predictions --device cpu
