# Release validation

Validation date: 2026-10-29. This records software checks performed during release
preparation, not a replication study.

## Environment and results

- Python 3.10, PyTorch 2.6.0+cpu, NumPy 2.2.6, PyYAML 6.0.3.
- pytest 9.2.0; optional NIfTI checks used SimpleITK 2.5.6.
- Full test suite: **36 passed, 3 subtests passed** (CPU).
- Loss checks include analytical EMA weights and gradients, absent groups,
  foreground gating, biological/style behavior, and the alignment reduction scale.
- Data checks include patient leakage, duplicate identifiers, missing metadata,
  box scaling, and inference with no mask or demographic columns.
- Preprocessing checks include NPY conversion and NIfTI geometry mismatch rejection.
- The pipeline test trains through warm-up, resumes with preserved EMA/optimizer/RNG
  state, matches an uninterrupted CPU run exactly, and runs evaluation/inference.
- Real SAM/ConMoAE components pass forward/backward, freezing, task-table expansion,
  and state-loading tests at reduced dimensions.
- A separate full ViT-B check at 256 × 256 passed one-sample forward/backward and
  two-sample inference using random initialization. The output was 1 × 256 × 256,
  the encoder grid 256 × 16 × 16, and the trainable parameter count 5,407,516.
- A synthetic checkpoint with original SAM positional tensor sizes passed the
  1024-to-256 position-embedding adaptation path.

The main commands are `pytest -q`, `python -m build`, and `bash scripts/smoke.sh`.
NIfTI coverage is optional in environments without `.[preprocess]`; that test is
skipped there. Use `pip install -e '.[dev,preprocess]'` to run the full suite.

The source distribution and wheel were built successfully. The wheel was installed
into a separate directory, and `scripts/smoke.sh` completed synthetic generation,
two-epoch training, evaluation, and box-provided inference using that installed
package without importing from the development source tree. The source archive
was checked to include configuration files, documentation, and runnable examples.

## Evidence boundaries

The full-backbone checks use random/synthetic tensors. The end-to-end software
example uses a tiny convolutional model and artificial data. Neither establishes
clinical accuracy, pretrained-checkpoint numerical equivalence, or reproduction
of the paper's tables. No clinical training or GPU benchmarking was performed.
Actual SAM/MedSegX pretrained binaries and original AEGIS experiment checkpoints
were not available for this validation. CUDA behavior remains untested here.

The GitHub workflow is supplied for future pushes; a local pass is not described
as a completed GitHub Actions run. Clinical data, private metadata CSVs, model
weights, local outputs, and the manuscript PDF are excluded from the source
release. See METHOD.md for scientific reconstruction choices and DATA.md for
preprocessing and evaluation assumptions.
