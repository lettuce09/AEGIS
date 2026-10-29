# Data and evaluation protocol

AEGIS accepts explicit manifests of 2D images, binary masks, patient identifiers,
and demographic group labels. Clinical images, patient metadata, original split
lists, and trained study checkpoints are not included in this repository. Keep
all real manifests, images, labels, predictions, and patient-level results in
local ignored data/output directories.

This release reconstructs a usable data interface from the supplied research
scripts. The paper describes normalized axial MRI slices resized to 256 × 256
and an 8:2 training/validation split. The supplied material does not identify the
exact original split membership, whether the paper's random split was at patient
or slice level, or which of the conflicting empty-slice filters was used. The
patient-level split checks and preprocessing choices below are explicit release
behavior, not evidence that the original experiments used the same choices.

## Slice manifest

Use a UTF-8 CSV with one row per image/mask pair:

```csv
sample_id,patient_id,image,mask,split,gender,age,site,box
example001,person001,images/example001.npy,masks/example001.npy,train,0,1,0,
example002,person002,images/example002.npy,masks/example002.npy,val,1,0,1,
example003,person003,images/example003.npy,masks/example003.npy,test,0,0,0,"[12, 20, 96, 120]"
```

These are invented identifiers and paths; the example is a schema, not a dataset.
All paths are relative to the CSV's directory. Absolute paths also work locally.
Use stable pseudonymous patient identifiers shared across every slice, scan,
modality, and timepoint of the same person. If identifiers are only unique within
an institution, create a globally unique local patient key first. The code never
infers a patient identifier from a filename or infers demographics from an image.
Identifiers stay strings, so leading zeros are preserved; surrounding whitespace
is trimmed.

| Column | Meaning |
| --- | --- |
| `sample_id` | Unique string identifying one slice; required. |
| `patient_id` | Stable patient string; required. Every occurrence must have the same split. |
| `image` | Path to a NumPy `.npy` image; required. |
| `mask` | Path to the corresponding binary `.npy` mask; required for training/evaluation. Ignored entirely for image-only inference. |
| `split` | `train`, `val`, or `test`; required. The split utility additionally accepts empty cells before assignment. |
| `gender` | Integer group code, or `-1`/empty for missing. |
| `age` | Integer age **group**, not age in years; `-1`/empty for missing. |
| `site` | Integer institution/domain code, or `-1`/empty for missing. |
| `box` | Optional JSON array `[x_min, y_min, x_max, y_max]` in source-image pixel coordinates. Required when using provided prompts. |

The recovered HNSCC conventions are:

| Attribute | Group 0 | Group 1 |
| --- | --- | --- |
| `gender` | Female | Male |
| `age` | Age ≤ 60 years | Age > 60 years |
| `site` | Institution/domain A | Institution/domain B |

The site letters are local placeholders; keep their mapping fixed across splits
and record it in your own protocol. These gender codes reflect the supplied
research metadata and do not define every possible gender category. Do not map
unknown values to a known group. Other tasks may define their own nonnegative
integer labels. Training requires the selected attribute's observed labels to
be exactly `0 .. fairness.num_groups - 1`, with all configured groups present.
Missing labels for that selected attribute are rejected during training. Other
attributes may be missing and are counted separately in evaluation.

### Array contract

- Images: `(H, W, 3)` RGB or `(H, W)` grayscale NumPy arrays. Preprocessing writes
  `uint8`; the loader accepts finite numeric values in `[0,255]`. Grayscale is
  repeated across three channels. Do not divide by 255 before loading: the SAM
  backbone performs its own normalization.
- Masks: `(H, W)` with only `0` and `1`, matching the source image's spatial shape.
  Convert multiclass labels to an explicitly chosen binary target beforehand.
- The loader resizes images with bilinear interpolation and masks with nearest
  interpolation to the configured square input size. It scales x and y box
  coordinates independently, including for non-square images.
- A box must have positive area and remain within the source image:
  `0 ≤ x_min < x_max ≤ W` and `0 ≤ y_min < y_max ≤ H`.
- Training/evaluation return `image` float CHW in `[0,255]`, `mask` float 1HW,
  `box` float `[4]`, `modal` integer, `organ` integer `[4]`, the selected `group`
  integer, all three attribute codes, and string sample/patient identifiers.
  Image-only inference omits `mask`.

The loader rejects duplicate sample identifiers, duplicate image or mask paths,
missing required files, invalid group codes, and patients appearing in different
splits. Paths are resolved before duplicate detection. It checks image and mask
contents when each sample is loaded. Distinct copies of the same scan cannot be
identified automatically; curate the manifest before running an experiment.

## Prompts and annotation-free inference

`data.prompt_mode: mask` constructs a tight box from the ground-truth foreground
for a **box-prompted segmentation benchmark**. The right and bottom box edges are
one past the largest foreground coordinates. An empty mask gets a full-image
box. This setting uses target annotations to supply localization; it is not an
automatic tumor detection experiment.

`data.prompt_mode: provided` reads boxes from the manifest. If training uses a
positive `box_jitter`, each edge is expanded by a random integer from zero
through that value, clipped to the source image. Validation and evaluation do
not apply this perturbation.

For actual image-only inference, provide externally obtained boxes and use:

```bash
python -m aegis.infer \
  --checkpoint outputs/hnscc/best.pt \
  --manifest data/inference.csv \
  --split test \
  --output outputs/predictions
```

An inference CSV can omit mask and demographic columns:

```csv
sample_id,patient_id,image,split,box
example001,person001,images/example001.npy,test,"[12, 20, 96, 120]"
```

The inference path does not read or validate the mask column, even if a reused
manifest contains stale mask paths. Prediction arrays are on the resized model
input grid. Reconstructing original-resolution 3D volumes and physical geometry
is outside this interface.

## Synthetic software check

Generate 12 artificial images with different patients in all three splits and
both groups present in each split:

```bash
python -m aegis.prepare synthetic --output data/synthetic --image-size 64 --seed 42
python -m aegis.train \
  --config configs/smoke.yaml \
  --manifest data/synthetic/manifest.csv \
  --output outputs/smoke \
  --device cpu
python -m aegis.evaluate \
  --checkpoint outputs/smoke/best.pt \
  --manifest data/synthetic/manifest.csv \
  --split test \
  --output outputs/smoke-evaluation \
  --device cpu
```

`configs/smoke.yaml` uses a tiny random model and 32 × 32 inputs. The synthetic
images are resized by the loader. This tests software execution; its numbers
have no relationship to the paper's clinical performance. Output directories
must be new or empty.

## Patient-level splitting

Prefer a reviewed, existing patient split whenever one is available. To create
a new split, first prepare an explicit slice manifest. Its `split` cells may be
empty or contain a valid existing patient-disjoint split:

```bash
python -m aegis.prepare split \
  --manifest data/all_slices.csv \
  --output data/splits/manifest.csv \
  --ratios 0.8 0.2 0.0 \
  --seed 2025
```

The three ratios correspond to train/validation/test and must be nonnegative
and sum to one. Patients are sorted, shuffled with a local seeded generator,
and allocated using largest-remainder rounding. All of a patient's slices
receive the same split. The output uses paths relative to its new directory;
image files are not copied. The utility refuses to overwrite an existing CSV.

This is a patient-level **random** split, without demographic stratification.
Inspect group counts in every split. Too few patients may make a requested split
empty, in which case the utility raises an error. An 8:2:0 split contains no test
set: evaluate its held-out part with `--split val` and call it validation.
For an independently held-out test set, define that protocol explicitly, for
example `--ratios 0.6 0.2 0.2`; do not label that example as the paper's original
split. The trainer selects checkpoints on validation Dice. Keep the test set
separate from model and hyperparameter selection.

## Optional volume preprocessing

The generic converter accepts matched 3D NumPy or NIfTI image/label pairs.
Install the optional NIfTI dependency when needed:

```bash
pip install -e '.[preprocess]'
```

Create a **volume** CSV, distinct from the resulting slice manifest:

```csv
volume_id,patient_id,image,mask,split,gender,age,site
volume001,person001,raw/image001.nii.gz,raw/label001.nii.gz,train,0,1,0
volume002,person002,raw/image002.nii.gz,raw/label002.nii.gz,val,1,0,1
```

`volume_id` must be unique and contain only ASCII letters, numbers, underscores,
or hyphens. Image and label paths are relative to this CSV. Use a preassigned
patient-disjoint split; the converter does not derive or repair clinical
metadata.

```bash
python -m aegis.prepare volumes \
  --manifest data/volumes.csv \
  --output data/preprocessed \
  --image-size 256 \
  --foreground-label 1 \
  --min-foreground-pixels 1
```

The converter performs these steps:

1. Read matched scalar volumes. `.npy` inputs must already be ordered as axial
   `(depth, height, width)`. For NIfTI, require equal size, spacing, origin, and
   direction, then reorient the matched pair together to LPS. It does not
   register images, repair annotation geometry, or resample oblique acquisitions
   into a new physical grid.
2. Select `label == foreground_label`. Label arrays must contain finite integer
   label IDs. Image arrays must be finite.
3. By default, discard empty target slices and slices with positive foreground
   smaller than `min_foreground_pixels`, measured **before resizing**. Add
   `--include-empty` to retain empty slices. A positive target that disappears
   during resizing is excluded and counted separately.
4. Normalize each retained image slice independently by min–max scaling to
   `uint8 [0,255]`; constant slices become zero. Resize to the specified square
   dimensions with PyTorch bilinear interpolation; resize masks with nearest
   interpolation. Repeat grayscale into three channels.
5. Write `images/`, `masks/`, `manifest.csv`, and `preprocessing.json`. The JSON
   records options and counts of retained/filtered slices.

The supplied historical scripts disagree about retaining empty slices, and do
not establish the exact inclusion policy used for every reported result. The
converter's defaults are therefore declared release choices. Its PyTorch
interpolation can differ numerically from the historical OpenCV implementation.
There is no established original foreground-pixel threshold to recover from the
paper. Changing the target label, threshold, or empty-slice policy changes the
evaluation population; record those choices and apply them consistently to
comparison methods. Never silently copy label geometry onto an image to hide a
misalignment.

## Evaluation outputs and units

`python -m aegis.evaluate` writes a local `per_sample.csv` and `summary.json`.
Predicted probabilities are thresholded at `> 0.5`. Dice is computed separately
for each 2D sample. Two empty masks score 1; if exactly one mask is empty, Dice
is 0. The summary uses the arithmetic mean of sample Dice, so patients with more
slices contribute more samples. This is not volumetric Dice or a patient-level
macro average.

For each attribute, the summary reports subgroup Dice means and counts,
missing-label counts, worst-group Dice, and:

```text
EA gap = highest observed subgroup mean Dice − lowest observed subgroup mean Dice
```

Summary Dice values are percentages and EA gaps are **percentage points**.
`per_sample.csv` stores individual Dice as fractions in `[0,1]`. Missing labels
remain in overall Dice but are excluded from that attribute's subgroup
comparison. If fewer than two groups are observed, EA is `null`, not zero.
Always inspect group coverage and counts before comparing gaps. The release does
not implement inferential significance tests, patient bootstrap intervals, HD95,
or a replacement for the paper's unavailable evaluation records.

## Existing MedSegX exports and public datasets

Existing `npy_imgs/` and `npy_gts/` exports can be referenced directly in a slice
manifest; no directory reorganization is needed. Populate patient IDs, splits,
and group labels from your reviewed metadata. Do not derive these fields from
filename prefixes without validating the mapping. The original private HNSCC
metadata tables are intentionally not shipped.

Harvard-FairSeg and PI-CAI are not complete benchmark integrations in this
release. Their pre-exported **binary image/mask pairs** may be represented by the
same interface after dataset-specific curation. Raw dataset ingestion, official
split reconstruction, demographic joins, task-specific preprocessing, multiple
structures/classes, and exact original benchmark checkpoint/evaluation protocols
are not provided. Use the appropriate modality and anatomy/task tokens from
`src/aegis/backbones/hierarchy.py`; the default `MRI`/`HNSCC` context is not valid
for every external target. Converting a public dataset into this schema does
not establish reproduction of the paper's external results.
