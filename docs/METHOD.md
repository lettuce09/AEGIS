# Method and implementation scope

This release organizes the supplied AEGIS research code around the method in
**AEGIS: Anatomy-Embedded Group-Invariant Segmentation for Fair Medical Foundation
Models** (MICCAI 2026, accepted manuscript supplied by the authors). It is a
corrected, simplified implementation. It is not an archived copy of the exact
training environment that produced the paper's tables.

## Architecture and training information

The model uses SAM ViT-B with MedSegX contextual mixture-of-adapter experts
(ConMoAE). The SAM image backbone and prompt encoder remain frozen; adapter
parameters and the mask decoder are fine-tuned. MedSegX supplies task/modality
context. Demographic labels enter the training losses and evaluation summaries;
they do not enter the model's forward pass. Bounding-box prompts are still needed
at inference. Attribute-agnostic inference does not mean prompt-free inference.

The original scripts use adapter bottleneck dimension 16, embedding dimension 16,
and four experts. These architecture settings are inherited from code; the AEGIS
paper does not enumerate them. The paper specifies 256 × 256 inputs. With the
provided patch size of 16, the encoder feature grid is 16 × 16; several inherited
comments incorrectly describe it as 64 × 64.

## Objectives

For each binary segmentation sample, the task loss is soft Dice loss plus mean
binary cross entropy with logits. The Dice denominator uses squared predictions,
as in the supplied MONAI-based task loss. Numerical smoothing is 1e-5. The batch
task objective is the mean of these per-sample losses.

The total objective follows Eq. (1):

```text
L_total = mean(L_task_i) + active * (lambda_align * L_align + lambda_fair * L_fair)
```

`active` is zero for the first five completed epochs and one thereafter. In a
zero-based training loop, epochs 0–4 use only the task loss; epoch 5 activates
both fairness terms. Historical subgroup losses are not updated during warm-up.
There is no separate alignment-only stage: both terms become active together.

### Attribute-conditional alignment (Eqs. 2–3)

Binary masks are resized to the encoder feature grid with nearest-neighbor
interpolation. A foreground prototype is a spatial mean of features inside the
mask. A sample contributes to foreground subgroup means only if its foreground
area on that grid is **strictly greater than 2 pixels**. Background prototypes
use the complementary mask; samples without any background pixels are excluded
from background subgroup means. Only groups with valid prototypes are paired.

The default objective is the literal squared Euclidean sum in Eq. (3):

```text
L_align = sum_{a != b} ||mean_fg[a] - mean_fg[b]||_2^2
        + lambda_bg * is_site * sum_{a != b} ||mean_bg[a] - mean_bg[b]||_2^2
```

Foreground alignment applies to any selected attribute. Background alignment
applies only to the site/style attribute. Biological attributes such as age and
gender leave the background term inactive. A region with fewer than two valid
groups contributes a differentiable zero.

**Scaling is explicit.** The paper does not specify prototype L2 normalization,
so the default is `normalize=False`. Its `a != b` notation is implemented as an
ordered sum, counting each unordered pair twice, with no division by channel
count. The old experiment files instead normalize each sample's prototype and
sum channel-mean MSE over unordered pairs. To inspect that alternative, use
`AttributeConditionalAlignment(pair_reduction="legacy_mse", normalize=True)`.
It is not mathematically equivalent to the default and may require retuning
`lambda_align`. The default does not claim to recover unpublished training
choices from the old files.

### History-aware weighting (Eqs. 4–5)

Each subgroup has a persistent register `history[k]`, initially zero. For groups
present in a batch:

```text
history[k] = beta * history[k] + (1 - beta) * mean_current_task_loss[k]
weights = softmax(history / temperature)
L_fair = sum_{k present in batch} weights[k] * mean_current_task_loss[k]
```

The EMA update and its weights are detached from autograd. Absent groups keep
their histories and remain in the softmax denominator, but contribute no current
loss. We do not renormalize weights over the observed groups. The state is saved
with training checkpoints. This is the paper's history-aware weighting rule;
it is not the multiplicative adversarial-weight update of classical GroupDRO.

The supplied training scripts optimize one selected sensitive attribute at a
time. The default release setting uses site; age or gender can be selected in
configuration. The paper reports metrics for all attributes but does not specify
a multi-attribute loss-combination rule. This release does not invent one.

## Reported settings

| Setting | Value | Evidence |
| --- | --- | --- |
| Backbone | SAM ViT-B + MedSegX ConMoAE | Paper §§2.1, 3.2 |
| Image size | 256 × 256 | Paper §3.1 |
| Optimizer | AdamW | Paper §3.2 |
| Learning rate | 5e-5 | Paper §3.2 |
| Weight decay | 0.01 | Paper §3.2 |
| Epochs | 80 | Paper §3.2 |
| Batch size | 64 | Paper §§2.3, 3.2 |
| Warm-up | 5 epochs | Paper §3.2 |
| `lambda_align` | 0.1 | Paper §3.2 |
| `lambda_fair` | 1.0 | Paper §3.2 |
| `lambda_bg` | 0.05 | Paper §3.2 |
| Foreground validity threshold | > 2 feature-grid pixels | Paper Eq. (2), §3.2 |
| EMA momentum `beta` | 0.9 | Paper Eq. (4), §3.2 |
| Temperature | 0.1 | Paper Eq. (5), §3.2 |
| Age grouping | ≤ 60 versus > 60 years | Paper §3.1 |
| HNSCC split | 80% training / 20% validation | Paper §3.1 |

The paper says the HNSCC split is random but does not identify the split unit or
provide patient lists. The release uses explicit patient IDs to prevent a patient
from appearing in multiple splits. Newly generated splits are therefore a safe
new protocol, not a claim to reconstruct the original split. Metadata group IDs
must be mapped consistently across splits; missing labels must not silently be
mapped to a majority group.

## Evaluation protocol

The paper reports the mean sample Dice, subgroup mean Dice, and the equal
accuracy gap:

```text
EA_gap = max(subgroup_mean_Dice) - min(subgroup_mean_Dice)
```

In this 2D implementation, a sample is a slice. The mean is therefore
slice-weighted, not a pooled-pixel Dice or a patient-level 3D Dice. Report subgroup
sample counts with the means and report percentage-point gaps when Dice is
expressed as a percentage. A group with no samples has no estimable mean; a gap
with fewer than two observed groups is undefined, not evidence of fairness.

Use the model's single output mask, sigmoid threshold 0.5, and a fixed checkpoint
selected by validation Dice. There is no ground-truth-based selection among
candidate masks. Both-empty Dice is 1, consistently with the original smoothed
metric; empty-positive-slice policies affect results and should be reported.
Ground-truth-derived bounding boxes, when used for prompted evaluation, are
additional oracle information and must be distinguished from independently
supplied boxes or prompt-free segmentation.

## Corrections relative to the supplied development files

The original filenames below refer to the authors' local development snapshot;
the redundant scripts themselves are intentionally not part of this release.

| Original file / behavior | Correction or boundary |
| --- | --- |
| `finetune_new-method.py` imports one loss interface but calls another; it also drops the standalone task term and ignores `lambda_fair` | One explicit objective and matching loss interfaces |
| `utils/fairness_loss.py` computes softmax weights from current losses and differentiates through them | Detached EMA-based weights from the paper |
| `utils/fairness_loss-method.py` divides `[B,C]` prototypes by `[B,1,1,1]` areas | Areas remain `[B,1]`, preventing accidental cross-sample broadcasting |
| The method variant defaults to EMA momentum 0.99 and threshold 5 | Paper defaults 0.9 and 2 |
| Training/validation subsets share one dataset whose augmentation flag is modified | Independent split datasets and explicit split verification |
| Several paths assume `model.module` exists | CPU/single-device operation without `.module` assumptions |
| Old evaluation code selects the best ground-truth Dice among candidate masks | Fixed single-mask evaluation |
| A copied checkpoint loader references the unrelated `CervicalCancer` task | Explicit compatible weight loading, with task-table adaptation documented |
| Inherited README describes MedSegX pretraining and references missing files | AEGIS-specific preparation, training, evaluation, and inference instructions |

The original HNSCC task mapping assigns `head&neck → neck → throat`, whereas the
paper's schematic labels the hierarchy as `Human Body → Head & Neck → Head`.
The release preserves the code's task vocabulary and records this discrepancy;
changing the hierarchy silently would change model conditioning.

## Pretrained task-table compatibility

The official MedSegX source has 111 tasks, while the local experiment snapshot
appends HNSCC as task 111 (zero based). The shared task ordering and all three
hierarchy-index mappings match the official source. The task embedding table
also contains a spare row. When a pre-HNSCC checkpoint is loaded, the release
configuration explicitly initializes the new HNSCC row from PharynxCancer and
moves the original spare row to the new final position. This initialization is
required to make the available upstream weights usable, but the original paper
initialization was not recoverable. HNSCC-expanded checkpoints retain their
existing task embeddings. Other incompatible shapes or missing tensors fail.

## Reproducibility limits

The supplied folder does not contain trained AEGIS checkpoints, immutable
experiment configurations, exact patient splits, or complete training logs that
establish which development revision produced the reported results. The paper
also does not specify every preprocessing decision, the unit behind its cohort
counts, how multi-attribute objectives were combined (if at all), or the complete
Harvard-FairSeg and PI-CAI adaptation/split protocol. Those dataset names alone
are insufficient to reconstruct a benchmark.

Accordingly, this release supplies an executable method and software checks. It
does **not** claim that unit tests, a synthetic smoke run, or untrained-model
inference reproduce the paper's Dice scores. Clinical data, patient-level
metadata, and pretrained model weights are not bundled. Obtain external datasets
and base weights from their original providers under their applicable terms.
