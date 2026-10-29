"""AEGIS objectives, following Eqs. (2)–(5) of the accompanying paper.

The training loop is responsible for Eq. (1), including the five-epoch warm-up.
Sensitive attributes are training metadata, never model inputs.
"""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _check_groups(groups: Tensor, batch_size: int, device: torch.device) -> None:
    if groups.ndim != 1 or groups.shape[0] != batch_size:
        raise ValueError("groups must have shape [B], matching the batch size")
    if groups.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError("groups must contain integer group IDs")
    if groups.device != device:
        raise ValueError("groups and inputs must be on the same device")
    if torch.any(groups < 0):
        raise ValueError("negative or unknown group IDs are not allowed")


def _binary_masks(masks: Tensor, batch_size: int, device: torch.device) -> Tensor:
    if masks.ndim == 3:
        masks = masks.unsqueeze(1)
    if masks.ndim != 4 or masks.shape[0] != batch_size or masks.shape[1] != 1:
        raise ValueError("binary masks must have shape [B, 1, H, W] or [B, H, W]")
    if masks.device != device:
        raise ValueError("masks and inputs must be on the same device")
    if torch.any((masks != 0) & (masks != 1)):
        raise ValueError("masks must contain only binary values 0 and 1")
    return masks.float()


def per_sample_segmentation_loss(logits: Tensor, masks: Tensor) -> Tensor:
    """Return squared-denominator soft Dice plus mean BCE for each sample.

    Logits have shape [B, 1, H, W]; no batch or sample dimension is squeezed.
    Both Dice smoothing constants are 1e-5, matching the original MONAI defaults.
    """
    if logits.ndim != 4 or logits.shape[1] != 1 or logits.shape[0] == 0:
        raise ValueError("logits must have nonempty shape [B, 1, H, W]")
    masks = _binary_masks(masks, logits.shape[0], logits.device)
    if masks.shape != logits.shape:
        raise ValueError("logits and masks must have the same spatial dimensions")
    logits = logits.float()
    probabilities = logits.sigmoid()
    dims = (1, 2, 3)
    intersection = (probabilities * masks).sum(dim=dims)
    denominator = probabilities.square().sum(dim=dims) + masks.square().sum(dim=dims)
    dice = 1 - (2 * intersection + 1e-5) / (denominator + 1e-5)
    bce = F.binary_cross_entropy_with_logits(logits, masks, reduction="none").mean(dim=dims)
    return dice + bce


class HistoryAwareWeighting(nn.Module):
    """EMA-stabilized subgroup weighting from Eqs. (4) and (5).

    Histories start at zero. Absent groups retain their histories and participate
    in the softmax denominator, but contribute no current loss. Consequently,
    weights are *not* renormalized over the observed groups. Saving state_dict()
    preserves the history required to resume training.
    """

    def __init__(self, num_groups: int, beta: float = 0.9, temperature: float = 0.1):
        super().__init__()
        if isinstance(num_groups, bool) or not isinstance(num_groups, int) or num_groups < 1:
            raise ValueError("num_groups must be a positive integer")
        if not math.isfinite(beta) or not 0 <= beta < 1:
            raise ValueError("beta must be finite and in [0, 1)")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        self.num_groups = num_groups
        self.beta = beta
        self.temperature = temperature
        self.register_buffer("history", torch.zeros(num_groups, dtype=torch.float32))

    @property
    def weights(self) -> Tensor:
        return torch.softmax(self.history.detach() / self.temperature, dim=0)

    def forward(self, losses: Tensor, groups: Tensor, update: bool = True) -> Tensor:
        if losses.ndim != 1 or losses.numel() == 0:
            raise ValueError("losses must have nonempty shape [B]")
        _check_groups(groups, losses.shape[0], losses.device)
        if torch.any(groups >= self.num_groups):
            raise ValueError("group ID exceeds configured num_groups")
        if self.history.device != losses.device:
            raise ValueError("move the weighting module to the same device as losses")
        if not torch.isfinite(losses).all():
            raise ValueError("per-sample losses must be finite")
        group_losses = {
            int(group): losses[groups == group].mean()
            for group in torch.unique(groups)
        }
        if update:
            with torch.no_grad():
                for group, mean_loss in group_losses.items():
                    self.history[group].mul_(self.beta).add_(
                        mean_loss.detach().to(self.history.dtype), alpha=1 - self.beta
                    )
        weights = self.weights
        return sum(weights[group] * mean_loss for group, mean_loss in group_losses.items())


class AttributeConditionalAlignment(nn.Module):
    """Foreground alignment, plus background alignment for style attributes.

    Default behavior follows the written equations: unnormalized prototypes and
    the sum of squared Euclidean distances over ordered group pairs. To inspect
    the earlier code's scaling, set pair_reduction='legacy_mse' and normalize=True.
    That option is an explicit alternative, not an equivalent paper setting.
    """

    def __init__(
        self,
        background_weight: float = 0.05,
        pixel_threshold: float = 2,
        pair_reduction: str = "ordered_sum",
        normalize: bool = False,
    ):
        super().__init__()
        if not math.isfinite(background_weight) or background_weight < 0:
            raise ValueError("background_weight must be finite and nonnegative")
        if not math.isfinite(pixel_threshold) or pixel_threshold < 0:
            raise ValueError("pixel_threshold must be finite and nonnegative")
        if pair_reduction not in {"ordered_sum", "legacy_mse"}:
            raise ValueError("pair_reduction must be 'ordered_sum' or 'legacy_mse'")
        self.background_weight = background_weight
        self.pixel_threshold = pixel_threshold
        self.pair_reduction = pair_reduction
        self.normalize = normalize

    def _align_region(self, features: Tensor, region: Tensor, groups: Tensor,
                      threshold: float) -> Tensor:
        area = region.sum(dim=(2, 3))  # [B, 1], broadcasts only over channels
        valid = area[:, 0] > threshold
        prototypes = (features * region).sum(dim=(2, 3)) / (area + 1e-6)
        if self.normalize:
            prototypes = F.normalize(prototypes, p=2, dim=1)
        means = [
            prototypes[(groups == group) & valid].mean(dim=0)
            for group in torch.unique(groups)
            if torch.any((groups == group) & valid)
        ]
        result = features.sum() * 0.0
        for i in range(len(means)):
            for j in range(i + 1, len(means)):
                squared_distance = (means[i] - means[j]).square()
                if self.pair_reduction == "ordered_sum":
                    result = result + 2 * squared_distance.sum()
                else:
                    result = result + squared_distance.mean()
        return result

    def forward(self, features: Tensor, masks: Tensor, groups: Tensor,
                is_style_attribute: bool = False) -> Tensor:
        if features.ndim != 4 or features.shape[0] == 0 or features.shape[1] == 0:
            raise ValueError("features must have nonempty shape [B, C, H, W]")
        _check_groups(groups, features.shape[0], features.device)
        masks = _binary_masks(masks, features.shape[0], features.device)
        masks = F.interpolate(masks, size=features.shape[-2:], mode="nearest")
        features = features.float()
        result = self._align_region(features, masks, groups, self.pixel_threshold)
        if is_style_attribute and self.background_weight:
            result = result + self.background_weight * self._align_region(
                features, 1 - masks, groups, threshold=0
            )
        return result
