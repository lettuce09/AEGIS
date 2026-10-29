"""AEGIS model construction and strict pretrained checkpoint loading."""

from pathlib import Path
import warnings

import torch
from torch import nn

from .backbones.build_sam import build_sam
from .backbones.medsegx import MedSegX
from .backbones import hierarchy as h


def hierarchy_indices(modality="MRI", task="HNSCC"):
    """Return (modality, [body part, subregion, organ, task]) integer indices."""
    modality = h.modal_dict.get(modality, modality)
    if modality not in h.modal_map:
        raise ValueError(f"Unknown modality {modality!r}; choose {tuple(h.modal_map)}")
    if task not in h.task_idx:
        raise ValueError(f"Unknown task {task!r}; use a task listed in backbones/hierarchy.py")
    organ = []
    for level in (1, 2, 3):
        groups = getattr(h, f"organ_level_{level}_dict")
        mapping = getattr(h, f"organ_level_{level}_map")
        matches = [mapping[name] for name, tasks in groups.items() if task in tasks]
        if len(matches) != 1:
            raise ValueError(f"Task {task!r} has ambiguous/missing hierarchy level {level}.")
        organ.append(matches[0])
    return h.modal_map[modality], organ + [h.task_idx[task]]


class TinySmokeModel(nn.Module):
    """Small convolutional test double; this is NOT the AEGIS architecture."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.GELU(),
                                     nn.Conv2d(8, 8, 3, padding=1), nn.GELU())
        self.decoder = nn.Conv2d(8, 1, 1)

    def forward(self, batch):
        features = self.encoder(batch["image"] / 255.0)
        return self.decoder(features), features


def load_medsegx(model, checkpoint, task_init_from=None):
    """Restore every adapter, hierarchy embedding and decoder tensor.

    Official MedSegX predates the local HNSCC task. Expanding that table is only
    allowed when an explicit ``task_init_from`` task is provided. This option
    is an engineering initialization choice, not a recovered paper setting.
    """
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"MedSegX checkpoint does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict) or not all(isinstance(k, str) for k in state):
        raise ValueError("MedSegX checkpoint must contain a raw state dict or {'model': state_dict}.")
    # The upstream save_parameters() writes keys relative to model.sam.
    # A full AEGIS model.state_dict() is restored by the trainer, not here.
    target = model.sam.state_dict()
    required = {k for k in target if any(part in k for part in
                ("adapter", "modal_embed", "organ_embed", "mask_decoder"))}
    missing = required - state.keys()
    unexpected = state.keys() - target.keys()
    if missing or unexpected:
        raise RuntimeError(f"MedSegX checkpoint keys: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    loaded = {}
    task_key = "image_encoder.organ_embed.4.weight"
    for key in sorted(required):
        value, expected = state[key], target[key]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"MedSegX checkpoint value {key} must be a tensor.")
        if value.shape != expected.shape:
            can_expand = (key == task_key and value.ndim == 2 and
                          value.shape[0] == len(h.task_list) and
                          expected.shape[0] == len(h.task_list) + 1 and
                          value.shape[1] == expected.shape[1])
            if not can_expand or task_init_from is None:
                hint = " Set model.task_init_from explicitly for the official pre-HNSCC checkpoint." if can_expand else ""
                raise RuntimeError(f"MedSegX shape mismatch for {key}: {tuple(value.shape)} vs {tuple(expected.shape)}.{hint}")
            if task_init_from not in h.task_idx or h.task_idx[task_init_from] >= h.task_idx["HNSCC"]:
                raise ValueError("task_init_from must name a task present in the official MedSegX table.")
            expanded = expected.clone()
            hn_index = h.task_idx["HNSCC"]
            expanded[:hn_index].copy_(value[:hn_index])
            expanded[hn_index].copy_(value[h.task_idx[task_init_from]])
            expanded[-1].copy_(value[-1])  # preserve upstream spare/unknown row
            value = expanded
            warnings.warn(f"Expanded official MedSegX task table: HNSCC initialized from {task_init_from}; "
                          "this explicit release choice is not verified as the paper initialization.", stacklevel=2)
        if value.shape != expected.shape:
            raise RuntimeError(f"MedSegX shape mismatch for {key} after adaptation.")
        loaded[key] = value
    target.update(loaded)
    model.sam.load_state_dict(target, strict=True)


def build_model(config, smoke=False, load_pretrained=True):
    """Build a model from the model section of the YAML configuration.

    Set load_pretrained=False only immediately before restoring a complete,
    strictly validated AEGIS model.state_dict() or in architecture unit tests.
    """
    if smoke or config.get("smoke", False):
        return TinySmokeModel()
    if load_pretrained:
        for key in ("sam_checkpoint", "medsegx_checkpoint"):
            if not config.get(key) or not Path(config[key]).is_file():
                raise FileNotFoundError(f"model.{key} must point to a local pretrained checkpoint.")
    sam = build_sam(config.get("model_type", "vit_b"), config.get("image_size", 256),
                    config.get("sam_checkpoint") if load_pretrained else None)
    model = MedSegX(sam, config.get("bottleneck_dim", 16),
                   config.get("embedding_dim", 16), config.get("expert_num", 4))
    if load_pretrained:
        load_medsegx(model, config["medsegx_checkpoint"], config.get("task_init_from"))
    return model
