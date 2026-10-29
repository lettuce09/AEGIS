# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Derived from Segment Anything and MedSegX; licensed under Apache-2.0.
# Modified for AEGIS: explicit local checkpoint loading and strict encoder checks.
"""Construct the MedSegX variant of SAM without interactive downloads."""

from functools import partial
from pathlib import Path

import torch
from torch.nn import functional as F

from .modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer


SPECS = {
    "vit_b": (768, 12, 12, [2, 5, 8, 11]),
    "vit_l": (1024, 24, 16, [5, 11, 17, 23]),
    "vit_h": (1280, 32, 16, [7, 15, 23, 31]),
}


def build_sam(model_type="vit_b", image_size=256, checkpoint=None):
    if model_type not in SPECS:
        raise ValueError(f"Unsupported model_type: {model_type!r}; choose {tuple(SPECS)}")
    dim, depth, heads, global_layers = SPECS[model_type]
    return _build_sam(dim, depth, heads, global_layers, image_size, checkpoint)


def _build_sam(encoder_embed_dim, encoder_depth, encoder_num_heads,
               encoder_global_attn_indexes, image_size=256, checkpoint=None):
    """Internal constructor; checkpoint=None is for strict full-state restore/tests."""
    if image_size < 16 or image_size % 16:
        raise ValueError("image_size must be a positive multiple of 16 (at least 16).")
    prompt_dim, patch = 256, 16
    grid = image_size // patch
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth, embed_dim=encoder_embed_dim, img_size=image_size,
            mlp_ratio=4, norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads, patch_size=patch, qkv_bias=True,
            use_rel_pos=True, global_attn_indexes=encoder_global_attn_indexes,
            window_size=14, out_chans=prompt_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_dim, image_embedding_size=(grid, grid),
            input_image_size=(image_size, image_size), mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(depth=2, embedding_dim=prompt_dim,
                                         mlp_dim=2048, num_heads=8),
            transformer_dim=prompt_dim, iou_head_depth=3,
            iou_head_hidden_dim=256, keep_resolution=True,
        ),
        pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375],
    )
    if checkpoint is not None:
        load_sam_encoders(sam, checkpoint, encoder_global_attn_indexes)
    return sam


def load_sam_encoders(sam, checkpoint, global_layers):
    """Load all frozen SAM weights, resizing only documented positional tensors.

    MedSegX's full-resolution decoder differs from SAM's decoder. Its complete
    state is subsequently loaded from the mandatory MedSegX checkpoint.
    """
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"SAM checkpoint does not exist: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("SAM checkpoint must be a raw state dictionary.")
    target = sam.state_dict()
    required = {k for k in target if k.startswith(("image_encoder.", "prompt_encoder."))}
    missing = required - state.keys()
    unexpected = state.keys() - target.keys()
    if missing or unexpected:
        raise RuntimeError(f"SAM checkpoint keys: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    loaded = {}
    for key in sorted(required):
        value, expected = state[key], target[key]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"SAM checkpoint value {key} must be a tensor.")
        if value.shape != expected.shape:
            if key == "image_encoder.pos_embed" and value.ndim == 4 and value.shape[-1] == expected.shape[-1]:
                value = F.interpolate(value.permute(0, 3, 1, 2).float(),
                                      size=expected.shape[1:3], mode="bilinear",
                                      align_corners=False).permute(0, 2, 3, 1)
            elif ".attn.rel_pos_" in key and int(key.split(".")[2]) in global_layers and value.ndim == 2 and value.shape[1] == expected.shape[1]:
                value = F.interpolate(value[None, None].float(), size=expected.shape,
                                      mode="bilinear", align_corners=False)[0, 0]
            else:
                raise RuntimeError(f"SAM shape mismatch for {key}: {tuple(value.shape)} vs {tuple(expected.shape)}")
        if value.shape != expected.shape:
            raise RuntimeError(f"Failed to adapt SAM positional tensor {key}.")
        loaded[key] = value
    target.update(loaded)
    sam.load_state_dict(target, strict=True)
