"""Architecture and checkpoint contract tests without downloaded weights."""

import pytest
import torch

from aegis.model import build_model, hierarchy_indices, load_medsegx
from aegis.backbones import hierarchy as h
from aegis.backbones.build_sam import _build_sam
from aegis.backbones.medsegx import MedSegX


def small_real_model():
    sam = _build_sam(32, 2, 4, [1], image_size=32)
    return MedSegX(sam, bottleneck_dim=4, embedding_dim=4, expert_num=2)


def batch():
    modal, organ = hierarchy_indices()
    return {"image": torch.rand(2, 3, 32, 32) * 255,
            "box": torch.tensor([[4., 4., 24., 24.]]).repeat(2, 1),
            "modal": torch.tensor([modal, modal]),
            "organ": torch.tensor([organ, organ])}


def adaptation_state(model):
    return {k: v.clone() for k, v in model.sam.state_dict().items()
            if any(s in k for s in ("adapter", "modal_embed", "organ_embed", "mask_decoder"))}


def test_real_sam_adapter_forward_and_freezing():
    model = small_real_model()
    for name, param in model.named_parameters():
        assert param.requires_grad == ("adapter" in name or "mask_decoder" in name)
    logits, features = model(batch())
    assert logits.shape == (2, 1, 32, 32)
    assert features.shape == (2, 256, 2, 2)
    (logits.square().mean() + features.square().mean()).backward()
    assert any(p.grad is not None for n, p in model.named_parameters() if "adapter_up" in n)
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)


def test_hierarchy_and_smoke_model_require_no_sensitive_attribute():
    modal, organ = hierarchy_indices("MRI", "HNSCC")
    assert modal == 7
    assert organ[:3] == [0, 2, 7]
    model = build_model({}, smoke=True)
    data = batch()
    logits, _ = model(data)
    assert logits.shape == (2, 1, 32, 32)
    with pytest.raises(ValueError, match="Unknown task"):
        hierarchy_indices(task="unknown")


def test_production_requires_weights():
    with pytest.raises(FileNotFoundError, match="sam_checkpoint"):
        build_model({})


def test_missing_checkpoint_key_fails(tmp_path):
    model = small_real_model()
    state = adaptation_state(model)
    del state[next(iter(state))]
    path = tmp_path / "incomplete.pt"
    torch.save({"model": state}, path)
    with pytest.raises(RuntimeError, match="missing="):
        load_medsegx(model, path)


def test_official_task_expansion_is_explicit(tmp_path):
    model = small_real_model()
    state = adaptation_state(model)
    key = "image_encoder.organ_embed.4.weight"
    rows = torch.randn_like(state[key])
    official = torch.cat([rows[:h.task_idx["HNSCC"]], rows[-1:]], dim=0)
    state[key] = official
    path = tmp_path / "official_shape.pt"
    torch.save({"model": state}, path)
    with pytest.raises(RuntimeError, match="task_init_from"):
        load_medsegx(model, path)
    with pytest.warns(UserWarning, match="HNSCC initialized"):
        load_medsegx(model, path, task_init_from="PharynxCancer")
    actual = model.sam.state_dict()[key]
    torch.testing.assert_close(actual[h.task_idx["HNSCC"]], official[h.task_idx["PharynxCancer"]])
    torch.testing.assert_close(actual[-1], official[-1])


def test_strict_adaptation_round_trip(tmp_path):
    model = small_real_model()
    state = adaptation_state(model)
    path = tmp_path / "adaptation.pt"
    torch.save(state, path)
    load_medsegx(model, path)
    for name, value in state.items():
        torch.testing.assert_close(model.sam.state_dict()[name], value)
