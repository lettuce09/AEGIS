# Derived from MedSegX (https://github.com/MedSegX/MedSegX-code), Apache-2.0.
# Modified for AEGIS: explicit batch schema and PEFT freeze policy; checkpoint
# handling is provided by aegis.model. Original ConMoAE parameter names retained.

import math
from typing import Type

import torch
import torch.nn as nn

from .modeling import Sam
from .modeling.common import MLPBlock
from .hierarchy import (modal_map_idx, organ_level_1_map_idx, organ_level_2_map_idx,
                        organ_level_3_map_idx, task_list)


def get_index(map_idx):
    """Build the lookup without mutating the shared hierarchy constants."""
    return torch.tensor([map_idx.get(i, 0) for i in range(max(map_idx) + 1)],
                        dtype=torch.long)


class MoEAdaptMLPBlock(nn.Module):
    def __init__(
        self,
        mlp: MLPBlock,
        expert_num: int,
        embedding_num: int,
        embedding_dim: int,
        adapter_bn: int = 64,
        adapter_act: Type[nn.Module] = nn.GELU,
        adapter_dropout: float = 0.1,
        adapter_scalar: float = 0.1,
    ) -> None:
        super().__init__()
        self.mlp = mlp
        self.expert_num = expert_num

        dim = mlp.embedding_dim
        self.adapter_input_embed = nn.Linear(dim, embedding_dim)
        self.adapter_gate = nn.Linear(embedding_dim * embedding_num, expert_num)
        self.adapter_organ_gate = nn.Linear(embedding_dim * 4, 4)

        self.adapter_down = nn.ModuleList([
            nn.Linear(dim, adapter_bn) for _ in range(expert_num)
        ])
        self.adapter_up = nn.ModuleList([
            nn.Linear(adapter_bn, dim) for _ in range(expert_num)
        ])
        self.adapter_act = adapter_act()
        self.adapter_dropout = adapter_dropout
        if adapter_scalar is None:
            self.adapter_scale = nn.Parameter(torch.ones(1))
        else:
            self.adapter_scale = adapter_scalar

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for i in range(self.expert_num):
            nn.init.kaiming_uniform_(self.adapter_down[i].weight, a=math.sqrt(5))
            nn.init.zeros_(self.adapter_down[i].bias)
            nn.init.zeros_(self.adapter_up[i].weight)
            nn.init.zeros_(self.adapter_up[i].bias)

    def forward(self, x: torch.Tensor,
                modal: torch.Tensor, organ: torch.Tensor) -> torch.Tensor:
        adpt = []
        for i in range(self.expert_num):
            adpt.append(self.adapter_up[i](nn.functional.dropout(
                self.adapter_act(self.adapter_down[i](x)),
                p=self.adapter_dropout, training=self.training)))
        adpt = torch.stack(adpt, dim=1)

        organ_0, organ_1, organ_2, organ_3, task = organ
        organ = torch.cat([organ_0, organ_1, organ_2, organ_3], dim=-1)
        organ_gate = self.adapter_organ_gate(organ)
        organ_gate = torch.softmax(organ_gate, dim=-1)
        organ = torch.stack([organ_0, organ_1, organ_2, organ_3], dim=1)
        organ = torch.einsum('bec,be->bc', organ, organ_gate)

        input = x.mean(dim=1).mean(dim=1)
        input = self.adapter_input_embed(input)

        gate = torch.cat([organ, task, modal, input], dim=-1)
        gate = self.adapter_gate(gate)
        gate = torch.softmax(gate, dim=1)

        adpt = torch.einsum('bemnc,be->bmnc', adpt, gate)
        return self.mlp(x) + adpt * self.adapter_scale, gate


class MedSegX(nn.Module):
    """Applies Tree MoE Adapter to SAM's image encoder.

    Args:
        sam: segment anything model, see 'segment_anything' dir
        bottleneck_dim: bottleneck dimension of adapter
        embedding_dim: modal and organ embedding dimension
        expert_num: number of experts in MoE adapter
        pos: which layer to apply adapter
    """

    def __init__(self, sam: Sam, bottleneck_dim: int, embedding_dim: int,
                 expert_num: int, pos: list = None):
        super(MedSegX, self).__init__()

        assert bottleneck_dim > 0
        assert embedding_dim > 0
        assert expert_num > 0

        # assign Adapter layer position (all layers by default)
        if pos:
            self.pos = pos
        else:
            self.pos = list(range(len(sam.image_encoder.blocks)))

        # freeze SAM image and prompt encoder
        for param in sam.image_encoder.parameters():
            param.requires_grad = False
        for param in sam.prompt_encoder.parameters():
            param.requires_grad = False

        # modality and organ embedding index
        modal_index = get_index(modal_map_idx)
        organ_index_1 = get_index(organ_level_1_map_idx)
        organ_index_2 = get_index(organ_level_2_map_idx)
        organ_index_3 = get_index(organ_level_3_map_idx)

        sam.image_encoder.register_buffer('modal_index', modal_index, False)
        sam.image_encoder.register_buffer('organ_index_1', organ_index_1, False)
        sam.image_encoder.register_buffer('organ_index_2', organ_index_2, False)
        sam.image_encoder.register_buffer('organ_index_3', organ_index_3, False)

        modal_embed = nn.Embedding(len(modal_map_idx), embedding_dim)
        organ_embed_0 = nn.Embedding(1, embedding_dim)
        organ_embed_1 = nn.Embedding(len(organ_level_1_map_idx), embedding_dim)
        organ_embed_2 = nn.Embedding(len(organ_level_2_map_idx), embedding_dim)
        organ_embed_3 = nn.Embedding(len(organ_level_3_map_idx), embedding_dim)
        organ_embed_4 = nn.Embedding(len(task_list)+1, embedding_dim)
        nn.init.zeros_(modal_embed.weight)
        nn.init.zeros_(organ_embed_0.weight)
        nn.init.zeros_(organ_embed_1.weight)
        nn.init.zeros_(organ_embed_2.weight)
        nn.init.zeros_(organ_embed_3.weight)
        nn.init.zeros_(organ_embed_4.weight)

        sam.image_encoder.modal_embed = modal_embed
        sam.image_encoder.organ_embed = nn.ModuleList([
            organ_embed_0, organ_embed_1,
            organ_embed_2, organ_embed_3,
            organ_embed_4,
        ])

        # apply Adapter to SAM image encoder
        for idx, blk in enumerate(sam.image_encoder.blocks):
            if idx not in self.pos:
                continue

            # create moe adapter layers
            blk.mlp = MoEAdaptMLPBlock(
                blk.mlp,
                expert_num=expert_num,
                embedding_num=4,
                embedding_dim=embedding_dim,
                adapter_bn=bottleneck_dim,
            )

        self.sam = sam
        # AEGIS freezes hierarchy embeddings as well as the SAM encoders.
        for name, param in self.named_parameters():
            param.requires_grad = "adapter" in name or "mask_decoder" in name

    def forward(self, data):
        img, box = data['image'], data['box']
        modal, organ = data['modal'], data['organ']

        # modal and organ embedding
        B = img.shape[0]

        modal_index = self.sam.image_encoder.modal_index[modal]
        modal_embed = self.sam.image_encoder.modal_embed(modal_index)

        organ_1, organ_2, organ_3, organ_4 = organ.unbind(dim=1)
        organ_index_0 = torch.zeros(B, dtype=torch.long, device=img.device)
        organ_embed_0 = self.sam.image_encoder.organ_embed[0](organ_index_0)
        organ_index_1 = self.sam.image_encoder.organ_index_1[organ_1]
        organ_embed_1 = self.sam.image_encoder.organ_embed[1](organ_index_1)
        organ_index_2 = self.sam.image_encoder.organ_index_2[organ_2]
        organ_embed_2 = self.sam.image_encoder.organ_embed[2](organ_index_2)
        organ_index_3 = self.sam.image_encoder.organ_index_3[organ_3]
        organ_embed_3 = self.sam.image_encoder.organ_embed[3](organ_index_3)

        organ_embed_4 = self.sam.image_encoder.organ_embed[4](organ_4)
        organ_embed = (organ_embed_0, organ_embed_1, organ_embed_2, organ_embed_3, organ_embed_4)

        # prompt encoder
        if len(box.shape) == 2:
            box = box[:, None, :]  # (B, 1, 4)

        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=None,
            boxes=box,
            masks=None,
        )

        # adapter image encoder
        input_image = self.sam.preprocess(img) # input range [0, 255], normalized by SAM
        image_embedding, expert_activation = self.sam.image_encoder(input_image, modal_embed, organ_embed) # (B, 256, H/16, W/16)

        # predicted masks
        mask_predictions, _ = self.sam.mask_decoder(
            image_embeddings=image_embedding, # (B, 256, H/16, W/16)
            image_pe=self.sam.prompt_encoder.get_dense_pe(), # (B, 256, H/16, W/16)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, H/16, W/16)
            multimask_output=False,
          )

        return mask_predictions, image_embedding
