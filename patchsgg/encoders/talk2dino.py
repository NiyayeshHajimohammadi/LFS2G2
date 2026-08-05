"""Talk2DINO projection layer (CLIP-text -> DINOv2 space).

Adapted from Talk2DINO / Patch-ioner (https://github.com/Ruggero1912/Patch-ioner). Kept
weight-compatible (module names, ``load_state_dict`` remap) so published checkpoints load as-is.
We only use ``project_clip_txt``; the image-side attention parameters are constructed for
state-dict compatibility but not exercised here. See NOTICE.md for attribution.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import yaml


class ProjectionLayer(nn.Module):
    """Projection on top of the CLIP-text encoder, mapping CLIP text embeddings into DINO space."""

    def __init__(self, act=nn.Tanh(), hidden_layer=False, cosine=True, dino_embed_dim=1024,
                 clip_embed_dim=512, num_attn_head=16, weight_attn_heads=None,
                 alignment_strategy="max_score", alpha=0.6, keep_cls=False, keep_end_seq=False):
        super().__init__()
        self.num_attn_head = num_attn_head
        self.linear_layer = nn.Linear(clip_embed_dim, dino_embed_dim)
        if hidden_layer:
            hidden_layer = 1 if hidden_layer is True else hidden_layer
            self.hidden_layers = nn.ModuleList(
                [nn.Linear(dino_embed_dim, dino_embed_dim) for _ in range(hidden_layer)]
            )
        self.act = act
        self.cosine = cosine
        self.weight_attn_heads = weight_attn_heads
        if weight_attn_heads == "static":
            self.attn_weights = nn.Parameter(torch.rand(self.num_attn_head))
        elif weight_attn_heads == "conditioned":
            self.weight_layer1 = nn.Linear(dino_embed_dim, dino_embed_dim)
            self.weight_layer2 = nn.Linear(dino_embed_dim, self.num_attn_head)
        self.alignment_strategy = alignment_strategy
        self.keep_cls = keep_cls
        self.keep_end_seq = keep_end_seq
        self.alpha = alpha

    @classmethod
    def from_config(cls, config):
        #My comment: builds a model from a YAML file or dictionary.
        if isinstance(config, str):
            with open(config, "r") as f:
                config = yaml.safe_load(f)["model"]
        act = config.get("act", None)
        act = {"tanh": nn.Tanh(), "relu": nn.ReLU(), "sigmoid": nn.Sigmoid(), None: None}.get(act, "?")
        if act == "?":
            raise ValueError("Unknown activation function")
        model = cls(
            act=act,
            hidden_layer=config.get("hidden_layer", False),
            cosine=config.get("cosine", True),
            dino_embed_dim=config.get("dino_embed_dim", 1024),
            num_attn_head=config.get("num_attn_head", 16),
            clip_embed_dim=config.get("clip_embed_dim", 512),
            weight_attn_heads=config.get("weight_attn_heads", None),
            alignment_strategy=config.get("alignment_strategy", "max_score"),
            alpha=config.get("alpha", 0.6),
            keep_cls=config.get("keep_cls", None),
            keep_end_seq=config.get("keep_end_seq", None),
        )
        if config.get("starting_checkpoint", None) is not None:
            model.load_state_dict(torch.load(config["starting_checkpoint"], "cpu"))
        return model

    def project_clip_txt(self, textual_embedding):
        textual_embedding = textual_embedding.float()
        x = self.linear_layer(textual_embedding)
        if hasattr(self, "hidden_layers"):
            for hidden_layer in self.hidden_layers:
                if self.act:
                    x = self.act(x)
                x = hidden_layer(x)
        return x

    def load_state_dict(self, state_dict, strict=True):
        if "linear_layer2.weight" in state_dict:  # old-checkpoint compatibility
            state_dict["hidden_layers.0.weight"] = state_dict.pop("linear_layer2.weight")
            state_dict["hidden_layers.0.bias"] = state_dict.pop("linear_layer2.bias")
        return super().load_state_dict(state_dict, strict)
#My comment:
# Several attributes are unused in your actual path
# load_state_dict() should return the superclass result
# No output normalization
# Config inconsistency for keep_cls and keep_end_se
# No input shape validation
# Ambiguous training/frozen status
