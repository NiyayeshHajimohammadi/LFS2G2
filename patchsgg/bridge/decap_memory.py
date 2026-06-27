"""Native DeCap support-memory projector.

Projects an embedding onto a softmax-weighted combination of a memory bank of *text* embeddings
(in the same space as the decoder's conditioning), pulling image features toward the text manifold
the decoder was trained on. This is a focused reimplementation of Patch-ioner's ``Im2TxtProjector``
(we keep only the projection + memory bank; the multi-encoder machinery is out of scope).

The memory bank is a tensor ``[M, D]`` saved with ``torch.save`` (e.g. CLIP-text or
Talk2DINO-projected caption embeddings). Build it once offline with ``tools/build_memory_bank.py``.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class DecapMemoryProjector:
    def __init__(self, memory_path: str, device: str = "cpu", normalize: bool = True):
        bank = torch.load(memory_path, map_location=device)
        if isinstance(bank, dict):
            bank = bank.get("embeddings", next(iter(bank.values())))
        self.bank = bank.float().to(device)            # [M, D]
        if normalize:
            self.bank = F.normalize(self.bank, dim=-1)
        self.normalize = normalize

    @torch.no_grad()
    def project(self, emb: torch.Tensor, temperature: float = 0.01) -> torch.Tensor:
        """``emb``: [*, D] -> softmax-weighted memory combination, same shape."""
        e = F.normalize(emb.float(), dim=-1) if self.normalize else emb.float()
        sim = e @ self.bank.t()                        # [*, M]
        w = (sim / temperature).softmax(dim=-1)
        return (w @ self.bank).to(emb.dtype)
