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

from patchsgg.utils.huggingface import get_model_path_with_hf_fallback


class DecapMemoryProjector:
    def __init__(
        self,
        memory_path: str,
        device: str = "cpu",
        normalize: bool = True,
        hf_repo_id: str | None = None,
        hf_filename: str | None = None,
        cache_dir: str | None = None,
    ):
        resolved_path = get_model_path_with_hf_fallback(
            memory_path, hf_repo_id=hf_repo_id, filename=hf_filename, cache_dir=cache_dir
        )
        bank = torch.load(resolved_path, map_location=device)
        if isinstance(bank, dict):
            if "embeddings" in bank:
                bank = bank["embeddings"]
            elif bank:
                bank = next(iter(bank.values()))
            else:
                raise ValueError("DeCap checkpoint dictionary is empty")
        if not isinstance(bank, torch.Tensor) or bank.ndim != 2:
            raise ValueError(f"DeCap memory bank must be a [M, D] tensor, got {type(bank)!r} shape={getattr(bank, 'shape', None)}")
        self.bank = bank.float().to(device)            # [M, D]
        if normalize:
            self.bank = F.normalize(self.bank, dim=-1)
        self.normalize = normalize

    @torch.no_grad()
    def project(self, emb: torch.Tensor, temperature: float = 0.01) -> torch.Tensor:
        """``emb``: [*, D] -> softmax-weighted memory combination, same shape."""
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if emb.shape[-1] != self.bank.shape[-1]:
            raise ValueError(f"embedding dim {emb.shape[-1]} does not match memory dim {self.bank.shape[-1]}")
        e = F.normalize(emb.float(), dim=-1) if self.normalize else emb.float()
        sim = e @ self.bank.t()                        # [*, M]
        w = (sim / temperature).softmax(dim=-1)
        return (w @ self.bank).to(emb.dtype)
