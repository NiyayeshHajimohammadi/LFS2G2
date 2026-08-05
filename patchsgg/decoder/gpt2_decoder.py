"""Hugging Face GPT-2 graph decoder with visual/text cross-attention.

The pretrained GPT-2 checkpoint supplies the causal self-attention blocks, MLPs,
layer norms, and the original positional embeddings. PatchSGG keeps its own small
scene-graph vocabulary, so graph-token embeddings and the graph output head are
learned for this task rather than reusing GPT-2's English BPE vocabulary.

GPT-2 checkpoints do not contain cross-attention layers. Those layers are added
from the GPT-2 configuration and initialized randomly; all compatible pretrained
weights are copied from the downloaded checkpoint.
"""
from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn

from patchsgg.decoder.base import GraphDecoder
from patchsgg.decoder.sampling import GenConfig, sample_constrained_token
from patchsgg.encoders.base import ConditioningSet
from patchsgg.graph_seq.vocab import TOKENS_PER_REL, GraphVocab


def _optional(value):
    """Normalize OmegaConf/AttrDict null-like values for Hugging Face kwargs."""
    return None if value in (None, "", "null", "None") else value


class GPT2CrossAttnDecoder(GraphDecoder):
    """A pretrained GPT-2 causal decoder conditioned through cross-attention.

    Notes
    -----
    * ``inputs_embeds`` is used, because PatchSGG graph token IDs are not GPT-2
      BPE token IDs.
    * The downloaded GPT-2 lexical embedding table is discarded after loading;
      it is never used by this model.
    * Cross-attention layers, the conditioning projection, graph embeddings,
      and graph output head are task-specific and start from random weights.
    * When the requested graph sequence is longer than GPT-2's native context,
      pretrained positional embeddings are copied and only the extra positions
      are randomly initialized.
    """

    def __init__(
        self,
        vocab: GraphVocab,
        cond_dim: int,
        max_seq_len: int,
        model_name: str = "openai-community/gpt2",
        revision: str = "main",
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
        gradient_checkpointing: bool = True,
        freeze_pretrained: bool = False,
        tie_graph_embeddings: bool = True,
        extend_positions: bool = True,
        dropout: Optional[float] = None,
    ):
        try:
            from transformers import GPT2Model
        except ImportError as exc:  # pragma: no cover - exercised only without optional dependency
            raise ImportError(
                "decoder.type='gpt2_cross_attn' requires Hugging Face Transformers. "
                "Install it with `pip install transformers huggingface_hub`."
            ) from exc

        load_kwargs = {
            "revision": revision,
            "local_files_only": bool(local_files_only),
        }
        cache_dir = _optional(cache_dir)
        if cache_dir is not None:
            load_kwargs["cache_dir"] = cache_dir

        # Load the ordinary pretrained GPT-2 once. We then construct a second
        # GPT-2 with cross-attention and copy every shape-compatible weight.
        pretrained = GPT2Model.from_pretrained(model_name, **load_kwargs)
        hf_cfg = copy.deepcopy(pretrained.config)
        native_positions = int(getattr(hf_cfg, "n_positions", 1024))

        if max_seq_len > native_positions and not extend_positions:
            del pretrained
            raise ValueError(
                f"PatchSGG needs {max_seq_len} GPT-2 positions, but {model_name!r} "
                f"provides {native_positions}. Set decoder.extend_positions=true or "
                "reduce eval.max_rels."
            )

        target_positions = max(native_positions, int(max_seq_len))
        hf_cfg.add_cross_attention = True
        hf_cfg.is_decoder = True
        hf_cfg.n_positions = target_positions
        hf_cfg.n_ctx = target_positions
        hf_cfg.use_cache = True
        if dropout is not None:
            p = float(dropout)
            hf_cfg.resid_pdrop = p
            hf_cfg.embd_pdrop = p
            hf_cfg.attn_pdrop = p

        hidden_size = int(hf_cfg.n_embd)
        super().__init__(
            vocab=vocab,
            cond_dim=cond_dim,
            d_model=hidden_size,
            max_seq_len=target_positions,
        )

        self.model_name = model_name
        self.native_max_positions = native_positions
        self.max_seq_len = target_positions
        self.gpt2 = GPT2Model(hf_cfg)
        self._copy_pretrained_weights(pretrained)
        del pretrained

        # We always pass inputs_embeds, so GPT-2's 50k-token English embedding
        # table would only consume memory. Keep a one-row placeholder required
        # by the model API instead.
        self.gpt2.wte = nn.Embedding(1, hidden_size)
        self.gpt2.wte.weight.requires_grad_(False)

        # GPT2Model already applies its final layer norm. The base decoder's
        # extra positional table is also unnecessary because GPT-2 owns wpe.
        self.norm = nn.Identity()
        del self.pos_embed

        self.head = nn.Linear(hidden_size, vocab.vocab_size, bias=False)
        self._initialize_task_layers(float(hf_cfg.initializer_range))
        if tie_graph_embeddings:
            self.head.weight = self.token_embed.weight

        if freeze_pretrained:
            self._freeze_pretrained_submodules()

        if gradient_checkpointing:
            self.gpt2.gradient_checkpointing_enable()

    def _copy_pretrained_weights(self, source: nn.Module) -> None:
        """Copy all compatible GPT-2 weights, preserving the native wpe prefix."""
        source_state = source.state_dict()
        target_state = self.gpt2.state_dict()
        with torch.no_grad():
            for name, src in source_state.items():
                dst = target_state.get(name)
                if dst is None:
                    continue
                if dst.shape == src.shape:
                    dst.copy_(src)
                    continue
                if (
                    name == "wpe.weight"
                    and dst.ndim == 2
                    and src.ndim == 2
                    and dst.shape[1] == src.shape[1]
                    and dst.shape[0] >= src.shape[0]
                ):
                    dst[: src.shape[0]].copy_(src)

    def _initialize_task_layers(self, std: float) -> None:
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=std)
        nn.init.normal_(self.cond_proj.weight, mean=0.0, std=std)
        nn.init.zeros_(self.cond_proj.bias)
        nn.init.normal_(self.head.weight, mean=0.0, std=std)

    def _freeze_pretrained_submodules(self) -> None:
        """Freeze copied GPT-2 components while leaving new cross-attention trainable."""
        for parameter in self.gpt2.parameters():
            parameter.requires_grad_(False)

        # Cross-attention did not exist in the source checkpoint and must learn.
        for block in self.gpt2.h:
            if hasattr(block, "crossattention"):
                for parameter in block.crossattention.parameters():
                    parameter.requires_grad_(True)
            if hasattr(block, "ln_cross_attn"):
                for parameter in block.ln_cross_attn.parameters():
                    parameter.requires_grad_(True)

    def _memory(self, cond: ConditioningSet) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        memory = self.cond_proj(cond.tokens)
        # Hugging Face uses 1/True for valid encoder positions and 0/False for padding.
        memory_mask = None if cond.mask is None else cond.mask.to(device=memory.device)
        return memory, memory_mask

    def _hidden(self, cond: ConditioningSet, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1] > self.max_seq_len:
            raise ValueError(
                f"token sequence length {tokens.shape[1]} exceeds GPT-2 context {self.max_seq_len}"
            )
        memory, memory_mask = self._memory(cond)
        outputs = self.gpt2(
            inputs_embeds=self.token_embed(tokens),
            encoder_hidden_states=memory,
            encoder_attention_mask=memory_mask,
            use_cache=False,
            return_dict=True,
        )
        return outputs.last_hidden_state

    @torch.no_grad()
    def generate(self, cond: ConditioningSet, gen_cfg: GenConfig):
        """Cached constrained generation; avoids recomputing the full prefix each step."""
        total_steps = int(gen_cfg.max_rels) * TOKENS_PER_REL
        required_positions = 1 + total_steps  # START plus generated graph tokens
        if required_positions > self.max_seq_len:
            raise ValueError(
                f"generation needs {required_positions} positions but GPT-2 has {self.max_seq_len}"
            )

        device = cond.tokens.device
        batch_size = cond.batch_size
        current = torch.full(
            (batch_size, 1), self.vocab.start_token, dtype=torch.long, device=device
        )
        memory, memory_mask = self._memory(cond)
        past_key_values = None
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        out_tokens = []
        out_scores = []

        for step_index in range(total_steps):
            outputs = self.gpt2(
                inputs_embeds=self.token_embed(current),
                encoder_hidden_states=memory,
                encoder_attention_mask=memory_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            logits = self.head(outputs.last_hidden_state[:, -1])
            current, score, finished = sample_constrained_token(
                logits=logits,
                step_index=step_index,
                vocab=self.vocab,
                cfg=gen_cfg,
                finished=finished,
            )
            out_tokens.append(current)
            out_scores.append(score)
            if bool(finished.all()):
                break

        return torch.cat(out_tokens, dim=1), torch.cat(out_scores, dim=1)
