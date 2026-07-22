"""Autoregressive graph decoders over the flat 5-tuple vocabulary.

Shared machinery (token/positional embeddings, output head, constrained generation) lives here;
subclasses only implement ``_hidden(cond, tokens) -> [B, T, d_model]``. Both consume a
:class:`ConditioningSet`, so pooled vs patch-set conditioning is just ``N==1`` vs ``N>1``.
"""
#My comment: defines common machinery that both cross_attn_decoder.py and prefix_decoder.py reuse->provides a level of abstraction just needs hidden states
from __future__ import annotations

import torch
import torch.nn as nn

from patchsgg.encoders.base import ConditioningSet #My comment: The obj that caontains tokens, pooled and mask-> expected shapes: [B,N,D],[B,D],[B,N]
from patchsgg.decoder.sampling import GenConfig, constrained_generate #My comment: GenConfig stores inference settings
#My comment: constrained_generate: performs token-by-token generation while enforcing the five-token grammar.
from patchsgg.graph_seq.vocab import GraphVocab #My comment: GraphVocab defines what every integer token means.


class GraphDecoder(nn.Module):
    def __init__(self, vocab: GraphVocab, cond_dim: int, d_model: int = 512, max_seq_len: int = 512):
        super().__init__() 
        self.vocab = vocab
        self.d_model = d_model
        self.cond_proj = nn.Linear(cond_dim, d_model)#My comment: a learnable linear projection from encoder feature dimension to decoder dimension-> [B, N, cond_dim] → [B, N, d_model]
        self.token_embed = nn.Embedding(vocab.vocab_size, d_model)#My comment: converts discrete graph token IDs into continuous vectors:[B, T] → [B, T, d_model]
        self.pos_embed = nn.Embedding(max_seq_len, d_model)#My comment: creates a learnable positional embedding table-> shape pos_embed: [max_seq_len, d_model]
        self.norm = nn.LayerNorm(d_model)#My comment:  normalizes hidden vectors.
        self.head = nn.Linear(d_model, vocab.vocab_size)#My comment: maps every hidden vector to one logit per vocabulary token [B, T, d_model] → [B, T, vocab_size]
        self.max_seq_len = max_seq_len

    # --- subclasses implement this -----------------------------------------------------------
    def _hidden(self, cond: ConditioningSet, tokens: torch.Tensor) -> torch.Tensor:#My comment: expected output [B, T, d_model]
        raise NotImplementedError

    def _embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor: #My comment: This internal helper combines graph-token and positional embeddings.
        #My comment: Expected inout: tokens: [B, T], dtype=torch.long/ Expected output: Zero-shotCross-modalTransferForLocation-freeSceneGraphGeneration
        pos = torch.arange(tokens.shape[1], device=tokens.device)
        return self.token_embed(tokens) + self.pos_embed(pos)[None]#My comment: [B, T, d_model] + [1, T, d_model]

    def logits(self, cond: ConditioningSet, tokens: torch.Tensor) -> torch.Tensor:
        #My comment: takes the decoder’s internal hidden representations and converts them into scores for every token in the graph vocabulary.
        return self.head(self.norm(self._hidden(cond, tokens)))

    def forward(self, cond: ConditioningSet, input_tokens: torch.Tensor) -> torch.Tensor:
        """Teacher-forced logits ``[B, T, V]`` aligned to the target sequence."""
        return self.logits(cond, input_tokens)
        #My comment: teacher-forcing shift itself. That shift is created by build_train_pair() in graph_seq/linearize.py.

    @torch.no_grad()
    def generate(self, cond: ConditioningSet, gen_cfg: GenConfig):
        B = cond.batch_size
        start = torch.full((B, 1), self.vocab.start_token, dtype=torch.long, device=cond.tokens.device)

        def step_fn(tokens: torch.Tensor) -> torch.Tensor:
            return self.logits(cond, tokens)[:, -1]

        return constrained_generate(step_fn, start, self.vocab, gen_cfg)

    @staticmethod
    def causal_mask(T: int, device) -> torch.Tensor:
        return torch.triu(torch.full((T, T), float("-inf"), device=device), diagonal=1)


def build_decoder(cfg, vocab: GraphVocab, cond_dim: int) -> GraphDecoder:
    kind = cfg.decoder.type
    # positional table must cover both teacher-forced training length (max_num_rels) and the
    # autoregressive generation length (eval.max_rels), whichever is larger.
    max_rels = max(vocab.max_num_rels, int(cfg.eval.get("max_rels", 100)))
    common = dict(
        vocab=vocab,
        cond_dim=cond_dim,
        d_model=int(cfg.decoder.d_model),
        max_seq_len=2 + max_rels * 5,
    )
    if kind == "cross_attn":
        from patchsgg.decoder.cross_attn_decoder import CrossAttnDecoder

        return CrossAttnDecoder(
            n_layers=int(cfg.decoder.n_layers),
            n_heads=int(cfg.decoder.n_heads),
            dim_ff=int(cfg.decoder.dim_ff),
            dropout=float(cfg.decoder.dropout),
            **common,
        )
    if kind == "prefix":
        from patchsgg.decoder.prefix_decoder import PrefixDecoder

        return PrefixDecoder(
            n_layers=int(cfg.decoder.n_layers),
            n_heads=int(cfg.decoder.n_heads),
            dim_ff=int(cfg.decoder.dim_ff),
            dropout=float(cfg.decoder.dropout),
            **common,
        )
    raise ValueError(f"unknown decoder.type {kind!r}")
