"""Turn generated token sequences into score-ranked matcher tuples for evaluation."""
from __future__ import annotations

from typing import List

import torch

from patchsgg.graph_seq.linearize import canonical_tuple, tokens_to_relation
from patchsgg.graph_seq.vocab import TOKENS_PER_REL, VG_VOCAB, GraphVocab

MatcherTuple = tuple


def sequences_to_predictions(
    seq: torch.Tensor,
    scores: torch.Tensor,
    vocab: GraphVocab = VG_VOCAB,
) -> List[List[MatcherTuple]]:
    """``seq``/``scores``: [B, T] (START already excluded). Returns per-sample matcher tuples
    sorted by descending per-tuple score."""
    out: List[List[MatcherTuple]] = []
    for b in range(seq.shape[0]):
        toks = seq[b].tolist()
        scs = scores[b].tolist()
        if vocab.end_token in toks:
            cut = toks.index(vocab.end_token)
            toks, scs = toks[:cut], scs[:cut]
        tuples = []
        for i in range(0, len(toks) - TOKENS_PER_REL + 1, TOKENS_PER_REL):
            block = toks[i : i + TOKENS_PER_REL]
            block_score = sum(scs[i : i + TOKENS_PER_REL]) / TOKENS_PER_REL
            try:
                rel = tokens_to_relation(block, vocab)
                tuples.append((canonical_tuple(rel, vocab), block_score))
            except Exception:
                continue
        tuples.sort(key=lambda x: x[1], reverse=True)
        out.append([t for t, _ in tuples])
    return out
