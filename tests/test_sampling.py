import torch

from patchsgg.decoder.sampling import GenConfig, constrained_generate, top_k_top_p_filter
from patchsgg.graph_seq.vocab import TOKENS_PER_REL, VG_VOCAB


def test_top_p_filter_restores_original_token_order():
    logits = torch.tensor([[1.0, 4.0, 3.0, 2.0]])
    filtered = top_k_top_p_filter(logits, top_p=0.8)
    assert torch.isfinite(filtered[0, 1])
    assert torch.isfinite(filtered[0, 2])
    assert torch.isneginf(filtered[0, 0])
    assert torch.isneginf(filtered[0, 3])


def test_stateful_constrained_generation_uses_shared_grammar():
    calls = []

    def step_fn(sequence, state):
        state = 0 if state is None else state
        calls.append((sequence.shape[1], state))
        logits = torch.arange(VG_VOCAB.vocab_size, dtype=torch.float).unsqueeze(0)
        return logits, state + 1

    sequence, scores = constrained_generate(
        step_fn=step_fn,
        start_tokens=torch.tensor([[VG_VOCAB.start_token]]),
        vocab=VG_VOCAB,
        cfg=GenConfig(max_rels=2, entity_sampling="greedy", allow_end=False),
        initial_state=None,
        stateful=True,
    )

    assert sequence.shape == scores.shape == (1, 2 * TOKENS_PER_REL)
    assert len(calls) == 2 * TOKENS_PER_REL
    for position, token in enumerate(sequence[0].tolist()):
        lo, hi = VG_VOCAB.range_for_role(VG_VOCAB.role_at(position))
        assert lo <= token < hi
