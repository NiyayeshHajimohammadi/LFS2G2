import torch

from patchsgg.config import load_config
from patchsgg.decoder import build_decoder
from patchsgg.encoders.base import ConditioningSet
from patchsgg.graph_seq.vocab import VG_VOCAB


def _cfg(decoder_type):
    cfg = load_config(
        "patchsgg/configs/diagnostic_text2text.yaml",
        [f"decoder.type={decoder_type}", "decoder.d_model=64", "decoder.n_layers=1", "decoder.n_heads=2"],
    )
    return cfg


def _cond(B, N, D):
    return ConditioningSet(tokens=torch.randn(B, N, D), pooled=torch.randn(B, D))


def test_cross_attn_decoder_forward_and_generate():
    cfg = _cfg("cross_attn")
    dec = build_decoder(cfg, VG_VOCAB, cond_dim=32)
    cond = _cond(2, 5, 32)
    tokens = torch.randint(0, VG_VOCAB.vocab_size, (2, 11))
    logits = dec(cond, tokens)
    assert logits.shape == (2, 11, VG_VOCAB.vocab_size)
    from patchsgg.decoder import GenConfig

    seq, scores = dec.generate(cond, GenConfig(max_rels=3, entity_sampling="greedy"))
    assert seq.shape == scores.shape
    assert seq.shape[0] == 2


def test_prefix_decoder_forward():
    cfg = _cfg("prefix")
    dec = build_decoder(cfg, VG_VOCAB, cond_dim=32)
    cond = _cond(2, 4, 32)
    tokens = torch.randint(0, VG_VOCAB.vocab_size, (2, 7))
    logits = dec(cond, tokens)
    assert logits.shape == (2, 7, VG_VOCAB.vocab_size)


def test_prefix_decoder_respects_mask():
    cfg = _cfg("prefix")
    dec = build_decoder(cfg, VG_VOCAB, cond_dim=16)
    mask = torch.tensor([[True, False, False]])
    cond = ConditioningSet(tokens=torch.randn(1, 3, 16), pooled=torch.randn(1, 16), mask=mask)
    tokens = torch.randint(0, VG_VOCAB.vocab_size, (1, 6))
    logits = dec(cond, tokens)
    assert torch.isfinite(logits).all()
