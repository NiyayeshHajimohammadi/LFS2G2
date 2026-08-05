import numpy as np

from patchsgg.graph_seq import VG_VOCAB, sequence_to_graph
from patchsgg.graph_seq.linearize import (
    Relation,
    build_train_pair,
    canonical_tuple,
    graph_to_sequence,
    relation_to_tokens,
    tokens_to_relation,
)


def _sample_graph():
    # man(0) riding(10) horse(5), man#1 wearing(20) hat(7)
    return [
        Relation(subj_cls=0, subj_inst=0, predicate=10, obj_cls=5, obj_inst=0),
        Relation(subj_cls=0, subj_inst=1, predicate=20, obj_cls=7, obj_inst=0),
    ]


def test_vocab_layout_matches_lfsgg():
    v = VG_VOCAB
    assert v.entity_start == 51
    assert v.instance_start == 202
    assert v.noise_token == 240
    assert v.end_token == 242
    assert v.start_token == 243
    assert v.no_known_token == 244
    assert v.vocab_size == 245


def test_relation_token_roundtrip():
    for rel in _sample_graph():
        toks = relation_to_tokens(rel)
        assert len(toks) == 5
        assert tokens_to_relation(toks) == rel


def test_graph_sequence_roundtrip():
    graph = _sample_graph()
    seq = graph_to_sequence(graph)
    assert seq[0] == VG_VOCAB.start_token
    assert seq[-1] == VG_VOCAB.end_token
    assert sequence_to_graph(seq) == graph


def test_sequence_to_graph_ignores_partial_tail():
    graph = _sample_graph()
    seq = graph_to_sequence(graph, add_end=False)
    seq = seq + [VG_VOCAB.entity_token(3)]  # dangling partial relation
    assert sequence_to_graph(seq) == graph


def test_canonical_tuple_format():
    rel = Relation(subj_cls=0, subj_inst=2, predicate=10, obj_cls=5, obj_inst=1)
    sub_tok, sub_inst, pred_tok, obj_tok, obj_inst = canonical_tuple(rel)
    assert sub_tok == VG_VOCAB.entity_token(0)
    assert pred_tok == 10  # 0-based predicate
    assert sub_inst == 2 and obj_inst == 1


def test_build_train_pair_lengths_and_masking():
    graph = _sample_graph()
    rng = np.random.default_rng(0)
    inp, tgt = build_train_pair(graph, rng=rng)
    # input length = 1 (START) + 5 * MAX_NUM_RELS ; target length = 5 * MAX_NUM_RELS + 1
    assert len(inp) == 1 + 5 * VG_VOCAB.max_num_rels
    assert len(inp) == len(tgt)
    # real tuple tokens appear at the front of the target, then END
    real_tokens = []
    for r in graph:
        real_tokens.extend(relation_to_tokens(r))
    assert list(tgt[: len(real_tokens)]) == real_tokens
    assert tgt[len(real_tokens)] == VG_VOCAB.end_token
    # padded positions contain NO_KNOWN (masked out of loss)
    assert (tgt == VG_VOCAB.no_known_token).sum() > 0


def test_build_train_pair_no_padding():
    graph = _sample_graph()
    inp, tgt = build_train_pair(graph, pad_to_max=False)
    assert len(inp) == 1 + 5 * len(graph)
    assert tgt[-1] == VG_VOCAB.end_token


def test_permute_and_reindex_assigns_ids_by_first_appearance():
    from patchsgg.graph_seq.linearize import permute_and_reindex_graph

    graph = [
        Relation(subj_cls=4, subj_inst=8, predicate=1, obj_cls=9, obj_inst=3),
        Relation(subj_cls=4, subj_inst=2, predicate=2, obj_cls=9, obj_inst=3),
        Relation(subj_cls=4, subj_inst=8, predicate=3, obj_cls=4, obj_inst=2),
    ]
    remapped = permute_and_reindex_graph(graph, shuffle=False)

    assert remapped[0].subj_inst == 0
    assert remapped[1].subj_inst == 1
    assert remapped[2].subj_inst == 0
    assert remapped[2].obj_inst == 1
    assert remapped[0].obj_inst == remapped[1].obj_inst == 0
