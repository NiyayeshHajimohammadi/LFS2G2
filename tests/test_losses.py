import torch

from patchsgg.graph_seq.linearize import Relation, build_train_pair, relation_to_tokens
from patchsgg.graph_seq.vocab import VG_VOCAB
from patchsgg.losses.order_agnostic import OrderAgnosticCELoss


def _one_hot_logits(target, vocab, n_blocks):
    """Build logits that argmax (within each role range) to the GT tuples in REVERSED order."""
    T = target.shape[1]
    V = vocab.vocab_size
    logits = torch.full((1, T, V), -10.0)
    blocks = [target[0, k * 5 : (k + 1) * 5].tolist() for k in range(n_blocks)]
    rev = list(reversed(blocks))
    for k, blk in enumerate(rev):
        for j, tok in enumerate(blk):
            logits[0, k * 5 + j, tok] = 10.0
    return logits


def test_order_agnostic_reorders_to_match_prediction():
    graph = [
        Relation(0, 0, 10, 5, 0),
        Relation(1, 0, 20, 6, 0),
        Relation(2, 0, 30, 7, 0),
    ]
    _, tgt = build_train_pair(graph, VG_VOCAB, pad_to_max=False)
    tgt = torch.as_tensor(tgt).unsqueeze(0)
    n_blocks = 3
    logits = _one_hot_logits(tgt, VG_VOCAB, n_blocks)

    loss = OrderAgnosticCELoss(VG_VOCAB)
    reordered = loss._reorder_target(logits, tgt)

    # the model "predicts" reversed order -> reordered GT should be reversed too
    orig_blocks = [tgt[0, k * 5 : (k + 1) * 5].tolist() for k in range(n_blocks)]
    new_blocks = [reordered[0, k * 5 : (k + 1) * 5].tolist() for k in range(n_blocks)]
    assert new_blocks == list(reversed(orig_blocks))
    # END / tail untouched
    assert reordered[0, n_blocks * 5] == VG_VOCAB.end_token


def test_order_agnostic_loss_lower_after_reorder():
    graph = [Relation(0, 0, 10, 5, 0), Relation(1, 0, 20, 6, 0)]
    _, tgt = build_train_pair(graph, VG_VOCAB, pad_to_max=False)
    tgt = torch.as_tensor(tgt).unsqueeze(0)
    logits = _one_hot_logits(tgt, VG_VOCAB, 2)

    oa = OrderAgnosticCELoss(VG_VOCAB)
    plain = oa._ce(logits, tgt)            # penalises the (reversed) order
    reordered_loss = oa(logits, tgt)       # order-agnostic
    assert reordered_loss < plain
