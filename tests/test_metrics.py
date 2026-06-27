from patchsgg.eval.evaluate import evaluate_graphs
from patchsgg.eval.matcher import InstanceMatcher
from patchsgg.eval.metrics import (
    apply_mapping_to_predictions,
    global_recall,
    mean_over_predicates,
    mean_recall_helper,
    set_recall,
)

# matcher tuples: (sub_entity_token, sub_inst, pred_token, obj_entity_token, obj_inst)
A = (51, 0, 10, 56, 0)
B = (51, 1, 20, 58, 0)
C = (60, 0, 5, 61, 0)


def test_global_recall_exact():
    gts = [A, B, C]
    assert global_recall(gts, [A, B, C]) == 1.0
    assert global_recall(gts, [A]) == 1 / 3
    assert global_recall(gts, []) == 0.0


def test_global_recall_duplicates_counted_by_multiplicity():
    gts = [A, A, B]
    assert global_recall(gts, [A]) == 1 / 3          # one of two A's recalled
    assert global_recall(gts, [A, A]) == 2 / 3


def test_set_recall_ignores_instances():
    gts = [A, B]
    # same classes/predicates but different instance ids -> still matches under set recall
    pred = [(51, 5, 10, 56, 9), (51, 7, 20, 58, 3)]
    assert set_recall(gts, pred, "triplet") == 1.0
    assert set_recall(gts, pred, "pred") == 1.0


def test_mean_recall_weights_predicates_equally():
    gts = [A, B]  # predicates 10 and 20
    per = mean_recall_helper(gts, [A])  # only predicate 10 recalled
    assert mean_over_predicates(per) == 0.5


def test_apply_mapping():
    preds = [(51, 9, 10, 56, 4)]
    mapping = {(51, 9): 0, (56, 4): 0}
    assert apply_mapping_to_predictions(preds, mapping) == [(51, 0, 10, 56, 0)]
    # unmatched -> None
    assert apply_mapping_to_predictions(preds, {})[0][1] is None


def test_evaluate_graphs_identity_fallback():
    matcher = InstanceMatcher(allow_identity_fallback=True)
    samples = [([A, B, C], [A, B, C]), ([A, B], [A])]
    out = evaluate_graphs(samples, ks=(20,), matcher=matcher)
    assert out["R@20"] == (1.0 + 0.5) / 2
    assert out["set/triplet"] == (1.0 + 0.5) / 2
