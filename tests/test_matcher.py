"""Instance-matcher tests. The branched test needs the compiled extension."""
import pytest

from patchsgg.eval.matcher import InstanceMatcher

A = (51, 0, 10, 56, 0)
B = (51, 1, 20, 58, 0)


def test_identity_fallback_mapping():
    m = InstanceMatcher(allow_identity_fallback=True)
    mapping = m.match([A, B], [A, B])
    assert mapping[(51, 0)] == 0
    assert mapping[(51, 1)] == 1


@pytest.mark.skipif(not InstanceMatcher().available, reason="branched_ssg_matcher not compiled")
def test_branched_matching_recovers_permuted_instances():
    from patchsgg.eval.metrics import apply_mapping_to_predictions, global_recall

    m = InstanceMatcher()
    gt = [A, B]
    pred = [(51, 5, 10, 56, 7), (51, 9, 20, 58, 3)]  # same graph, permuted instance ids
    mapping = m.match(gt, pred)
    mapped = apply_mapping_to_predictions(pred, mapping)
    assert global_recall(gt, mapped) == 1.0
