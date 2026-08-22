import math

import numpy as np
import torch
from torch.utils.data import DataLoader

from patchsgg.config import load_config
from patchsgg.data.collate import GraphCollator
from patchsgg.data.factory import build_dataset
from patchsgg.graph_seq.linearize import Relation
from patchsgg.graph_seq.pmar import (
    build_pmar_candidates,
    residual_permutation_count,
    structural_refine,
)
from patchsgg.graph_seq.vocab import GraphVocab, VG_VOCAB
from patchsgg.losses.pmar import PMARLoss
from patchsgg.model import PatchSGGModel, build_vocab


def _example_graph():
    """Create a small graph with a known residual ambiguity.

    There are four people:

        person 0 --holding--> cup
        person 1 --wearing--> hat
        person 2 --beside----> dog
        person 3 --beside----> dog

    Structurally:

        person 0 is unique
        person 1 is unique

    but:

        person 2
        person 3

    are structurally indistinguishable.

    Therefore the raw person permutation space would be:

        4! = 24

    but structural refinement should reduce it to:

        2! = 2
    """

    person = 1
    cup = 2
    hat = 3
    dog = 4

    holding = 1
    wearing = 2
    beside = 3

    return [
        Relation(
            person,
            0,
            holding,
            cup,
            0,
        ),
        Relation(
            person,
            1,
            wearing,
            hat,
            0,
        ),
        Relation(
            person,
            2,
            beside,
            dog,
            0,
        ),
        Relation(
            person,
            3,
            beside,
            dog,
            0,
        ),
    ]


def test_refinement_reduces_four_person_factorial_to_two():
    """Structural refinement should leave only the true ambiguity."""

    graph = _example_graph()

    cells = structural_refine(
        graph,
        VG_VOCAB,
    )

    # Extract only the person-class refinement cells.
    person_cells = [
        cell
        for cell in cells
        if cell.class_id == 1
    ]

    # We expect:
    #
    # person 0 -> singleton cell
    # person 1 -> singleton cell
    # person 2/person 3 -> ambiguity cell
    #
    # Therefore cell sizes should be:
    #
    # [1, 1, 2]
    assert sorted(
        len(cell.nodes)
        for cell in person_cells
    ) == [1, 1, 2]

    # All other nodes are singletons, so the residual group is:
    #
    # 1! * 1! * 2! * 1! * 1! * 1! = 2
    assert (
        residual_permutation_count(cells)
        == 2
    )


def test_exact_candidates_invariant_to_edge_order_and_annotation_renaming():
    """Exact PMAR candidates must ignore arbitrary annotation IDs.

    This checks two invariances simultaneously:

    1. Input relation order should not matter.
    2. Renaming same-class instance IDs should not matter.
    """

    graph = _example_graph()

    # Arbitrarily rename the four person IDs:
    #
    # 0 -> 3
    # 1 -> 2
    # 2 -> 1
    # 3 -> 0
    person_map = {
        0: 3,
        1: 2,
        2: 1,
        3: 0,
    }

    renamed = []

    # Reverse the relation order too.
    for relation in reversed(graph):

        renamed.append(
            Relation(
                relation.subj_cls,
                (
                    person_map[
                        relation.subj_inst
                    ]
                    if relation.subj_cls == 1
                    else relation.subj_inst
                ),
                relation.predicate,
                relation.obj_cls,
                relation.obj_inst,
            )
        )

    original_candidates = (
        build_pmar_candidates(
            graph,
            VG_VOCAB,
            exact_threshold=64,
        )
    )

    renamed_candidates = (
        build_pmar_candidates(
            renamed,
            VG_VOCAB,
            exact_threshold=64,
        )
    )

    assert (
        original_candidates.mode
        ==
        renamed_candidates.mode
        ==
        "exact"
    )

    assert (
        original_candidates
        .residual_permutation_count
        ==
        renamed_candidates
        .residual_permutation_count
        ==
        2
    )

    # This is the important test:
    #
    # arbitrary annotation naming and input relation order should
    # produce exactly the same canonical PMAR candidate sequences.
    assert (
        original_candidates.graphs
        ==
        renamed_candidates.graphs
    )


def test_exact_mode_deduplicates_identical_serializations():
    """Different permutations can produce the same canonical sequence.

    Consider:

        person A --relation--> person B
        person B --relation--> person A

    Swapping A and B gives the same sorted graph.

    Therefore:

        residual permutation count = 2

    but:

        number of unique serializations = 1
    """

    graph = [
        Relation(
            1,
            0,
            5,
            1,
            1,
        ),
        Relation(
            1,
            1,
            5,
            1,
            0,
        ),
    ]

    candidates = build_pmar_candidates(
        graph,
        VG_VOCAB,
        exact_threshold=64,
    )

    assert (
        candidates
        .residual_permutation_count
        == 2
    )

    # Exact mode must deduplicate identical serialized graphs.
    assert candidates.num_candidates == 1


def test_sampled_mode_keeps_duplicate_draws():
    """Sampled PMAR should retain repeated IID permutation draws."""

    graph = _example_graph()

    # The graph has M_residual = 2.
    #
    # exact_threshold=1 forces sampled mode.
    #
    # Sampling 20 times from only two possible assignments guarantees
    # repeated graph candidates.
    candidates = build_pmar_candidates(
        graph,
        VG_VOCAB,
        exact_threshold=1,
        num_samples=20,
        rng=np.random.default_rng(0),
    )

    assert candidates.mode == "sampled"

    # Sampled mode must retain all 20 IID draws.
    assert candidates.num_candidates == 20

    # Since there are only two possible assignments, there must be
    # duplicate serialized candidates.
    #
    # This verifies that sampled mode does NOT deduplicate them.
    assert (
        len(
            set(
                candidates.graphs
            )
        )
        < 20
    )


def test_pmar_math_uses_summed_nll_then_logsumexp():
    """PMAR must sum token NLLs before marginalization."""

    # Create a tiny artificial vocabulary so the expected probability
    # can be calculated analytically.
    vocab = GraphVocab(
        n_preds=2,
        n_entities=2,
        max_instance_id=2,
        random_max_instance_id=1,
        max_num_rels=1,
    )

    loss = PMARLoss(
        vocab
    )

    # Two candidate sequences, each two tokens long.
    target = torch.tensor(
        [
            [0, 1],
            [1, 0],
        ]
    )

    # Uniform logits:
    #
    # p(token) = 1 / vocab_size
    #
    # at every position.
    logits = torch.zeros(
        2,
        2,
        vocab.vocab_size,
    )

    candidate_nll = (
        loss.candidate_nll(
            logits,
            target,
        )
    )

    # Each sequence has two tokens.
    #
    # Therefore:
    #
    # NLL =
    #     -log(1/V)
    #     -log(1/V)
    #
    #     = 2 log(V)
    expected_each = (
        2.0
        * math.log(
            vocab.vocab_size
        )
    )

    assert torch.allclose(
        candidate_nll,
        torch.full_like(
            candidate_nll,
            expected_each,
        ),
        atol=1e-6,
    )

    # With two equally probable candidates:
    #
    # L =
    #   -log(
    #       exp(-L1)
    #       +
    #       exp(-L2)
    #   )
    #
    # Since L1 = L2:
    #
    # L =
    #   L1 - log(2)
    marginal = loss.marginalize(
        candidate_nll
    )

    expected_marginal = torch.tensor(
        expected_each
        - math.log(2.0)
    )

    assert torch.allclose(
        marginal,
        expected_marginal,
        atol=1e-6,
    )


def test_model_pmar_path_runs_and_backpropagates_on_toy_cpu():
    """End-to-end PMAR smoke test using the project's toy setup.

    This verifies that:

    - PMAR can be selected from config.
    - GraphCollator provides train_graphs.
    - PatchSGGModel enters the PMAR path.
    - Candidate sequences are constructed.
    - The decoder evaluates them.
    - PMAR returns a finite scalar loss.
    - Backpropagation reaches decoder parameters.
    """

    cfg = load_config(
        "patchsgg/configs/diagnostic_text2text.yaml",
        [
            "loss.type=pmar",
            "loss.pmar_exact_threshold=8",
            "loss.pmar_num_samples=4",
            "loss.pmar_candidate_batch_size=2",
            "vocab.max_num_rels=2",
            "data.toy_n_train=4",
            "data.toy_max_rels=2",
            "train.batch_size=2",
            "eval.max_rels=2",
        ],
    )

    vocab = build_vocab(
        cfg
    )

    dataset = build_dataset(
        cfg,
        "train",
        vocab,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=GraphCollator(
            vocab,
            seed=0,
        ),
    )

    batch = next(
        iter(
            dataloader
        )
    )

    # This confirms your GraphCollator modification was made correctly.
    assert "train_graphs" in batch

    assert len(
        batch["train_graphs"]
    ) == 2

    model = PatchSGGModel(
        cfg
    )

    loss = model.compute_loss(
        batch,
        modality="text",
    )

    # Training loss should be one scalar.
    assert loss.ndim == 0

    # Make sure PMAR has not produced NaN or inf.
    assert torch.isfinite(
        loss
    )

    # Make sure the new PMAR computation remains differentiable.
    loss.backward()

    # At least one trainable decoder parameter should now have a gradient.
    assert any(
        parameter.grad is not None
        for parameter
        in model.decoder.parameters()
        if parameter.requires_grad
    )