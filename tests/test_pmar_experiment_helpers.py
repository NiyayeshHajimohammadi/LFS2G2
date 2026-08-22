"""Tests for the controlled PMAR experiment helpers.

Copy to tests/test_pmar_experiment_helpers.py and keep benchmark_pmar_ablation.py in tools/.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from patchsgg.graph_seq.linearize import Relation, build_train_pair, graph_to_sequence
from patchsgg.graph_seq.pmar import build_pmar_candidates
from patchsgg.graph_seq.vocab import VG_VOCAB


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "benchmark_pmar_ablation.py"
SPEC = importlib.util.spec_from_file_location("benchmark_pmar_ablation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
exact_canonical_graph = MODULE.exact_canonical_graph


def cycle_graph(rename=(0, 1, 2, 3), edge_order=(0, 1, 2, 3)):
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    graph = [
        Relation(1, int(rename[s]), 1, 1, int(rename[o]))
        for s, o in edges
    ]
    return [graph[i] for i in edge_order]


def seq(graph):
    return tuple(graph_to_sequence(graph, VG_VOCAB, add_start=False, add_end=True))


def test_canonical_ce_target_invariant_to_annotation_rename_and_edge_order():
    a = cycle_graph()
    b = cycle_graph(rename=(3, 1, 0, 2), edge_order=(2, 0, 3, 1))

    ca = exact_canonical_graph(a, VG_VOCAB, exact_threshold=64)
    cb = exact_canonical_graph(b, VG_VOCAB, exact_threshold=64)

    assert seq(ca) == seq(cb)


def test_canonical_ce_target_is_member_of_exact_pmar_set():
    graph = cycle_graph(rename=(2, 0, 3, 1), edge_order=(1, 3, 0, 2))
    canonical = exact_canonical_graph(graph, VG_VOCAB, exact_threshold=64)

    candidates = build_pmar_candidates(
        graph,
        VG_VOCAB,
        exact_threshold=64,
        num_samples=8,
        rng=np.random.default_rng(0),
    )

    assert candidates.mode == "exact"
    assert seq(canonical) in {seq(candidate) for candidate in candidates.graphs}


def test_four_cycle_has_expected_ambiguity():
    candidates = build_pmar_candidates(
        cycle_graph(),
        VG_VOCAB,
        exact_threshold=64,
        num_samples=8,
        rng=np.random.default_rng(0),
    )

    assert candidates.residual_permutation_count == 24
    assert candidates.mode == "exact"
    assert candidates.num_candidates == 6


def test_post_marginal_length_normalization_only_rescales_gradient():
    # Fixed sequence length T=21 for a 4-relation graph + EOS target.
    _, target = build_train_pair(cycle_graph(), VG_VOCAB, pad_to_max=False)
    T = len(target)
    assert T == 21

    raw = torch.tensor([3.0, 3.7, 5.1], requires_grad=True)
    exact = -torch.logsumexp(-raw, dim=0)
    grad_exact = torch.autograd.grad(exact, raw, retain_graph=True)[0]

    normalized = exact / T
    grad_norm = torch.autograd.grad(normalized, raw)[0]

    assert torch.allclose(grad_norm, grad_exact / T, atol=1e-7)


def test_mean_ce_equals_sum_nll_divided_by_length_without_masking():
    torch.manual_seed(0)
    T = 21
    V = VG_VOCAB.vocab_size
    logits = torch.randn(1, T, V)
    target = torch.randint(0, V, (1, T))

    mean_ce = F.cross_entropy(
        logits.reshape(-1, V),
        target.reshape(-1),
        reduction="mean",
    )
    log_probs = F.log_softmax(logits.float(), dim=-1)
    sum_nll = -log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1).sum()

    assert torch.allclose(mean_ce, sum_nll / T, atol=1e-6)