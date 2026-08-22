"""Controlled PMAR ablation benchmark.

This experiment isolates three questions on an ambiguity-heavy synthetic graph task:

1. Does deterministic canonicalization help compared with the original LF-SGG CE target?
2. Does marginalizing multiple equivalent serializations help beyond canonicalization alone?
3. Is any PMAR difference caused mainly by sequence-NLL gradient scale?

Compared methods
---------------
ce
    Existing project CE path: GraphCollator's shuffled LF-SGG serialization + mean token CE.

canonical_ce
    One deterministic graph-canonical target chosen as the lexicographically smallest exact
    PMAR serialization, trained with mean token CE. This isolates deterministic
    canonicalization without marginalization.

pmar_sK
    Sampled PMAR with K residual assignments (duplicates intentionally retained by PMAR).

pmar_exact
    Exact PMAR over all unique exact candidate serializations.

pmar_exact_norm
    Same exact PMAR objective, divided by the fixed target sequence length *after*
    marginalization. This does not change candidate responsibilities or the optimum on this
    fixed-length benchmark; it only makes gradient scale comparable to mean-token CE.

The default synthetic graph is a directed 4-cycle whose nodes all have the same class and whose
edges all have the same predicate. Its residual permutation count is 4! = 24 and, after exact
serialization deduplication, it has 6 unique serializations.

Example
-------
python tools/benchmark_pmar_ablation.py \
  --device cuda \
  --batch-size 8 \
  --steps 300 \
  --eval-every 25 \
  --candidate-batch-size 32 \
  --sampled-ks 1 2 4 8 \
  --seeds 0 1 2 3 4 \
  --csv outputs/pmar_ablation.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import statistics
import time
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from patchsgg.config import load_config
from patchsgg.data.collate import GraphCollator
from patchsgg.eval.evaluate import evaluate_graphs
from patchsgg.eval.matcher import InstanceMatcher
from patchsgg.graph_seq.linearize import (
    Relation,
    build_train_pair,
    graph_to_sequence,
)
from patchsgg.graph_seq.pmar import build_pmar_candidates
from patchsgg.model import PatchSGGModel, build_vocab


# -----------------------------------------------------------------------------
# Synthetic ambiguity dataset
# -----------------------------------------------------------------------------


class AmbiguousCycleDataset(Dataset):
    """Synthetic directed cycles with arbitrary same-class instance identities."""

    def __init__(
        self,
        n: int,
        vocab,
        *,
        cycle_size: int = 4,
        n_patterns: int = 8,
        seed: int = 0,
    ):
        self.n = int(n)
        self.vocab = vocab
        self.cycle_size = int(cycle_size)
        self.n_patterns = int(n_patterns)
        self.seed = int(seed)

        if self.cycle_size < 3:
            raise ValueError("cycle_size must be >= 3")
        if self.cycle_size > int(vocab.max_instance_id):
            raise ValueError(
                f"cycle_size={self.cycle_size} exceeds vocab.max_instance_id={vocab.max_instance_id}"
            )
        if self.n_patterns >= min(vocab.n_entities, vocab.n_preds):
            raise ValueError("n_patterns is too large for the configured vocabulary")

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int):
        pattern_id = index % self.n_patterns
        entity_class = 1 + pattern_id
        predicate = 1 + pattern_id

        rng = np.random.default_rng(self.seed + index * 104729)
        rename = rng.permutation(self.cycle_size).tolist()

        edges = [
            (node, (node + 1) % self.cycle_size)
            for node in range(self.cycle_size)
        ]
        rng.shuffle(edges)

        graph = [
            Relation(
                entity_class,
                int(rename[subj]),
                predicate,
                entity_class,
                int(rename[obj]),
            )
            for subj, obj in edges
        ]

        image_id = self.seed + index
        generator = torch.Generator().manual_seed(image_id)
        image = torch.rand(3, 32, 32, generator=generator)

        # No annotation instance IDs are exposed in the conditioning text.
        text = f"ambiguous directed-cycle pattern {pattern_id} size {self.cycle_size}"

        return {
            "image": image,
            "text": text,
            "graph": graph,
            "image_id": image_id,
        }


# -----------------------------------------------------------------------------
# Reproducibility / configuration
# -----------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_config(
    args,
    loss_type: str,
    *,
    sampled_k: int | None = None,
):
    overrides = [
        f"device={args.device}",
        "encoders.build_only_required=true",
        "train.train_modality=text",
        "eval.eval_modality=text",
        f"train.batch_size={args.batch_size}",
        f"eval.batch_size={args.batch_size}",
        f"train.lr={args.lr}",
        "train.weight_decay=0.0",
        "decoder.dropout=0.0",
        "loss.label_smoothing=0.0",
        f"vocab.max_num_rels={args.cycle_size}",
        f"eval.max_rels={args.cycle_size}",
        "eval.entity_sampling=greedy",
        "eval.allow_end=true",
        "eval.matcher_identity_fallback=true",
        f"loss.type={loss_type}",
    ]

    if loss_type == "pmar":
        overrides.append(
            f"loss.pmar_candidate_batch_size={args.candidate_batch_size}"
        )
        if sampled_k is None:
            overrides.extend(
                [
                    f"loss.pmar_exact_threshold={args.exact_threshold}",
                    "loss.pmar_num_samples=8",
                ]
            )
        else:
            # Force sampled mode for any genuinely ambiguous cycle.
            overrides.extend(
                [
                    "loss.pmar_exact_threshold=1",
                    f"loss.pmar_num_samples={sampled_k}",
                ]
            )

    return load_config(
        "patchsgg/configs/diagnostic_text2text.yaml",
        overrides,
    )


def make_loaders(cfg, args, seed: int):
    vocab = build_vocab(cfg)

    train_dataset = AmbiguousCycleDataset(
        args.train_examples,
        vocab,
        cycle_size=args.cycle_size,
        n_patterns=args.patterns,
        seed=1000 + seed * 100000,
    )
    val_dataset = AmbiguousCycleDataset(
        args.val_examples,
        vocab,
        cycle_size=args.cycle_size,
        n_patterns=args.patterns,
        seed=500000 + seed * 100000,
    )

    shuffle_generator = torch.Generator().manual_seed(seed + 77)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=shuffle_generator,
        num_workers=0,
        collate_fn=GraphCollator(
            vocab,
            seed=seed + 101,
            deterministic=False,
        ),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=GraphCollator(
            vocab,
            seed=seed + 202,
            deterministic=True,
        ),
    )
    return vocab, train_loader, val_loader


# -----------------------------------------------------------------------------
# Canonical-CE helpers
# -----------------------------------------------------------------------------


def _candidate_sequence_key(graph, vocab) -> tuple[int, ...]:
    return tuple(
        int(token)
        for token in graph_to_sequence(
            list(graph),
            vocab,
            add_start=False,
            add_end=True,
        )
    )


def exact_canonical_graph(graph, vocab, *, exact_threshold: int):
    """Return one rename/order-invariant exact canonical representative.

    We ask PMAR for its *exact unique* serialization set and choose the lexicographically
    smallest serialization. This baseline uses PMAR's equivalence definition but does not
    marginalize over candidates.
    """

    candidates = build_pmar_candidates(
        graph,
        vocab,
        exact_threshold=exact_threshold,
        num_samples=1,
        rng=np.random.default_rng(0),
    )

    if candidates.mode != "exact":
        raise RuntimeError(
            "canonical_ce requires exact candidate enumeration, but this graph entered "
            f"{candidates.mode!r} mode with residual permutation count "
            f"{candidates.residual_permutation_count}. Increase --exact-threshold."
        )
    if not candidates.graphs:
        raise RuntimeError("PMAR produced no exact candidates")

    return min(
        candidates.graphs,
        key=lambda candidate: _candidate_sequence_key(candidate, vocab),
    )


def canonical_ce_loss(model, batch, *, exact_threshold: int) -> torch.Tensor:
    """Mean-token CE on one deterministic exact canonical serialization per graph."""

    if "train_graphs" not in batch:
        raise KeyError(
            "canonical_ce requires batch['train_graphs']; use the PMAR-enabled GraphCollator"
        )

    conditioning = model.encode(batch, modality="text", training=True)
    device = conditioning.tokens.device

    input_rows = []
    target_rows = []

    for graph in batch["train_graphs"]:
        canonical_graph = exact_canonical_graph(
            graph,
            model.vocab,
            exact_threshold=exact_threshold,
        )
        input_array, target_array = build_train_pair(
            list(canonical_graph),
            model.vocab,
            pad_to_max=False,
        )
        input_rows.append(torch.from_numpy(input_array).long())
        target_rows.append(torch.from_numpy(target_array).long())

    lengths = {int(row.shape[0]) for row in input_rows}
    if len(lengths) != 1:
        raise RuntimeError(
            "canonical_ce benchmark expected equal sequence lengths inside the synthetic batch"
        )

    input_tokens = torch.stack(input_rows, dim=0).to(device)
    target_tokens = torch.stack(target_rows, dim=0).to(device)
    logits = model.decoder(conditioning, input_tokens)

    # Explicit mean-token CE so this baseline matches ordinary CE gradient scale.
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target_tokens.reshape(-1),
        ignore_index=model.vocab.no_known_token,
        reduction="mean",
    )


def fixed_target_length(batch, vocab) -> int:
    lengths = set()
    for graph in batch["train_graphs"]:
        _, target_array = build_train_pair(
            list(graph),
            vocab,
            pad_to_max=False,
        )
        lengths.add(int(target_array.shape[0]))
    if len(lengths) != 1:
        raise RuntimeError(
            "pmar_exact_norm is defined here only for fixed-length synthetic batches"
        )
    return next(iter(lengths))


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------


@torch.no_grad()
def evaluate_model(model, loader, cfg, recall_k: int):
    model.eval()
    samples = []

    matcher = InstanceMatcher(
        n=int(cfg.eval.get("matcher_n", 3)),
        depth_limit=int(cfg.eval.get("matcher_depth", 10)),
        allow_identity_fallback=True,
    )

    for batch in loader:
        predictions = model.predict(batch, modality="text")
        samples.extend(zip(batch["gt_graphs"], predictions))

    return evaluate_graphs(samples, ks=(recall_k,), matcher=matcher)


# -----------------------------------------------------------------------------
# Methods / training
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class MethodSpec:
    name: str
    loss_type: str
    sampled_k: int | None = None
    canonical_ce: bool = False
    normalize_after_marginal: bool = False


def build_methods(sampled_ks: Sequence[int]) -> list[MethodSpec]:
    methods = [
        MethodSpec(name="ce", loss_type="ce"),
        MethodSpec(name="canonical_ce", loss_type="ce", canonical_ce=True),
    ]

    for k in sampled_ks:
        if int(k) < 1:
            raise ValueError("sampled PMAR K must be >= 1")
        methods.append(
            MethodSpec(name=f"pmar_s{int(k)}", loss_type="pmar", sampled_k=int(k))
        )

    methods.extend(
        [
            MethodSpec(name="pmar_exact", loss_type="pmar"),
            MethodSpec(
                name="pmar_exact_norm",
                loss_type="pmar",
                normalize_after_marginal=True,
            ),
        ]
    )
    return methods


def compute_training_loss(model, batch, method: MethodSpec, args) -> torch.Tensor:
    if method.canonical_ce:
        return canonical_ce_loss(
            model,
            batch,
            exact_threshold=args.exact_threshold,
        )

    loss = model.compute_loss(batch, modality="text")

    if method.normalize_after_marginal:
        target_length = fixed_target_length(batch, model.vocab)
        loss = loss / float(target_length)

    return loss


def train_one(args, method: MethodSpec, seed: int):
    cfg = make_config(
        args,
        method.loss_type,
        sampled_k=method.sampled_k,
    )

    seed_everything(seed)
    vocab, train_loader, val_loader = make_loaders(cfg, args, seed)

    # Identical initialization across methods for each seed.
    seed_everything(seed)
    device = torch.device(args.device)
    model = PatchSGGModel(cfg).to(device)

    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=args.lr,
        weight_decay=0.0,
    )

    recall_name = f"R@{args.cycle_size}"
    mean_recall_name = f"mR@{args.cycle_size}"

    initial = evaluate_model(model, val_loader, cfg, args.cycle_size)
    print(
        f"[{method.name} seed={seed}] initial "
        f"{recall_name}={initial[recall_name]:.3f} "
        f"{mean_recall_name}={initial[mean_recall_name]:.3f} "
        f"triplet={initial['set/triplet']:.3f}"
    )

    history = [
        {
            "step": 0,
            "R": initial[recall_name],
            "mR": initial[mean_recall_name],
            "triplet": initial["set/triplet"],
        }
    ]

    train_iterator = iter(train_loader)
    train_losses = []
    start_time = time.perf_counter()

    for step in range(1, args.steps + 1):
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        model.train()
        optimizer.zero_grad(set_to_none=True)

        loss = compute_training_loss(model, batch, method, args)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(
                f"{method.name}: non-finite loss at step {step}: {loss}"
            )

        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.trainable_parameters(),
                args.grad_clip,
            )
        optimizer.step()
        train_losses.append(float(loss.detach().cpu()))

        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate_model(model, val_loader, cfg, args.cycle_size)
            history.append(
                {
                    "step": step,
                    "R": metrics[recall_name],
                    "mR": metrics[mean_recall_name],
                    "triplet": metrics["set/triplet"],
                }
            )
            print(
                f"[{method.name} seed={seed}] step={step:4d} "
                f"{recall_name}={metrics[recall_name]:.3f} "
                f"{mean_recall_name}={metrics[mean_recall_name]:.3f} "
                f"triplet={metrics['set/triplet']:.3f}"
            )

    elapsed = time.perf_counter() - start_time
    final_metrics = evaluate_model(model, val_loader, cfg, args.cycle_size)
    tail = train_losses[-min(20, len(train_losses)) :]

    return {
        "method": method.name,
        "seed": seed,
        "R": final_metrics[recall_name],
        "mR": final_metrics[mean_recall_name],
        "triplet": final_metrics["set/triplet"],
        "best_R": max(point["R"] for point in history),
        "curve_mean_R": statistics.mean(point["R"] for point in history[1:]),
        "elapsed": elapsed,
        "mean_train_loss": statistics.mean(tail),
        "history": history,
    }


# -----------------------------------------------------------------------------
# Statistics / reporting
# -----------------------------------------------------------------------------


def mean_std(results: Sequence[dict], key: str) -> tuple[float, float]:
    values = [float(result[key]) for result in results]
    return (
        statistics.mean(values),
        statistics.stdev(values) if len(values) > 1 else 0.0,
    )


def paired_bootstrap_ci(
    baseline: Sequence[float],
    treatment: Sequence[float],
    *,
    seed: int = 12345,
    draws: int = 20000,
) -> tuple[float, float, float]:
    """Paired mean difference and percentile bootstrap CI.

    This is descriptive for small n; do not treat it as definitive significance testing.
    """

    if len(baseline) != len(treatment):
        raise ValueError("paired arrays must have the same length")
    diffs = np.asarray(treatment, dtype=np.float64) - np.asarray(
        baseline, dtype=np.float64
    )
    mean_diff = float(diffs.mean())
    if len(diffs) <= 1:
        return mean_diff, mean_diff, mean_diff

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(diffs), size=(draws, len(diffs)))
    boot = diffs[indices].mean(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    return mean_diff, float(low), float(high)


def save_csv(path: str, results: Sequence[dict]) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "seed", "step", "R", "mR", "triplet"])
        for result in results:
            for point in result["history"]:
                writer.writerow(
                    [
                        result["method"],
                        result["seed"],
                        point["step"],
                        point["R"],
                        point["mR"],
                        point["triplet"],
                    ]
                )


def print_summary(results: Sequence[dict], methods: Sequence[MethodSpec], recall_name: str):
    print("\n" + "=" * 100)
    print("FINAL ABLATION RESULTS")
    print("=" * 100)
    print()
    print(
        f"{'Method':<18}"
        f"{recall_name:>15}"
        f"{'Best R':>15}"
        f"{'Curve mean':>15}"
        f"{'Triplet':>15}"
        f"{'Train sec':>15}"
    )
    print("-" * 93)

    by_method: dict[str, list[dict]] = {}
    for method in methods:
        rows = [r for r in results if r["method"] == method.name]
        by_method[method.name] = rows

        r_mean, r_std = mean_std(rows, "R")
        best_mean, best_std = mean_std(rows, "best_R")
        curve_mean, curve_std = mean_std(rows, "curve_mean_R")
        trip_mean, trip_std = mean_std(rows, "triplet")
        time_mean, time_std = mean_std(rows, "elapsed")

        print(
            f"{method.name:<18}"
            f"{r_mean:>8.3f}±{r_std:<6.3f}"
            f"{best_mean:>8.3f}±{best_std:<6.3f}"
            f"{curve_mean:>8.3f}±{curve_std:<6.3f}"
            f"{trip_mean:>8.3f}±{trip_std:<6.3f}"
            f"{time_mean:>8.2f}±{time_std:<6.2f}"
        )

    baseline_rows = by_method.get("ce", [])
    baseline_by_seed = {int(r["seed"]): float(r["R"]) for r in baseline_rows}

    print("\nPaired final-R differences vs CE (bootstrap 95% CI; descriptive):")
    print(f"{'Method':<18}{'Delta R':>12}{'95% CI':>24}")
    print("-" * 54)

    for method in methods:
        if method.name == "ce":
            continue
        rows = by_method[method.name]
        treatment_by_seed = {int(r["seed"]): float(r["R"]) for r in rows}
        common = sorted(set(baseline_by_seed) & set(treatment_by_seed))
        baseline = [baseline_by_seed[s] for s in common]
        treatment = [treatment_by_seed[s] for s in common]
        delta, low, high = paired_bootstrap_ci(baseline, treatment)
        print(f"{method.name:<18}{delta:>+12.3f}{f'[{low:+.3f}, {high:+.3f}]':>24}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--train-examples", type=int, default=128)
    parser.add_argument("--val-examples", type=int, default=64)
    parser.add_argument("--patterns", type=int, default=8)
    parser.add_argument("--cycle-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--candidate-batch-size", type=int, default=32)
    parser.add_argument("--exact-threshold", type=int, default=256)
    parser.add_argument("--sampled-ks", type=int, nargs="*", default=[1, 2, 4, 8])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--csv", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    print("\n" + "=" * 100)
    print("PMAR CONTROLLED ABLATION: SERIALIZATION vs MARGINALIZATION vs LOSS SCALE")
    print("=" * 100)
    print(f"Device:                 {args.device}")
    print(f"Batch size:             {args.batch_size}")
    print(f"Optimizer steps:        {args.steps}")
    print(f"Evaluation every:       {args.eval_every} steps")
    print(f"Cycle size:             {args.cycle_size}")
    print(f"Candidate batch size:   {args.candidate_batch_size}")
    print(f"Exact threshold:        {args.exact_threshold}")
    print(f"Seeds:                  {args.seeds}")

    # Probe the actual ambiguity used by the benchmark.
    probe_cfg = make_config(args, "pmar")
    probe_vocab = build_vocab(probe_cfg)
    probe_graph = AmbiguousCycleDataset(
        1,
        probe_vocab,
        cycle_size=args.cycle_size,
        n_patterns=args.patterns,
        seed=123,
    )[0]["graph"]
    probe = build_pmar_candidates(
        probe_graph,
        probe_vocab,
        exact_threshold=args.exact_threshold,
        num_samples=8,
        rng=np.random.default_rng(0),
    )

    print("\nSynthetic ambiguity check:")
    print(f"    M_residual:              {probe.residual_permutation_count}")
    print(f"    candidate mode:          {probe.mode}")
    print(f"    exact unique candidates: {probe.num_candidates if probe.mode == 'exact' else 'not enumerated'}")

    if probe.mode != "exact":
        raise RuntimeError(
            "This ablation needs exact candidates for canonical_ce and pmar_exact. "
            "Increase --exact-threshold."
        )

    methods = build_methods(args.sampled_ks)
    print("\nMethods:")
    for method in methods:
        print(f"    {method.name}")

    results = []
    for seed in args.seeds:
        print("\n" + "=" * 100)
        print(f"SEED {seed}")
        print("=" * 100)
        for method in methods:
            print("\n" + "-" * 100)
            print(f"Training {method.name}")
            print("-" * 100)
            results.append(train_one(args, method, seed))

    recall_name = f"R@{args.cycle_size}"
    print_summary(results, methods, recall_name)

    if args.csv:
        save_csv(args.csv, results)
        print(f"\nSaved learning curves to: {args.csv}")

    print("\nInterpretation:")
    print("  ce vs canonical_ce       -> deterministic canonicalization effect")
    print("  canonical_ce vs PMAR     -> marginalization effect")
    print("  pmar_exact vs _norm      -> optimization / gradient-scale effect")
    print("  PMAR K sweep             -> candidate-count quality/cost tradeoff")
    print("  Best R / curve mean      -> avoids relying only on one noisy final checkpoint")
    print("\nDo not compare raw CE and PMAR loss magnitudes directly.")


if __name__ == "__main__":
    main()