"""Controlled learning-performance benchmark: CE vs PMAR.

This benchmark asks:

    Does PMAR learn an ambiguous scene graph more effectively than CE?

Unlike benchmark_pmar_vs_ce.py, which measures runtime/memory using random
tensors, this script actually TRAINS models.

The synthetic dataset deliberately contains graphs with same-class objects
whose instance IDs are arbitrary.

Each graph is a directed four-node cycle:

    A -> B -> C -> D -> A

All four nodes have the same entity class and all edges have the same
predicate class.

For these graphs:

    raw / residual assignments = 4! = 24

but exact PMAR produces only 6 unique canonical serializations after
deduplication.

The conditioning text is invariant to instance numbering, while the
GraphCollator still randomizes relation order during training.

Therefore:

CE
    sees different arbitrary autoregressive targets for the same
    semantic conditioning.

PMAR
    marginalizes over graph-equivalent serializations.

Metrics are evaluated with the project's LF-SGG matcher, so arbitrary
instance numbering at prediction time is not penalized.

Example:

    python tools/benchmark_pmar_vs_ce_quality.py \
        --device cuda \
        --batch-size 8 \
        --steps 200 \
        --eval-every 20 \
        --candidate-batch-size 32 \
        --seeds 0

To additionally compare sampled PMAR with different K:

    python tools/benchmark_pmar_vs_ce_quality.py \
        --device cuda \
        --batch-size 8 \
        --steps 200 \
        --eval-every 20 \
        --candidate-batch-size 32 \
        --sampled-ks 1 2 4 8 \
        --seeds 0

For a more reliable experiment:

    python tools/benchmark_pmar_vs_ce_quality.py \
        --device cuda \
        --batch-size 8 \
        --steps 300 \
        --eval-every 25 \
        --candidate-batch-size 32 \
        --sampled-ks 1 2 4 8 \
        --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import statistics
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from patchsgg.config import load_config
from patchsgg.data.collate import GraphCollator
from patchsgg.eval.evaluate import evaluate_graphs
from patchsgg.eval.matcher import InstanceMatcher
from patchsgg.graph_seq.linearize import Relation
from patchsgg.graph_seq.pmar import build_pmar_candidates
from patchsgg.model import PatchSGGModel, build_vocab


# ============================================================================
# Synthetic ambiguity dataset
# ============================================================================


class AmbiguousCycleDataset(Dataset):
    """Synthetic dataset specifically designed to exercise PMAR.

    Every underlying graph is a directed 4-cycle:

        0 -> 1
        1 -> 2
        2 -> 3
        3 -> 0

    All four nodes have the same entity class.

    The annotation instance IDs are randomly renamed for every dataset item.

    Crucially, the text conditioning DOES NOT contain those instance IDs.

    Therefore arbitrary annotation naming cannot be recovered from the
    conditioning input and should not be treated as semantic information.
    """

    def __init__(
        self,
        n: int,
        vocab,
        *,
        n_patterns: int = 8,
        seed: int = 0,
    ):
        self.n = int(n)
        self.vocab = vocab
        self.n_patterns = int(n_patterns)
        self.seed = int(seed)

        # Each pattern gets a separate entity class and predicate class.
        #
        # Pattern 0:
        #   entity class 1, predicate 1
        #
        # Pattern 1:
        #   entity class 2, predicate 2
        #
        # etc.
        if self.n_patterns >= min(
            vocab.n_entities,
            vocab.n_preds,
        ):
            raise ValueError(
                "n_patterns is too large for the configured vocabulary"
            )

    def __len__(self) -> int:
        return self.n

    def __getitem__(
        self,
        index: int,
    ):
        pattern_id = (
            index % self.n_patterns
        )

        entity_class = (
            1 + pattern_id
        )

        predicate = (
            1 + pattern_id
        )

        rng = np.random.default_rng(
            self.seed
            + index * 104729
        )

        # ------------------------------------------------------------
        # Arbitrarily rename the four graph nodes.
        #
        # This changes annotation identity but not graph semantics.
        # ------------------------------------------------------------

        rename = (
            rng.permutation(4)
            .tolist()
        )

        # Semantic cycle.
        edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
        ]

        # Also vary raw relation storage order.
        rng.shuffle(
            edges
        )

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

        image_id = (
            self.seed + index
        )

        # GraphCollator expects an image even though this experiment
        # conditions on text.
        generator = (
            torch.Generator()
            .manual_seed(
                image_id
            )
        )

        image = torch.rand(
            3,
            32,
            32,
            generator=generator,
        )

        # ------------------------------------------------------------
        # IMPORTANT:
        #
        # Same semantic pattern -> same conditioning text.
        #
        # The text contains NO instance IDs.
        #
        # This means the network cannot infer arbitrary annotation IDs
        # from the input.
        # ------------------------------------------------------------

        text = (
            f"ambiguous directed-cycle "
            f"pattern {pattern_id}"
        )

        return {
            "image": image,
            "text": text,
            "graph": graph,
            "image_id": image_id,
        }


# ============================================================================
# Reproducibility
# ============================================================================


def seed_everything(
    seed: int,
):
    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


# ============================================================================
# Configuration
# ============================================================================


def make_config(
    args,
    loss_type: str,
    *,
    sampled_k: int | None = None,
):
    """Create an otherwise-identical CE or PMAR configuration."""

    overrides = [
        f"device={args.device}",

        # Only text is required.
        "encoders.build_only_required=true",

        "train.train_modality=text",
        "eval.eval_modality=text",

        f"train.batch_size={args.batch_size}",
        f"eval.batch_size={args.batch_size}",

        f"train.lr={args.lr}",
        "train.weight_decay=0.0",

        # Remove dropout noise from the controlled experiment.
        "decoder.dropout=0.0",

        # CE baseline should be ordinary likelihood CE.
        "loss.label_smoothing=0.0",

        # Every synthetic graph has four relations.
        "vocab.max_num_rels=4",
        "eval.max_rels=4",

        # Fully deterministic generation.
        "eval.entity_sampling=greedy",

        # Allow normal EOS behavior.
        "eval.allow_end=true",

        # Helpful for synthetic matcher evaluation.
        "eval.matcher_identity_fallback=true",

        f"loss.type={loss_type}",
    ]

    if loss_type == "pmar":

        overrides.append(
            "loss.pmar_candidate_batch_size="
            f"{args.candidate_batch_size}"
        )

        # ------------------------------------------------------------
        # Exact PMAR
        # ------------------------------------------------------------

        if sampled_k is None:

            # Four-cycle:
            #
            # M_residual = 24
            #
            # so threshold 64 makes it exact.
            overrides.extend(
                [
                    "loss.pmar_exact_threshold=64",
                    "loss.pmar_num_samples=8",
                ]
            )

        # ------------------------------------------------------------
        # Sampled PMAR
        # ------------------------------------------------------------

        else:

            # Force sampling because 24 > 1.
            overrides.extend(
                [
                    "loss.pmar_exact_threshold=1",
                    f"loss.pmar_num_samples={sampled_k}",
                ]
            )

    return load_config(
        "patchsgg/configs/"
        "diagnostic_text2text.yaml",
        overrides,
    )


# ============================================================================
# Data
# ============================================================================


def make_loaders(
    cfg,
    args,
    seed: int,
):
    vocab = build_vocab(
        cfg
    )

    train_dataset = (
        AmbiguousCycleDataset(
            args.train_examples,
            vocab,
            n_patterns=args.patterns,
            seed=(
                1000
                + seed * 100000
            ),
        )
    )

    val_dataset = (
        AmbiguousCycleDataset(
            args.val_examples,
            vocab,
            n_patterns=args.patterns,
            seed=(
                500000
                + seed * 100000
            ),
        )
    )

    # ---------------------------------------------------------------
    # Resetting both the DataLoader generator and GraphCollator seed
    # for every method gives CE and PMAR equivalent data randomness.
    # ---------------------------------------------------------------

    shuffle_generator = (
        torch.Generator()
        .manual_seed(
            seed + 77
        )
    )

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

    return (
        vocab,
        train_loader,
        val_loader,
    )


# ============================================================================
# Evaluation
# ============================================================================


@torch.no_grad()
def evaluate_model(
    model,
    loader,
    cfg,
):
    """Evaluate graph prediction, not token identity."""

    model.eval()

    samples = []

    matcher = InstanceMatcher(
        n=int(
            cfg.eval.get(
                "matcher_n",
                3,
            )
        ),
        depth_limit=int(
            cfg.eval.get(
                "matcher_depth",
                10,
            )
        ),
        allow_identity_fallback=True,
    )

    for batch in loader:

        predictions = model.predict(
            batch,
            modality="text",
        )

        for gt, pred in zip(
            batch["gt_graphs"],
            predictions,
        ):
            samples.append(
                (
                    gt,
                    pred,
                )
            )

    # Every GT graph has four relations.
    #
    # R@4 therefore asks:
    #
    # How much of the entire GT graph was recovered
    # in the model's four generated relations?
    return evaluate_graphs(
        samples,
        ks=(4,),
        matcher=matcher,
    )


# ============================================================================
# One training experiment
# ============================================================================


def train_one(
    args,
    method: str,
    seed: int,
):
    sampled_k = None

    if method.startswith(
        "pmar_s"
    ):
        sampled_k = int(
            method.split(
                "s",
                1,
            )[1]
        )

    loss_type = (
        "ce"
        if method == "ce"
        else "pmar"
    )

    cfg = make_config(
        args,
        loss_type,
        sampled_k=sampled_k,
    )

    # ---------------------------------------------------------------
    # Build identical data streams.
    # ---------------------------------------------------------------

    seed_everything(
        seed
    )

    (
        vocab,
        train_loader,
        val_loader,
    ) = make_loaders(
        cfg,
        args,
        seed,
    )

    # ---------------------------------------------------------------
    # IMPORTANT:
    # Reset seed immediately before model construction.
    #
    # CE and PMAR therefore start from identical decoder weights.
    # ---------------------------------------------------------------

    seed_everything(
        seed
    )

    device = torch.device(
        args.device
    )

    model = PatchSGGModel(
        cfg
    ).to(
        device
    )

    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=args.lr,
        weight_decay=0.0,
    )

    # ---------------------------------------------------------------
    # Initial performance.
    # ---------------------------------------------------------------

    initial_metrics = (
        evaluate_model(
            model,
            val_loader,
            cfg,
        )
    )

    print()
    print(
        f"[{method} seed={seed}] "
        f"initial "
        f"R@4={initial_metrics['R@4']:.3f} "
        f"mR@4={initial_metrics['mR@4']:.3f} "
        f"triplet={initial_metrics['set/triplet']:.3f}"
    )

    history = [
        {
            "step": 0,
            "R@4": initial_metrics[
                "R@4"
            ],
            "mR@4": initial_metrics[
                "mR@4"
            ],
            "triplet": initial_metrics[
                "set/triplet"
            ],
        }
    ]

    train_iterator = iter(
        train_loader
    )

    train_losses = []

    start_time = (
        time.perf_counter()
    )

    # ---------------------------------------------------------------
    # Fixed optimizer-step budget.
    # ---------------------------------------------------------------

    for step in range(
        1,
        args.steps + 1,
    ):

        try:
            batch = next(
                train_iterator
            )

        except StopIteration:
            train_iterator = iter(
                train_loader
            )

            batch = next(
                train_iterator
            )

        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        loss = model.compute_loss(
            batch,
            modality="text",
        )

        if not bool(
            torch.isfinite(
                loss
            )
        ):
            raise RuntimeError(
                f"{method}: non-finite loss "
                f"at step {step}: {loss}"
            )

        loss.backward()

        if args.grad_clip > 0:

            torch.nn.utils.clip_grad_norm_(
                model.trainable_parameters(),
                args.grad_clip,
            )

        optimizer.step()

        train_losses.append(
            float(
                loss.detach()
                .cpu()
            )
        )

        # -----------------------------------------------------------
        # Evaluate learning curve.
        # -----------------------------------------------------------

        if (
            step % args.eval_every
            == 0
            or step == args.steps
        ):

            metrics = (
                evaluate_model(
                    model,
                    val_loader,
                    cfg,
                )
            )

            history.append(
                {
                    "step": step,
                    "R@4": metrics[
                        "R@4"
                    ],
                    "mR@4": metrics[
                        "mR@4"
                    ],
                    "triplet": metrics[
                        "set/triplet"
                    ],
                }
            )

            print(
                f"[{method} seed={seed}] "
                f"step={step:4d} "
                f"R@4={metrics['R@4']:.3f} "
                f"mR@4={metrics['mR@4']:.3f} "
                f"triplet={metrics['set/triplet']:.3f}"
            )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    final_metrics = (
        evaluate_model(
            model,
            val_loader,
            cfg,
        )
    )

    tail = train_losses[
        -min(
            20,
            len(train_losses),
        ):
    ]

    return {
        "method": method,
        "seed": seed,

        "initial_R@4":
            initial_metrics[
                "R@4"
            ],

        "R@4":
            final_metrics[
                "R@4"
            ],

        "mR@4":
            final_metrics[
                "mR@4"
            ],

        "triplet":
            final_metrics[
                "set/triplet"
            ],

        "elapsed":
            elapsed,

        # Do NOT directly compare CE and PMAR loss magnitude.
        # This is logged only as a within-method diagnostic.
        "mean_train_loss":
            statistics.mean(
                tail
            ),

        "history":
            history,
    }


# ============================================================================
# Statistics
# ============================================================================


def mean_std(
    results,
    key,
):
    values = [
        result[key]
        for result in results
    ]

    mean = statistics.mean(
        values
    )

    std = (
        statistics.stdev(
            values
        )
        if len(values) > 1
        else 0.0
    )

    return (
        mean,
        std,
    )


# ============================================================================
# Optional CSV
# ============================================================================


def save_csv(
    path,
    results,
):
    directory = os.path.dirname(
        path
    )

    if directory:
        os.makedirs(
            directory,
            exist_ok=True,
        )

    with open(
        path,
        "w",
        newline="",
    ) as handle:

        writer = csv.writer(
            handle
        )

        writer.writerow(
            [
                "method",
                "seed",
                "step",
                "R@4",
                "mR@4",
                "triplet",
            ]
        )

        for result in results:

            for point in result[
                "history"
            ]:

                writer.writerow(
                    [
                        result[
                            "method"
                        ],
                        result[
                            "seed"
                        ],
                        point[
                            "step"
                        ],
                        point[
                            "R@4"
                        ],
                        point[
                            "mR@4"
                        ],
                        point[
                            "triplet"
                        ],
                    ]
                )

    print(
        f"\nSaved learning curves to: "
        f"{path}"
    )


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--eval-every",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--train-examples",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--val-examples",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--patterns",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--candidate-batch-size",
        type=int,
        default=32,
    )

    # Optional sampled PMAR experiments.
    #
    # Example:
    #
    #   --sampled-ks 1 2 4 8
    #
    # will run:
    #
    #   CE
    #   PMAR sampled K=1
    #   PMAR sampled K=2
    #   PMAR sampled K=4
    #   PMAR sampled K=8
    #   PMAR exact
    parser.add_argument(
        "--sampled-ks",
        type=int,
        nargs="*",
        default=[],
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0],
    )

    parser.add_argument(
        "--csv",
        default=None,
    )

    args = parser.parse_args()

    print()
    print("=" * 80)
    print(
        "CE vs PMAR LEARNING-PERFORMANCE BENCHMARK"
    )
    print("=" * 80)

    print()
    print(
        f"Device:                 "
        f"{args.device}"
    )

    print(
        f"Batch size:             "
        f"{args.batch_size}"
    )

    print(
        f"Optimizer steps:        "
        f"{args.steps}"
    )

    print(
        f"Evaluation every:       "
        f"{args.eval_every} steps"
    )

    print(
        f"Train examples:         "
        f"{args.train_examples}"
    )

    print(
        f"Validation examples:    "
        f"{args.val_examples}"
    )

    print(
        f"Semantic patterns:      "
        f"{args.patterns}"
    )

    print(
        f"Candidate batch size:   "
        f"{args.candidate_batch_size}"
    )

    print(
        f"Seeds:                  "
        f"{args.seeds}"
    )

    # ----------------------------------------------------------------
    # Verify that the benchmark graph genuinely exercises PMAR.
    # ----------------------------------------------------------------

    probe_cfg = make_config(
        args,
        "pmar",
    )

    probe_vocab = build_vocab(
        probe_cfg
    )

    probe_graph = (
        AmbiguousCycleDataset(
            1,
            probe_vocab,
            seed=123,
        )[0]["graph"]
    )

    probe_candidates = (
        build_pmar_candidates(
            probe_graph,
            probe_vocab,
            exact_threshold=64,
            num_samples=8,
            rng=np.random.default_rng(
                0
            ),
        )
    )

    print()
    print(
        "Synthetic ambiguity check:"
    )

    print(
        "    M_residual:             "
        f"{probe_candidates.residual_permutation_count}"
    )

    print(
        "    exact unique candidates:"
        f" {probe_candidates.num_candidates}"
    )

    if (
        probe_candidates
        .residual_permutation_count
        <= 1
    ):
        raise RuntimeError(
            "benchmark graph has no PMAR ambiguity"
        )

    if (
        probe_candidates
        .num_candidates
        <= 1
    ):
        raise RuntimeError(
            "benchmark graph does not produce "
            "multiple unique PMAR serializations"
        )

    # ----------------------------------------------------------------
    # Methods.
    # ----------------------------------------------------------------

    methods = [
        "ce"
    ]

    for k in args.sampled_ks:

        if k < 1:
            raise ValueError(
                "sampled PMAR K must be >= 1"
            )

        methods.append(
            f"pmar_s{k}"
        )

    methods.append(
        "pmar_exact"
    )

    print()
    print(
        "Methods:"
    )

    for method in methods:
        print(
            f"    {method}"
        )

    results = []

    # ----------------------------------------------------------------
    # Run every method from the same seed.
    # ----------------------------------------------------------------

    for seed in args.seeds:

        print()
        print("=" * 80)
        print(
            f"SEED {seed}"
        )
        print("=" * 80)

        for method in methods:

            print()
            print("-" * 80)

            print(
                f"Training {method}"
            )

            print("-" * 80)

            result = train_one(
                args,
                method,
                seed,
            )

            results.append(
                result
            )

    # ----------------------------------------------------------------
    # Final summary.
    # ----------------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "FINAL PERFORMANCE RESULTS"
    )
    print("=" * 80)

    print()

    print(
        f"{'Method':<14}"
        f"{'R@4':>14}"
        f"{'mR@4':>14}"
        f"{'Triplet':>14}"
        f"{'Train sec':>14}"
    )

    print(
        "-" * 70
    )

    for method in methods:

        method_results = [
            result
            for result in results
            if result["method"]
            == method
        ]

        recall_mean, recall_std = (
            mean_std(
                method_results,
                "R@4",
            )
        )

        mr_mean, mr_std = (
            mean_std(
                method_results,
                "mR@4",
            )
        )

        triplet_mean, triplet_std = (
            mean_std(
                method_results,
                "triplet",
            )
        )

        time_mean, time_std = (
            mean_std(
                method_results,
                "elapsed",
            )
        )

        print(
            f"{method:<14}"
            f"{recall_mean:>8.3f}"
            f"±{recall_std:<5.3f}"
            f"{mr_mean:>8.3f}"
            f"±{mr_std:<5.3f}"
            f"{triplet_mean:>8.3f}"
            f"±{triplet_std:<5.3f}"
            f"{time_mean:>8.2f}"
            f"±{time_std:<5.2f}"
        )

    if args.csv:

        save_csv(
            args.csv,
            results,
        )

    print()
    print("=" * 80)
    print(
        "HOW TO INTERPRET THIS"
    )
    print("=" * 80)

    print()
    print(
        "Higher R@4, mR@4, and triplet recall are better."
    )

    print()
    print(
        "The primary comparison is performance after the SAME number "
        "of optimizer steps."
    )

    print()
    print(
        "Do not compare the raw CE and PMAR training-loss values directly."
    )

    print(
        "CE uses mean token cross-entropy while PMAR uses summed "
        "autoregressive sequence NLL before marginalization."
    )

    print()
    print(
        "This benchmark deliberately tests representation ambiguity."
    )

    print(
        "It is NOT a claim about final Visual Genome performance."
    )


if __name__ == "__main__":
    main()