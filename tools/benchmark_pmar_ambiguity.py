"""Sweep graph ambiguity strength for CE vs canonical CE vs PMAR.

This script reuses the controlled synthetic experiment from benchmark_pmar_ablation.py and changes
only the size of a same-class directed cycle. For a directed n-cycle with identical node/predicate
classes, residual label ambiguity grows factorially (n!), while exact unique serialized candidates
grow much more slowly because graph symmetries deduplicate many assignments.

The purpose is to test the hypothesis:

    PMAR should become more useful as arbitrary serialization ambiguity increases.

Example:
python tools/benchmark_pmar_ambiguity.py \
  --device cuda \
  --cycle-sizes 3 4 5 \
  --batch-size 8 \
  --steps 300 \
  --eval-every 25 \
  --seeds 0 1 2 3 4 \
  --csv outputs/pmar_ambiguity.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
from types import SimpleNamespace

import numpy as np
import torch

# This file is intended to live beside benchmark_pmar_ablation.py in tools/.
from benchmark_pmar_ablation import (
    AmbiguousCycleDataset,
    MethodSpec,
    build_vocab,
    make_config,
    paired_bootstrap_ci,
    train_one,
)
from patchsgg.graph_seq.pmar import build_pmar_candidates


def method_specs() -> list[MethodSpec]:
    return [
        MethodSpec(name="ce", loss_type="ce"),
        MethodSpec(name="canonical_ce", loss_type="ce", canonical_ce=True),
        MethodSpec(name="pmar_s2", loss_type="pmar", sampled_k=2),
        MethodSpec(name="pmar_exact", loss_type="pmar"),
        MethodSpec(
            name="pmar_exact_norm",
            loss_type="pmar",
            normalize_after_marginal=True,
        ),
    ]


def mean_std(values):
    values = [float(v) for v in values]
    return (
        statistics.mean(values),
        statistics.stdev(values) if len(values) > 1 else 0.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--cycle-sizes", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--train-examples", type=int, default=128)
    parser.add_argument("--val-examples", type=int, default=64)
    parser.add_argument("--patterns", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--candidate-batch-size", type=int, default=64)
    parser.add_argument("--exact-threshold", type=int, default=1024)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    methods = method_specs()
    all_rows = []

    print("\n" + "=" * 100)
    print("PMAR AMBIGUITY-STRENGTH SWEEP")
    print("=" * 100)
    print(f"Cycle sizes: {args.cycle_sizes}")
    print(f"Seeds:       {args.seeds}")

    for cycle_size in args.cycle_sizes:
        run_args = SimpleNamespace(**vars(args))
        run_args.cycle_size = int(cycle_size)
        # make_config expects sampled_ks to exist only elsewhere; provide it for completeness.
        run_args.sampled_ks = [2]

        probe_cfg = make_config(run_args, "pmar")
        probe_vocab = build_vocab(probe_cfg)
        probe_graph = AmbiguousCycleDataset(
            1,
            probe_vocab,
            cycle_size=cycle_size,
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

        print("\n" + "#" * 100)
        print(
            f"CYCLE SIZE {cycle_size}: M_residual={probe.residual_permutation_count}, "
            f"mode={probe.mode}, exact_unique={probe.num_candidates if probe.mode == 'exact' else 'N/A'}"
        )
        print("#" * 100)

        if probe.mode != "exact":
            raise RuntimeError(
                f"cycle size {cycle_size} exceeded exact threshold; increase --exact-threshold"
            )

        for seed in args.seeds:
            for method in methods:
                print(
                    f"\n[cycle={cycle_size}] training {method.name}, seed={seed}"
                )
                result = train_one(run_args, method, seed)
                result["cycle_size"] = int(cycle_size)
                result["M_residual"] = int(probe.residual_permutation_count)
                result["exact_unique"] = int(probe.num_candidates)
                all_rows.append(result)

    print("\n" + "=" * 110)
    print("AMBIGUITY SWEEP SUMMARY")
    print("=" * 110)
    print(
        f"{'Cycle':>6}{'M_resid':>12}{'Unique':>10}{'Method':<20}"
        f"{'Final R':>16}{'Best R':>16}{'Delta vs CE':>16}"
    )
    print("-" * 110)

    for cycle_size in args.cycle_sizes:
        cycle_rows = [r for r in all_rows if r["cycle_size"] == cycle_size]
        ce_by_seed = {
            int(r["seed"]): float(r["R"])
            for r in cycle_rows
            if r["method"] == "ce"
        }

        for method in methods:
            rows = [r for r in cycle_rows if r["method"] == method.name]
            final_mean, final_std = mean_std([r["R"] for r in rows])
            best_mean, best_std = mean_std([r["best_R"] for r in rows])

            if method.name == "ce":
                delta_text = "--"
            else:
                by_seed = {int(r["seed"]): float(r["R"]) for r in rows}
                common = sorted(set(ce_by_seed) & set(by_seed))
                delta, low, high = paired_bootstrap_ci(
                    [ce_by_seed[s] for s in common],
                    [by_seed[s] for s in common],
                )
                delta_text = f"{delta:+.3f} [{low:+.3f},{high:+.3f}]"

            print(
                f"{cycle_size:>6}"
                f"{rows[0]['M_residual']:>12}"
                f"{rows[0]['exact_unique']:>10}"
                f"{method.name:<20}"
                f"{final_mean:>8.3f}±{final_std:<7.3f}"
                f"{best_mean:>8.3f}±{best_std:<7.3f}"
                f"{delta_text:>16}"
            )

    if args.csv:
        directory = os.path.dirname(args.csv)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(args.csv, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "cycle_size",
                    "M_residual",
                    "exact_unique",
                    "method",
                    "seed",
                    "final_R",
                    "best_R",
                    "curve_mean_R",
                    "triplet",
                    "elapsed_sec",
                ]
            )
            for row in all_rows:
                writer.writerow(
                    [
                        row["cycle_size"],
                        row["M_residual"],
                        row["exact_unique"],
                        row["method"],
                        row["seed"],
                        row["R"],
                        row["best_R"],
                        row["curve_mean_R"],
                        row["triplet"],
                        row["elapsed"],
                    ]
                )
        print(f"\nSaved summary rows to: {args.csv}")

    print("\nPrimary question: does PMAR's delta over CE increase as M_residual / exact ambiguity grows?")


if __name__ == "__main__":
    main()