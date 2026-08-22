"""Measure PMAR ambiguity on real Visual Genome training graphs.

This script does not train a model. It scans the configured VG training split and reports how much
same-class instance-label ambiguity remains after PMAR structural refinement.

It deliberately uses VGGraphDataset._graph_for directly so it does not decode image files; only the
annotation HDF5 is read. For graphs with more than vocab.max_num_rels, it optionally mimics one
training-time relation shuffle/truncation before PMAR analysis.

Example:
python tools/scan_vg_pmar_ambiguity.py \
  --config patchsgg/configs/location_free_paper.yaml \
  --split train \
  --max-graphs 10000 \
  --exact-threshold 64 \
  --csv outputs/vg_pmar_ambiguity.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import time
from collections import Counter

import numpy as np

from patchsgg.config import load_config
from patchsgg.data.vg_dataset import VGGraphDataset
from patchsgg.graph_seq.linearize import permute_and_reindex_graph
from patchsgg.graph_seq.pmar import (
    build_pmar_candidates,
    residual_permutation_count,
    structural_refine,
)
from patchsgg.model import build_vocab


def raw_permutation_count(graph) -> int:
    nodes_by_class = {}
    for relation in graph:
        nodes_by_class.setdefault(int(relation.subj_cls), set()).add(int(relation.subj_inst))
        nodes_by_class.setdefault(int(relation.obj_cls), set()).add(int(relation.obj_inst))
    total = 1
    for instances in nodes_by_class.values():
        total *= math.factorial(len(instances))
    return total


def safe_log10(value: int) -> float:
    return math.log10(value) if value > 0 else float("-inf")


def bucket_residual(value: int) -> str:
    if value == 1:
        return "1"
    for upper in (2, 4, 8, 16, 32, 64):
        if value <= upper:
            return f"<={upper}"
    return ">64"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="patchsgg/configs/location_free_paper.yaml",
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--max-graphs", type=int, default=-1)
    parser.add_argument("--exact-threshold", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mimic-training-subgraph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Shuffle/reindex then truncate to vocab.max_num_rels, matching the training collator "
            "when graphs are longer than the model target budget."
        ),
    )
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, [])
    vocab = build_vocab(cfg)
    dataset = VGGraphDataset(cfg, args.split, vocab)

    total_available = len(dataset)
    limit = total_available if args.max_graphs < 0 else min(args.max_graphs, total_available)

    print("\n" + "=" * 100)
    print("VISUAL GENOME PMAR AMBIGUITY SCAN")
    print("=" * 100)
    print(f"Config:                  {args.config}")
    print(f"Split:                   {args.split}")
    print(f"Graphs scanned:          {limit} / {total_available}")
    print(f"max_num_rels:            {vocab.max_num_rels}")
    print(f"Exact threshold:         {args.exact_threshold}")
    print(f"Mimic training subgraph: {args.mimic_training_subgraph}")

    rows = []
    bucket_counts = Counter()
    exact_unique_values = []
    raw_values = []
    residual_values = []
    refine_cell_values = []
    max_cell_values = []
    relation_counts = []
    exact_count = 0
    sampled_count = 0

    start = time.perf_counter()
    roi = dataset._roi()

    try:
        for local_index in range(limit):
            img_idx = dataset.indices[local_index]
            image_id = int(dataset.image_meta[img_idx]["image_id"])
            graph = dataset._graph_for(roi, img_idx)

            if args.mimic_training_subgraph:
                rng = np.random.default_rng(
                    (args.seed * 1_000_003 + image_id * 97) % (2**63 - 1)
                )
                graph = permute_and_reindex_graph(
                    graph,
                    vocab=vocab,
                    rng=rng,
                    shuffle=True,
                )[: vocab.max_num_rels]
            else:
                graph = list(graph[: vocab.max_num_rels])

            raw_count = raw_permutation_count(graph)
            cells = structural_refine(graph, vocab)
            residual_from_cells = int(residual_permutation_count(cells))
            max_cell_size = max((len(cell.nodes) for cell in cells), default=0)

            candidate_info = build_pmar_candidates(
                graph,
                vocab,
                exact_threshold=args.exact_threshold,
                num_samples=1,
                rng=np.random.default_rng(args.seed + image_id),
            )

            residual = int(candidate_info.residual_permutation_count)
            if residual != residual_from_cells:
                raise RuntimeError("PMAR residual count disagrees with structural_refine")
            num_cells = int(candidate_info.num_refinement_cells)
            exact = candidate_info.mode == "exact"
            unique_exact = int(candidate_info.num_candidates) if exact else None

            raw_values.append(raw_count)
            residual_values.append(residual)
            refine_cell_values.append(num_cells)
            max_cell_values.append(max_cell_size)
            relation_counts.append(len(graph))
            bucket_counts[bucket_residual(residual)] += 1

            if exact:
                exact_count += 1
                exact_unique_values.append(unique_exact)
            else:
                sampled_count += 1

            rows.append(
                {
                    "image_id": image_id,
                    "num_relations": len(graph),
                    "num_refinement_cells": num_cells,
                    "max_residual_cell_size": max_cell_size,
                    "M_raw": raw_count,
                    "M_residual": residual,
                    "log10_M_raw": safe_log10(raw_count),
                    "log10_M_residual": safe_log10(residual),
                    "reduction_factor": (float(raw_count) / float(residual)) if residual else float("inf"),
                    "mode": candidate_info.mode,
                    "exact_unique_candidates": unique_exact,
                }
            )

            if (local_index + 1) % 1000 == 0:
                elapsed = time.perf_counter() - start
                print(
                    f"  scanned {local_index + 1:>7}/{limit} graphs "
                    f"({elapsed:.1f}s)"
                )
    finally:
        dataset.close()

    elapsed = time.perf_counter() - start
    n = len(rows)

    print("\n" + "=" * 100)
    print("RESULTS")
    print("=" * 100)
    print(f"Elapsed:                 {elapsed:.2f} s")
    print(f"Mean relations:          {statistics.mean(relation_counts) if relation_counts else 0:.2f}")
    print(f"Median M_raw:            {statistics.median(raw_values) if raw_values else 0}")
    print(f"Median M_residual:       {statistics.median(residual_values) if residual_values else 0}")
    print(f"Max M_raw:               {max(raw_values) if raw_values else 0}")
    print(f"Max M_residual:          {max(residual_values) if residual_values else 0}")
    print(f"Mean refinement cells:   {statistics.mean(refine_cell_values) if refine_cell_values else 0:.2f}")
    print(f"Mean max residual cell:  {statistics.mean(max_cell_values) if max_cell_values else 0:.2f}")
    if raw_values and residual_values:
        log_reductions = [safe_log10(r) - safe_log10(m) for r, m in zip(raw_values, residual_values)]
        print(f"Mean log10 reduction:    {statistics.mean(log_reductions):.2f} orders")
    print(f"Exact at threshold:      {exact_count}/{n} ({100.0 * exact_count / max(n, 1):.2f}%)")
    print(f"Sampled at threshold:    {sampled_count}/{n} ({100.0 * sampled_count / max(n, 1):.2f}%)")

    print("\nResidual-permutation distribution:")
    ordered_buckets = ["1", "<=2", "<=4", "<=8", "<=16", "<=32", "<=64", ">64"]
    for bucket in ordered_buckets:
        count = bucket_counts[bucket]
        print(f"  {bucket:>4}: {count:>7}  ({100.0 * count / max(n, 1):6.2f}%)")

    cumulative_thresholds = [1, 2, 4, 8, 16, 32, 64]
    print("\nCumulative fraction with M_residual <= threshold:")
    for threshold in cumulative_thresholds:
        count = sum(1 for value in residual_values if value <= threshold)
        print(f"  <= {threshold:>2}: {count:>7}  ({100.0 * count / max(n, 1):6.2f}%)")

    if exact_unique_values:
        print("\nExact-mode unique serialized candidate counts:")
        print(f"  mean:   {statistics.mean(exact_unique_values):.2f}")
        print(f"  median: {statistics.median(exact_unique_values):.2f}")
        print(f"  max:    {max(exact_unique_values)}")

    if args.csv:
        directory = os.path.dirname(args.csv)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(args.csv, "w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "image_id",
                    "num_relations",
                    "num_refinement_cells",
                    "max_residual_cell_size",
                    "M_raw",
                    "M_residual",
                    "log10_M_raw",
                    "log10_M_residual",
                    "reduction_factor",
                    "mode",
                    "exact_unique_candidates",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved per-graph statistics to: {args.csv}")

    print("\nUse this distribution together with the decoder time/memory benchmark to estimate practical PMAR cost.")


if __name__ == "__main__":
    main()