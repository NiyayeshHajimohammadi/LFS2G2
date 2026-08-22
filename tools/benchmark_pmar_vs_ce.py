"""Controlled CE vs PMAR computational benchmark.

This benchmark measures the EXTRA training cost introduced by PMAR.

It intentionally:

- uses the project's real decoder;
- uses real ConditioningSet objects from the toy text encoder;
- uses random autoregressive token tensors;
- keeps batch size, sequence length, vocabulary, and decoder identical;
- varies only the number of PMAR candidate serializations K;
- measures forward + loss + backward;
- optionally reports CUDA peak-memory increase.

It does NOT measure graph structural-refinement/candidate-generation cost.
That should be measured separately on the real VG dataset.

Run:

    python tools/benchmark_pmar_vs_ce.py

Example GPU run:

    python tools/benchmark_pmar_vs_ce.py \
        --device cuda \
        --batch-size 4 \
        --repeats 20 \
        --warmup 5 \
        --candidate-batch-size 4 \
        --ks 1 2 4 8 16 32 64
"""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from patchsgg.config import load_config
from patchsgg.data.collate import GraphCollator
from patchsgg.data.factory import build_dataset
from patchsgg.losses.pmar import PMARLoss
from patchsgg.model import PatchSGGModel, build_vocab


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def format_ms(seconds: float) -> str:
    return f"{seconds * 1000.0:.2f}"


def benchmark_function(
    fn,
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
):
    """Benchmark one forward+backward function."""

    for _ in range(warmup):
        fn()

    synchronize(device)

    timings = []

    for _ in range(repeats):

        synchronize(device)

        start = time.perf_counter()

        fn()

        synchronize(device)

        timings.append(
            time.perf_counter() - start
        )

    return {
        "median": statistics.median(timings),
        "mean": statistics.mean(timings),
        "min": min(timings),
        "max": max(timings),
    }


def benchmark_peak_cuda_memory(
    fn,
    *,
    model,
    device: torch.device,
):
    """Measure incremental CUDA peak allocated memory."""

    if device.type != "cuda":
        return None

    model.zero_grad(
        set_to_none=True
    )

    torch.cuda.empty_cache()

    synchronize(device)

    baseline = torch.cuda.memory_allocated(
        device
    )

    torch.cuda.reset_peak_memory_stats(
        device
    )

    fn()

    synchronize(device)

    peak = torch.cuda.max_memory_allocated(
        device
    )

    incremental = max(
        0,
        peak - baseline,
    )

    return incremental / (1024**2)


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


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
        default=2,
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--candidate-batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=[
            1,
            2,
            4,
            8,
            16,
        ],
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    device = torch.device(
        args.device
    )

    torch.manual_seed(
        args.seed
    )

    np.random.seed(
        args.seed
    )

    if device.type == "cuda":
        torch.cuda.manual_seed_all(
            args.seed
        )

    print()
    print("=" * 80)
    print("CE vs PMAR COMPUTATIONAL BENCHMARK")
    print("=" * 80)

    print()
    print(f"Device:                 {device}")
    print(f"Batch size:             {args.batch_size}")
    print(f"Warmup steps:           {args.warmup}")
    print(f"Measured steps:         {args.repeats}")
    print(
        "PMAR candidate batch:   "
        f"{args.candidate_batch_size}"
    )
    print(f"K values:               {args.ks}")

    # ------------------------------------------------------------------
    # Small diagnostic model.
    #
    # We explicitly configure CE here because the decoder itself is shared
    # by both benchmark paths.
    # ------------------------------------------------------------------

    cfg = load_config(
        "patchsgg/configs/diagnostic_text2text.yaml",
        [
            f"device={args.device}",
            "loss.type=ce",
            "loss.label_smoothing=0.0",
            f"train.batch_size={args.batch_size}",
            f"data.toy_n_train={max(args.batch_size * 2, 8)}",
            "data.toy_max_rels=4",
            "vocab.max_num_rels=4",
            "eval.max_rels=4",
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
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=GraphCollator(
            vocab,
            seed=args.seed,
        ),
    )

    batch = next(
        iter(
            dataloader
        )
    )

    model = PatchSGGModel(
        cfg
    ).to(device)

    model.train()

    # ------------------------------------------------------------------
    # Obtain one real conditioning batch.
    #
    # The encoder is run ONCE and excluded from timing.
    #
    # This is intentional:
    #
    # CE and PMAR both encode each image/text once.
    #
    # The differential cost we care about comes from evaluating multiple
    # autoregressive candidate sequences through the decoder.
    # ------------------------------------------------------------------

    with torch.no_grad():

        conditioning = model.encode(
            batch,
            modality="text",
            training=True,
        ).to(device)

    batch_size = (
        conditioning.batch_size
    )

    # Use the same sequence length as the actual toy training batch.
    sequence_length = int(
        batch["input_tokens"].shape[1]
    )

    vocab_size = int(
        vocab.vocab_size
    )

    max_k = max(
        args.ks
    )

    print()
    print("Synthetic tensor dimensions:")
    print(f"    B = {batch_size}")
    print(f"    T = {sequence_length}")
    print(f"    V = {vocab_size}")
    print(f"    max K = {max_k}")

    # ------------------------------------------------------------------
    # Fixed random tensors.
    #
    # Keeping them fixed avoids measuring random-number generation.
    # ------------------------------------------------------------------

    ce_inputs = torch.randint(
        0,
        vocab_size,
        (
            batch_size,
            sequence_length,
        ),
        device=device,
        dtype=torch.long,
    )

    ce_targets = torch.randint(
        0,
        vocab_size,
        (
            batch_size,
            sequence_length,
        ),
        device=device,
        dtype=torch.long,
    )

    pmar_inputs = torch.randint(
        0,
        vocab_size,
        (
            batch_size,
            max_k,
            sequence_length,
        ),
        device=device,
        dtype=torch.long,
    )

    pmar_targets = torch.randint(
        0,
        vocab_size,
        (
            batch_size,
            max_k,
            sequence_length,
        ),
        device=device,
        dtype=torch.long,
    )

    pmar_loss = PMARLoss(
        vocab,
        exact_threshold=max_k,
        num_samples=max_k,
        candidate_batch_size=(
            args.candidate_batch_size
        ),
        seed=args.seed,
    )

    # ------------------------------------------------------------------
    # CE training step
    # ------------------------------------------------------------------

    def ce_step():

        model.zero_grad(
            set_to_none=True
        )

        logits = model.decoder(
            conditioning,
            ce_inputs,
        )

        # Standard CE baseline.
        #
        # Mean reduction matches normal token-level CE behavior.
        loss = F.cross_entropy(
            logits.reshape(
                -1,
                logits.shape[-1],
            ),
            ce_targets.reshape(-1),
            reduction="mean",
        )

        loss.backward()

        return loss

    print()
    print("-" * 80)
    print("Benchmarking CE baseline...")
    print("-" * 80)

    ce_timing = benchmark_function(
        ce_step,
        device=device,
        warmup=args.warmup,
        repeats=args.repeats,
    )

    ce_memory = benchmark_peak_cuda_memory(
        ce_step,
        model=model,
        device=device,
    )

    print()
    print(
        "CE median forward+backward:"
        f" {format_ms(ce_timing['median'])} ms"
    )

    if ce_memory is not None:
        print(
            "CE incremental peak memory:"
            f" {ce_memory:.2f} MB"
        )

    # ------------------------------------------------------------------
    # PMAR benchmark
    # ------------------------------------------------------------------

    results = []

    for k in args.ks:

        def pmar_step(
            current_k=k,
        ):
            model.zero_grad(
                set_to_none=True
            )

            # ---------------------------------------------------------------
            # Flatten candidates from ALL examples into one global pool.
            #
            # Original:
            #   [B, K, T]
            #
            # Flattened:
            #   [B*K, T]
            # ---------------------------------------------------------------

            flat_inputs = (
                pmar_inputs[
                    :,
                    :current_k,
                    :,
                ]
                .reshape(
                    batch_size * current_k,
                    sequence_length,
                )
            )

            flat_targets = (
                pmar_targets[
                    :,
                    :current_k,
                    :,
                ]
                .reshape(
                    batch_size * current_k,
                    sequence_length,
                )
            )

            # owner[i] tells us which original training example
            # candidate i belongs to.
            owners = (
                torch.arange(
                    batch_size,
                    device=device,
                    dtype=torch.long,
                )
                .repeat_interleave(
                    current_k
                )
            )

            # Keep candidate NLLs attached to autograd.
            candidate_nlls = []

            # ---------------------------------------------------------------
            # Process candidates globally rather than one example at a time.
            # ---------------------------------------------------------------

            for start in range(
                0,
                flat_inputs.shape[0],
                args.candidate_batch_size,
            ):
                stop = min(
                    start
                    + args.candidate_batch_size,
                    flat_inputs.shape[0],
                )

                chunk_owners = owners[
                    start:stop
                ]

                candidate_conditioning = (
                    model._select_conditioning(
                        conditioning,
                        chunk_owners,
                    )
                )

                logits = model.decoder(
                    candidate_conditioning,
                    flat_inputs[start:stop],
                )

                nll = pmar_loss.candidate_nll(
                    logits,
                    flat_targets[start:stop],
                )

                candidate_nlls.append(
                    nll
                )

            all_candidate_nll = torch.cat(
                candidate_nlls,
                dim=0,
            )

            # ---------------------------------------------------------------
            # Regroup candidates by original training example.
            #
            # Because our synthetic benchmark has exactly current_k candidates
            # per example, reshaping is enough.
            # ---------------------------------------------------------------

            per_example_nll = (
                all_candidate_nll.reshape(
                    batch_size,
                    current_k,
                )
            )

            # ---------------------------------------------------------------
            # PMAR separately for every graph:
            #
            #   L_i = -logsumexp_k(-NLL_ik)
            # ---------------------------------------------------------------

            graph_losses = (
                -torch.logsumexp(
                    -per_example_nll,
                    dim=1,
                )
            )

            loss = graph_losses.mean()

            loss.backward()

            return loss

        print()
        print(
            f"Benchmarking PMAR with K={k}..."
        )

        timing = benchmark_function(
            pmar_step,
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
        )

        memory = benchmark_peak_cuda_memory(
            pmar_step,
            model=model,
            device=device,
        )

        slowdown = (
            timing["median"]
            /
            ce_timing["median"]
        )

        memory_ratio = None

        if (
            memory is not None
            and ce_memory is not None
            and ce_memory > 0
        ):
            memory_ratio = (
                memory / ce_memory
            )

        results.append(
            {
                "k": k,
                "median": timing["median"],
                "slowdown": slowdown,
                "memory": memory,
                "memory_ratio": memory_ratio,
            }
        )

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)

    if device.type == "cuda":

        print()
        print(
            f"{'Method':<12}"
            f"{'K':>6}"
            f"{'Time ms':>14}"
            f"{'Time x':>12}"
            f"{'Peak MB':>14}"
            f"{'Mem x':>12}"
        )

        print("-" * 70)

        print(
            f"{'CE':<12}"
            f"{1:>6}"
            f"{ce_timing['median'] * 1000:>14.2f}"
            f"{1.0:>12.2f}"
            f"{ce_memory:>14.2f}"
            f"{1.0:>12.2f}"
        )

        for result in results:

            memory_text = (
                f"{result['memory']:.2f}"
                if result["memory"] is not None
                else "N/A"
            )

            ratio_text = (
                f"{result['memory_ratio']:.2f}"
                if result["memory_ratio"] is not None
                else "N/A"
            )

            print(
                f"{'PMAR':<12}"
                f"{result['k']:>6}"
                f"{result['median'] * 1000:>14.2f}"
                f"{result['slowdown']:>12.2f}"
                f"{memory_text:>14}"
                f"{ratio_text:>12}"
            )

    else:

        print()
        print(
            f"{'Method':<12}"
            f"{'K':>6}"
            f"{'Time ms':>14}"
            f"{'Time x':>12}"
        )

        print("-" * 46)

        print(
            f"{'CE':<12}"
            f"{1:>6}"
            f"{ce_timing['median'] * 1000:>14.2f}"
            f"{1.0:>12.2f}"
        )

        for result in results:

            print(
                f"{'PMAR':<12}"
                f"{result['k']:>6}"
                f"{result['median'] * 1000:>14.2f}"
                f"{result['slowdown']:>12.2f}"
            )

    
if __name__ == "__main__":
    main()