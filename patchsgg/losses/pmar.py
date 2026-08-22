"""Permutation-Marginalized Autoregressive (PMAR) loss.

PMAR differs from the existing token-level losses in this project.

The existing CE-style losses receive:

    logits[B, T, V]
    target[B, T]

after one teacher-forced decoder forward pass.

PMAR cannot work that way because every graph-equivalent serialization has
a different autoregressive prefix. Therefore each PMAR candidate must be
teacher-forced through the decoder separately.

This module is responsible only for:

1. constructing PMAR graph candidates through graph_seq.pmar;
2. computing the summed autoregressive NLL of each candidate;
3. marginalizing candidate sequence probabilities using log-sum-exp.

The actual repeated decoder forwards are performed by PatchSGGModel.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from patchsgg.graph_seq.pmar import (
    PMARCandidates,
    build_pmar_candidates,
)
from patchsgg.graph_seq.vocab import GraphVocab
from patchsgg.losses.base import Loss


class PMARLoss(Loss):
    """Permutation-Marginalized Autoregressive loss.

    For a graph G with valid candidate serializations:

        s_1, s_2, ..., s_K

    each candidate has autoregressive negative log-likelihood:

        L_k =
            - sum_t
              log p(
                  s_{k,t}
                  |
                  s_{k,<t},
                  conditioning
              )

    PMAR then computes:

        L_G =
            -log sum_k exp(-L_k)

    which gives probability mass to the complete equivalence class of
    valid graph serializations.

    Notes
    -----
    PMAR intentionally does NOT use:

        - class-frequency weighting;
        - label smoothing;
        - averaged token CE.

    Those would break the clean interpretation of L_k as a sequence
    negative log-likelihood.

    The model must also build PMAR teacher-forcing pairs with:

        pad_to_max=False

    so that each sequence probability consists only of:

        graph tokens + EOS

    rather than LF-SGG's random training padding.
    """

    # PatchSGGModel checks this flag to decide whether this loss needs
    # the special multiple-candidate decoder path.
    requires_candidate_sequences = True

    def __init__(
        self,
        vocab: GraphVocab,
        *,
        exact_threshold: int = 64,
        num_samples: int = 8,
        candidate_batch_size: int = 8,
        seed: int = 42,
    ):
        """Create PMAR loss.

        Parameters
        ----------
        vocab:
            Project graph vocabulary.

        exact_threshold:
            Maximum residual permutation count that will be enumerated
            exactly.

            Example:

                exact_threshold = 64

            gives:

                M_residual <= 64
                    -> exact PMAR

                M_residual > 64
                    -> sampled PMAR

        num_samples:
            Number of IID residual assignments used when exact
            enumeration would exceed exact_threshold.

        candidate_batch_size:
            Number of candidate serializations sent through the decoder
            simultaneously.

            This is only a memory/performance setting. It does not alter
            the mathematical PMAR objective.

        seed:
            Seed for sampled residual permutations.
        """

        # PMAR must use raw sequence likelihood.
        #
        # Therefore:
        #
        #   weight = None
        #   label_smoothing = 0
        #
        # Loss will internally register an all-ones weight vector.
        super().__init__(
            vocab,
            weight=None,
            label_smoothing=0.0,
        )

        if exact_threshold < 1:
            raise ValueError(
                "PMAR exact_threshold must be >= 1, "
                f"got {exact_threshold}"
            )

        if num_samples < 1:
            raise ValueError(
                "PMAR num_samples must be >= 1, "
                f"got {num_samples}"
            )

        if candidate_batch_size < 1:
            raise ValueError(
                "PMAR candidate_batch_size must be >= 1, "
                f"got {candidate_batch_size}"
            )

        self.exact_threshold = int(
            exact_threshold
        )

        self.num_samples = int(
            num_samples
        )

        self.candidate_batch_size = int(
            candidate_batch_size
        )

        # Persistent RNG used by sampled PMAR.
        #
        # Keeping the generator on the loss object means we do not restart
        # from the same permutation samples every call.
        self._rng = np.random.default_rng(
            int(seed)
        )

    # ------------------------------------------------------------------
    # RNG checkpoint support
    # ------------------------------------------------------------------

    def get_extra_state(self):
        """Return NumPy RNG state for PyTorch checkpoints.

        nn.Module includes extra state in state_dict() when these methods
        are implemented.

        This means sampled PMAR can resume from a checkpoint without
        resetting its permutation sampler.
        """

        return self._rng.bit_generator.state

    def set_extra_state(
        self,
        state,
    ):
        """Restore NumPy RNG state from a checkpoint."""

        self._rng = (
            np.random.default_rng()
        )

        self._rng.bit_generator.state = (
            state
        )

    # ------------------------------------------------------------------
    # Candidate construction
    # ------------------------------------------------------------------

    def build_candidates(
        self,
        graph,
    ) -> PMARCandidates:
        """Construct exact or sampled PMAR graph candidates.

        Structural refinement and graph-equivalence logic live in:

            patchsgg.graph_seq.pmar

        rather than inside this loss module.

        This separation is intentional:

            graph_seq.pmar
                -> representation / graph invariance

            losses.pmar
                -> probability / likelihood mathematics
        """

        return build_pmar_candidates(
            graph,
            self.vocab,
            exact_threshold=self.exact_threshold,
            num_samples=self.num_samples,
            rng=self._rng,
        )

    # ------------------------------------------------------------------
    # Candidate sequence likelihood
    # ------------------------------------------------------------------

    @staticmethod
    def candidate_nll(
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate one summed autoregressive NLL per candidate.

        Parameters
        ----------
        logits:
            Decoder output of shape:

                [K, T, V]

            where:

                K = number of PMAR candidates
                T = sequence length
                V = vocabulary size

        target:
            Target token IDs of shape:

                [K, T]

        Returns
        -------
        torch.Tensor
            Shape:

                [K]

            containing one summed sequence NLL for every candidate.

        Important
        ---------
        The loss is SUMMED over tokens, not averaged.

        For candidate k:

            NLL_k =
                -sum_t log p(y_t | y_<t)

        This is required before PMAR's log-sum-exp marginalization.
        """

        if logits.ndim != 3:
            raise ValueError(
                "PMAR candidate_nll expected logits "
                f"with shape [K,T,V], got {tuple(logits.shape)}"
            )

        if target.ndim != 2:
            raise ValueError(
                "PMAR candidate_nll expected target "
                f"with shape [K,T], got {tuple(target.shape)}"
            )

        if logits.shape[0] != target.shape[0]:
            raise ValueError(
                "PMAR logits/target candidate counts differ: "
                f"{logits.shape[0]} vs {target.shape[0]}"
            )

        if logits.shape[1] != target.shape[1]:
            raise ValueError(
                "PMAR logits/target sequence lengths differ: "
                f"{logits.shape[1]} vs {target.shape[1]}"
            )

        # --------------------------------------------------------------
        # Calculate log probabilities.
        #
        # Cast to float32 before log_softmax. This is useful under AMP,
        # where logits may otherwise be float16/bfloat16.
        #
        # log-sum-exp over full sequence NLLs is more numerically sensitive
        # than ordinary token-level CE.
        # --------------------------------------------------------------

        log_probs = F.log_softmax(
            logits.float(),
            dim=-1,
        )

        # --------------------------------------------------------------
        # Select the probability of the target token at each position.
        #
        # log_probs:
        #
        #   [K, T, V]
        #
        # target.unsqueeze(-1):
        #
        #   [K, T, 1]
        #
        # result:
        #
        #   [K, T]
        # --------------------------------------------------------------

        target_log_probs = (
            log_probs.gather(
                dim=-1,
                index=target.unsqueeze(-1),
            )
            .squeeze(-1)
        )

        # --------------------------------------------------------------
        # Sum token negative log probabilities to obtain a true sequence
        # NLL.
        #
        # DO NOT use .mean(dim=-1) here.
        # --------------------------------------------------------------

        sequence_nll = (
            -target_log_probs.sum(
                dim=-1
            )
        )

        return sequence_nll

    # ------------------------------------------------------------------
    # PMAR marginalization
    # ------------------------------------------------------------------

    @staticmethod
    def marginalize(
        candidate_nll: torch.Tensor,
    ) -> torch.Tensor:
        """Marginalize over graph-equivalent candidate sequences.

        Given:

            candidate_nll[k] = L_k

        computes:

            -log sum_k exp(-L_k)

        This is equivalent to:

            -log sum_k p(s_k | conditioning)

        when each L_k is the proper autoregressive sequence NLL.
        """

        if candidate_nll.ndim != 1:
            raise ValueError(
                "PMAR marginalize expected a 1-D candidate NLL "
                f"tensor, got shape {tuple(candidate_nll.shape)}"
            )

        if candidate_nll.numel() == 0:
            raise ValueError(
                "PMAR cannot marginalize over zero candidates"
            )

        return -torch.logsumexp(
            -candidate_nll,
            dim=0,
        )

    # ------------------------------------------------------------------
    # Normal Loss.forward is intentionally unavailable
    # ------------------------------------------------------------------

    def forward(
        self,
        logits,
        target,
        input_tokens=None,
    ):
        """Prevent accidental use through the ordinary one-forward path.

        Existing losses can be calculated as:

            logits = decoder(conditioning, input_tokens)

            loss = loss_fn(
                logits,
                target,
                input_tokens,
            )

        PMAR cannot.

        Every candidate serialization has a different autoregressive
        prefix and therefore requires its own teacher-forced decoder
        evaluation.

        PatchSGGModel.compute_loss() must detect:

            requires_candidate_sequences = True

        and use its dedicated PMAR candidate path instead.
        """

        raise RuntimeError(
            "PMARLoss cannot be called through the ordinary "
            "single-sequence Loss.forward() path. "
            "Each PMAR candidate requires its own teacher-forced "
            "decoder forward. Use PatchSGGModel.compute_loss(), "
            "which handles losses with "
            "requires_candidate_sequences=True."
        )