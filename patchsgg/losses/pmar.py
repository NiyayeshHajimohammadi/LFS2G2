"""
Permutation-Marginalized Autoregressive Loss (PMAR)

Computes:

    -log sum_k p(sequence_k | image)

over graph-equivalent candidate serializations.

The loss expects multiple teacher-forced
candidate sequences per image.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F



class PMARLoss(nn.Module):
    """
    PMAR sequence likelihood loss.

    Inputs:

        logits:
            [N_candidates, T, V]

        targets:
            [N_candidates, T]

        mask:
            [N_candidates, T]

            1:
                token contributes

            0:
                ignored


        candidate_owner:
            [N_candidates]

            tells which original image
            each candidate belongs to.


    Example:

        owner =
            [0,0,0,1,2,2]

        means:

            image 0:
                candidates 0,1,2

            image 1:
                candidate 3

            image 2:
                candidates 4,5
    """

    def __init__(
        self,
        reduction: str = "mean",
    ):
        super().__init__()

        assert reduction in (
            "mean",
            "sum",
        )

        self.reduction = reduction



    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        candidate_owner: torch.Tensor,
        batch_size: Optional[int] = None,
    ):
        """
        Compute PMAR loss.
        """

        if logits.ndim != 3:
            raise ValueError(
                f"logits must be [N,T,V], got {logits.shape}"
            )

        if targets.ndim != 2:
            raise ValueError(
                f"targets must be [N,T], got {targets.shape}"
            )


        N, T, V = logits.shape


        if targets.shape != (N,T):
            raise ValueError(
                "targets shape mismatch"
            )


        if mask.shape != (N,T):
            raise ValueError(
                "mask shape mismatch"
            )


        if candidate_owner.shape[0] != N:
            raise ValueError(
                "candidate_owner mismatch"
            )



        # -------------------------------------------------
        # Token log probability
        # -------------------------------------------------

        log_probs = F.log_softmax(
            logits,
            dim=-1,
        )


        # Select probability of the true token

        token_log_probs = log_probs.gather(
            dim=-1,
            index=targets.unsqueeze(-1),
        ).squeeze(-1)


        # -------------------------------------------------
        # Remove padding / tokens after EOS
        # -------------------------------------------------

        token_log_probs = (
            token_log_probs
            *
            mask.float()
        )


        # -------------------------------------------------
        # Sequence likelihood
        #
        # log p(sequence)
        #
        # = sum_t log p(token_t)
        #
        # -------------------------------------------------

        sequence_log_probs = (
            token_log_probs
            .sum(dim=-1)
        )


        # Shape:

        # [N_candidates]


        # -------------------------------------------------
        # Marginalize candidates belonging
        # to the same graph
        # -------------------------------------------------

        losses = []


        unique_graphs = torch.unique(
            candidate_owner,
            sorted=True,
        )


        for graph_id in unique_graphs:

            candidate_scores = (
                sequence_log_probs[
                    candidate_owner == graph_id
                ]
            )


            # candidate_scores are:
            #
            # log p(s1)
            # log p(s2)
            # ...
            #
            # therefore:
            #
            # log(sum(exp(scores)))

            graph_log_probability = (
                torch.logsumexp(
                    candidate_scores,
                    dim=0,
                )
            )


            losses.append(
                -graph_log_probability
            )


        losses = torch.stack(
            losses
        )


        # -------------------------------------------------
        # Batch reduction
        # -------------------------------------------------

        if self.reduction == "mean":

            return losses.mean()

        else:

            return losses.sum()