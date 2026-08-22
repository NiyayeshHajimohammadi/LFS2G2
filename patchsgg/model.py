"""PatchSGG core model: frozen encoders, bridge, graph decoder, and loss."""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from patchsgg.bridge import build_bridge
from patchsgg.decoder import GenConfig, build_decoder
from patchsgg.encoders import build_encoders
from patchsgg.encoders.base import ConditioningSet
from patchsgg.graph_seq.vocab import TOKENS_PER_REL, GraphVocab
from patchsgg.losses import build_loss
from patchsgg.postprocess import sequences_to_predictions
from patchsgg.graph_seq.linearize import (build_train_pair,)

def build_vocab(cfg) -> GraphVocab:
    vocab_cfg = cfg.get("vocab", {}) if hasattr(cfg, "get") else {}
    return GraphVocab(
        n_preds=int(vocab_cfg.get("n_preds", 51)),
        n_entities=int(vocab_cfg.get("n_entities", 151)),
        max_instance_id=int(vocab_cfg.get("max_instance_id", 30)),
        random_max_instance_id=int(
            vocab_cfg.get("random_max_instance_id", 10)
        ),
        max_num_rels=int(vocab_cfg.get("max_num_rels", 55)),
    )


class PatchSGGModel(nn.Module):
    """Complete encoder-to-location-free-scene-graph model."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.vocab = build_vocab(cfg)

        train_modality = str(cfg.train.get("train_modality", "text"))
        eval_modality = str(cfg.eval.get("eval_modality", "image"))
        requested_modalities = {train_modality, eval_modality}
        unknown = requested_modalities - {"text", "image"}
        if unknown:
            raise ValueError(f"unsupported conditioning modalities: {sorted(unknown)}")

        build_only_required = bool(
            cfg.encoders.get("build_only_required", False)
        )
        self.text_encoder, self.image_encoder = build_encoders(
            cfg,
            need_text=("text" in requested_modalities) if build_only_required else True,
            need_image=("image" in requested_modalities) if build_only_required else True,
        )

        dimensions = []
        if self.text_encoder is not None:
            dimensions.append(("text", int(self.text_encoder.embed_dim)))
        if self.image_encoder is not None:
            dimensions.append(("image", int(self.image_encoder.embed_dim)))

        cond_dim = dimensions[0][1]
        if any(dim != cond_dim for _, dim in dimensions[1:]):
            details = ", ".join(f"{name}={dim}" for name, dim in dimensions)
            raise ValueError(
                f"encoder dimensions differ ({details}); use a shared feature space "
                "or add an explicit modality projection"
            )

        self.bridge = build_bridge(cfg)
        self.decoder = build_decoder(cfg, self.vocab, cond_dim)
        self.loss_fn = build_loss(cfg, self.vocab)
        self.set_generation_config(cfg.eval)

    def set_generation_config(self, eval_cfg) -> None:
        """Update generation settings, including post-checkpoint CLI overrides."""
        max_rels = int(eval_cfg.get("max_rels", 100))
        if max_rels < 0:
            raise ValueError(f"eval.max_rels must be non-negative, got {max_rels}")

        required_positions = 1 + max_rels * TOKENS_PER_REL
        if required_positions > int(self.decoder.max_seq_len):
            raise ValueError(
                f"Generating {max_rels} relations requires {required_positions} "
                f"positions, but the decoder supports {self.decoder.max_seq_len}"
            )

        self.gen_cfg = GenConfig(
            max_rels=max_rels,
            temperature=float(eval_cfg.get("temperature", 1.75)),
            top_p=float(eval_cfg.get("top_p", 0.95)),
            top_k=int(eval_cfg.get("top_k", 0)),
            entity_sampling=str(
                eval_cfg.get("entity_sampling", "stochastic")
            ),
            allow_end=bool(eval_cfg.get("allow_end", True)),
        )

    def encode(
        self,
        batch: Dict,
        modality: str,
        training: bool,
    ) -> ConditioningSet:
        if modality == "text":
            if self.text_encoder is None:
                raise RuntimeError("the text encoder was not built for this configuration")
            if "texts" not in batch:
                raise KeyError("text conditioning requires batch['texts']")
            conditioning = self.text_encoder.encode(batch["texts"])
        elif modality == "image":
            if self.image_encoder is None:
                raise RuntimeError("the image encoder was not built for this configuration")
            if "images" not in batch:
                raise KeyError("image conditioning requires batch['images']")
            conditioning = self.image_encoder.encode(batch["images"])
        else:
            raise ValueError(
                f"modality must be 'text' or 'image', got {modality!r}"
            )

        return self.bridge(
            conditioning,
            training=training,
            modality=modality,
        )

    # def compute_loss(
    #     self,
    #     batch: Dict,
    #     modality: str = "text",
    # ) -> torch.Tensor:
    #     conditioning = self.encode(batch, modality=modality, training=True)
    #     device = conditioning.tokens.device
    #     input_tokens = batch["input_tokens"].to(device=device, dtype=torch.long)
    #     target_tokens = batch["target_tokens"].to(device=device, dtype=torch.long)

    #     logits = self.decoder(conditioning, input_tokens)
    #     return self.loss_fn(logits, target_tokens, input_tokens)
    def compute_loss(
        self,
        batch: Dict,
        modality: str = "text",
    ) -> torch.Tensor:
        conditioning = self.encode(batch,modality=modality,training=True,)

        if getattr(self.loss_fn,"requires_candidate_sequences",False,):
            return self._compute_candidate_sequence_loss(batch,conditioning,)
        # EXACT EXISTING PATH
        device = conditioning.tokens.device
        input_tokens = batch["input_tokens"].to(device=device,dtype=torch.long,)
        target_tokens = batch["target_tokens"].to(device=device,dtype=torch.long,)

        logits = self.decoder(conditioning,input_tokens,)

        return self.loss_fn(logits,target_tokens,input_tokens,)

    @torch.no_grad()
    def predict(
        self,
        batch: Dict,
        modality: str,
    ) -> List[List[tuple]]:
        conditioning = self.encode(batch, modality=modality, training=False)
        sequences, scores = self.decoder.generate(conditioning, self.gen_cfg)
        return sequences_to_predictions(sequences, scores, self.vocab)

    def trainable_parameters(self):
        """Return every parameter that this configuration has marked trainable."""
        modules = [
            self.decoder,
            self.bridge,
            self.text_encoder,
            self.image_encoder,
        ]

        parameters = []

        for module in modules:
            if module is None:
                continue

            parameters.extend(
                parameter
                for parameter in module.parameters()
                if parameter.requires_grad
            )

        return parameters
    def _repeat_conditioning(
        self,
        conditioning: ConditioningSet,
        index: int,
        count: int,
    ) -> ConditioningSet:

        tokens = conditioning.tokens[ index : index + 1].expand( count, -1,-1,)
        pooled = conditioning.pooled[ index : index + 1].expand( count,-1,)
        mask = None
        if conditioning.mask is not None:
            mask = conditioning.mask[index : index + 1 ].expand(count,-1,)

        return ConditioningSet(
            tokens=tokens,
            pooled=pooled,
            mask=mask,
        )
    def _compute_candidate_sequence_loss(
        self,
        batch: Dict,
        conditioning: ConditioningSet,
    ) -> torch.Tensor:
        """Compute PMAR while batching candidates across training examples."""

        if "train_graphs" not in batch:
            raise KeyError(
                "candidate-sequence losses require "
                "batch['train_graphs']; use GraphCollator"
            )

        if (
            len(batch["train_graphs"])
            != conditioning.batch_size
        ):
            raise ValueError(
                "train_graphs batch size does not match "
                "conditioning batch size"
            )

        device = conditioning.tokens.device
        batch_size = conditioning.batch_size

        # ---------------------------------------------------------------
        # Candidate records are pooled across the ENTIRE training batch.
        #
        # Each record contains:
        #
        #   owner example index
        #   input sequence
        #   target sequence
        #
        # Candidates with equal sequence length can be stacked together.
        # ---------------------------------------------------------------

        buckets = {}

        for example_index, graph in enumerate(
            batch["train_graphs"]
        ):
            candidates = (
                self.loss_fn.build_candidates(
                    graph
                )
            )

            if not candidates.graphs:
                raise RuntimeError(
                    "PMAR returned no candidates"
                )

            for candidate_graph in candidates.graphs:

                input_array, target_array = (
                    build_train_pair(
                        list(candidate_graph),
                        self.vocab,
                        pad_to_max=False,
                    )
                )

                input_tensor = torch.from_numpy(
                    input_array
                ).long()

                target_tensor = torch.from_numpy(
                    target_array
                ).long()

                sequence_length = int(
                    input_tensor.shape[0]
                )

                buckets.setdefault(
                    sequence_length,
                    [],
                ).append(
                    (
                        example_index,
                        input_tensor,
                        target_tensor,
                    )
                )

        # ---------------------------------------------------------------
        # Store candidate NLLs by the graph/example they belong to.
        #
        # Important:
        # these tensors remain attached to autograd.
        # ---------------------------------------------------------------

        per_example_nlls = [
            []
            for _ in range(
                batch_size
            )
        ]

        candidate_batch_size = int(
            self.loss_fn.candidate_batch_size
        )

        # ---------------------------------------------------------------
        # Candidates from DIFFERENT examples are now allowed to share the
        # same decoder call.
        #
        # We bucket by sequence length because torch.stack requires equal
        # sequence lengths.
        # ---------------------------------------------------------------

        for sequence_length, records in buckets.items():

            for start in range(
                0,
                len(records),
                candidate_batch_size,
            ):
                stop = min(
                    start + candidate_batch_size,
                    len(records),
                )

                chunk = records[
                    start:stop
                ]

                owner_indices = [
                    record[0]
                    for record in chunk
                ]

                input_tokens = torch.stack(
                    [
                        record[1]
                        for record in chunk
                    ],
                    dim=0,
                ).to(
                    device=device,
                    dtype=torch.long,
                )

                target_tokens = torch.stack(
                    [
                        record[2]
                        for record in chunk
                    ],
                    dim=0,
                ).to(
                    device=device,
                    dtype=torch.long,
                )

                candidate_conditioning = (
                    self._select_conditioning(
                        conditioning,
                        owner_indices,
                    )
                )

                logits = self.decoder(
                    candidate_conditioning,
                    input_tokens,
                )

                candidate_nll = (
                    self.loss_fn.candidate_nll(
                        logits,
                        target_tokens,
                    )
                )

                # Put each NLL back into its corresponding graph.
                for local_index, owner in enumerate(
                    owner_indices
                ):
                    per_example_nlls[
                        owner
                    ].append(
                        candidate_nll[
                            local_index
                        ]
                    )

        # ---------------------------------------------------------------
        # PMAR marginalization still happens separately for each graph.
        #
        # L_i = -logsumexp_k(-NLL_ik)
        # ---------------------------------------------------------------

        graph_losses = []

        for example_index, nll_list in enumerate(
            per_example_nlls
        ):

            if not nll_list:
                raise RuntimeError(
                    f"PMAR example {example_index} "
                    "has no candidate NLLs"
                )

            candidate_nll = torch.stack(
                nll_list,
                dim=0,
            )

            graph_loss = (
                self.loss_fn.marginalize(
                    candidate_nll
                )
            )

            graph_losses.append(
                graph_loss
            )

        # ---------------------------------------------------------------
        # L_batch = mean_i L_i
        # ---------------------------------------------------------------

        return torch.stack(
            graph_losses,
            dim=0,
        ).mean()
    def _select_conditioning(
        self,
        conditioning: ConditioningSet,
        example_indices,
    ) -> ConditioningSet:
        """Select/repeat conditioning rows for a global PMAR candidate batch."""
        index = torch.as_tensor(example_indices,device=conditioning.tokens.device,dtype=torch.long,)

        return ConditioningSet(tokens=conditioning.tokens.index_select(0,index,),
            pooled=conditioning.pooled.index_select(0,index,),
            mask=(None if conditioning.mask is None else conditioning.mask.index_select(0, index,)), )
