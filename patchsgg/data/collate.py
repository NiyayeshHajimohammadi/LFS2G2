"""Batch collation for graph-token training and location-free evaluation."""
from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import get_worker_info

from patchsgg.graph_seq.linearize import (
    Relation,
    build_train_pair,
    graph_to_matcher_tuples,
    permute_and_reindex_graph,
)
from patchsgg.graph_seq.vocab import GraphVocab


class GraphCollator:
    """Create the batch contract consumed by :class:`PatchSGGModel`."""

    def __init__(self, vocab: GraphVocab, seed: int = 42, deterministic: bool = False):
        self.vocab = vocab
        self.seed = int(seed)
        self.deterministic = bool(deterministic)
        self._rngs: dict[int, np.random.Generator] = {}

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_rngs"] = {}
        return state

    def _worker_rng(self) -> np.random.Generator:
        info = get_worker_info()
        worker_key = -1 if info is None else int(info.id)
        if worker_key not in self._rngs:
            worker_seed = (
                self.seed
                if info is None
                else (self.seed + int(info.seed)) % (2**63 - 1)
            )
            self._rngs[worker_key] = np.random.default_rng(worker_seed)
        return self._rngs[worker_key]

    @staticmethod
    def _normalize_graph(value: Sequence) -> list[Relation]:
        graph: list[Relation] = []
        for item in value:
            relation = item if isinstance(item, Relation) else Relation(*map(int, item))
            graph.append(Relation(*map(int, relation)))
        return graph

    def __call__(self, samples: Sequence[Mapping]) -> Dict:
        if not samples:
            raise ValueError("cannot collate an empty batch")

        texts: list[str] = []
        images: list[torch.Tensor] = []
        image_ids: list[int] = []
        input_tokens: list[torch.Tensor] = []
        target_tokens: list[torch.Tensor] = []
        gt_graphs: list[list[tuple]] = []
        train_graphs: list[list[Relation]] = []

        shared_rng = None if self.deterministic else self._worker_rng()

        for sample in samples:
            missing = {"image", "text", "graph", "image_id"} - set(sample)
            if missing:
                raise KeyError(
                    f"dataset sample is missing required keys: {sorted(missing)}"
                )

            image = sample["image"]
            if not isinstance(image, torch.Tensor) or image.ndim != 3:
                raise TypeError(
                    "sample image must be a CHW tensor, "
                    f"got {type(image)!r} shape={getattr(image, 'shape', None)}"
                )

            graph = self._normalize_graph(sample["graph"])
            image_id = int(sample["image_id"])

            if self.deterministic:
                sample_seed = (self.seed * 1_000_003 + image_id * 97) % (2**63 - 1)
                rng = np.random.default_rng(sample_seed)
            else:
                rng = shared_rng

            # LF-SGG target representation: random relation order during training,
            # followed by per-class instance IDs assigned by first appearance.
            training_graph = permute_and_reindex_graph(
                graph,
                vocab=self.vocab,
                rng=rng,
                shuffle=not self.deterministic,
            )
            input_array, target_array = build_train_pair(
                training_graph,
                self.vocab,
                rng=rng,
            )

            texts.append(str(sample["text"]))
            images.append(image.float())
            image_ids.append(image_id)
            input_tokens.append(torch.from_numpy(input_array))
            target_tokens.append(torch.from_numpy(target_array))

            # Evaluation retains the dataset graph; the matcher handles arbitrary
            # predicted instance numbering.
            gt_graphs.append(graph_to_matcher_tuples(graph, self.vocab))
            train_graphs.append(list( training_graph[ : self.vocab.max_num_rels]))
        shapes = {tuple(image.shape) for image in images}
        if len(shapes) != 1:
            raise ValueError(
                f"all preprocessed images must have the same CHW shape, got {sorted(shapes)}"
            )

        return {
            "texts": texts,
            "images": torch.stack(images, dim=0),
            "input_tokens": torch.stack(input_tokens, dim=0).long(),
            "target_tokens": torch.stack(target_tokens, dim=0).long(),
            "gt_graphs": gt_graphs,
            "train_graphs": train_graphs,
            "image_ids": image_ids,
        }
