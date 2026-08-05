"""Batch collation for graph-conditioned training and location-free evaluation."""
from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import get_worker_info

from patchsgg.graph_seq.linearize import Relation, build_train_pair, graph_to_matcher_tuples
from patchsgg.graph_seq.vocab import GraphVocab


class GraphCollator:
    """Create the exact batch contract consumed by :class:`PatchSGGModel`."""

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
            worker_seed = self.seed if info is None else (self.seed + int(info.seed)) % (2**63 - 1)
            self._rngs[worker_key] = np.random.default_rng(worker_seed)
        return self._rngs[worker_key]

    @staticmethod
    def _normalize_graph(value: Sequence) -> list[Relation]:
        graph: list[Relation] = []
        for item in value:
            rel = item if isinstance(item, Relation) else Relation(*map(int, item))
            graph.append(Relation(*map(int, rel)))
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

        shared_rng = None if self.deterministic else self._worker_rng()
        for batch_index, sample in enumerate(samples):
            missing = {"image", "text", "graph", "image_id"} - set(sample)
            if missing:
                raise KeyError(f"dataset sample is missing required keys: {sorted(missing)}")
            image = sample["image"]
            if not isinstance(image, torch.Tensor) or image.ndim != 3:
                raise TypeError(f"sample image must be a CHW tensor, got {type(image)!r} shape={getattr(image, 'shape', None)}")
            graph = self._normalize_graph(sample["graph"])
            image_id = int(sample["image_id"])
            if self.deterministic:
                sample_seed = (self.seed * 1_000_003 + image_id * 97) % (2**63 - 1)
                rng = np.random.default_rng(sample_seed)
            else:
                rng = shared_rng
            inp, tgt = build_train_pair(graph, self.vocab, rng=rng)

            texts.append(str(sample["text"]))
            images.append(image.float())
            image_ids.append(image_id)
            input_tokens.append(torch.from_numpy(inp))
            target_tokens.append(torch.from_numpy(tgt))
            gt_graphs.append(graph_to_matcher_tuples(graph, self.vocab))

        shapes = {tuple(image.shape) for image in images}
        if len(shapes) != 1:
            raise ValueError(f"all preprocessed images must have the same CHW shape, got {sorted(shapes)}")
        return {
            "texts": texts,
            "images": torch.stack(images, dim=0),
            "input_tokens": torch.stack(input_tokens, dim=0).long(),
            "target_tokens": torch.stack(target_tokens, dim=0).long(),
            "gt_graphs": gt_graphs,
            "image_ids": image_ids,
        }
