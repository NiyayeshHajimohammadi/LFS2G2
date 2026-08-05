"""Deterministic synthetic dataset used by smoke and overfit tests."""
from __future__ import annotations

from typing import Dict

import torch
from torch.utils.data import Dataset

from patchsgg.data.graph_text_views import serialize_graph
from patchsgg.graph_seq.linearize import Graph, Relation
from patchsgg.graph_seq.vocab import GraphVocab


class ToyGraphDataset(Dataset):
    def __init__(self, cfg, split: str, vocab: GraphVocab):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"unknown split {split!r}")
        self.cfg = cfg
        self.split = split
        self.vocab = vocab
        key = "toy_n_train" if split == "train" else "toy_n_val"
        self.n = int(cfg.data.get(key, 64 if split == "train" else 16))
        self.max_rels = max(1, min(int(cfg.data.get("toy_max_rels", 3)), vocab.max_num_rels))
        self.image_size = int(cfg.encoders.get("toy_image_size", 32))
        self.seed = int(cfg.get("seed", 42)) + (0 if split == "train" else 10_000)
        self.ind_to_classes = ["__background__"] + [f"entity_{i}" for i in range(1, vocab.n_entities)]
        self.ind_to_predicates = ["__background__"] + [f"predicate_{i}" for i in range(1, vocab.n_preds)]
        self.indices = list(range(self.n))
        self.image_meta = [{"image_id": self.seed + i} for i in range(self.n)]

    def __len__(self) -> int:
        return self.n

    def _graph(self, index: int) -> Graph:
        # Reserve index zero for the background classes used by VG.
        n_entity_real = max(1, self.vocab.n_entities - 1)
        n_pred_real = max(1, self.vocab.n_preds - 1)
        n_rels = 1 + (index % self.max_rels)
        graph: Graph = []
        for j in range(n_rels):
            subj = 1 + ((index * 7 + j * 3) % n_entity_real)
            obj = 1 + ((index * 11 + j * 5 + 1) % n_entity_real)
            pred = 1 + ((index * 13 + j * 2) % n_pred_real)
            graph.append(Relation(subj, 0, pred, obj, 0))
        return graph

    def __getitem__(self, index: int) -> Dict:
        if not 0 <= index < self.n:
            raise IndexError(index)
        graph = self._graph(index)
        image_id = int(self.image_meta[index]["image_id"])
        generator = torch.Generator().manual_seed(image_id)
        image = torch.rand(3, self.image_size, self.image_size, generator=generator)
        text = serialize_graph(graph, self.ind_to_classes, self.ind_to_predicates, with_instances=True)
        # Include the ID so identical relation strings in tiny vocabularies remain separable.
        text = f"image {image_id}: {text}"
        return {"image": image, "text": text, "graph": graph, "image_id": image_id}
