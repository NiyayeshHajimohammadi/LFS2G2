"""Dataset factory."""
from __future__ import annotations

from patchsgg.graph_seq.vocab import GraphVocab


def build_dataset(cfg, split, vocab):
    dataset_name = str(cfg.data.dataset)

    if dataset_name == "vg":
        from patchsgg.data.vg_dataset import VGGraphDataset

        return VGGraphDataset(cfg, split, vocab)

    if dataset_name == "toy":
        from patchsgg.data.toy_dataset import ToyGraphDataset

        return ToyGraphDataset(cfg, split, vocab)

    raise ValueError(f"Unknown dataset: {dataset_name!r}")