"""Visual Genome scene-graph dataset for location-free graph generation.

The loader follows the standard ``VG-SGG.h5`` layout. Visual Genome's label IDs are already
background-inclusive (objects 0..150, predicates 0..50), so they are intentionally *not* shifted.
Per-class instance IDs are assigned from bounding boxes using the LF-SGG IoU rule.
"""
from __future__ import annotations

import json
import os
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, Mapping, Sequence

import h5py
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from patchsgg.data.graph_text_views import select_text_view, serialize_graph
from patchsgg.data.transforms import build_image_transform
from patchsgg.graph_seq.linearize import Graph, Relation
from patchsgg.graph_seq.vocab import GraphVocab


_CORRUPTED_IMAGE_IDS = {1592, 1722, 4616, 4617}
_BOX_SCALE = 1024


def _mapping_to_list(mapping: Mapping | Sequence, label: str) -> list[str]:
    if isinstance(mapping, list):
        return [str(x) for x in mapping]
    if not isinstance(mapping, Mapping):
        raise TypeError(f"{label} must be a list or index mapping, got {type(mapping)!r}")
    indexed = {int(k): str(v) for k, v in mapping.items()}
    if not indexed:
        return []
    result = [""] * (max(indexed) + 1)
    for index, value in indexed.items():
        result[index] = value
    return result


def _box_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    left_top = np.maximum(box1[:2], box2[:2])
    right_bottom = np.minimum(box1[2:], box2[2:])
    wh = np.maximum(right_bottom - left_top, 0.0)
    intersection = float(wh[0] * wh[1])
    area1 = float(max(box1[2] - box1[0], 0.0) * max(box1[3] - box1[1], 0.0))
    area2 = float(max(box2[2] - box2[0], 0.0) * max(box2[3] - box2[1], 0.0))
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0


class VGGraphDataset(Dataset):
    """Read standard Visual Genome annotations and emit semantic location-free graphs."""

    def __init__(self, cfg, split: str, vocab: GraphVocab):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"unknown split {split!r}")
        self.cfg = cfg
        self.split = split
        self.vocab = vocab
        vg_cfg = cfg.data.vg
        self.roidb_file = str(vg_cfg.roidb_file)
        self.dict_file = str(vg_cfg.dict_file)
        self.image_meta_file = str(vg_cfg.image_meta)
        self.image_dir = str(vg_cfg.image_dir)
        self.instance_iou = float(vg_cfg.get("instance_iou", 0.5))
        self.seed = int(cfg.get("seed", 42))
        self.text_view = str(cfg.data.get("text_view", "serialize"))
        self.serialize_with_instances = bool(cfg.data.get("serialize_with_instances", False))
        self.filter_duplicate_rels = bool(vg_cfg.get("filter_duplicate_rels", False))
        self.transform = build_image_transform(cfg)
        self._roi_handle: h5py.File | None = None
        self._roi_pid: int | None = None
        self._warned_instance_overflow = False

        for path, description in [
            (self.roidb_file, "VG HDF5 annotation file"),
            (self.dict_file, "VG dictionary file"),
            (self.image_meta_file, "VG image metadata file"),
        ]:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"{description} not found: {path}")
        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"VG image directory not found: {self.image_dir}")

        with open(self.dict_file, encoding="utf-8") as handle:
            info = json.load(handle)
        self.ind_to_classes = _mapping_to_list(info.get("idx_to_label", info.get("ind_to_classes", [])), "idx_to_label")
        self.ind_to_predicates = _mapping_to_list(
            info.get("idx_to_predicate", info.get("ind_to_predicates", [])), "idx_to_predicate"
        )
        if len(self.ind_to_classes) != vocab.n_entities:
            raise ValueError(
                f"vocab.n_entities={vocab.n_entities}, but {self.dict_file} contains {len(self.ind_to_classes)} classes"
            )
        if len(self.ind_to_predicates) != vocab.n_preds:
            raise ValueError(
                f"vocab.n_preds={vocab.n_preds}, but {self.dict_file} contains {len(self.ind_to_predicates)} predicates"
            )

        with h5py.File(self.roidb_file, "r") as roi:
            required = {
                "split", "img_to_first_box", "img_to_last_box", "img_to_first_rel", "img_to_last_rel",
                "labels", "boxes_1024", "relationships", "predicates",
            }
            missing = required - set(roi.keys())
            if missing:
                raise KeyError(f"{self.roidb_file} is missing required datasets: {sorted(missing)}")
            split_values = np.asarray(roi["split"][:], dtype=np.int64)
            has_boxes = np.asarray(roi["img_to_first_box"][:]) >= 0
            has_relations = np.asarray(roi["img_to_first_rel"][:]) >= 0
            n_images = len(split_values)

        with open(self.image_meta_file, encoding="utf-8") as handle:
            raw_meta = json.load(handle)
        filtered_meta = [m for m in raw_meta if int(m["image_id"]) not in _CORRUPTED_IMAGE_IDS]
        if len(filtered_meta) == n_images:
            self.image_meta = filtered_meta
        elif len(raw_meta) == n_images:
            self.image_meta = raw_meta
        else:
            raise ValueError(
                f"image metadata/HDF5 length mismatch: metadata={len(raw_meta)}, filtered={len(filtered_meta)}, h5={n_images}"
            )

        valid = has_boxes & has_relations
        if split == "test":
            mask = valid & (split_values == int(vg_cfg.get("test_split_flag", 2)))
            indices = np.flatnonzero(mask)
        elif split == "val" and np.any(split_values == int(vg_cfg.get("val_split_flag", 1))):
            mask = valid & (split_values == int(vg_cfg.get("val_split_flag", 1)))
            indices = np.flatnonzero(mask)
        else:
            train_flag = int(vg_cfg.get("train_split_flag", 0))
            train_pool = np.flatnonzero(valid & (split_values == train_flag))
            n_val = max(0, int(vg_cfg.get("num_val_images", 5000)))
            indices = train_pool[:n_val] if split == "val" else train_pool[n_val:]

        num_images = int(vg_cfg.get("num_images", -1))
        if num_images >= 0:
            indices = indices[:num_images]
        self.indices = [int(i) for i in indices]

        captions_path = cfg.data.get("captions_cache", None)
        self.captions: Dict[int, list[str]] = {}
        if captions_path:
            if not os.path.isfile(captions_path):
                raise FileNotFoundError(f"captions cache not found: {captions_path}")
            with open(captions_path, encoding="utf-8") as handle:
                self.captions = {int(k): list(v) for k, v in json.load(handle).items()}

    def __len__(self) -> int:
        return len(self.indices)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_roi_handle"] = None
        state["_roi_pid"] = None
        return state

    def close(self) -> None:
        if self._roi_handle is not None:
            try:
                self._roi_handle.close()
            finally:
                self._roi_handle = None
                self._roi_pid = None

    def __del__(self):  # pragma: no cover - best-effort cleanup
        self.close()

    def _roi(self) -> h5py.File:
        """Open one read-only HDF5 handle per process/DataLoader worker."""
        pid = os.getpid()
        if self._roi_handle is None or self._roi_pid != pid:
            self.close()
            self._roi_handle = h5py.File(self.roidb_file, "r", swmr=True)
            self._roi_pid = pid
        return self._roi_handle

    def _image_path(self, img_idx: int) -> str:
        meta = self.image_meta[img_idx]
        image_id = int(meta["image_id"])
        candidates = [
            Path(self.image_dir) / f"{image_id}.jpg",
            Path(self.image_dir) / str(meta.get("file_name", "")),
        ]
        for path in candidates:
            if path.name and path.is_file():
                return str(path)
        raise FileNotFoundError(f"image {image_id} not found under {self.image_dir}")

    def _graph_for(self, roi: h5py.File, img_idx: int) -> Graph:
        first_box = int(roi["img_to_first_box"][img_idx])
        last_box = int(roi["img_to_last_box"][img_idx])
        first_rel = int(roi["img_to_first_rel"][img_idx])
        last_rel = int(roi["img_to_last_rel"][img_idx])
        if first_box < 0 or first_rel < 0:
            return []

        labels = np.asarray(roi["labels"][first_box : last_box + 1]).reshape(-1).astype(np.int64)
        boxes = np.asarray(roi[f"boxes_{_BOX_SCALE}"][first_box : last_box + 1], dtype=np.float32).copy()
        boxes[:, :2] -= boxes[:, 2:] / 2.0
        boxes[:, 2:] += boxes[:, :2]
        pairs = np.asarray(roi["relationships"][first_rel : last_rel + 1], dtype=np.int64) - first_box
        predicates = np.asarray(roi["predicates"][first_rel : last_rel + 1]).reshape(-1).astype(np.int64)
        if np.any(pairs < 0) or np.any(pairs >= len(labels)):
            raise ValueError(f"relationship box index outside image {img_idx}'s box range")

        relations = [(int(s), int(o), int(p)) for (s, o), p in zip(pairs, predicates)]
        if self.filter_duplicate_rels:
            grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
            for subj_index, obj_index, pred in relations:
                grouped[(subj_index, obj_index)].append(pred)
            # Deterministic choice avoids validation drift and worker-dependent Python RNG state.
            relations = [(s, o, preds[0]) for (s, o), preds in grouped.items()]

        clusters: dict[int, list[np.ndarray]] = defaultdict(list)
        node_instances: dict[tuple[int, int], int | None] = {}

        def instance_for(class_id: int, local_box_index: int) -> int | None:
            key = (class_id, local_box_index)
            if key in node_instances:
                return node_instances[key]
            box = boxes[local_box_index]
            for instance_id, known_box in enumerate(clusters[class_id]):
                if _box_iou(known_box, box) > self.instance_iou:
                    node_instances[key] = instance_id
                    return instance_id
            if len(clusters[class_id]) >= self.vocab.max_instance_id:
                if not self._warned_instance_overflow:
                    warnings.warn(
                        "an image contains more same-class instances than vocab.max_instance_id; overflowing relations are skipped",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    self._warned_instance_overflow = True
                node_instances[key] = None
                return None
            instance_id = len(clusters[class_id])
            clusters[class_id].append(box.copy())
            node_instances[key] = instance_id
            return instance_id

        graph: Graph = []
        for subj_index, obj_index, predicate in relations:
            subj_class = int(labels[subj_index])
            obj_class = int(labels[obj_index])
            # VG IDs include the background at zero. Do not subtract one.
            if not (0 <= subj_class < self.vocab.n_entities and 0 <= obj_class < self.vocab.n_entities):
                raise ValueError(f"object label outside configured vocabulary in image {img_idx}")
            if not (0 <= predicate < self.vocab.n_preds):
                raise ValueError(f"predicate label {predicate} outside configured vocabulary in image {img_idx}")
            subj_instance = instance_for(subj_class, subj_index)
            obj_instance = instance_for(obj_class, obj_index)
            if subj_instance is None or obj_instance is None:
                continue
            graph.append(Relation(subj_class, subj_instance, predicate, obj_class, obj_instance))
        return graph

    def __getitem__(self, index: int) -> Dict:
        img_idx = self.indices[index]
        image_id = int(self.image_meta[img_idx]["image_id"])
        with Image.open(self._image_path(img_idx)) as image:
            image_tensor = self.transform(image)
        graph = self._graph_for(self._roi(), img_idx)
        serialized = serialize_graph(
            graph,
            self.ind_to_classes,
            self.ind_to_predicates,
            with_instances=self.serialize_with_instances,
        )
        text = select_text_view(
            serialized=serialized,
            image_id=image_id,
            mode=self.text_view,
            captions=self.captions,
            seed=self.seed,
        )
        return {"image": image_tensor, "text": text, "graph": graph, "image_id": image_id}
