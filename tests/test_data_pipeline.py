"""Data-package contract tests, including a miniature standard VG-SGG fixture."""
from __future__ import annotations

import json
import pickle

import h5py
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

from patchsgg.config import as_config, load_config
from patchsgg.data.collate import GraphCollator
from patchsgg.data.factory import build_dataset
from patchsgg.data.vg_dataset import VGGraphDataset
from patchsgg.model import build_vocab
from patchsgg.utils.huggingface import get_model_path_with_hf_fallback


def test_toy_collator_contract():
    cfg = load_config(
        "patchsgg/configs/diagnostic_text2text.yaml",
        ["vocab.max_num_rels=4", "data.toy_n_train=4", "encoders.toy_image_size=16"],
    )
    vocab = build_vocab(cfg)
    dataset = build_dataset(cfg, "train", vocab)
    batch = next(iter(DataLoader(dataset, batch_size=3, collate_fn=GraphCollator(vocab, seed=7))))
    assert batch["images"].shape == (3, 3, 16, 16)
    assert batch["input_tokens"].shape == batch["target_tokens"].shape == (3, 1 + 5 * 4)
    assert str(batch["input_tokens"].dtype) == "torch.int64"
    assert all(isinstance(x, int) for x in batch["image_ids"])
    assert all(len(t) == 5 for graph in batch["gt_graphs"] for t in graph)


def _write_vg_fixture(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for image_id in (101, 102, 103):
        Image.new("RGB", (24, 20), (image_id % 255, 30, 60)).save(image_dir / f"{image_id}.jpg")

    metadata = [
        {"image_id": 101, "width": 24, "height": 20},
        {"image_id": 102, "width": 24, "height": 20},
        {"image_id": 103, "width": 24, "height": 20},
    ]
    meta_path = tmp_path / "image_data.json"
    meta_path.write_text(json.dumps(metadata))

    dictionaries = {
        "idx_to_label": {"0": "__background__", "1": "person", "2": "horse", "3": "tree"},
        "idx_to_predicate": {"0": "__background__", "1": "on", "2": "near"},
    }
    dict_path = tmp_path / "VG-SGG-dicts.json"
    dict_path.write_text(json.dumps(dictionaries))

    # Image 0 has three person boxes: boxes 0/1 overlap (>0.5 IoU) and must share instance 0;
    # box 2 is separate and must become instance 1. Box 3 is the horse object.
    boxes = np.asarray(
        [
            [100, 100, 80, 80], [104, 104, 80, 80], [400, 400, 80, 80], [600, 600, 100, 100],
            [100, 100, 80, 80], [300, 300, 80, 80],
            [100, 100, 80, 80], [300, 300, 80, 80],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([[1], [1], [1], [2], [1], [3], [2], [3]], dtype=np.int64)
    relationships = np.asarray([[0, 3], [1, 3], [2, 3], [4, 5], [6, 7]], dtype=np.int64)
    predicates = np.asarray([[1], [2], [1], [2], [1]], dtype=np.int64)

    h5_path = tmp_path / "VG-SGG.h5"
    with h5py.File(h5_path, "w") as roi:
        roi["split"] = np.asarray([0, 0, 2], dtype=np.int64)
        roi["img_to_first_box"] = np.asarray([0, 4, 6], dtype=np.int64)
        roi["img_to_last_box"] = np.asarray([3, 5, 7], dtype=np.int64)
        roi["img_to_first_rel"] = np.asarray([0, 3, 4], dtype=np.int64)
        roi["img_to_last_rel"] = np.asarray([2, 3, 4], dtype=np.int64)
        roi["labels"] = labels
        roi["boxes_1024"] = boxes
        roi["relationships"] = relationships
        roi["predicates"] = predicates
    return h5_path, dict_path, meta_path, image_dir


def _vg_cfg(tmp_path):
    h5_path, dict_path, meta_path, image_dir = _write_vg_fixture(tmp_path)
    return as_config(
        {
            "seed": 3,
            "device": "cpu",
            "num_workers": 0,
            "vocab": {
                "n_preds": 3,
                "n_entities": 4,
                "max_instance_id": 4,
                "random_max_instance_id": 2,
                "max_num_rels": 5,
            },
            "encoders": {"space": "toy", "toy_image_size": 16},
            "data": {
                "dataset": "vg",
                "text_view": "serialize",
                "serialize_with_instances": True,
                "captions_cache": None,
                "vg": {
                    "roidb_file": str(h5_path),
                    "dict_file": str(dict_path),
                    "image_meta": str(meta_path),
                    "image_dir": str(image_dir),
                    "instance_iou": 0.5,
                    "num_val_images": 1,
                    "num_images": -1,
                    "filter_duplicate_rels": False,
                },
            },
        }
    )


def test_vg_ids_instances_splits_and_worker_safe_handle(tmp_path):
    cfg = _vg_cfg(tmp_path)
    vocab = build_vocab(cfg)
    val = VGGraphDataset(cfg, "val", vocab)
    train = VGGraphDataset(cfg, "train", vocab)
    test = VGGraphDataset(cfg, "test", vocab)
    assert val.indices == [0]
    assert train.indices == [1]
    assert test.indices == [2]

    sample = val[0]
    graph = sample["graph"]
    # Raw VG IDs are preserved: class 1 and predicate 1 never become background zero.
    assert graph[0].subj_cls == 1 and graph[0].predicate == 1 and graph[0].obj_cls == 2
    assert [rel.subj_inst for rel in graph] == [0, 0, 1]
    assert sample["image"].shape == (3, 16, 16)
    assert "person#0" in sample["text"] and "person#1" in sample["text"]

    # An open h5py handle is removed from pickled dataset state, as DataLoader workers require.
    val._roi()
    restored = pickle.loads(pickle.dumps(val))
    assert restored._roi_handle is None
    assert restored[0]["image_id"] == 101

    worker_batch = next(iter(DataLoader(val, batch_size=1, num_workers=2, collate_fn=GraphCollator(vocab, seed=9))))
    assert worker_batch["image_ids"] == [101]


def test_hf_fallback_prefers_existing_local_file(tmp_path):
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"local")
    assert get_model_path_with_hf_fallback(str(checkpoint), hf_repo_id="unused/repo") == str(checkpoint)
