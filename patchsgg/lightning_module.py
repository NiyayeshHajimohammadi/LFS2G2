"""PyTorch Lightning wrapper around :class:`PatchSGGModel` and a matching DataModule.

Training runs on one conditioning modality (text by default) and validation on another (image for
the real task, text for the diagnostic). Recall metrics are computed at validation-epoch end using
LF-SGG's branched matcher.
"""
from __future__ import annotations

from typing import Dict, List

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from patchsgg.config import as_config, as_container
from patchsgg.data.collate import GraphCollator
from patchsgg.data.factory import build_dataset
from patchsgg.eval.evaluate import evaluate_graphs
from patchsgg.eval.matcher import InstanceMatcher
from patchsgg.model import PatchSGGModel, build_vocab


class SGGLightning(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        cfg = as_config(cfg) if isinstance(cfg, dict) else cfg
        # store a plain-dict copy so load_from_checkpoint can rebuild the model
        self.save_hyperparameters({"cfg": as_container(cfg)})
        self.cfg = cfg
        self.model = PatchSGGModel(cfg)
        self._val_samples: List = []
        self._matcher = InstanceMatcher(
            n=int(cfg.eval
                  .get("matcher_n", 3)),
            depth_limit=int(cfg.eval.get("matcher_depth", 10)),
            allow_identity_fallback=bool(cfg.eval.get("matcher_identity_fallback", False)),
        )

    @classmethod
    def from_checkpoint(cls, path: str, map_location="cpu") -> "SGGLightning":
        """Load a checkpoint while overriding its stale construction device.

        Lightning checkpoints often store ``device: cuda`` in the training config. Rebuilding that
        config unchanged on a CPU inference host would try to construct frozen encoders on CUDA
        before ``map_location`` can help. The runtime device therefore follows ``map_location``.
        """
        ckpt = torch.load(path, map_location=map_location)
        cfg_dict = dict(ckpt["hyper_parameters"]["cfg"])
        if isinstance(map_location, (str, torch.device)):
            runtime_device = str(map_location)
            if runtime_device.startswith("cuda"):
                runtime_device = "cuda"
            elif runtime_device.startswith("cpu"):
                runtime_device = "cpu"
            cfg_dict["device"] = runtime_device
        model = cls(cfg_dict)
        model.load_state_dict(ckpt["state_dict"])
        return model

    def training_step(self, batch: Dict, _):
        loss = self.model.compute_loss(batch, modality=self.cfg.train.train_modality)
        self.log("train/loss", loss, prog_bar=True, batch_size=len(batch["gt_graphs"]))
        return loss

    def validation_step(self, batch: Dict, _):
        preds = self.model.predict(batch, modality=self.cfg.eval.eval_modality)
        for gt, pred in zip(batch["gt_graphs"], preds):
            self._val_samples.append((gt, pred))

    def on_validation_epoch_end(self):
        if not self._val_samples:
            return
        metrics = evaluate_graphs(
            self._val_samples, ks=tuple(self.cfg.eval.get("ks", [20, 50, 100])), matcher=self._matcher
        )
        for k, v in metrics.items():
            self.log(f"val/{k}", v, prog_bar=k in ("R@20", "mR@20"))
        self._val_samples = []

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.model.trainable_parameters(),
            lr=float(self.cfg.train.lr),
            weight_decay=float(self.cfg.train.get("weight_decay", 1e-4)),
        )


class SGGDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.vocab = build_vocab(cfg)
        seed = int(cfg.get("seed", 42))
        self.train_collate = GraphCollator(vocab=self.vocab,seed=seed,deterministic=False,pmar_enabled=(cfg.loss.type == "pmar"),
            pmar_exact_threshold=int(cfg.loss.get("exact_threshold",8)),
            pmar_num_samples=int(cfg.loss.get("num_samples",8)),)
        self.val_collate = GraphCollator(vocab=self.vocab, seed=seed + 1, deterministic=True)
        self.collate = self.train_collate  # backwards-compatible attribute

    def setup(self, stage=None):
        self.train_ds = build_dataset(self.cfg, "train", self.vocab)
        self.val_ds = build_dataset(self.cfg, "val", self.vocab)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=int(self.cfg.train.batch_size), shuffle=True,
            num_workers=int(self.cfg.get("num_workers", 0)), collate_fn=self.train_collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=int(self.cfg.eval.get("batch_size", self.cfg.train.batch_size)),
            shuffle=False, num_workers=int(self.cfg.get("num_workers", 0)), collate_fn=self.val_collate,
        )
