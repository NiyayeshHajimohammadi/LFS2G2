"""Inference / evaluation entry point (PyTorch Lightning).

    patchsgg-infer --config <cfg> --ckpt outputs/last.ckpt [--split val] [--dump preds.json]
"""
from __future__ import annotations

import argparse
import json
from typing import List

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from patchsgg.config import load_config
from patchsgg.data.collate import GraphCollator
from patchsgg.data.factory import build_dataset
from patchsgg.lightning_module import SGGLightning
from patchsgg.model import build_vocab
from patchsgg.train import _resolve_accelerator


def main(argv: List[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--dump", default=None, help="optional path to write predicted graphs (json)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.override)
    accelerator, device = _resolve_accelerator(cfg)
    cfg.device = device

    model = SGGLightning.from_checkpoint(args.ckpt, map_location=device)
    model.eval()

    vocab = build_vocab(cfg)
    collate = GraphCollator(vocab=vocab, seed=int(cfg.get("seed", 42)))
    ds = build_dataset(cfg, args.split, vocab)
    loader = DataLoader(ds, batch_size=int(cfg.eval.get("batch_size", 8)), shuffle=False,
                        num_workers=int(cfg.get("num_workers", 0)), collate_fn=collate)

    trainer = pl.Trainer(accelerator=accelerator, devices=1, logger=False)
    trainer.validate(model, dataloaders=loader)

    if args.dump:
        model.to(device)
        records = []
        for batch in loader:
            if isinstance(batch.get("images"), torch.Tensor):
                batch["images"] = batch["images"].to(device)
            preds = model.model.predict(batch, modality=model.cfg.eval.eval_modality)
            for image_id, gt, pred in zip(batch["image_ids"], batch["gt_graphs"], preds):
                records.append({"image_id": image_id, "gt": gt, "pred": pred})
        with open(args.dump, "w") as f:
            json.dump(records, f)
        print(f"wrote {len(records)} predictions -> {args.dump}")


if __name__ == "__main__":
    main()
