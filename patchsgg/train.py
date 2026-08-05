"""Training entry point (PyTorch Lightning).

    patchsgg-train --config patchsgg/configs/default_crossattn_talk2dino.yaml
"""
from __future__ import annotations

import argparse
from typing import List

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)

from patchsgg.config import load_config
from patchsgg.lightning_module import SGGDataModule, SGGLightning


def _resolve_accelerator(cfg):
    want = str(cfg.get("device", "cuda"))
    if want.startswith("cuda") and torch.cuda.is_available():
        return "gpu", "cuda"
    return "cpu", "cpu"


def main(argv: List[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--ckpt", default=None, help="resume from this Lightning checkpoint")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.override)
    accelerator, device = _resolve_accelerator(cfg)
    cfg.device = device  # encoders place frozen weights here; must match the trainer device
    pl.seed_everything(int(cfg.get("seed", 42)), workers=True)

    dm = SGGDataModule(cfg)
    model = SGGLightning(cfg)

    out_dir = cfg.get("output_dir", "outputs")
    ckpt_cb = ModelCheckpoint(dirpath=out_dir, save_last=True, save_top_k=1,
                              monitor="val/R@20", mode="max", filename="best-{epoch}")
    grad_clip = float(cfg.train.get("grad_clip", 0)) or None
    trainer = pl.Trainer(
        max_epochs=int(cfg.train.max_epochs),
        accelerator=accelerator,
        devices=1,
        default_root_dir=out_dir,
        check_val_every_n_epoch=int(
            cfg.train.get("eval_every_epochs", 1)
        ),
        log_every_n_steps=int(
            cfg.train.get("log_every_steps", 50)
        ),
        gradient_clip_val=grad_clip,
        precision=cfg.train.get("precision", "32-true"),
        accumulate_grad_batches=int(
            cfg.train.get("accumulate_grad_batches", 1)
        ),
        callbacks=callbacks,
    )
    trainer.fit(model, datamodule=dm, ckpt_path=args.ckpt)
    print(f"done. best={ckpt_cb.best_model_path or '(n/a)'}  last={ckpt_cb.last_model_path}")


if __name__ == "__main__":
    main()
