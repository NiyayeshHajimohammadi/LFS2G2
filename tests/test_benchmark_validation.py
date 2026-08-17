import argparse
import time

import pytorch_lightning as pl
import torch

from patchsgg.config import load_config
from patchsgg.lightning_module import SGGLightning, SGGDataModule


parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--ckpt", required=True)
parser.add_argument("--batches", type=int, default=10)
parser.add_argument("--max-rels", type=int, default=None)
parser.add_argument("--batch-size", type=int, default=None)
args = parser.parse_args()


cfg = load_config(args.config)

cfg.device = "cuda"

if args.max_rels is not None:
    cfg.eval.max_rels = args.max_rels

if args.batch_size is not None:
    cfg.eval.batch_size = args.batch_size


# Build model using current config.
model = SGGLightning(cfg)

checkpoint = torch.load(
    args.ckpt,
    map_location="cpu",
)

model.load_state_dict(
    checkpoint["state_dict"]
)


data = SGGDataModule(cfg)


trainer = pl.Trainer(
    accelerator="gpu",
    devices=1,

    # Important:
    limit_val_batches=args.batches,

    # No training-related overhead.
    logger=False,
    enable_checkpointing=False,
    num_sanity_val_steps=0,
)


torch.cuda.synchronize()
start = time.perf_counter()

trainer.validate(
    model,
    datamodule=data,
)

torch.cuda.synchronize()
elapsed = time.perf_counter() - start


images = args.batches * int(cfg.eval.batch_size)

print()
print("=" * 60)
print("VALIDATION BENCHMARK")
print("=" * 60)
print(f"batches:          {args.batches}")
print(f"approx images:    {images}")
print(f"max_rels:         {cfg.eval.max_rels}")
print(f"batch size:       {cfg.eval.batch_size}")
print(f"elapsed:          {elapsed:.2f} sec")
print(f"sec / batch:      {elapsed / args.batches:.2f}")
print(f"sec / image:      {elapsed / images:.3f}")

full_batches = (5000 + int(cfg.eval.batch_size) - 1) // int(cfg.eval.batch_size)
estimated = elapsed / args.batches * full_batches

print()
print(f"full val batches: {full_batches}")
print(f"estimated full:   {estimated / 60:.1f} min")
print(f"estimated full:   {estimated / 3600:.2f} hours")
print("=" * 60)