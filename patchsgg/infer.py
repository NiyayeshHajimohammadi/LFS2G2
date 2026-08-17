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

from patchsgg.config import as_config, as_container, load_config
from patchsgg.data.collate import GraphCollator
from patchsgg.data.factory import build_dataset
from patchsgg.lightning_module import SGGLightning
from patchsgg.model import build_vocab
from patchsgg.train import _resolve_accelerator
from patchsgg.graph_seq.vocab import VG_VOCAB
from patchsgg.data.graph_text_views import serialize_graph
from patchsgg.graph_seq.linearize import Relation

def matcher_to_graph(tuples):
    graph = []
    for t in tuples:
        subj_tok, subj_inst, pred_tok, obj_tok, obj_inst = t

        rel = Relation(
            VG_VOCAB.entity_idx(subj_tok),
            subj_inst,
            VG_VOCAB.predicate_idx(pred_tok),
            VG_VOCAB.entity_idx(obj_tok),
            obj_inst,
        )

        graph.append(rel)

    return graph

def _data_runtime_config(cli_cfg, checkpoint_cfg, device: str):
    """Use checkpoint model/encoder settings while allowing CLI data-path and loader overrides."""
    merged = as_container(checkpoint_cfg)
    cli = as_container(cli_cfg)
    merged["device"] = device
    merged["data"] = cli["data"]
    merged["seed"] = cli.get("seed", merged.get("seed", 42))
    merged["num_workers"] = cli.get("num_workers", merged.get("num_workers", 0))
    merged.setdefault("eval", {})["batch_size"] = cli.get("eval", {}).get(
        "batch_size", merged.get("eval", {}).get("batch_size", 8)
    )
    return as_config(merged)


def _assert_vocab_compatible(cli_cfg, model_vocab) -> None:
    cli_vocab = build_vocab(cli_cfg)
    fields = ("n_preds", "n_entities", "max_instance_id", "max_num_rels")
    mismatches = {name: (getattr(cli_vocab, name), getattr(model_vocab, name)) for name in fields
                  if getattr(cli_vocab, name) != getattr(model_vocab, name)}
    if mismatches:
        details = ", ".join(f"{name}: config={a}, checkpoint={b}" for name, (a, b) in mismatches.items())
        raise ValueError(f"inference config is incompatible with checkpoint vocabulary ({details})")


def main(argv: List[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--dump", default=None, help="optional path to write predicted graphs (json)")
    args = parser.parse_args(argv)

    cli_cfg = load_config(args.config, args.override)
    accelerator, device = _resolve_accelerator(cli_cfg)
    cli_cfg.device = device

    model = SGGLightning.from_checkpoint(
        args.ckpt,
        map_location=device,
    )

    model.eval()

    _assert_vocab_compatible(
        cli_cfg,
        model.model.vocab,
    )

    # Apply inference-time settings from the command-line config.
    #
    # This allows:
    #   --override eval.max_rels=300
    #   --override eval.top_p=0.95
    #   --override eval.temperature=1.75
    model.model.set_generation_config(
        cli_cfg.eval
    )

    data_cfg = _data_runtime_config(cli_cfg, model.cfg, device)
    vocab = model.model.vocab
    collate = GraphCollator(vocab=vocab, seed=int(data_cfg.get("seed", 42)), deterministic=True)
    ds = build_dataset(data_cfg, args.split, vocab)
    loader = DataLoader(
        ds,
        batch_size=int(data_cfg.eval.get("batch_size", 8)),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 0)),
        collate_fn=collate,
    )

    trainer = pl.Trainer(accelerator=accelerator, devices=1, logger=False)
    # trainer.validate(model, dataloaders=loader)

    # if args.dump:
    #     model.to(device)
    #     records = []
    #     for batch in loader:
    #         if isinstance(batch.get("images"), torch.Tensor):
    #             batch["images"] = batch["images"].to(device)
    #         preds = model.model.predict(batch, modality=model.cfg.eval.eval_modality)
    #         for image_id, gt, pred in zip(batch["image_ids"], batch["gt_graphs"], preds):
    #             records.append({"image_id": int(image_id), "gt": gt, "pred": pred})
    #     with open(args.dump, "w", encoding="utf-8") as handle:
    #         json.dump(records, handle)
    #     print(f"wrote {len(records)} predictions -> {args.dump}")
    if args.dump:
        model.to(device)
        records = []

        max_images = 10
        count = 0

        for batch in loader:
            if isinstance(batch.get("images"), torch.Tensor):
                batch["images"] = batch["images"].to(device)

            preds = model.model.predict(
                batch,
                modality=model.cfg.eval.eval_modality
            )

            for image_id, gt, pred in zip(
                batch["image_ids"],
                batch["gt_graphs"],
                preds
            ):
                records.append({
                    "image_id": int(image_id),
                    "gt": serialize_graph(
                        matcher_to_graph(gt),
                        ds.ind_to_classes,
                        ds.ind_to_predicates,
                        with_instances=True
                    ),
                    "pred": serialize_graph(
                        matcher_to_graph(pred),
                        ds.ind_to_classes,
                        ds.ind_to_predicates,
                        with_instances=True
                    )})

                count += 1
                if count >= max_images:
                    break

            if count >= max_images:
                break

        with open(args.dump, "w", encoding="utf-8") as handle:
            json.dump(records, handle)

        print(f"wrote {len(records)} predictions -> {args.dump}")


if __name__ == "__main__":
    main()
