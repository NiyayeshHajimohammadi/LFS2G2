"""Offline: turn VG scene graphs into fluent natural-language captions with an LLM.

Produces a resumable cache ``{image_id: [captions...]}`` consumed by the ``llm_caption`` / ``mix``
text views. Use ``--dry-run`` to fill the cache with deterministic serializations (no API calls),
which is handy for pipeline testing.

Default model: claude-opus-4-8 (Anthropic). Requires ANTHROPIC_API_KEY for real generation.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

from patchsgg.config import load_config
from patchsgg.data.graph_text_views import serialize_graph
from patchsgg.model import build_vocab

PROMPT = (
    "You are given the ground-truth scene graph of an image as a list of subject-predicate-object "
    "relations. Write {n} short, natural image captions that faithfully describe ONLY what the "
    "relations state -- do not invent objects, attributes, or relations not present. Vary phrasing "
    "across the captions. Return one caption per line, no numbering.\n\nRelations:\n{relations}"
)


def _generate(client, model: str, relations: str, n: int) -> List[str]:
    msg = client.messages.create(
        model=model,
        max_tokens=400,
        messages=[{"role": "user", "content": PROMPT.format(n=n, relations=relations)}],
    )
    text = "".join(block.text for block in msg.content if getattr(block, "type", None) == "text")
    return [line.strip() for line in text.splitlines() if line.strip()][:n]


def main(argv: List[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=3, help="captions per image")
    parser.add_argument("--model", default="claude-opus-4-8")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    cfg.device = "cpu"
    vocab = build_vocab(cfg)
    from patchsgg.data.vg_dataset import VGGraphDataset

    ds = VGGraphDataset(cfg, split=args.split, vocab=vocab)

    cache: Dict[int, List[str]] = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            cache = {int(k): v for k, v in json.load(f).items()}

    client = None
    if not args.dry_run:
        import anthropic

        client = anthropic.Anthropic()

    n_total = len(ds.indices) if args.limit < 0 else min(args.limit, len(ds.indices))
    for idx in range(n_total):
        img_idx = ds.indices[idx]
        image_id = int(ds.image_meta[img_idx]["image_id"])
        if image_id in cache:
            continue
        graph = ds._graph_for(ds._roi(), img_idx)
        relations = serialize_graph(graph, ds.ind_to_classes, ds.ind_to_predicates)
        if not graph:
            continue
        cache[image_id] = [relations] if args.dry_run else _generate(client, args.model, relations, args.n)
        if idx % 50 == 0:
            with open(args.out, "w") as f:
                json.dump(cache, f)
            print(f"[{idx}/{n_total}] cached image {image_id}")

    with open(args.out, "w") as f:
        json.dump(cache, f)
    print(f"done. {len(cache)} images -> {args.out}")


if __name__ == "__main__":
    main()
