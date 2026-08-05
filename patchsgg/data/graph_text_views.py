"""Text representations used to train the decoder from scene graphs."""
from __future__ import annotations

from typing import Mapping, Sequence

from patchsgg.graph_seq.linearize import Graph, Relation


def _lookup(names: Sequence[str], index: int, kind: str) -> str:
    if 0 <= index < len(names) and names[index]:
        return str(names[index])
    return f"{kind}_{index}"


def serialize_graph(
    graph: Graph,
    ind_to_classes: Sequence[str],
    ind_to_predicates: Sequence[str],
    with_instances: bool = False,
) -> str:
    """Serialize a semantic graph into a deterministic relation-list sentence."""
    parts: list[str] = []
    for rel in graph:
        rel = Relation(*map(int, rel))
        subject = _lookup(ind_to_classes, rel.subj_cls, "entity")
        object_ = _lookup(ind_to_classes, rel.obj_cls, "entity")
        predicate = _lookup(ind_to_predicates, rel.predicate, "predicate")
        if with_instances:
            subject = f"{subject}#{rel.subj_inst}"
            object_ = f"{object_}#{rel.obj_inst}"
        parts.append(f"{subject} {predicate} {object_}")
    return "; ".join(parts) if parts else "empty scene graph"


def select_text_view(
    *,
    serialized: str,
    image_id: int,
    mode: str,
    captions: Mapping[int, Sequence[str]] | None = None,
    seed: int = 0,
) -> str:
    """Select a stable text view; missing caption entries safely fall back to serialization."""
    mode = str(mode)
    choices = list((captions or {}).get(int(image_id), ()))
    choices = [str(c).strip() for c in choices if str(c).strip()]
    if mode == "serialize":
        return serialized
    if mode == "llm_caption":
        return choices[(int(image_id) + seed) % len(choices)] if choices else serialized
    if mode == "mix":
        if not choices or (int(image_id) + seed) % 2 == 0:
            return serialized
        return choices[(int(image_id) + seed) % len(choices)]
    raise ValueError(f"unknown data.text_view={mode!r}; expected serialize, llm_caption, or mix")
