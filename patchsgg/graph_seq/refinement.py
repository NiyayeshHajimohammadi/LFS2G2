"""
Structural refinement utilities for PMAR.

This module performs graph-only processing:

Graph
 -> node extraction
 -> predicate-aware directed WL refinement
 -> structural cells
 -> residual permutation blocks

No neural network code belongs here.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial
from typing import Dict, List, Tuple, Iterable


# A node is represented by:
# (class_id, original_instance_id)
Node = Tuple[int, int]


@dataclass(frozen=True)
class StructuralCell:
    """
    A group of nodes that remain indistinguishable
    after structural refinement.

    Example:

        class = person
        nodes = {
            (person,0),
            (person,3)
        }

    means these two persons have identical structural roles.
    """

    class_id: int

    nodes: Tuple[Node, ...]

    # final deterministic structural signature
    signature: Tuple


@dataclass
class RefinementResult:
    """
    Output of structural refinement.
    """

    cells: List[StructuralCell]

    # mapping:
    # node -> final color/signature
    colors: Dict[Node, Tuple]


def extract_nodes(graph) -> List[Node]:
    """
    Extract unique graph nodes.

    Expected graph format:
        iterable of relations

    Each relation should have:

        subj_cls
        subj_instance
        obj_cls
        obj_instance

    compatible with current PatchSGG Relation objects.
    """

    nodes = set()

    for rel in graph:

        nodes.add(
            (
                int(rel.subj_cls),
                int(rel.subj_instance),
            )
        )

        nodes.add(
            (
                int(rel.obj_cls),
                int(rel.obj_instance),
            )
        )

    return sorted(nodes)


def build_adjacency(graph):
    """
    Construct directed predicate-aware adjacency.

    Returns:

        outgoing[node] =
            [
                (predicate, neighbor),
                ...
            ]

        incoming[node] =
            [
                (predicate, neighbor),
                ...
            ]

    Direction matters.

    A --holding--> B

    is different from

    B --holding--> A
    """

    outgoing = {}
    incoming = {}

    for rel in graph:

        subj = (
            int(rel.subj_cls),
            int(rel.subj_instance),
        )

        obj = (
            int(rel.obj_cls),
            int(rel.obj_instance),
        )

        predicate = int(rel.predicate)

        outgoing.setdefault(subj, []).append(
            (predicate, obj)
        )

        incoming.setdefault(obj, []).append(
            (predicate, subj)
        )

        # ensure isolated nodes exist
        outgoing.setdefault(obj, [])
        incoming.setdefault(subj, [])

    return outgoing, incoming


def initial_colors(nodes: Iterable[Node]):
    """
    Initial WL colors.

    At iteration 0 only the semantic class is known.

    Example:

        person#0
        person#1

    both start as:

        color = person
    """

    return {
        node: ("class", node[0])
        for node in nodes
    }


def refinement_signature(
    node: Node,
    colors: Dict[Node, Tuple],
    outgoing,
    incoming,
):
    """
    Compute one directed predicate-aware WL update.

    Signature:

        (
          current color,
          outgoing relations,
          incoming relations
        )

    Multiplicity is preserved because lists are sorted,
    not converted to sets.
    """

    outgoing_signature = tuple(
        sorted(
            (
                predicate,
                colors[neighbor],
            )
            for predicate, neighbor in outgoing[node]
        )
    )

    incoming_signature = tuple(
        sorted(
            (
                predicate,
                colors[neighbor],
            )
            for predicate, neighbor in incoming[node]
        )
    )

    return (
        colors[node],
        ("out", outgoing_signature),
        ("in", incoming_signature),
    )


def canonicalize_signatures(signatures):
    """
    Convert arbitrary signatures into deterministic compact colors.

    Important:

    We do NOT use Python hash() because it is randomized
    between processes.

    Ordering is based on the actual tuple representation.
    """

    unique = sorted(
        set(signatures.values()),
        key=lambda x: repr(x),
    )

    mapping = {
        signature: idx
        for idx, signature in enumerate(unique)
    }

    return {
        node: (
            "color",
            mapping[signature],
        )
        for node, signature in signatures.items()
    }


def structural_refine(
    graph,
    max_iterations: int = 20,
) -> RefinementResult:
    """
    Perform predicate-aware directed WL refinement.

    Stops when colors stop changing.
    """

    nodes = extract_nodes(graph)

    outgoing, incoming = build_adjacency(graph)

    colors = initial_colors(nodes)

    for _ in range(max_iterations):

        signatures = {
            node: refinement_signature(
                node,
                colors,
                outgoing,
                incoming,
            )
            for node in nodes
        }

        new_colors = canonicalize_signatures(
            signatures
        )

        if new_colors == colors:
            break

        colors = new_colors

    # group nodes by final color AND class

    groups = {}

    for node, color in colors.items():

        key = (
            node[0],   # class
            color,
        )

        groups.setdefault(
            key,
            []
        ).append(node)


    cells = []

    for (class_id, color), members in groups.items():

        cells.append(
            StructuralCell(
                class_id=class_id,
                nodes=tuple(
                    sorted(members)
                ),
                signature=color,
            )
        )


    # deterministic cell ordering

    cells.sort(
        key=lambda c: (
            c.class_id,
            repr(c.signature),
        )
    )


    return RefinementResult(
        cells=cells,
        colors=colors,
    )


def assign_index_blocks(
    cells: List[StructuralCell],
):
    """
    Assign deterministic instance-ID ranges.

    Example:

        cell size 1
        cell size 3
        cell size 2


    becomes:

        cell0 -> [0]
        cell1 -> [1,2,3]
        cell2 -> [4,5]
    """

    blocks = {}

    current = 0

    for cell in cells:

        size = len(cell.nodes)

        blocks[cell] = list(
            range(
                current,
                current + size,
            )
        )

        current += size


    return blocks


def residual_permutation_count(
    cells: List[StructuralCell],
) -> int:
    """
    Compute:

        M_residual =
            product |cell|!

    """

    result = 1

    for cell in cells:
        result *= factorial(
            len(cell.nodes)
        )

    return result