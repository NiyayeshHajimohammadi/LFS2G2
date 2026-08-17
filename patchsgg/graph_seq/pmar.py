"""
PMAR graph serialization utilities.

This module generates graph-equivalent candidate
serializations for Permutation-Marginalized
Autoregressive training.

Pipeline:

graph
 -> structural refinement
 -> residual assignments
 -> relabel graph
 -> canonical sort
 -> token sequence
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product
import random
from typing import Dict, List, Tuple


from patchsgg.graph_seq.refinement import (
    StructuralCell,
    RefinementResult,
    structural_refine,
    residual_permutation_count,
)


Node = Tuple[int, int]


@dataclass
class PMARCandidate:
    """
    One candidate serialization.

    sequence:
        token IDs after vocabulary conversion

    assignment:
        mapping:

        (class, original_instance)
            ->
        new instance index
    """

    sequence: List[int]

    assignment: Dict[Node, int]


@dataclass
class PMARGenerationResult:
    """
    Output of candidate generation.
    """

    candidates: List[PMARCandidate]

    exact: bool

    residual_count: int



# ---------------------------------------------------------
# Assignment generation
# ---------------------------------------------------------


def _cell_permutations(
    cell: StructuralCell,
    block: List[int],
):
    """
    Generate all assignments inside one ambiguity cell.

    Example:

    nodes:

        A,B

    block:

        [1,2]


    returns:

        A->1,B->2
        A->2,B->1
    """

    nodes = list(cell.nodes)

    for perm in permutations(block):

        yield {
            node: idx
            for node, idx in zip(
                nodes,
                perm,
            )
        }



def enumerate_assignments(
    cells: List[StructuralCell],
):
    """
    Exact enumeration of residual
    permutation group.

    Returns:

        dictionaries:

        node -> canonical instance id
    """

    block_start = 0

    all_cell_assignments = []

    for cell in cells:

        size = len(cell.nodes)

        block = list(
            range(
                block_start,
                block_start + size,
            )
        )

        block_start += size

        all_cell_assignments.append(
            list(
                _cell_permutations(
                    cell,
                    block,
                )
            )
        )


    for choices in product(
        *all_cell_assignments
    ):

        merged = {}

        for choice in choices:
            merged.update(choice)

        yield merged



def sample_assignments(
    cells: List[StructuralCell],
    num_samples: int,
    seed: int = 0,
):
    """
    Sample residual assignments.

    Sampling is uniform inside each
    ambiguity cell.

    Used when exact enumeration
    becomes too expensive.
    """

    rng = random.Random(seed)


    blocks = {}

    start = 0

    for cell in cells:

        size = len(cell.nodes)

        blocks[cell] = list(
            range(
                start,
                start + size,
            )
        )

        start += size



    for _ in range(num_samples):

        assignment = {}

        for cell in cells:

            nodes = list(cell.nodes)

            indices = blocks[cell].copy()

            rng.shuffle(indices)

            assignment.update(
                {
                    node: idx
                    for node, idx
                    in zip(
                        nodes,
                        indices,
                    )
                }
            )

        yield assignment



# ---------------------------------------------------------
# Graph relabeling
# ---------------------------------------------------------


def relabel_graph(
    graph,
    assignment: Dict[Node, int],
):
    """
    Replace original instance IDs
    by PMAR canonical IDs.

    Example:

        person#17

    becomes:

        person#0
    """

    new_graph = []


    for rel in graph:

        subj_old = (
            int(rel.subj_cls),
            int(rel.subj_instance),
        )

        obj_old = (
            int(rel.obj_cls),
            int(rel.obj_instance),
        )


        new_graph.append(
            {
                "subj_cls":
                    int(rel.subj_cls),

                "subj_instance":
                    assignment[subj_old],

                "predicate":
                    int(rel.predicate),

                "obj_cls":
                    int(rel.obj_cls),

                "obj_instance":
                    assignment[obj_old],
            }
        )


    return new_graph



# ---------------------------------------------------------
# Canonical relation ordering
# ---------------------------------------------------------


def canonical_sort_relations(
    graph,
):
    """
    Remove dependence on annotation
    relation storage order.

    Keeps duplicate relations.

    """

    return sorted(
        graph,
        key=lambda r: (
            r["subj_cls"],
            r["subj_instance"],
            r["obj_cls"],
            r["obj_instance"],
            r["predicate"],
        ),
    )



# ---------------------------------------------------------
# Serialization
# ---------------------------------------------------------


def serialize_graph(
    graph,
    vocab,
):
    """
    Convert a relabeled graph into
    Pix2SG-style token sequence.

    One relation:

        subj_cls
        subj_instance
        obj_cls
        obj_instance
        predicate


    """

    sequence = []

    graph = canonical_sort_relations(
        graph
    )


    for rel in graph:

        sequence.extend(
            [
                vocab.entity_idx(
                    rel["subj_cls"]
                ),

                vocab.instance_token(
                    rel["subj_instance"]
                ),

                vocab.entity_idx(
                    rel["obj_cls"]
                ),

                vocab.instance_token(
                    rel["obj_instance"]
                ),

                vocab.predicate_idx(
                    rel["predicate"]
                ),
            ]
        )


    sequence.append(
        vocab.end_token
    )


    return sequence



# ---------------------------------------------------------
# Main PMAR candidate builder
# ---------------------------------------------------------


def build_pmar_candidates(
    graph,
    vocab,
    exact_threshold: int = 8,
    num_samples: int = 8,
    seed: int = 0,
):
    """
    Generate PMAR candidates.

    Exact mode:

        enumerate every residual
        permutation.

    Sampling mode:

        sample assignments.

    """

    refinement = structural_refine(
        graph
    )

    cells = refinement.cells


    residual_count = residual_permutation_count(
        cells
    )


    if residual_count <= exact_threshold:

        assignments = enumerate_assignments(
            cells
        )

        exact = True

    else:

        assignments = sample_assignments(
            cells,
            num_samples=num_samples,
            seed=seed,
        )

        exact = False



    candidates = []


    seen = set()


    for assignment in assignments:

        relabeled = relabel_graph(
            graph,
            assignment,
        )

        sequence = serialize_graph(
            relabeled,
            vocab,
        )


        # Exact mode removes duplicate
        # serializations.
        #
        # Sampling mode keeps duplicates
        # intentionally.

        if exact:

            key = tuple(sequence)

            if key in seen:
                continue

            seen.add(key)


        candidates.append(
            PMARCandidate(
                sequence=sequence,
                assignment=assignment,
            )
        )


    return PMARGenerationResult(
        candidates=candidates,
        exact=exact,
        residual_count=residual_count,
    )