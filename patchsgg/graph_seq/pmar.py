"""Graph-equivalent serializations for PMAR training.

This module is deliberately model-free. It takes a semantic location-free graph and:

1. refines same-class vertices with directed, predicate-aware 1-WL colour refinement;
2. assigns each stable refinement cell a canonical contiguous instance-id block;
3. enumerates or samples only the residual permutations inside ambiguous cells;
4. relabels the graph and deterministically sorts relations in autoregressive tuple order.

Crucially, annotation instance ids are used only as opaque node identities. They are never used
inside refinement signatures or to order refinement cells, so exact candidate sets are invariant to
same-class instance renaming and input edge order.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product
import math
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from patchsgg.graph_seq.linearize import Graph, Relation
from patchsgg.graph_seq.vocab import GraphVocab, VG_VOCAB


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

# A graph node is identified by:
#
#   (entity class id, annotation instance id)
#
# Example:
#
#   (1, 0) -> person instance 0
#   (1, 1) -> person instance 1
#
# IMPORTANT:
# The second number is treated only as an opaque identity.
# It must never influence the structural-refinement signature.
Node = Tuple[int, int]


# Maps an original graph node to its new PMAR instance index.
#
# Example:
#
#   {
#       (person_class, original_person_7): 0,
#       (person_class, original_person_3): 1,
#   }
Assignment = Dict[Node, int]


@dataclass(frozen=True)
class RefinementCell:
    """One stable same-class refinement cell.

    Attributes
    ----------
    class_id:
        Entity class shared by every node in this cell.

    color:
        Final structural-refinement colour.

    nodes:
        Original graph nodes belonging to this ambiguity cell.

    block:
        Canonical instance-id block assigned to this cell.

        For example, if a class has three structural cells of sizes:

            1, 2, 1

        then their blocks might be:

            (0,)
            (1, 2)
            (3,)

        Only permutations *inside* each block remain ambiguous.
    """

    class_id: int
    color: int
    nodes: Tuple[Node, ...]
    block: Tuple[int, ...]


@dataclass(frozen=True)
class PMARCandidates:
    """Candidate canonical graphs for one training example.

    Attributes
    ----------
    graphs:
        Candidate relation sequences after structural refinement,
        residual relabeling, and canonical relation sorting.

    mode:
        Either:

            "exact"

        or:

            "sampled"

    residual_permutation_count:
        Size of the residual permutation group before exact-mode
        serialization deduplication.

    num_refinement_cells:
        Number of stable structural cells in the graph.
    """

    graphs: Tuple[Tuple[Relation, ...], ...]
    mode: str
    residual_permutation_count: int
    num_refinement_cells: int

    @property
    def num_candidates(self) -> int:
        """Number of actual candidate graph serializations."""
        return len(self.graphs)


# ---------------------------------------------------------------------------
# Basic graph helpers
# ---------------------------------------------------------------------------


def _normalize_graph(
    graph: Sequence[Relation | Sequence[int]],
) -> Graph:
    """Convert graph entries into Relation objects.

    The rest of this module works with the project's Relation type.

    This helper also allows tests or callers to provide simple tuples/lists.
    """

    return [
        relation
        if isinstance(relation, Relation)
        else Relation(*map(int, relation))
        for relation in graph
    ]


def _nodes(
    graph: Graph,
) -> Tuple[Node, ...]:
    """Extract all unique graph nodes.

    Annotation IDs are used here only to distinguish node identities.

    They are NOT used as structural features.
    """

    nodes = {
        node
        for relation in graph
        for node in (
            (
                int(relation.subj_cls),
                int(relation.subj_inst),
            ),
            (
                int(relation.obj_cls),
                int(relation.obj_inst),
            ),
        )
    }

    # Sorting here is only an implementation convenience so Python iteration
    # is deterministic.
    #
    # The annotation instance id is NOT used inside the refinement signature.
    return tuple(sorted(nodes))


# ---------------------------------------------------------------------------
# Structural refinement
# ---------------------------------------------------------------------------


def structural_refine(
    graph: Sequence[Relation | Sequence[int]],
    vocab: GraphVocab = VG_VOCAB,
) -> Tuple[RefinementCell, ...]:
    """Perform directed, predicate-aware 1-WL-style structural refinement.

    Refinement begins using only the entity class:

        q_v^(0) = class(v)

    Each following signature contains:

        - entity class,
        - previous structural colour,
        - outgoing (predicate, neighbour-colour) multiset,
        - incoming (predicate, neighbour-colour) multiset.

    In conceptual form:

        signature(v) = (
            class(v),
            old_colour(v),
            sorted outgoing structure,
            sorted incoming structure,
        )

    Including the previous colour ensures refinement is monotonic:
    previously separated cells cannot merge again.

    Predicate labels, direction, and multiplicity are all preserved.

    Returns
    -------
    Tuple[RefinementCell, ...]
        Stable refinement cells with canonical class-local instance blocks.
    """

    normalized = _normalize_graph(graph)

    nodes = _nodes(normalized)

    # Empty scene graph.
    if not nodes:
        return tuple()

    # ---------------------------------------------------------------
    # Build directed adjacency information.
    # ---------------------------------------------------------------

    outgoing: Dict[
        Node,
        List[Tuple[int, Node]],
    ] = {
        node: []
        for node in nodes
    }

    incoming: Dict[
        Node,
        List[Tuple[int, Node]],
    ] = {
        node: []
        for node in nodes
    }

    for relation in normalized:

        subj = (
            int(relation.subj_cls),
            int(relation.subj_inst),
        )

        obj = (
            int(relation.obj_cls),
            int(relation.obj_inst),
        )

        predicate = int(
            relation.predicate
        )

        outgoing[subj].append(
            (
                predicate,
                obj,
            )
        )

        incoming[obj].append(
            (
                predicate,
                subj,
            )
        )

    # ---------------------------------------------------------------
    # Initial partition:
    #
    # nodes are distinguished only by entity class.
    # ---------------------------------------------------------------

    colors: Dict[Node, int] = {
        node: int(node[0])
        for node in nodes
    }

    # ---------------------------------------------------------------
    # Iterative structural refinement.
    #
    # A partition of |V| nodes can strictly refine only finitely many
    # times. len(nodes) + 1 is therefore a defensive bound.
    # ---------------------------------------------------------------

    for _ in range(
        len(nodes) + 1
    ):

        signatures = {}

        for node in nodes:

            # Directed outgoing structural signature.
            #
            # Example:
            #
            #   holding -> colour 5
            #   next_to -> colour 8
            #
            # Sorting gives deterministic multiset representation.
            # Duplicate tuples are retained, so multiplicity matters.
            out_sig = tuple(
                sorted(
                    (
                        pred,
                        colors[other],
                    )
                    for pred, other
                    in outgoing[node]
                )
            )

            # Directed incoming structural signature.
            in_sig = tuple(
                sorted(
                    (
                        pred,
                        colors[other],
                    )
                    for pred, other
                    in incoming[node]
                )
            )

            # CRITICAL:
            #
            # node[1], the original annotation instance id,
            # MUST NOT appear here.
            #
            # Otherwise arbitrary VG instance numbering would leak into
            # the structural partition and destroy permutation invariance.
            signatures[node] = (
                int(node[0]),
                int(colors[node]),
                out_sig,
                in_sig,
            )

        # -----------------------------------------------------------
        # Convert structural signatures into deterministic integer
        # colours.
        #
        # Sorting signatures means colour assignment is deterministic
        # and independent of dictionary iteration order.
        # -----------------------------------------------------------

        unique_signatures = sorted(
            set(
                signatures.values()
            )
        )

        signature_to_color = {
            signature: index
            for index, signature
            in enumerate(
                unique_signatures
            )
        }

        new_colors = {
            node: signature_to_color[
                signatures[node]
            ]
            for node in nodes
        }

        # -----------------------------------------------------------
        # Because the old colour is part of the new signature,
        # refinement cannot merge previously distinct cells.
        #
        # Therefore if the number of colours no longer increases,
        # the partition has stabilized.
        # -----------------------------------------------------------

        stable = (
            len(
                set(
                    new_colors.values()
                )
            )
            ==
            len(
                set(
                    colors.values()
                )
            )
        )

        colors = new_colors

        if stable:
            break

    else:
        # This should theoretically never happen because the partition
        # is monotonic and finite.
        raise RuntimeError(
            "structural refinement did not converge"
        )

    # ----------------------------------------------------------------
    # Group nodes by:
    #
    #   (entity class, final structural colour)
    #
    # The class condition is intentionally kept explicit because
    # PMAR permutations are class-local.
    # ----------------------------------------------------------------

    by_class_and_color: Dict[
        Tuple[int, int],
        List[Node],
    ] = {}

    for node in nodes:

        key = (
            int(node[0]),
            int(colors[node]),
        )

        by_class_and_color.setdefault(
            key,
            [],
        ).append(
            node
        )

    cells: List[
        RefinementCell
    ] = []

    classes = sorted(
        {
            class_id
            for class_id, _
            in by_class_and_color
        }
    )

    # ----------------------------------------------------------------
    # Assign contiguous canonical instance-id blocks separately
    # inside each entity class.
    #
    # Example:
    #
    # person structural cells:
    #
    #   cell A size 1 -> block (0,)
    #   cell B size 2 -> block (1, 2)
    #   cell C size 1 -> block (3,)
    #
    # Only the internal bijection of cell B remains ambiguous.
    # ----------------------------------------------------------------

    for class_id in classes:

        class_keys = sorted(
            (
                key
                for key
                in by_class_and_color
                if key[0]
                == class_id
            ),
            key=lambda key: key[1],
        )

        next_index = 0

        for _, color in class_keys:

            # Sorting these nodes is only used to provide a stable
            # container ordering for enumeration.
            #
            # Exact candidate sets remain invariant because every
            # possible permutation inside the cell is considered.
            cell_nodes = tuple(
                sorted(
                    by_class_and_color[
                        (
                            class_id,
                            color,
                        )
                    ]
                )
            )

            size = len(
                cell_nodes
            )

            # Instance IDs occupy:
            #
            #   0 ... vocab.max_instance_id - 1
            #
            # hence the number of nodes required must not exceed
            # vocab.max_instance_id.
            if (
                next_index + size
                >
                vocab.max_instance_id
            ):
                raise ValueError(
                    f"class {class_id} needs "
                    f"{next_index + size} instance ids, "
                    f"but vocab.max_instance_id="
                    f"{vocab.max_instance_id}"
                )

            block = tuple(
                range(
                    next_index,
                    next_index + size,
                )
            )

            cells.append(
                RefinementCell(
                    class_id=class_id,
                    color=color,
                    nodes=cell_nodes,
                    block=block,
                )
            )

            next_index += size

    return tuple(
        cells
    )


# ---------------------------------------------------------------------------
# Residual permutation space
# ---------------------------------------------------------------------------


def residual_permutation_count(
    cells: Sequence[
        RefinementCell
    ],
) -> int:
    """Calculate the residual permutation-group size.

    If the stable cells have sizes:

        n_1, n_2, ..., n_m

    then:

        M_residual =
            n_1! * n_2! * ... * n_m!

    Singleton cells contribute:

        1! = 1.
    """

    count = 1

    for cell in cells:

        count *= math.factorial(
            len(
                cell.nodes
            )
        )

    return count


def _assignment_from_choices(
    cells: Sequence[
        RefinementCell
    ],
    choices: Sequence[
        Sequence[int]
    ],
) -> Assignment:
    """Construct one node->instance assignment from cell choices."""

    assignment: Assignment = {}

    for cell, ids in zip(
        cells,
        choices,
    ):

        if (
            len(ids)
            !=
            len(cell.nodes)
        ):
            raise ValueError(
                "assignment choice has "
                "the wrong cell size"
            )

        for node, new_instance in zip(
            cell.nodes,
            ids,
        ):

            assignment[node] = int(
                new_instance
            )

    return assignment


def enumerate_residual_assignments(
    cells: Sequence[
        RefinementCell
    ],
) -> Iterable[
    Assignment
]:
    """Enumerate every residual assignment exactly.

    For one cell:

        nodes = (A, B)
        block = (1, 2)

    possible assignments are:

        A -> 1, B -> 2
        A -> 2, B -> 1

    For multiple cells, their permutation spaces are combined using
    a Cartesian product.
    """

    per_cell = [
        tuple(
            permutations(
                cell.block
            )
        )
        for cell in cells
    ]

    for choices in product(
        *per_cell
    ):

        yield _assignment_from_choices(
            cells,
            choices,
        )


def sample_residual_assignments(
    cells: Sequence[
        RefinementCell
    ],
    num_samples: int,
    rng: np.random.Generator,
) -> Iterable[
    Assignment
]:
    """Draw IID uniform residual assignments.

    Unlike exact mode, duplicate assignments are intentionally retained.

    This means if the same assignment happens to be sampled multiple
    times, all of those draws remain in the sampled PMAR approximation.
    """

    if num_samples <= 0:
        raise ValueError(
            f"num_samples must be positive, "
            f"got {num_samples}"
        )

    for _ in range(
        int(num_samples)
    ):

        choices = []

        for cell in cells:

            ids = np.asarray(
                cell.block,
                dtype=np.int64,
            )

            shuffled = (
                rng.permutation(
                    ids
                )
            )

            choices.append(
                tuple(
                    int(x)
                    for x
                    in shuffled
                )
            )

        yield _assignment_from_choices(
            cells,
            choices,
        )


# ---------------------------------------------------------------------------
# Canonical graph serialization
# ---------------------------------------------------------------------------


def relabel_and_canonical_sort(
    graph: Sequence[
        Relation | Sequence[int]
    ],
    assignment: Mapping[
        Node,
        int,
    ],
) -> Graph:
    """Relabel graph instance IDs and canonical-sort its relations.

    IMPORTANT:

    Relation itself is semantically stored as:

        (
            subj_cls,
            subj_inst,
            predicate,
            obj_cls,
            obj_inst,
        )

    But the project's autoregressive sequence order is:

        (
            subj_cls,
            subj_inst,
            obj_cls,
            obj_inst,
            predicate,
        )

    PMAR canonicalization must therefore explicitly sort according
    to the autoregressive order rather than relying on Python's default
    NamedTuple sorting.
    """

    normalized = _normalize_graph(
        graph
    )

    relabeled: Graph = []

    for relation in normalized:

        subj = (
            int(
                relation.subj_cls
            ),
            int(
                relation.subj_inst
            ),
        )

        obj = (
            int(
                relation.obj_cls
            ),
            int(
                relation.obj_inst
            ),
        )

        if (
            subj not in assignment
            or obj not in assignment
        ):
            raise KeyError(
                "assignment does not cover "
                "every graph node"
            )

        relabeled.append(
            Relation(
                subj_cls=int(
                    relation.subj_cls
                ),
                subj_inst=int(
                    assignment[subj]
                ),
                predicate=int(
                    relation.predicate
                ),
                obj_cls=int(
                    relation.obj_cls
                ),
                obj_inst=int(
                    assignment[obj]
                ),
            )
        )

    # ---------------------------------------------------------------
    # Canonical relation order.
    #
    # The autoregressive tuple order is:
    #
    #   subject class
    #   subject instance
    #   object class
    #   object instance
    #   predicate
    #
    # Predicate intentionally comes last.
    # ---------------------------------------------------------------

    relabeled.sort(
        key=lambda relation: (
            int(
                relation.subj_cls
            ),
            int(
                relation.subj_inst
            ),
            int(
                relation.obj_cls
            ),
            int(
                relation.obj_inst
            ),
            int(
                relation.predicate
            ),
        )
    )

    return relabeled


# ---------------------------------------------------------------------------
# Top-level PMAR candidate construction
# ---------------------------------------------------------------------------


def build_pmar_candidates(
    graph: Sequence[
        Relation | Sequence[int]
    ],
    vocab: GraphVocab = VG_VOCAB,
    *,
    exact_threshold: int = 64,
    num_samples: int = 8,
    rng: np.random.Generator | None = None,
) -> PMARCandidates:
    """Build exact or sampled PMAR candidate serializations.

    Procedure
    ---------
    1. Perform structural refinement.
    2. Calculate residual permutation count.
    3. If count <= exact_threshold:
           enumerate every residual assignment.
       Otherwise:
           sample `num_samples` IID assignments.
    4. Relabel each graph.
    5. Canonically sort relations.
    6. In exact mode only, deduplicate identical serializations.

    Parameters
    ----------
    graph:
        Semantic graph for one training example.

    vocab:
        Project GraphVocab.

    exact_threshold:
        Maximum residual permutation count that will be enumerated
        exactly.

        Example:

            exact_threshold = 64

        means:

            M_residual <= 64
                -> exact enumeration

            M_residual > 64
                -> sampled approximation

    num_samples:
        Number of IID residual assignments sampled when exact
        enumeration is too expensive.

    rng:
        NumPy random number generator.

        A persistent generator should normally be supplied by PMARLoss
        so sampled training can be reproducible and checkpointable.

    Returns
    -------
    PMARCandidates
    """

    if exact_threshold < 1:
        raise ValueError(
            f"exact_threshold must be >= 1, "
            f"got {exact_threshold}"
        )

    if num_samples < 1:
        raise ValueError(
            f"num_samples must be >= 1, "
            f"got {num_samples}"
        )

    normalized = _normalize_graph(
        graph
    )

    # ---------------------------------------------------------------
    # Structural refinement determines which nodes are still
    # genuinely ambiguous.
    # ---------------------------------------------------------------

    cells = structural_refine(
        normalized,
        vocab,
    )

    # ---------------------------------------------------------------
    # Number of remaining class-local bijections:
    #
    #   product over cells of |cell|!
    # ---------------------------------------------------------------

    total = residual_permutation_count(
        cells
    )

    # ---------------------------------------------------------------
    # Exact PMAR
    # ---------------------------------------------------------------

    if total <= int(
        exact_threshold
    ):

        mode = "exact"

        assignments = (
            enumerate_residual_assignments(
                cells
            )
        )

        # Different residual assignments can occasionally serialize
        # into the exact same canonical sequence.
        #
        # Exact graph likelihood should count each unique serialized
        # representation once.
        unique: Dict[
            Tuple[Relation, ...],
            None,
        ] = {}

        for assignment in assignments:

            candidate = tuple(
                relabel_and_canonical_sort(
                    normalized,
                    assignment,
                )
            )

            unique.setdefault(
                candidate,
                None,
            )

        candidate_graphs = tuple(
            unique.keys()
        )

    # ---------------------------------------------------------------
    # Sampled PMAR
    # ---------------------------------------------------------------

    else:

        mode = "sampled"

        if rng is None:
            rng = (
                np.random.default_rng()
            )

        # IMPORTANT:
        #
        # Sampled duplicates are NOT deduplicated.
        #
        # Each item represents one IID sample from the residual
        # permutation distribution.
        candidate_graphs = tuple(
            tuple(
                relabel_and_canonical_sort(
                    normalized,
                    assignment,
                )
            )
            for assignment
            in sample_residual_assignments(
                cells,
                num_samples,
                rng,
            )
        )

    # ---------------------------------------------------------------
    # Empty graph case.
    #
    # An empty graph still corresponds to one valid autoregressive
    # sequence:
    #
    #   START -> EOS
    #
    # build_train_pair(..., pad_to_max=False) will handle the actual
    # START/EOS token construction later.
    # ---------------------------------------------------------------

    if not candidate_graphs:

        candidate_graphs = (
            tuple(),
        )

    return PMARCandidates(
        graphs=candidate_graphs,
        mode=mode,
        residual_permutation_count=total,
        num_refinement_cells=len(
            cells
        ),
    )