"""Flat token vocabulary for location-free 5-tuple scene graphs.

Layout follows LF-SGG (Pix2SG) so the vendored Cython ``BranchedSSGMatcher`` and the recall
metrics stay drop-in compatible. A single 5-tuple relation is

    canonical order (metrics / matcher): (subj_class, subj_instance, predicate, obj_class, obj_instance)
    sequence  order (generation/AR)    :  subj_class, subj_instance, obj_class, obj_instance, predicate

The two orders differ: LF-SGG generates *both entities first, predicate last*. Keep this
distinction explicit everywhere -- mixing them silently corrupts both training and evaluation.

The VG defaults below reproduce LF-SGG ``detr_configs.py`` exactly:
    PRED   tokens : [0, 51)            (51 predicates incl. __background__)
    ENTITY tokens : [51, 202)          (151 object classes)
    INSTANCE tokens: [202, 232)        (ASSUMED_MAX_INSTANCE_ID = 30)
    NOISE=240  END=242  START=243  NO_KNOWN=244     (vocab size = 245)
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class TokenType(str, Enum):
    """Token role at a given position in the flattened sequence.

    Drives constrained decoding: at each step only logits for the valid range are kept.
    """

    ENTITY = "entity"          # an object-class token (subject or object)
    INSTANCE = "instance"      # an instance-id token disambiguating same-class entities
    PREDICATE = "predicate"    # a relation/predicate token
    SPECIAL = "special"        # START / END / NOISE / NO_KNOWN


# Position (mod 5) -> role, for the *sequence* order: sub_cls, sub_inst, obj_cls, obj_inst, pred
_SEQ_ROLE = [
    TokenType.ENTITY,    # 0 subject class
    TokenType.INSTANCE,  # 1 subject instance
    TokenType.ENTITY,    # 2 object class
    TokenType.INSTANCE,  # 3 object instance
    TokenType.PREDICATE, # 4 predicate
]

TOKENS_PER_REL = 5


@dataclass(frozen=True)
class GraphVocab:
    """Configurable flat vocabulary. Defaults reproduce LF-SGG/VG.

    All ``*_start`` values are inclusive lower bounds; counts give the size of each block.
    Special tokens are placed after the instance block, leaving LF-SGG's exact gaps so the
    integer ids match the upstream matcher.
    """

    n_preds: int = 51
    n_entities: int = 151
    max_instance_id: int = 30          # ASSUMED_MAX_INSTANCE_ID
    random_max_instance_id: int = 10   # RANDOM_MAX_INSTANCE_ID (used for noise padding)
    max_num_rels: int = 55             # MAX_NUM_RELS

    pred_start: int = 0

    def __post_init__(self) -> None:
        for name in ("n_preds", "n_entities", "max_instance_id", "max_num_rels"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if not 0 < self.random_max_instance_id <= self.max_instance_id:
            raise ValueError(
                "random_max_instance_id must be in [1, max_instance_id], got "
                f"{self.random_max_instance_id} and {self.max_instance_id}"
            )
        if self.pred_start < 0:
            raise ValueError(f"pred_start must be non-negative, got {self.pred_start}")

    @property
    def entity_start(self) -> int:
        return self.pred_start + self.n_preds

    @property
    def instance_start(self) -> int:
        return self.entity_start + self.n_entities

    # Special tokens -- offsets relative to (instance_start + max_instance_id), per LF-SGG.
    @property
    def _spec_base(self) -> int:
        return self.instance_start + self.max_instance_id

    @property
    def noise_token(self) -> int:
        return self._spec_base + 8

    @property
    def end_token(self) -> int:
        return self._spec_base + 10

    @property
    def start_token(self) -> int:
        return self._spec_base + 11

    @property
    def no_known_token(self) -> int:
        return self._spec_base + 12

    @property
    def vocab_size(self) -> int:
        return self.no_known_token + 1

    # ---- range helpers -------------------------------------------------------------------
    @property
    def pred_range(self) -> Tuple[int, int]:
        return self.pred_start, self.pred_start + self.n_preds

    @property
    def entity_range(self) -> Tuple[int, int]:
        return self.entity_start, self.entity_start + self.n_entities

    @property
    def instance_range(self) -> Tuple[int, int]:
        return self.instance_start, self.instance_start + self.max_instance_id

    def role_at(self, position_in_seq: int) -> TokenType:
        """Role of the token at ``position_in_seq`` (0-based, *excluding* the START token)."""
        return _SEQ_ROLE[position_in_seq % TOKENS_PER_REL]

    def range_for_role(self, role: TokenType) -> Tuple[int, int]:
        if role is TokenType.ENTITY:
            return self.entity_range
        if role is TokenType.INSTANCE:
            return self.instance_range
        if role is TokenType.PREDICATE:
            return self.pred_range
        raise ValueError(f"No contiguous logit range for role {role}")

    # ---- (de)tokenizing single fields ----------------------------------------------------
    def entity_token(self, class_idx: int) -> int:
        if not 0 <= int(class_idx) < self.n_entities:
            raise ValueError(f"entity index outside [0, {self.n_entities}): {class_idx}")
        return self.entity_start + int(class_idx)

    def predicate_token(self, pred_idx: int) -> int:
        if not 0 <= int(pred_idx) < self.n_preds:
            raise ValueError(f"predicate index outside [0, {self.n_preds}): {pred_idx}")
        return self.pred_start + int(pred_idx)

    def instance_token(self, instance_idx: int) -> int:
        if not 0 <= int(instance_idx) < self.max_instance_id:
            raise ValueError(f"instance index outside [0, {self.max_instance_id}): {instance_idx}")
        return self.instance_start + int(instance_idx)

    def entity_idx(self, token: int) -> int:
        if not self.entity_range[0] <= int(token) < self.entity_range[1]:
            raise ValueError(f"token {token} is not an entity token")
        return int(token) - self.entity_start

    def predicate_idx(self, token: int) -> int:
        if not self.pred_range[0] <= int(token) < self.pred_range[1]:
            raise ValueError(f"token {token} is not a predicate token")
        return int(token) - self.pred_start

    def instance_idx(self, token: int) -> int:
        if not self.instance_range[0] <= int(token) < self.instance_range[1]:
            raise ValueError(f"token {token} is not an instance token")
        return int(token) - self.instance_start


# Module-level default used throughout the codebase / tests.
VG_VOCAB = GraphVocab()

# Problems:
# No input validation. Methods like entity_token(), predicate_token(), and instance_token() accept any integer. Passing -1 or an out-of-range value silently produces an invalid token. Adding assertions or raising ValueError would make debugging easier.
# Magic offsets (+8, +10, +11, +12). They correctly reproduce LF-SGG, but the rationale isn't encoded in the code itself. A named constant or a short explanation of the reserved gaps would improve maintainability.
# Fixed max_instance_id. This inherited design limits the number of distinguishable same-class instances. It's acceptable for compatibility but should be explicitly discussed as a limitation in the project or paper.
# SPECIAL is defined but not handled by range_for_role(). This is intentional because special tokens are not a contiguous prediction range during structured decoding, but a short comment in the method would prevent confusion for future contributors.

#My comment: this component answers to the quesion "What does each integer token mean?"
