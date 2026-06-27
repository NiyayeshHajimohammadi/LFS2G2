"""Instance matcher used at evaluation (and optionally by matching-CE).

Wraps LF-SGG's compiled branched matcher (``branched_ssg_matcher``, Cython/C++20). Build it with
``pip install -e .`` (needs a C++20 compiler) or ``python setup.py build_ext --inplace``.
``allow_identity_fallback`` is kept for the text->text diagnostic, where GT and predicted instance
ids already coincide and no extension is needed.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Dict, Optional, Sequence, Tuple

MatcherTuple = Tuple[int, int, int, int, int]


class InstanceMatcher:
    def __init__(self, n: int = 3, depth_limit: int = 10, allow_identity_fallback: bool = False):
        self.n = n
        self.depth_limit = depth_limit
        self.allow_identity_fallback = allow_identity_fallback
        self._impl = None
        self._load_error: Optional[Exception] = None
        try:
            from branched_ssg_matcher import PyBranchedSSGMatcher  # compiled extension

            self._impl = PyBranchedSSGMatcher()
        except Exception as exc:  # pragma: no cover - depends on compiled ext
            self._load_error = exc

    @property
    def available(self) -> bool:
        return self._impl is not None

    def match(
        self, gts: Sequence[MatcherTuple], preds: Sequence[MatcherTuple]
    ) -> Dict[Tuple[int, int], Optional[int]]:
        if self.allow_identity_fallback and self._impl is None:
            mapping: Dict[Tuple[int, int], Optional[int]] = {}
            for sub_id, sub_inst, _pred, obj_id, obj_inst in preds:
                mapping[(sub_id, sub_inst)] = sub_inst
                mapping[(obj_id, obj_inst)] = obj_inst
            return mapping
        if self._impl is None:
            raise RuntimeError(
                "branched_ssg_matcher extension not built. Run `pip install -e .` with a C++20 "
                "compiler (gcc>=11/clang), or set allow_identity_fallback=True for the text->text "
                f"diagnostic. Import error: {self._load_error}"
            )
        return self._impl.branched_matching(
            deepcopy([list(g) for g in gts]),
            deepcopy([list(p) for p in preds]),
            N=self.n,
            depth_limit=self.depth_limit,
        )
