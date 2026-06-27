# Attribution

PatchSGG reimplements / adapts ideas and a small amount of code from two upstream projects.
Please cite both if you use this work.

## Patch-ioner — "One Patch to Caption Them All"
- https://github.com/Ruggero1912/Patch-ioner
- `patchsgg/encoders/talk2dino.py` is adapted (kept weight-compatible) from its Talk2DINO
  `ProjectionLayer`.
- `patchsgg/bridge/decap_memory.py` is a focused reimplementation of its DeCap `Im2TxtProjector`
  (support-memory projection only).
- The DINOv2 patch-token usage mirrors its `dino_extraction` (we use DINOv2's native
  `forward_features` output directly).

## LF-SGG — Location-Free Scene Graph Generation (Pix2SG)
- https://github.com/egeozsoy/LF-SGG
- The 5-tuple vocabulary, sequence linearization, and recall metrics are reimplemented natively in
  `patchsgg/graph_seq` and `patchsgg/eval/metrics.py` (faithful ports).
- The branched instance matcher (`patchsgg/eval/branched/branched_ssg_matcher.{h,pyx,pxd}`) is
  LF-SGG's original Cython/C++20 code, built as the `branched_ssg_matcher` extension. The `.pxd`
  cimports were rejoined onto single lines (upstream had stray line breaks); semantics unchanged.
