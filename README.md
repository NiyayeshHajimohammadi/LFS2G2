# PatchSGG — Location-Free Zero-Shot Scene Graph Generation

Extends **Patch-ioner** (DeCap-style text-only-trained autoregressive decoder over frozen vision
encoders) to the structured-output setting of **LF-SGG** (Pix2Seq-style 5-tuple scene graph
generation without bounding boxes).

**Idea.** Train an AR decoder to emit a linearized scene graph
(`⟨subj_class, subj_instance, predicate, obj_class, obj_instance⟩` 5-tuples) conditioned on a *text*
view of a Visual Genome graph; at inference, condition on a *DINOv2 image* instead. A shared
Talk2DINO space bridges the text→image modality gap.

The codebase is built to *experiment*: every research axis is a config switch.

| Axis | Options (config) |
|---|---|
| conditioning | `pooled` · **`patch_set`** (cross-attn, default) · `resampler` |
| loss | `ce` · `ce_weighted` · **`matching_ce`** (instance-permutation-invariant) · `order_agnostic` |
| bridge | `identity` · `capdec_noise` · `decap_projection` |
| shared space | **`dinov2_talk2dino`** (default) · `clip` |
| text view | `serialize` · `llm_caption` · `mix` |
| modality | train `{text,image}` × eval `{text,image}` (enables text→text diagnostic) |

## Layout
```
patchsgg/
  encoders/    text & image encoders -> unified ConditioningSet
  bridge/      modality-gap bridges (composable)
  graph_seq/   token vocabulary + graph<->sequence linearization
  decoder/     AR graph decoders (cross-attn default, prefix baseline) + constrained sampling
  losses/      ce / weighted / matching / order-agnostic
  data/        VG dataset wrapper, graph text views, collation
  eval/        recall metrics + LF-SGG branched matcher (Cython/C++20, compiled)
  encoders/    text/image/toy encoders + Talk2DINO projection
  lightning_module.py (SGGLightning + DataModule)  train.py  infer.py
  configs/     base + presets/ablations
  tools/       llm_graph_to_caption.py
tests/
```

Training uses **PyTorch Lightning**. Evaluation uses LF-SGG's original branched instance matcher
(Cython/C++20), built as the `branched_ssg_matcher` extension at install time. The Talk2DINO
projection and DeCap memory projection are vendored/reimplemented in-package; see `NOTICE.md`.

## Setup (Linux training box)
```bash
pip install -e .[encoders,dev]    # compiles branched_ssg_matcher (needs gcc>=11 / clang, C++20)
# DINOv2 loads via torch.hub; install OpenAI CLIP for the clip/dinov2_talk2dino encoders:
pip install git+https://github.com/openai/CLIP.git
# Place Talk2DINO weights/config under checkpoints/talk2dino/ (see configs/base.yaml).
# If the extension didn't build: python setup.py build_ext --inplace
```

## Quickstart
```bash
# 1) sanity: graph<->sequence round-trip + metrics
pytest -q

# 2) diagnostic: train on text, eval on text (measures the information-bottleneck ceiling)
patchsgg-train --config patchsgg/configs/diagnostic_text2text.yaml

# 3) the real task: train on text, eval on image
patchsgg-train --config patchsgg/configs/default_crossattn_talk2dino.yaml
patchsgg-infer --config patchsgg/configs/default_crossattn_talk2dino.yaml --ckpt outputs/last.ckpt
```

See `NOTICE.md` for upstream attribution.
