# GeoRCP-MIRAGE v1 — Frozen Scientific Design

## Status

Scientific architecture, targets, loss families, evaluation rules, and
ablation plan are frozen before definitive proposed-model outer-test runs.

Physical batch size and final optimization engineering remain pending the
RTX 4080 profile.

## Input

- Native RVO-ME raster: 570 x 380
- Model canvas: 576 x 384
- Padding only: L3/R3/T2/B2
- No resize or interpolation
- Near-native SHA: `ac5ce015e1a05a3dc47f054c76f8706d42ac9a9c9506543cbf4ca8e621d53714`

## Primary architecture

**Global path:** official MIRAGE-Base pretrained retinal OCT foundation encoder.

Selected MIRAGE blocks: **3, 6, 9, 12**.

**Local detail path:** shallow CNN/ConvNeXt-style native-detail encoder at
stride 2 and stride 4.

**Task interaction:** three region/curve/point token streams interact through
a 2-layer, 8-head Geometry-Conditioned Task Interaction Transformer at the
12 x 18 patch grid.

**Decoders:**

- Region: background/SRF/IRF full-resolution logits + SRF/IRF boundary heads.
- Curve: ELM/EZ centerline heatmap, vertical distance field, column presence,
  and masked per-column y distribution.
- Point: HF Gaussian center heatmap + subpixel offset at stride 2 (192 x 288).

## Geometry decisions

### Region — SRF / IRF

2D region masks remain the primary representation. Raw region vertices are
preserved. The 25 native pixel-center-OOB vertices are only IRF/SRF vertices
that reach the continuous right image boundary; no HF/ELM/EZ annotations are
affected.

### Curve — ELM / EZ

The annotation audit found >99.9% x-monotonic valid linestrips with extremely
rare multi-level column ambiguity. Therefore a hybrid dense-field +
column-distribution representation is used.

Global connectivity is **not** forced. EZ can be truly ruptured/discontinuous,
so the model must preserve pathological gaps rather than bridge them.

### Point — HF

Stride 2 is frozen because it produces **zero cell collisions** for all 9,212
valid HF points. Stride 4 already merges 18 points.

## Loss strategy

Region:
- cross entropy + foreground Dice
- auxiliary boundary BCE + Dice

Curve:
- Gaussian centerline BCE
- masked vertical-distance SmoothL1
- column-presence BCE
- masked column-y distribution loss
- ELM-above-EZ ordering loss where both are valid
- **no global clDice/connectivity loss**

HF:
- CenterNet-style focal heatmap loss
- positive-center SmoothL1 offset loss

Region/curve/point task groups use learned homoscedastic uncertainty weighting.

## Evaluation

Same frozen patient-disjoint 5-fold roles for all models.

Primary curve metrics are geometric distances and column-presence accuracy;
Dice/IoU remain secondary comparability metrics.

HF primary metrics are point-localization metrics, FROC and count error.

Patient-level bootstrap confidence intervals will be used.

## Frozen identifiers

- Target protocol SHA: `74af7ecd608c54056c72eafefe3392375107f2f3fe9a43a50db45641e8c81ddd`
- Architecture protocol SHA: `7201fc90b534251d9395fda20c432eec3443e5860594a8c59e606465f27a3c21`
- Loss protocol SHA: `8e32825a24a978c27dadd8d924450b4a363ec1fe8b4433ea89f575b713b43f73`
- Evaluation protocol SHA: `1fba9f37467177f1363a0b10e5240fda78c569801a24c8546f17ed76fb8aaa4a`
- Ablation plan SHA: `e9c292325f9ad9413e11a2f805b5854e9779c733b7ed9dc221bf79575a158233`
- Pending RTX4080 engineering SHA: `e073f5b0017c568146bd71f8159eb2733f4effbe723ffe28d6f00c933a7b034e`

## Next step

Profile the exact frozen model on the RTX 4080, freeze the remaining
optimization/engineering settings, then generate geometry_v2 targets and begin
modern baseline/proposed-model training without using any outer-test result for
tuning.
