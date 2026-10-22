# GeoRCP-RVO-ME

Code, patient-disjoint fold assignments, frozen protocols, and summary results for:

> H. A. Ahmed. *Geometry-Aware Multi-Task Learning for Heterogeneous Retinal OCT Biomarkers in Retinal Vein Occlusion: A Patient-Disjoint Evaluation.* Manuscript submitted for publication.

GeoRCP represents optical coherence tomography (OCT) biomarkers from the public **RVO-ME** dataset in their native geometry:

| Biomarker | Geometry | Prediction branch |
|---|---|---|
| Intraretinal fluid (IRF), subretinal fluid (SRF) | 2D regions | region branch |
| External limiting membrane (ELM), ellipsoid zone (EZ) | 1D curves | curve branch |
| Hyperreflective foci (HF) | 0D points | point branch (CenterNet-style) |

All branches share a pretrained **MIRAGE-Base** retinal encoder. Every model is evaluated with a five-fold **patient-disjoint** protocol (130 patients, 3,012 B-scans), because the official RVO-ME train/test split places the same patients (IDs 8 and 61) and one eye (Eye 72) in both training and test sets.

---

## Repository contents

```
project/
├── scripts/                              GeoRCP-MIRAGE, MIRAGE-Semantic, GeoRCP-A2, evaluation, figures
│   └── data_preparation_and_unetpp/      dataset audit, fold creation, target generation, U-Net++ baseline
├── splits/patient_5fold_v1/              patient-to-fold assignments (train / val / test lists per fold)
├── baselines/segmentation/
│   ├── patient5fold_protocol_v1/         canonical image- and patient-level roles for every fold
│   ├── trackB_conventional_v1/           U-Net++ training protocol and five-fold aggregated results
│   └── mirage_semantic_384x576_v1/       MIRAGE-Semantic protocol and training manifest
├── training/georcp_v1/                   GeoRCP-A2 (no task interaction) protocol and training manifest
├── design/geometry_aware_rcp_v1/         frozen scientific design, targets, losses, evaluation, ablation plan
├── preprocessing/near_native_576x384_v1/ near-native padding protocol (570x380 -> 576x384)
├── preflight/step07_georcp_v1/           implementation specification
└── evaluation/                           held-out summary metrics and paired patient-level bootstrap results
```

**Not included:** the RVO-ME images and annotations (please obtain them from the original source), generated target arrays, MIRAGE weights, and trained checkpoints.

---

## Requirements

The experiments were run with Python 3.8.20, PyTorch 2.4.1 (CUDA 12.1), and an NVIDIA GeForce RTX 4080 (16 GB).

```bash
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA version from <https://pytorch.org>.

## Data

1. Download RVO-ME from Figshare: <https://doi.org/10.6084/m9.figshare.29804435.v1>
2. Place it so that the folder layout is:

```
<ROOT>/
├── RVO-Lesion/
│   ├── RVO_Lesion_Labelme/      original LabelMe vector annotations (used for GeoRCP targets)
│   └── Image_Seg/               released images and semantic masks (used for U-Net++ and MIRAGE-Semantic)
└── project/                     this repository's project/ folder
```

## MIRAGE encoder

- Weights: `j-morano/MIRAGE-Base` on Hugging Face (downloaded automatically on first use).
- Wrapper: download `hf/mirage_hf.py` from the official MIRAGE repository (<https://github.com/j-morano/MIRAGE>) and save it as
  `project/foundation_models/mirage_base_preflight_v1/official_mirage_hf.py`.
  `data_preparation_and_unetpp/05E3_mirage_base_preflight.py` shows how it was fetched and checked.

## Paths and integrity checks

The scripts are released **exactly as they were run**, so that their SHA-256 fingerprints match the frozen protocol records in this repository. Consequently:

- Each script defines its root folder near the top (`ROOT = Path(r"D:\RVO_ME")` or `Path(r"E:\Papers\RVO_ME")`, because the project moved between two machines). Set `ROOT` to your own `<ROOT>` before running.
- Several scripts verify the fingerprints of frozen files and of earlier scripts. For example, Step 12 checks the Step 09 script, and the Fold 2–5 training script checks the Fold 1 completion record. If you edit a script, update the corresponding expected fingerprint constants.

---

## Pipeline

Run the scripts in this order. The fold assignments and canonical roles are already provided, so stage A is only needed to reproduce the splits from scratch.

| Stage | Script(s) | Purpose |
|---|---|---|
| **A. Audit and splits** | `data_preparation_and_unetpp/audit_rvo_me.py`, `audit02_integrity.py` | Dataset audit, including the patient/eye overlap in the official split |
| | `data_preparation_and_unetpp/01_create_patient_folds.py` | Stratified five-fold patient-disjoint assignment |
| | `data_preparation_and_unetpp/05A_freeze_patient5fold_protocol.py` | Canonical train/validation/test roles for every fold |
| **B. U-Net++ baseline** | `data_preparation_and_unetpp/02_generate_geometry_targets.py` | Reference targets required by the U-Net++ integrity checks |
| | `05B_train_unetpp_trackB_fold1.py` … `fold5_CORRECTED.py` | U-Net++ training per fold (256x256 inputs; scored at 256x256 and 570x380) |
| | `05D_aggregate_unetpp_trackB_5fold.py` | Pooled out-of-fold metrics and patient-level bootstrap CIs |
| **C. GeoRCP preparation** | `data_preparation_and_unetpp/05E2_freeze_near_native_576x384.py` | Near-native padding protocol |
| | `data_preparation_and_unetpp/05E3_mirage_base_preflight.py` | MIRAGE-Base download and check |
| | `data_preparation_and_unetpp/05E5A_freeze_georcp_mirage_scientific_design.py` | Freezes architecture, targets, losses, evaluation, and ablation plan |
| | `06_build_geometry_v2.py` | Region, curve, and point targets from the vector annotations |
| | `07_georcp_preflight_v2_ampfix_py38compat.py` | GeoRCP-MIRAGE implementation (imported by Steps 08–12); creates the corrected `geometry_v3` HF targets |
| **D. GeoRCP-MIRAGE training** | `08_georcp_definitive_fold1_v3_noaug.py` | Fold 1 (no augmentation) |
| | `08_georcp_final_train_folds2to5.py` | Folds 2–5 under the same protocol |
| | `08_georcp_fold1_photoaug_v4.py` | Predefined Fold 1 photometric-augmentation experiment (not adopted) |
| **E. Held-out evaluation** | `09_georcp_calibrate_and_evaluate.py` | Run 1: validation-only calibration and protocol freeze. Run 2: one-time evaluation of all five held-out folds |
| **F. Controlled attribution** | `10_mirage_semantic_baseline_train_resume_safe_v3.py` | MIRAGE-Semantic baseline (same encoder, semantic decoder) |
| | `11_georcp_A2_nointeraction_train_resume.py` | GeoRCP-A2 (without the TaskInteraction module) |
| | `12_attribution_calibrate_and_evaluate.py` | Run 1: validation-only calibration and protocol freeze. Run 2: held-out evaluation and paired patient-level bootstrap |
| **G. Figures** | `generate_figure1_rvome_v2.py`, `generate_figure2_georcp_mirage_v3.py`, `Generate_figure3_georcp_qualitative_final.py` | Figures 1–3 |

Outer-test patients are never used for training, early stopping, threshold calibration, or model selection.

## Results files

| File | Content |
|---|---|
| `project/evaluation/georcp_v1/trackB_fivefold_v1/heldout_summary.csv` | Pooled held-out metrics of the full GeoRCP-MIRAGE model |
| `project/evaluation/attribution_v1/attribution_summary.csv` | Pooled held-out metrics of MIRAGE-Semantic (B2) and GeoRCP-A2 |
| `project/evaluation/attribution_v1/paired_patient_bootstrap.csv` | Paired patient-level bootstrap comparisons (2,000 resamples) |
| `project/baselines/segmentation/trackB_conventional_v1/unetpp/aggregation_5fold_v1/` | U-Net++ results at 256x256 and at the native 570x380 resolution |

## Acknowledgments

This work uses the RVO-ME dataset (Xiong et al., *Scientific Data*, 2026) and the MIRAGE foundation model (Morano et al., *npj Digital Medicine*, 2025). Please cite these works if you use this repository.

## Citation

Citation details will be added after publication.

## License

Code in this repository is released under the MIT License (see `LICENSE`). The RVO-ME dataset and MIRAGE are subject to their own licenses.
