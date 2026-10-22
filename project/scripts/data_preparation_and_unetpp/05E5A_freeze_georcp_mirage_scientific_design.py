# =============================================================================
# RVO-ME STEP 05E5A
# Freeze the scientific architecture, target representations, loss families,
# evaluation rules, and ablation plan for the proposed geometry-aware model.
#
# IMPORTANT
# ---------
# This freezes SCIENTIFIC DESIGN choices before definitive training.
#
# It intentionally DOES NOT freeze:
#   - physical batch size
#   - gradient accumulation
#   - final learning-rate schedule
#   - gradient checkpointing
#   - AMP dtype
#
# Those engineering settings will be profiled on the RTX 4080 in Step 05E5B.
#
# No model training is performed.
# No frozen dataset asset is modified.
# =============================================================================

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(r"E:\Papers\RVO_ME")

PROJECT = ROOT / "project"

OUTPUT_DIR = (
    PROJECT
    / "design"
    / "geometry_aware_rcp_v1"
)

NEAR_NATIVE_JSON = (
    PROJECT
    / "preprocessing"
    / "near_native_576x384_v1"
    / "NEAR_NATIVE_PROTOCOL_FROZEN.json"
)

E4A_JSON = (
    PROJECT
    / "audits"
    / "step05e4a_geometry_representation_v1"
    / "STEP05E4A_COMPLETE.json"
)

E4C_JSON = (
    PROJECT
    / "audits"
    / "step05e4c_hf_resolution_v1"
    / "STEP05E4C_COMPLETE.json"
)

E4D_JSON = (
    PROJECT
    / "audits"
    / "step05e4d_native_oob_coordinates_v1"
    / "STEP05E4D_COMPLETE.json"
)

E4B_JSON = (
    PROJECT
    / "foundation_models"
    / "mirage_rcp_prototype_v1"
    / "STEP05E4B_COMPLETE.json"
)

UNETPP_5FOLD_JSON = (
    PROJECT
    / "baselines"
    / "segmentation"
    / "trackB_conventional_v1"
    / "unetpp"
    / "aggregation_5fold_v1"
    / "STEP05D_COMPLETE.json"
)


# =============================================================================
# FROZEN IDENTIFIERS
# =============================================================================

EXPECTED_NEAR_NATIVE_SHA = (
    "ac5ce015e1a05a3dc47f054c76f8706d42ac9a9c9506543cbf4ca8e621d53714"
)

EXPECTED_GEOMETRY_V1_SHA = (
    "a4029d5ab4570237ae607d1cc24d551ce438845eab1814e8a82735a122daeb55"
)

EXPECTED_PATIENT_PROTOCOL_SHA = (
    "52ed516dcc21f1c51068c59fb662c9a0030b143e6440d6f4a0380df713c9bcaf"
)

EXPECTED_UNETPP_AGG_SHA = (
    "944ee1620967920c19709c56ff172d32d756285b1ec971c4d3196fe9d5498051"
)

EXPECTED_MIRAGE_WRAPPER_SHA = (
    "07fa63e2766b67dbc0c07f0d9d29db1562f707425ef2fbf90a688f62a94e6ede"
)

EXPECTED_HF_POINTS = 9212
EXPECTED_HF_POSITIVE_IMAGES = 2286
EXPECTED_HF_STRIDE = 2

EXPECTED_ELM_VALID = 2384
EXPECTED_ELM_IMAGES = 1625

EXPECTED_EZ_VALID = 4474
EXPECTED_EZ_IMAGES = 2730


# =============================================================================
# HELPERS
# =============================================================================

def load_json(path: Path) -> Any:

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def save_json(
    path: Path,
    obj: Any,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            obj,
            f,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )


def stable_sha(
    obj: Any,
) -> str:

    blob = json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(
        blob
    ).hexdigest()


# =============================================================================
# VERIFY INPUT AUDITS
# =============================================================================

def verify_inputs():

    near = load_json(
        NEAR_NATIVE_JSON
    )

    e4a = load_json(
        E4A_JSON
    )

    e4c = load_json(
        E4C_JSON
    )

    e4d = load_json(
        E4D_JSON
    )

    e4b = load_json(
        E4B_JSON
    )

    unetpp = load_json(
        UNETPP_5FOLD_JSON
    )

    # Near-native
    near_sha = str(
        near.get(
            "near_native_protocol_sha256",
            "",
        )
    )

    if near_sha != EXPECTED_NEAR_NATIVE_SHA:

        raise RuntimeError(
            "Near-native protocol SHA mismatch."
        )

    # Geometry audit
    if e4a.get(
        "status"
    ) != "COMPLETE":

        raise RuntimeError(
            "05E4A is not COMPLETE."
        )

    curves = e4a[
        "curve_summary"
    ]

    if int(
        curves["ELM"][
            "valid_annotations"
        ]
    ) != EXPECTED_ELM_VALID:

        raise RuntimeError(
            "ELM count mismatch."
        )

    if int(
        curves["ELM"][
            "positive_images"
        ]
    ) != EXPECTED_ELM_IMAGES:

        raise RuntimeError(
            "ELM positive-image mismatch."
        )

    if int(
        curves["EZ"][
            "valid_annotations"
        ]
    ) != EXPECTED_EZ_VALID:

        raise RuntimeError(
            "EZ count mismatch."
        )

    if int(
        curves["EZ"][
            "positive_images"
        ]
    ) != EXPECTED_EZ_IMAGES:

        raise RuntimeError(
            "EZ positive-image mismatch."
        )

    # HF stride audit
    if int(
        e4c[
            "valid_hf_points"
        ]
    ) != EXPECTED_HF_POINTS:

        raise RuntimeError(
            "HF point count mismatch."
        )

    if int(
        e4c[
            "positive_images"
        ]
    ) != EXPECTED_HF_POSITIVE_IMAGES:

        raise RuntimeError(
            "HF positive-image mismatch."
        )

    found_stride = int(
        e4c[
            "recommendation"
        ][
            "recommended_stride"
        ]
    )

    if found_stride != EXPECTED_HF_STRIDE:

        raise RuntimeError(
            "HF recommended stride mismatch."
        )

    # OOB audit
    if int(
        e4d[
            "native_oob_vertices"
        ]
    ) != 25:

        raise RuntimeError(
            "Native-OOB count mismatch."
        )

    oob_rec = e4d[
        "recommendation"
    ]

    if int(
        oob_rec[
            "hf_oob_count"
        ]
    ) != 0:

        raise RuntimeError(
            "Unexpected HF OOB point found."
        )

    labels_oob = (
        oob_rec[
            "labels_with_oob_vertices"
        ]
    )

    if set(
        labels_oob
    ) - {
        "IRF",
        "SRF",
    }:

        raise RuntimeError(
            "Unexpected non-region OOB label."
        )

    # Prototype
    if e4b.get(
        "status"
    ) != "COMPLETE":

        raise RuntimeError(
            "05E4B prototype is not COMPLETE."
        )

    if (
        e4b[
            "full_finetune_batch2"
        ][
            "status"
        ]
        != "PASS"
    ):

        raise RuntimeError(
            "05E4B full-finetune batch-2 "
            "prototype did not pass."
        )

    if (
        str(
            e4b[
                "official_mirage_wrapper_sha256"
            ]
        )
        != EXPECTED_MIRAGE_WRAPPER_SHA
    ):

        raise RuntimeError(
            "MIRAGE wrapper SHA mismatch."
        )

    # U-Net++ frozen baseline
    if (
        str(
            unetpp[
                "aggregation_summary_sha256"
            ]
        )
        != EXPECTED_UNETPP_AGG_SHA
    ):

        raise RuntimeError(
            "Frozen U-Net++ aggregation SHA mismatch."
        )

    return {
        "near_native":
            near,

        "geometry_audit":
            e4a,

        "hf_stride_audit":
            e4c,

        "oob_audit":
            e4d,

        "prototype":
            e4b,

        "unetpp":
            unetpp,
    }


# =============================================================================
# FROZEN TARGET PROTOCOL
# =============================================================================

def make_target_protocol(
    inputs: Dict[str, Any],
) -> Dict[str, Any]:

    target = {
        "name":
            "RVO-ME Geometry-Aware Target Protocol v2",

        "version":
            2,

        "status":
            "FROZEN",

        "authoritative_source":
            (
                "Raw LabelMe float geometry for "
                "IRF/SRF/ELM/EZ/HF"
            ),

        "conventional_mask_source":
            (
                "Released Image_Seg masks retained only "
                "for conventional semantic-segmentation "
                "benchmark reproduction/comparison"
            ),

        "canvas": {
            "native_wh":
                [
                    570,
                    380,
                ],

            "model_wh":
                [
                    576,
                    384,
                ],

            "padding_lrtb":
                [
                    3,
                    3,
                    2,
                    2,
                ],

            "resize":
                False,

            "interpolation":
                None,

            "near_native_protocol_sha256":
                EXPECTED_NEAR_NATIVE_SHA,
        },

        # ---------------------------------------------------------------------
        # Regions
        # ---------------------------------------------------------------------

        "regions": {
            "labels":
                [
                    "SRF",
                    "IRF",
                ],

            "geometry_dimension":
                "2D",

            "primary_target":
                (
                    "two binary region masks rasterized "
                    "from valid raw geometry on 576x384"
                ),

            "semantic_relation":
                (
                    "The final region classifier is mutually "
                    "exclusive at inference: background/SRF/IRF."
                ),

            "auxiliary_target":
                (
                    "one-pixel class-specific region boundary "
                    "maps derived from the padded region masks"
                ),

            "rasterization":
                {
                    "polygon":
                        "closed filled region",

                    "linestrip_with_at_least_3_points":
                        (
                            "closed filled region, matching "
                            "validated geometry_v1 policy"
                        ),

                    "malformed_shape_policy":
                        (
                            "exclude malformed individual "
                            "annotation only; retain image"
                        ),
                },

            "validity_masks": {
                "region_interior_loss":
                    (
                        "native 570x380 field of view only"
                    ),

                "region_boundary_auxiliary_loss":
                    (
                        "native FOV plus annotation-supported "
                        "right-boundary padding where a raw "
                        "IRF/SRF vertex lies exactly on the "
                        "continuous x=570 image boundary"
                    ),
            },

            "oob_annotation_policy":
                {
                    "observed_vertices":
                        25,

                    "labels":
                        {
                            "IRF":
                                24,

                            "SRF":
                                1,
                        },

                    "interpretation":
                        (
                            "All are region vertices. Maximum "
                            "right-boundary excursion is 0; "
                            "they reach the continuous x=570 "
                            "boundary but do not extend beyond it."
                        ),

                    "preserve_raw_float_vertices":
                        True,
                },
        },

        # ---------------------------------------------------------------------
        # Curves
        # ---------------------------------------------------------------------

        "curves": {
            "labels":
                [
                    "ELM",
                    "EZ",
                ],

            "geometry_dimension":
                "1D",

            "valid_annotation_type":
                "linestrip with >=2 finite points",

            "data_support": {
                "ELM_valid_annotations":
                    EXPECTED_ELM_VALID,

                "ELM_positive_images":
                    EXPECTED_ELM_IMAGES,

                "ELM_x_monotonic_fraction":
                    float(
                        inputs[
                            "geometry_audit"
                        ][
                            "curve_summary"
                        ][
                            "ELM"
                        ][
                            "x_monotonic_annotation_fraction"
                        ]
                    ),

                "ELM_median_horizontal_coverage":
                    float(
                        inputs[
                            "geometry_audit"
                        ][
                            "curve_summary"
                        ][
                            "ELM"
                        ][
                            "image_horizontal_coverage_fraction"
                        ][
                            "median"
                        ]
                    ),

                "EZ_valid_annotations":
                    EXPECTED_EZ_VALID,

                "EZ_positive_images":
                    EXPECTED_EZ_IMAGES,

                "EZ_x_monotonic_fraction":
                    float(
                        inputs[
                            "geometry_audit"
                        ][
                            "curve_summary"
                        ][
                            "EZ"
                        ][
                            "x_monotonic_annotation_fraction"
                        ]
                    ),

                "EZ_median_horizontal_coverage":
                    float(
                        inputs[
                            "geometry_audit"
                        ][
                            "curve_summary"
                        ][
                            "EZ"
                        ][
                            "image_horizontal_coverage_fraction"
                        ][
                            "median"
                        ]
                    ),
            },

            "primary_targets": {
                "centerline_heatmap":
                    {
                        "description":
                            (
                                "Gaussian proximity map generated "
                                "from authoritative float curves "
                                "on annotated columns only"
                            ),

                        "sigma_native_px":
                            1.5,

                        "composition_for_multiple_levels":
                            "pixelwise maximum",
                    },

                "vertical_distance_field":
                    {
                        "description":
                            (
                                "per-column unsigned vertical "
                                "distance to nearest valid curve "
                                "level"
                            ),

                        "truncation_native_px":
                            16.0,

                        "normalization":
                            "clip(distance/16, 0, 1)",

                        "loss_validity":
                            (
                                "only columns covered by an "
                                "authoritative curve annotation"
                            ),
                    },
            },

            "auxiliary_targets": {
                "column_presence":
                    {
                        "shape":
                            "2 x 576",

                        "meaning":
                            (
                                "whether ELM/EZ is annotated/"
                                "identifiable in that column"
                            ),
                    },

                "column_y_distribution":
                    {
                        "shape":
                            "2 x 384 x 576",

                        "supervision":
                            (
                                "normalized vertical Gaussian "
                                "distribution, only on present "
                                "columns"
                            ),

                        "sigma_native_px":
                            1.5,

                        "multiple_levels":
                            (
                                "multi-peak target permitted in "
                                "rare ambiguous columns"
                            ),
                    },
            },

            "pathology_preservation_policy":
                {
                    "force_global_connectivity":
                        False,

                    "global_cldice_training_loss":
                        False,

                    "reason":
                        (
                            "EZ rupture/discontinuity can be "
                            "pathological and explicitly occurs "
                            "in the dataset annotation protocol. "
                            "The model must not be forced to "
                            "bridge true gaps."
                        ),

                    "gap_modeling":
                        (
                            "column-presence prediction explicitly "
                            "models visible/annotated gaps"
                        ),
            },

            "anatomical_order_constraint":
                {
                    "enabled":
                        True,

                    "condition":
                        (
                            "columns where both ELM and EZ "
                            "are present/valid"
                        ),

                    "rule":
                        (
                            "expected_y_ELM <= expected_y_EZ"
                        ),

                    "hard_margin_px":
                        0.0,

                    "loss":
                        (
                            "hinge/ReLU ordering penalty; no "
                            "fixed anatomical separation imposed"
                        ),
            },

            "validity_mask":
                (
                    "native FOV; no ELM/EZ raw coordinate "
                    "was observed outside native FOV"
                ),
        },

        # ---------------------------------------------------------------------
        # Points
        # ---------------------------------------------------------------------

        "points": {
            "label":
                "HF",

            "geometry_dimension":
                "0D",

            "valid_annotation_type":
                "single finite LabelMe point",

            "valid_points":
                EXPECTED_HF_POINTS,

            "positive_images":
                EXPECTED_HF_POSITIVE_IMAGES,

            "output_stride":
                EXPECTED_HF_STRIDE,

            "heatmap_hw":
                [
                    192,
                    288,
                ],

            "target":
                {
                    "heatmap":
                        (
                            "Gaussian center heatmap at stride 2"
                        ),

                    "gaussian_sigma_cells":
                        1.0,

                    "gaussian_composition":
                        (
                            "pixelwise maximum, preserving "
                            "multiple nearby peaks"
                        ),

                    "offset":
                        (
                            "two-channel local fractional "
                            "x/y offset from heatmap cell "
                            "to authoritative float coordinate"
                        ),
                },

            "collision_evidence":
                {
                    "stride_2_extra_collisions":
                        0,

                    "stride_2_extra_collision_fraction":
                        0.0,

                    "stride_4_extra_collisions":
                        18,

                    "stride_4_extra_collision_fraction":
                        0.001954,
                },

            "validity_mask":
                (
                    "native FOV; no valid HF center was "
                    "observed outside native FOV"
                ),
        },

        "malformed_annotation_policy":
            (
                "Retain the image and exclude only the malformed "
                "individual primary annotation, matching geometry_v1."
            ),

        "geometry_v1_dependency_sha256":
            EXPECTED_GEOMETRY_V1_SHA,
    }

    target[
        "target_protocol_sha256"
    ] = stable_sha(
        target
    )

    return target


# =============================================================================
# FROZEN ARCHITECTURE
# =============================================================================

def make_architecture_protocol(
    target_protocol: Dict[str, Any],
) -> Dict[str, Any]:

    architecture = {
        "name":
            "GeoRCP-MIRAGE v1",

        "status":
            "FROZEN_SCIENTIFIC_ARCHITECTURE",

        "primary_dimensionality":
            "2D",

        "2p5d":
            {
                "primary_model":
                    False,

                "reason":
                    (
                        "Released B-scans are strongly ordered "
                        "but physical adjacency/z-spacing is not "
                        "proven; representative pathological "
                        "slices were selected."
                    ),

                "future_status":
                    (
                        "exploratory only if original acquisition "
                        "metadata or author confirmation establishes "
                        "physical adjacency"
                    ),
            },

        # ---------------------------------------------------------------------
        # Input
        # ---------------------------------------------------------------------

        "input": {
            "channels":
                1,

            "tensor_hw":
                [
                    384,
                    576,
                ],

            "near_native_protocol_sha256":
                EXPECTED_NEAR_NATIVE_SHA,

            "normalization":
                (
                    "MIRAGE-compatible OCT normalization; exact "
                    "numeric normalization verified from official "
                    "wrapper during implementation"
                ),
        },

        # ---------------------------------------------------------------------
        # Global encoder
        # ---------------------------------------------------------------------

        "global_encoder": {
            "model":
                "MIRAGE-Base",

            "modality":
                "bscan",

            "pretraining":
                (
                    "official retinal OCT/SLO/layer multimodal "
                    "foundation-model weights"
                ),

            "parameters_observed":
                86039000,

            "patch_size":
                32,

            "input_patch_grid":
                [
                    12,
                    18,
                ],

            "image_tokens":
                216,

            "embedding_dim":
                768,

            "selected_blocks_one_based":
                [
                    3,
                    6,
                    9,
                    12,
                ],

            "fine_tuning_primary":
                "full encoder fine-tuning",

            "decoder_only_encoder_frozen":
                (
                    "required ablation, not the primary full model"
                ),

            "official_wrapper_sha256":
                EXPECTED_MIRAGE_WRAPPER_SHA,
        },

        # ---------------------------------------------------------------------
        # Native-detail path
        # ---------------------------------------------------------------------

        "local_detail_encoder": {
            "purpose":
                (
                    "restore high-frequency local information "
                    "that may be attenuated by 32x32 ViT patches; "
                    "particularly important for small IRF, ELM/EZ, "
                    "and HF"
                ),

            "input":
                "same 1-channel 384x576 OCT",

            "stage_stride2":
                {
                    "channels":
                        32,

                    "operator":
                        (
                            "3x3 stride-2 convolution + normalization "
                            "+ GELU + one ConvNeXt-style local block"
                        ),
                },

            "stage_stride4":
                {
                    "channels":
                        64,

                    "operator":
                        (
                            "3x3 stride-2 convolution from stride2 "
                            "+ two ConvNeXt-style local blocks"
                        ),
                },

            "ablation_required":
                True,
        },

        # ---------------------------------------------------------------------
        # Multi-layer token fusion
        # ---------------------------------------------------------------------

        "global_token_fusion": {
            "selected_layer_projection_dim":
                256,

            "method":
                (
                    "project each selected MIRAGE layer 768->256, "
                    "concatenate four layer features channel-wise, "
                    "then linear fuse 1024->256 per spatial token"
                ),

            "output":
                "216 x 256 shared patch tokens",

            "why_multilevel":
                (
                    "combine intermediate local/structural and "
                    "deeper contextual MIRAGE representations"
                ),

            "ablation":
                (
                    "final MIRAGE block only vs blocks 3/6/9/12"
                ),
        },

        # ---------------------------------------------------------------------
        # Geometry-conditioned interaction
        # ---------------------------------------------------------------------

        "geometry_task_interaction": {
            "name":
                "Geometry-Conditioned Task Interaction Transformer",

            "task_streams":
                [
                    "region",
                    "curve",
                    "point",
                ],

            "construction":
                (
                    "three learned task-specific linear adapters "
                    "and task-type embeddings applied to the same "
                    "216 fused global tokens"
                ),

            "concatenated_tokens":
                648,

            "token_dim":
                256,

            "transformer_layers":
                2,

            "attention_heads":
                8,

            "mlp_ratio":
                4.0,

            "dropout":
                0.1,

            "attention_scope":
                (
                    "joint self-attention across all region/curve/"
                    "point task tokens"
                ),

            "purpose":
                (
                    "learn spatial/anatomical dependencies among "
                    "heterogeneous biomarker tasks without imposing "
                    "unsafe hard region-point rules"
                ),

            "ablation_required":
                True,
        },

        # ---------------------------------------------------------------------
        # Reassembly / local fusion
        # ---------------------------------------------------------------------

        "task_reassembly": {
            "per_task_patch_expansion":
                {
                    "input":
                        "216 x 256 task tokens",

                    "expansion":
                        (
                            "learned 8x8 patch reassembly from "
                            "patch-32 tokens"
                        ),

                    "output_stride":
                        4,

                    "output_hw":
                        [
                            96,
                            144,
                        ],

                    "channels":
                        64,
                },

            "stride4_local_fusion":
                {
                    "inputs":
                        [
                            "task stride-4 global feature 64ch",
                            "local-detail stride-4 feature 64ch",
                        ],

                    "fused_channels":
                        128,

                    "refinement":
                        "two ConvNeXt-style blocks",
                },

            "stride2_local_fusion":
                {
                    "operation":
                        (
                            "bilinear upsample task feature x2, "
                            "concatenate local-detail stride-2 feature"
                        ),

                    "channels_after_fusion":
                        64,
                },
        },

        # ---------------------------------------------------------------------
        # Heads
        # ---------------------------------------------------------------------

        "region_head": {
            "source":
                "region task stride-2 fused feature",

            "full_resolution_hw":
                [
                    384,
                    576,
                ],

            "region_logits":
                {
                    "classes":
                        [
                            "background",
                            "SRF",
                            "IRF",
                        ],

                    "activation":
                        "softmax",
                },

            "boundary_logits":
                {
                    "channels":
                        [
                            "SRF_boundary",
                            "IRF_boundary",
                        ],

                    "activation":
                        "sigmoid",
                },
        },

        "curve_head": {
            "source":
                "curve task stride-2 fused feature",

            "full_resolution_hw":
                [
                    384,
                    576,
                ],

            "outputs":
                {
                    "centerline_logits":
                        "2 channels: ELM/EZ",

                    "distance_field":
                        (
                            "2 channels, sigmoid normalized "
                            "unsigned vertical distance"
                        ),

                    "column_presence_logits":
                        "2 x 576",

                    "column_y_distribution":
                        (
                            "derived by softmax over the y dimension "
                            "of centerline logits on valid columns"
                        ),
                },

            "postprocessing":
                (
                    "presence-gated per-column peaks/distributions; "
                    "do not fill unobserved gaps"
                ),
        },

        "point_head": {
            "source":
                "point task stride-2 fused feature",

            "output_stride":
                2,

            "output_hw":
                [
                    192,
                    288,
                ],

            "outputs":
                {
                    "heatmap_logits":
                        1,

                    "offset_channels":
                        2,
                },
        },

        "target_protocol_sha256":
            target_protocol[
                "target_protocol_sha256"
            ],
    }

    architecture[
        "architecture_protocol_sha256"
    ] = stable_sha(
        architecture
    )

    return architecture


# =============================================================================
# LOSS PROTOCOL
# =============================================================================

def make_loss_protocol(
    target_protocol: Dict[str, Any],
    architecture_protocol: Dict[str, Any],
) -> Dict[str, Any]:

    loss = {
        "name":
            "GeoRCP-MIRAGE Loss Protocol v1",

        "status":
            "FROZEN_LOSS_FAMILIES",

        "important_note":
            (
                "Numerical task-level balancing is dynamic; "
                "no outer-test result may be used to select "
                "or modify loss definitions."
            ),

        # ---------------------------------------------------------------------
        # Region
        # ---------------------------------------------------------------------

        "region_group": {
            "mask_loss":
                {
                    "components":
                        [
                            "multiclass cross entropy",
                            "foreground soft Dice",
                        ],

                    "combination":
                        "equal contribution after component normalization",
                },

            "boundary_auxiliary":
                {
                    "components":
                        [
                            "binary cross entropy",
                            "soft Dice",
                        ],

                    "relative_weight_within_region_group":
                        0.25,
                },
        },

        # ---------------------------------------------------------------------
        # Curve
        # ---------------------------------------------------------------------

        "curve_group": {
            "centerline_heatmap":
                {
                    "loss":
                        "binary cross entropy with soft Gaussian targets",

                    "relative_weight":
                        1.0,
                },

            "vertical_distance_field":
                {
                    "loss":
                        "SmoothL1",

                    "mask":
                        "annotated curve columns only",

                    "relative_weight":
                        0.5,
                },

            "column_presence":
                {
                    "loss":
                        "binary cross entropy",

                    "relative_weight":
                        0.25,
                },

            "column_y_distribution":
                {
                    "loss":
                        (
                            "cross entropy / KL from ground-truth "
                            "Gaussian vertical distribution to predicted "
                            "y-softmax"
                        ),

                    "mask":
                        "present/annotated columns only",

                    "relative_weight":
                        1.0,
                },

            "ELM_EZ_order":
                {
                    "loss":
                        (
                            "ReLU(expected_y_ELM - expected_y_EZ)"
                        ),

                    "mask":
                        "columns where both curves are valid",

                    "relative_weight":
                        0.1,
                },

            "global_connectivity_loss":
                {
                    "enabled":
                        False,

                    "reason":
                        (
                            "True EZ pathological ruptures/discontinuous "
                            "bands must not be artificially bridged."
                        ),
                },
        },

        # ---------------------------------------------------------------------
        # Point
        # ---------------------------------------------------------------------

        "point_group": {
            "heatmap":
                {
                    "loss":
                        "CenterNet-style modified focal heatmap loss",

                    "relative_weight":
                        1.0,
                },

            "offset":
                {
                    "loss":
                        "SmoothL1 at positive HF center cells only",

                    "relative_weight":
                        1.0,
                },
        },

        # ---------------------------------------------------------------------
        # Group balancing
        # ---------------------------------------------------------------------

        "task_group_balancing": {
            "groups":
                [
                    "region",
                    "curve",
                    "point",
                ],

            "method":
                (
                    "learned homoscedastic uncertainty weighting "
                    "with one trainable log-variance per task group"
                ),

            "form":
                (
                    "sum_t exp(-s_t)*L_t + s_t"
                ),

            "initial_log_variance":
                0.0,

            "reason":
                (
                    "region, curve, and point targets have radically "
                    "different support densities and raw loss scales; "
                    "dynamic group balancing avoids test-driven manual "
                    "weight tuning"
                ),
        },

        "padding_policy":
            (
                "spatial losses use target-specific validity masks "
                "defined in the target protocol"
            ),

        "target_protocol_sha256":
            target_protocol[
                "target_protocol_sha256"
            ],

        "architecture_protocol_sha256":
            architecture_protocol[
                "architecture_protocol_sha256"
            ],
    }

    loss[
        "loss_protocol_sha256"
    ] = stable_sha(
        loss
    )

    return loss


# =============================================================================
# EVALUATION PROTOCOL
# =============================================================================

def make_evaluation_protocol() -> Dict[str, Any]:

    evaluation = {
        "name":
            "GeoRCP-MIRAGE Evaluation Protocol v1",

        "status":
            "FROZEN",

        "cross_validation":
            {
                "patient_roles":
                    "frozen canonical 5-fold Track-B roles",

                "patient_protocol_sha256":
                    EXPECTED_PATIENT_PROTOCOL_SHA,

                "outer_test_usage":
                    (
                        "exactly once per patient/fold; never for "
                        "checkpoint selection or hyperparameter tuning"
                    ),

                "primary_oof_aggregation":
                    (
                        "pool additive patient-level sufficient statistics "
                        "where mathematically valid"
                    ),

                "confidence_intervals":
                    (
                        "patient-cluster bootstrap, 10,000 replicates"
                    ),
            },

        "regions": {
            "primary_metrics":
                [
                    "Dice",
                    "IoU",
                ],

            "boundary_metrics":
                [
                    "boundary F1 at prespecified tolerance",
                    "HD95",
                ],

            "evaluation_canvas":
                "exact native 570x380 crop",
        },

        "curves": {
            "primary_metrics":
                [
                    "mean absolute vertical boundary error (pixels)",
                    "median absolute vertical boundary error (pixels)",
                    "HD95 / robust curve distance",
                    "column-presence precision",
                    "column-presence recall",
                    "column-presence F1",
                ],

            "secondary_comparability_metrics":
                [
                    (
                        "Dice/IoU after rasterizing predicted curve "
                        "using the released-mask comparison convention"
                    )
                ],

            "gap_sensitive_analysis":
                [
                    "fragment count error",
                    "false bridge rate across GT gaps",
                    "missed annotated segment rate",
                ],

            "note":
                (
                    "Dice alone is not the primary curve endpoint "
                    "because 1-pixel shifts strongly alter thin-line overlap."
                ),
        },

        "points": {
            "primary_metrics":
                [
                    "precision/recall/F1 within 5-pixel tolerance",
                    "precision/recall/F1 within 10-pixel tolerance",
                    "mean localization error",
                    "median localization error",
                    "FROC",
                    "count MAE",
                ],

            "secondary_official_comparison":
                (
                    "convert point predictions to the paper-compatible "
                    "box convention only for secondary AP@0.5 comparison"
                ),
        },

        "unified_reporting":
            {
                "per_fold":
                    True,

                "fold_mean_sd":
                    True,

                "pooled_oof":
                    True,

                "patient_bootstrap_ci":
                    True,
            },

        "frozen_unetpp_reference": {
            "aggregation_sha256":
                EXPECTED_UNETPP_AGG_SHA,

            "native256_pooled_mean_dice":
                0.6561,

            "native570x380_pooled_mean_dice":
                0.5790,
        },
    }

    evaluation[
        "evaluation_protocol_sha256"
    ] = stable_sha(
        evaluation
    )

    return evaluation


# =============================================================================
# ABLATION / BASELINE PLAN
# =============================================================================

def make_ablation_plan() -> Dict[str, Any]:

    plan = {
        "name":
            "RVO-ME Modern Baseline and GeoRCP Ablation Plan v1",

        "status":
            "FROZEN_BEFORE_PROPOSED_MODEL_TESTING",

        "modern_baselines": [
            {
                "id":
                    "B0",

                "model":
                    "U-Net++",

                "input":
                    "256x256",

                "status":
                    "COMPLETE_FROZEN",

                "purpose":
                    "clean conventional reference",
            },

            {
                "id":
                    "B1",

                "model":
                    "nnU-Net v2 residual-encoder configuration",

                "input":
                    "near-native/native geometry preserving",

                "status":
                    "PENDING",

                "purpose":
                    (
                        "strong self-configuring specialist "
                        "segmentation baseline"
                    ),
            },

            {
                "id":
                    "B2",

                "model":
                    (
                        "MIRAGE-Base + standard published-style "
                        "ConvNeXt semantic segmentation decoder"
                    ),

                "input":
                    "384x576",

                "status":
                    "PENDING",

                "purpose":
                    (
                        "OCT foundation-model baseline; isolates "
                        "domain pretraining + near-native resolution"
                    ),
            },
        ],

        "proposed_ablation_sequence": [
            {
                "id":
                    "A0",

                "configuration":
                    (
                        "MIRAGE-Base + standard semantic decoder "
                        "at 384x576"
                    ),

                "tests":
                    "foundation encoder / resolution baseline",
            },

            {
                "id":
                    "A1",

                "configuration":
                    (
                        "A0 + local-detail encoder, still ordinary "
                        "semantic segmentation"
                    ),

                "tests":
                    "high-frequency local-detail contribution",
            },

            {
                "id":
                    "A2",

                "configuration":
                    (
                        "MIRAGE multilevel + local detail + "
                        "geometry-specific Region/Curve/Point heads, "
                        "NO task-interaction transformer"
                    ),

                "tests":
                    "geometry-specific representation contribution",
            },

            {
                "id":
                    "A3",

                "configuration":
                    (
                        "A2 + Geometry-Conditioned Task Interaction "
                        "Transformer"
                    ),

                "tests":
                    "cross-task interaction contribution",
            },

            {
                "id":
                    "A4",

                "configuration":
                    (
                        "A3 + curve column-y distribution + "
                        "ELM/EZ ordering constraint"
                    ),

                "tests":
                    "structured curve supervision contribution",
            },

            {
                "id":
                    "A5",

                "configuration":
                    "FULL GeoRCP-MIRAGE v1",

                "tests":
                    "final proposed architecture",
            },
        ],

        "mandatory_encoder_ablation":
            {
                "frozen_MIRAGE_vs_full_finetune":
                    True,

                "recommended_scope":
                    (
                        "at minimum one prespecified fold during "
                        "development; final choice selected only on "
                        "validation, never outer test"
                    ),
            },

        "multilevel_feature_ablation":
            {
                "final_block_only_vs_blocks_3_6_9_12":
                    True,
            },

        "resolution_control":
            {
                "required":
                    True,

                "reason":
                    (
                        "architecture gains must not be confounded "
                        "with 256x256 -> 384x576 geometry preservation"
                    ),
            },

        "outer_test_tuning_prohibited":
            True,
    }

    plan[
        "ablation_plan_sha256"
    ] = stable_sha(
        plan
    )

    return plan


# =============================================================================
# ENGINEERING SETTINGS RESERVED FOR RTX 4080
# =============================================================================

def make_rtx4080_pending_protocol() -> Dict[str, Any]:

    pending = {
        "status":
            "PENDING_EXACT_RTX4080_PROFILE",

        "must_be_frozen_before_first_definitive_training": [
            "physical batch size",
            "gradient accumulation",
            "effective batch size",
            "AMP dtype",
            "gradient checkpointing on/off",
            "encoder learning rate",
            "decoder/head learning rate",
            "weight decay",
            "scheduler",
            "warmup",
            "maximum epochs",
            "early stopping patience",
            "validation checkpoint metric",
            "data augmentation policy",
        ],

        "initial_profile_candidates_not_yet_frozen": {
            "physical_batch_sizes":
                [
                    2,
                    4,
                    6,
                    8,
                ],

            "AMP":
                True,

            "optimizer_candidate":
                "AdamW",

            "encoder_lr_candidate":
                1e-5,

            "decoder_lr_candidate":
                1e-4,

            "weight_decay_candidate":
                0.01,
        },

        "rule":
            (
                "Engineering settings are selected from GPU feasibility "
                "and training-fold validation only. They must be frozen "
                "before any outer-test evaluation of a proposed model."
            ),
    }

    pending[
        "pending_engineering_protocol_sha256"
    ] = stable_sha(
        pending
    )

    return pending


# =============================================================================
# HUMAN-READABLE PROTOCOL
# =============================================================================

def make_markdown(
    target: Dict[str, Any],
    architecture: Dict[str, Any],
    loss: Dict[str, Any],
    evaluation: Dict[str, Any],
    ablation: Dict[str, Any],
    pending: Dict[str, Any],
) -> str:

    return f"""# GeoRCP-MIRAGE v1 — Frozen Scientific Design

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
- Near-native SHA: `{EXPECTED_NEAR_NATIVE_SHA}`

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

- Target protocol SHA: `{target["target_protocol_sha256"]}`
- Architecture protocol SHA: `{architecture["architecture_protocol_sha256"]}`
- Loss protocol SHA: `{loss["loss_protocol_sha256"]}`
- Evaluation protocol SHA: `{evaluation["evaluation_protocol_sha256"]}`
- Ablation plan SHA: `{ablation["ablation_plan_sha256"]}`
- Pending RTX4080 engineering SHA: `{pending["pending_engineering_protocol_sha256"]}`

## Next step

Profile the exact frozen model on the RTX 4080, freeze the remaining
optimization/engineering settings, then generate geometry_v2 targets and begin
modern baseline/proposed-model training without using any outer-test result for
tuning.
"""


# =============================================================================
# MAIN
# =============================================================================

def main():

    print(
        "\n"
        + "=" * 100
    )

    print(
        "RVO-ME STEP 05E5A: "
        "FREEZE SCIENTIFIC GeoRCP-MIRAGE DESIGN"
    )

    print(
        "=" * 100
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\n[1/7] Verifying prerequisite "
        "audits and frozen baselines..."
    )

    inputs = verify_inputs()

    print(
        "    Near-native protocol: PASS"
    )

    print(
        "    Geometry representation audit: PASS"
    )

    print(
        "    HF stride-2 audit: PASS"
    )

    print(
        "    Boundary-coordinate audit: PASS"
    )

    print(
        "    MIRAGE RCP prototype: PASS"
    )

    print(
        "    Frozen U-Net++ 5-fold reference: PASS"
    )

    print(
        "\n[2/7] Freezing geometry-specific "
        "target protocol..."
    )

    target = make_target_protocol(
        inputs
    )

    save_json(
        OUTPUT_DIR
        / "TARGET_PROTOCOL_FROZEN.json",
        target,
    )

    print(
        "    Target protocol SHA:"
    )

    print(
        "    "
        + target[
            "target_protocol_sha256"
        ]
    )

    print(
        "\n[3/7] Freezing proposed "
        "scientific architecture..."
    )

    architecture = (
        make_architecture_protocol(
            target
        )
    )

    save_json(
        OUTPUT_DIR
        / "ARCHITECTURE_PROTOCOL_FROZEN.json",
        architecture,
    )

    print(
        "    Architecture protocol SHA:"
    )

    print(
        "    "
        + architecture[
            "architecture_protocol_sha256"
        ]
    )

    print(
        "\n[4/7] Freezing loss families..."
    )

    loss = make_loss_protocol(
        target,
        architecture,
    )

    save_json(
        OUTPUT_DIR
        / "LOSS_PROTOCOL_FROZEN.json",
        loss,
    )

    print(
        "    Loss protocol SHA:"
    )

    print(
        "    "
        + loss[
            "loss_protocol_sha256"
        ]
    )

    print(
        "\n[5/7] Freezing evaluation "
        "and ablation plans..."
    )

    evaluation = (
        make_evaluation_protocol()
    )

    ablation = (
        make_ablation_plan()
    )

    save_json(
        OUTPUT_DIR
        / "EVALUATION_PROTOCOL_FROZEN.json",
        evaluation,
    )

    save_json(
        OUTPUT_DIR
        / "ABLATION_PLAN_FROZEN.json",
        ablation,
    )

    print(
        "    Evaluation SHA:"
    )

    print(
        "    "
        + evaluation[
            "evaluation_protocol_sha256"
        ]
    )

    print(
        "    Ablation SHA:"
    )

    print(
        "    "
        + ablation[
            "ablation_plan_sha256"
        ]
    )

    print(
        "\n[6/7] Reserving RTX 4080-only "
        "engineering choices..."
    )

    pending = (
        make_rtx4080_pending_protocol()
    )

    save_json(
        OUTPUT_DIR
        / "RTX4080_ENGINEERING_PENDING.json",
        pending,
    )

    print(
        "    Physical batch size: NOT YET FROZEN"
    )

    print(
        "    LR/scheduler/checkpointing: NOT YET FROZEN"
    )

    print(
        "    These will be frozen before first "
        "definitive proposed-model training."
    )

    print(
        "\n[7/7] Writing master freeze record..."
    )

    master = {
        "status":
            "COMPLETE",

        "step":
            "05E5A",

        "design_name":
            "GeoRCP-MIRAGE v1",

        "scientific_design_frozen":
            True,

        "engineering_profile_pending":
            True,

        "frozen_dependencies": {
            "geometry_v1_sha256":
                EXPECTED_GEOMETRY_V1_SHA,

            "patient_5fold_protocol_sha256":
                EXPECTED_PATIENT_PROTOCOL_SHA,

            "near_native_protocol_sha256":
                EXPECTED_NEAR_NATIVE_SHA,

            "unetpp_trackB_5fold_aggregation_sha256":
                EXPECTED_UNETPP_AGG_SHA,

            "official_mirage_wrapper_sha256":
                EXPECTED_MIRAGE_WRAPPER_SHA,
        },

        "protocols": {
            "target_protocol_sha256":
                target[
                    "target_protocol_sha256"
                ],

            "architecture_protocol_sha256":
                architecture[
                    "architecture_protocol_sha256"
                ],

            "loss_protocol_sha256":
                loss[
                    "loss_protocol_sha256"
                ],

            "evaluation_protocol_sha256":
                evaluation[
                    "evaluation_protocol_sha256"
                ],

            "ablation_plan_sha256":
                ablation[
                    "ablation_plan_sha256"
                ],

            "pending_engineering_protocol_sha256":
                pending[
                    "pending_engineering_protocol_sha256"
                ],
        },

        "critical_design_decisions": [
            "2D primary model; no primary 2.5D claim",
            "near-native 384x576 model canvas",
            "MIRAGE-Base full-finetune primary encoder",
            "blocks 3/6/9/12 multilevel fusion",
            "native-detail CNN path",
            "geometry-conditioned region/curve/point task interaction",
            "IRF/SRF region masks + boundary auxiliary",
            "ELM/EZ hybrid dense curve + column-y targets",
            "no forced global EZ continuity",
            "ELM-above-EZ soft ordering constraint",
            "HF stride-2 heatmap + subpixel offsets",
            "patient-disjoint frozen 5-fold evaluation",
        ],
    }

    master[
        "scientific_design_master_sha256"
    ] = stable_sha(
        master
    )

    save_json(
        OUTPUT_DIR
        / "SCIENTIFIC_DESIGN_FROZEN.json",
        master,
    )

    markdown = make_markdown(
        target,
        architecture,
        loss,
        evaluation,
        ablation,
        pending,
    )

    (
        OUTPUT_DIR
        / "SCIENTIFIC_DESIGN.md"
    ).write_text(
        markdown,
        encoding="utf-8",
    )

    complete = {
        "status":
            "COMPLETE",

        "step":
            "05E5A",

        "scientific_design_master_sha256":
            master[
                "scientific_design_master_sha256"
            ],

        "output_dir":
            str(
                OUTPUT_DIR
            ),
    }

    save_json(
        OUTPUT_DIR
        / "STEP05E5A_COMPLETE.json",
        complete,
    )

    print(
        "\n"
        + "-" * 100
    )

    print(
        "SCIENTIFIC DESIGN FROZEN"
    )

    print(
        "-" * 100
    )

    print(
        "Primary model: GeoRCP-MIRAGE v1"
    )

    print(
        "Input: 384x576 near-native"
    )

    print(
        "Encoder: MIRAGE-Base, blocks 3/6/9/12"
    )

    print(
        "Local detail path: stride 2 + stride 4"
    )

    print(
        "Region: SRF/IRF masks + boundaries"
    )

    print(
        "Curve: ELM/EZ dense field + column distributions"
    )

    print(
        "Curve continuity: NOT forced globally"
    )

    print(
        "HF: stride-2 heatmap + offsets"
    )

    print(
        "Cross-task module: 2-layer geometry-conditioned transformer"
    )

    print(
        "Patient protocol: frozen 5-fold Track B"
    )

    print(
        "\nMaster scientific-design SHA-256:"
    )

    print(
        master[
            "scientific_design_master_sha256"
        ]
    )

    print(
        "\nResults saved to:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "STEP 05E5A COMPLETE"
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()
