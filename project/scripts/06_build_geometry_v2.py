# =============================================================================
# RVO-ME STEP 06
# CONSOLIDATED GEOMETRY_V2 TARGET BUILDER
#
# This single stage script replaces many small target-generation scripts.
#
# It performs:
#   1) prerequisite verification
#   2) GeoRCP-MIRAGE v1.1 target-validity amendment
#   3) complete geometry_v2 target generation
#   4) full-dataset validation
#   5) deterministic content hashing / manifest
#   6) visual QC generation
#   7) PROJECT_LOG.md append on successful completion
#
# NO MODEL TRAINING IS PERFORMED.
# RAW DATASET FILES ARE NEVER MODIFIED.
#
# ---------------------------------------------------------------------------
# FINAL TARGETS PER IMAGE (.npz)
# ---------------------------------------------------------------------------
#
# region_class:
#   uint8 [384,576]
#   0=background, 1=SRF, 2=IRF, 255=ignore
#
# region_valid:
#   uint8 [384,576]
#   1 for native-FOV region classification pixels except raw SRF/IRF overlap
#
# srf_mask / irf_mask:
#   uint8 [384,576]
#   independent raw-geometry region masks
#
# region_boundary:
#   uint8 [2,384,576]
#   channel 0=SRF, channel 1=IRF
#
# region_boundary_valid:
#   uint8 [384,576]
#   native FOV + annotation-supported positive boundary pixels in padding
#
# curve_centerline:
#   float16 [2,384,576]
#   channel 0=ELM, channel 1=EZ
#   Gaussian vertical proximity field (sigma=1.5 native px)
#
# curve_distance:
#   float16 [2,384,576]
#   normalized unsigned vertical distance, clipped at 16 native px
#   meaningful only on columns where that curve is present
#
# curve_presence:
#   uint8 [2,576]
#   whether each curve is annotated/visible at that model x-column
#
# curve_column_valid:
#   uint8 [576]
#   native horizontal FOV columns only
#
# hf_heatmap:
#   float16 [192,288]
#   stride-2 Gaussian heatmap, sigma=1.0 heatmap cells
#
# hf_offset:
#   float16 [2,192,288]
#   local fractional x/y offset at positive center cells
#
# hf_offset_mask:
#   uint8 [192,288]
#   1 exactly at HF center cells
#
# hf_valid:
#   uint8 [192,288]
#   stride-2 cells intersecting the native FOV
#
# Notes:
#   - ELM/EZ y-distribution target is derived at training time by normalizing
#     curve_centerline over y on columns where curve_presence==1.
#   - ELM/EZ distance/y losses are masked by curve_presence.
#   - centerline and presence losses use the native-FOV validity masks.
#
# =============================================================================

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(r"E:\Papers\RVO_ME")
DATASET_ROOT = ROOT / "RVO-Lesion"
PROJECT = ROOT / "project"

LABELME_ROOT = DATASET_ROOT / "RVO_Lesion_Labelme"
IMAGE_DIR = DATASET_ROOT / "Image_Seg" / "images"

DESIGN_DIR = (
    PROJECT
    / "design"
    / "geometry_aware_rcp_v1"
)

SCIENTIFIC_DESIGN_JSON = (
    DESIGN_DIR
    / "SCIENTIFIC_DESIGN_FROZEN.json"
)

E5C_JSON = (
    PROJECT
    / "audits"
    / "step05e5c_region_overlap_characterization_v1"
    / "STEP05E5C_COMPLETE.json"
)

OUTPUT_DIR = (
    PROJECT
    / "targets"
    / "geometry_v2"
)

TARGET_DIR = (
    OUTPUT_DIR
    / "targets"
)

QC_DIR = (
    OUTPUT_DIR
    / "qc"
)

PROJECT_LOG = (
    PROJECT
    / "PROJECT_LOG.md"
)


# =============================================================================
# FROZEN IDS / EXPECTATIONS
# =============================================================================

EXPECTED_V1_MASTER_SHA = (
    "9f1586d102523009443e2b88dca54a9ca8fa5c983f08fdde837184d436a95801"
)

EXPECTED_NEAR_NATIVE_SHA = (
    "ac5ce015e1a05a3dc47f054c76f8706d42ac9a9c9506543cbf4ca8e621d53714"
)

EXPECTED_GEOMETRY_V1_SHA = (
    "a4029d5ab4570237ae607d1cc24d551ce438845eab1814e8a82735a122daeb55"
)

EXPECTED_PATIENT_PROTOCOL_SHA = (
    "52ed516dcc21f1c51068c59fb662c9a0030b143e6440d6f4a0380df581a9ba63e31"
)

EXPECTED_IMAGES = 3012

EXPECTED_VALID = {
    "IRF": 39534,
    "SRF": 1288,
    "ELM": 2384,
    "EZ": 4474,
    "HF": 9212,
}

EXPECTED_POSITIVE_IMAGES = {
    "IRF": 2893,
    "SRF": 1215,
    "ELM": 1625,
    "EZ": 2730,
    "HF": 2286,
}

EXPECTED_OVERLAP_IMAGES = 17
EXPECTED_OVERLAP_PIXELS = 165

# Canvas.
NATIVE_W = 570
NATIVE_H = 380

MODEL_W = 576
MODEL_H = 384

PAD_LEFT = 3
PAD_RIGHT = 3
PAD_TOP = 2
PAD_BOTTOM = 2

# Channels / classes.
REGION_LABELS = ("SRF", "IRF")
CURVE_LABELS = ("ELM", "EZ")

REGION_CLASS_VALUE = {
    "SRF": 1,
    "IRF": 2,
}

IGNORE_INDEX = 255

# Curve target parameters frozen in 05E5A.
CURVE_SIGMA_PX = 1.5
CURVE_DISTANCE_TRUNC_PX = 16.0

# HF target parameters frozen in 05E5A.
HF_STRIDE = 2
HF_H = MODEL_H // HF_STRIDE
HF_W = MODEL_W // HF_STRIDE
HF_SIGMA_CELLS = 1.0

# QC
N_QC_IMAGES = 12

# PROJECT_LOG marker for idempotent append.
LOG_MARKER = "<!-- STEP06_GEOMETRY_V2 -->"


# =============================================================================
# GENERIC HELPERS
# =============================================================================

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


def load_json(
    path: Path,
) -> Any:

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def write_csv(
    path: Path,
    rows: List[Dict[str, Any]],
) -> None:

    if not rows:
        raise RuntimeError(
            f"Refusing to write empty CSV: {path}"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = list(
        rows[0].keys()
    )

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)


def stable_json_sha(
    obj: Any,
) -> str:

    blob = json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        blob
    ).hexdigest()


def array_content_sha(
    arrays: Dict[str, np.ndarray],
) -> str:
    """
    Stable array-content SHA independent of NPZ zip timestamps/compression.
    """

    h = hashlib.sha256()

    for name in sorted(
        arrays
    ):

        arr = np.ascontiguousarray(
            arrays[
                name
            ]
        )

        h.update(
            name.encode(
                "utf-8"
            )
        )

        h.update(
            str(
                arr.dtype
            ).encode(
                "ascii"
            )
        )

        h.update(
            json.dumps(
                list(
                    arr.shape
                )
            ).encode(
                "ascii"
            )
        )

        h.update(
            arr.tobytes(
                order="C"
            )
        )

    return h.hexdigest()


def directory_size_bytes(
    path: Path,
) -> int:

    total = 0

    for p in path.rglob("*"):

        if p.is_file():

            total += p.stat().st_size

    return total


def human_bytes(
    n: int,
) -> str:

    units = [
        "B",
        "KB",
        "MB",
        "GB",
        "TB",
    ]

    value = float(
        n
    )

    for unit in units:

        if (
            value < 1024.0
            or unit == units[-1]
        ):

            return (
                f"{value:.2f} {unit}"
            )

        value /= 1024.0

    return (
        f"{n} B"
    )


# =============================================================================
# FILE DISCOVERY
# =============================================================================

def find_images_by_stem(
    folder: Path,
) -> Dict[str, Path]:

    allowed = {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff",
    }

    out: Dict[str, Path] = {}

    for p in folder.iterdir():

        if (
            p.is_file()
            and p.suffix.lower() in allowed
        ):

            if p.stem in out:

                raise RuntimeError(
                    f"Duplicate image stem: {p.stem}"
                )

            out[
                p.stem
            ] = p

    return out


def find_json_by_stem() -> Dict[str, Path]:

    out: Dict[str, Path] = {}

    for p in LABELME_ROOT.rglob(
        "*.json"
    ):

        if p.stem in out:

            raise RuntimeError(
                "Duplicate LabelMe JSON stem:\n"
                f"{p.stem}\n"
                f"{out[p.stem]}\n"
                f"{p}"
            )

        out[
            p.stem
        ] = p

    return out


# =============================================================================
# PREREQUISITE VERIFICATION / v1.1 AMENDMENT
# =============================================================================

def verify_prerequisites():
    """
    Verifies only records required for target generation.
    """

    design = load_json(
        SCIENTIFIC_DESIGN_JSON
    )

    master_sha = str(
        design.get(
            "scientific_design_master_sha256",
            "",
        )
    )

    if (
        master_sha
        != EXPECTED_V1_MASTER_SHA
    ):

        raise RuntimeError(
            "GeoRCP-MIRAGE v1 scientific-design SHA mismatch.\n"
            f"Expected: {EXPECTED_V1_MASTER_SHA}\n"
            f"Found:    {master_sha}"
        )

    e5c = load_json(
        E5C_JSON
    )

    if (
        e5c.get(
            "status"
        )
        != "COMPLETE"
    ):

        raise RuntimeError(
            "Step05E5C is not COMPLETE."
        )

    if int(
        e5c[
            "overlap_images"
        ]
    ) != EXPECTED_OVERLAP_IMAGES:

        raise RuntimeError(
            "05E5C overlap-image count mismatch."
        )

    if int(
        e5c[
            "overlap_pixels"
        ]
    ) != EXPECTED_OVERLAP_PIXELS:

        raise RuntimeError(
            "05E5C overlap-pixel count mismatch."
        )

    decision = e5c[
        "decision"
    ]

    if (
        decision[
            "classification"
        ]
        != "BOUNDARY_RASTERIZATION_AMBIGUITY"
    ):

        raise RuntimeError(
            "05E5C does not support the frozen softmax-region amendment."
        )

    if (
        decision[
            "softmax_region_head_status"
        ]
        != "RETAIN"
    ):

        raise RuntimeError(
            "05E5C does not retain the region softmax head."
        )

    return (
        design,
        e5c,
    )


def create_v1_1_amendment(
    e5c: Dict[str, Any],
) -> Dict[str, Any]:

    amendment = {
        "name":
            "GeoRCP-MIRAGE v1.1 target-validity amendment",

        "status":
            "FROZEN",

        "parent_scientific_design":
            "GeoRCP-MIRAGE v1",

        "parent_master_sha256":
            EXPECTED_V1_MASTER_SHA,

        "reason":
            (
                "Independent raw IRF and SRF rasterization produced "
                "165 overlap pixels across 17/3012 images. "
                "Step05E5C demonstrated boundary/rasterization ambiguity: "
                "0 overlap pixels remain after 2-pixel simultaneous erosion "
                "and 0 pixels lie >2 px inside both regions."
            ),

        "architecture_change":
            False,

        "loss_family_change":
            False,

        "target_representation_change":
            False,

        "validity_mask_change_only":
            True,

        "region_policy": {
            "head":
                (
                    "retain mutually exclusive "
                    "background/SRF/IRF softmax"
                ),

            "raw_overlap_pixels":
                EXPECTED_OVERLAP_PIXELS,

            "raw_overlap_images":
                EXPECTED_OVERLAP_IMAGES,

            "interior_classification_policy":
                (
                    "set raw SRF/IRF overlap pixels to IGNORE_INDEX=255 "
                    "and region_valid=0"
                ),

            "class_precedence":
                None,

            "boundary_policy":
                (
                    "retain independent class-specific SRF and IRF "
                    "boundary targets"
                ),
        },

        "supporting_step05e5c": {
            "deep_inside_both_gt2px":
                int(
                    e5c[
                        "deep_inside_both_gt2px"
                    ]
                ),

            "erosion_survival_2px":
                int(
                    e5c[
                        "erosion_survival_totals"
                    ][
                        "2"
                    ]
                ),
        },
    }

    amendment[
        "amendment_sha256"
    ] = stable_json_sha(
        amendment
    )

    return amendment


# =============================================================================
# VALID SHAPE DEFINITIONS
# =============================================================================

def finite_native_points(
    shape: Dict[str, Any],
) -> List[
    Tuple[
        float,
        float,
    ]
] | None:

    points = shape.get(
        "points",
        [],
    )

    if not isinstance(
        points,
        list,
    ):

        return None

    out = []

    for p in points:

        if (
            not isinstance(
                p,
                (list, tuple),
            )
            or len(p) < 2
        ):

            return None

        try:

            x = float(
                p[0]
            )

            y = float(
                p[1]
            )

        except Exception:

            return None

        if not (
            math.isfinite(
                x
            )
            and math.isfinite(
                y
            )
        ):

            return None

        out.append(
            (
                x,
                y,
            )
        )

    return out


def valid_region_shape(
    shape: Dict[str, Any],
    label: str,
) -> bool:

    if (
        shape.get(
            "label"
        )
        != label
    ):

        return False

    if (
        shape.get(
            "shape_type"
        )
        not in {
            "polygon",
            "linestrip",
        }
    ):

        return False

    points = finite_native_points(
        shape
    )

    return (
        points is not None
        and len(
            points
        ) >= 3
    )


def valid_curve_shape(
    shape: Dict[str, Any],
    label: str,
) -> bool:

    if (
        shape.get(
            "label"
        )
        != label
    ):

        return False

    if (
        shape.get(
            "shape_type"
        )
        != "linestrip"
    ):

        return False

    points = finite_native_points(
        shape
    )

    return (
        points is not None
        and len(
            points
        ) >= 2
    )


def valid_hf_shape(
    shape: Dict[str, Any],
) -> bool:

    if (
        shape.get(
            "label"
        )
        != "HF"
    ):

        return False

    if (
        shape.get(
            "shape_type"
        )
        != "point"
    ):

        return False

    points = finite_native_points(
        shape
    )

    return (
        points is not None
        and len(
            points
        ) == 1
    )


# =============================================================================
# COMMON VALIDITY MASKS
# =============================================================================

def make_native_fov_mask() -> np.ndarray:

    mask = np.zeros(
        (
            MODEL_H,
            MODEL_W,
        ),
        dtype=np.uint8,
    )

    mask[
        PAD_TOP:
        PAD_TOP + NATIVE_H,

        PAD_LEFT:
        PAD_LEFT + NATIVE_W,
    ] = 1

    return mask


def make_curve_column_valid() -> np.ndarray:

    valid = np.zeros(
        (
            MODEL_W,
        ),
        dtype=np.uint8,
    )

    valid[
        PAD_LEFT:
        PAD_LEFT + NATIVE_W
    ] = 1

    return valid


def make_hf_valid(
    native_fov: np.ndarray,
) -> np.ndarray:
    """
    A stride-2 cell is valid if ANY pixel in its 2x2 model-canvas area
    intersects the native FOV.

    This preserves all valid HF centers despite asymmetric padding alignment.
    """

    if (
        MODEL_H % HF_STRIDE != 0
        or MODEL_W % HF_STRIDE != 0
    ):

        raise RuntimeError(
            "Model raster is not divisible by HF stride."
        )

    reshaped = (
        native_fov
        .reshape(
            HF_H,
            HF_STRIDE,
            HF_W,
            HF_STRIDE,
        )
    )

    valid = (
        reshaped.max(
            axis=(
                1,
                3,
            )
        )
        > 0
    ).astype(
        np.uint8
    )

    return valid


# =============================================================================
# REGION TARGETS
# =============================================================================

def rasterize_region(
    shapes: List[
        Dict[str, Any]
    ],
    label: str,
) -> Tuple[
    np.ndarray,
    int,
]:

    mask = np.zeros(
        (
            MODEL_H,
            MODEL_W,
        ),
        dtype=np.uint8,
    )

    valid_count = 0

    for shape in shapes:

        if not valid_region_shape(
            shape,
            label,
        ):

            continue

        points = finite_native_points(
            shape
        )

        assert points is not None

        transformed = np.asarray(
            [
                [
                    x + PAD_LEFT,
                    y + PAD_TOP,
                ]
                for (
                    x,
                    y,
                ) in points
            ],
            dtype=np.float64,
        )

        integer = np.rint(
            transformed
        ).astype(
            np.int32
        )

        if (
            integer[
                :,
                0,
            ].min()
            < 0
            or integer[
                :,
                0,
            ].max()
            >= MODEL_W
            or integer[
                :,
                1,
            ].min()
            < 0
            or integer[
                :,
                1,
            ].max()
            >= MODEL_H
        ):

            raise RuntimeError(
                f"{label}: transformed region vertex outside "
                "frozen model canvas."
            )

        cv2.fillPoly(
            mask,
            [
                integer
            ],
            color=1,
        )

        valid_count += 1

    return (
        mask,
        valid_count,
    )


def inner_boundary_1px(
    mask: np.ndarray,
) -> np.ndarray:

    kernel = np.ones(
        (
            3,
            3,
        ),
        dtype=np.uint8,
    )

    eroded = cv2.erode(
        mask,
        kernel,
        iterations=1,
    )

    boundary = (
        (
            mask > 0
        )
        & (
            eroded == 0
        )
    ).astype(
        np.uint8
    )

    return boundary


def build_region_targets(
    shapes: List[
        Dict[str, Any]
    ],
    native_fov: np.ndarray,
):

    srf, n_srf = rasterize_region(
        shapes,
        "SRF",
    )

    irf, n_irf = rasterize_region(
        shapes,
        "IRF",
    )

    overlap = (
        (srf > 0)
        & (irf > 0)
    )

    region_class = np.full(
        (
            MODEL_H,
            MODEL_W,
        ),
        IGNORE_INDEX,
        dtype=np.uint8,
    )

    native = (
        native_fov > 0
    )

    region_class[
        native
    ] = 0

    region_class[
        native
        & (srf > 0)
        & ~overlap
    ] = REGION_CLASS_VALUE[
        "SRF"
    ]

    region_class[
        native
        & (irf > 0)
        & ~overlap
    ] = REGION_CLASS_VALUE[
        "IRF"
    ]

    # Explicit v1.1 amendment:
    # raw overlap = ignore, with NO class precedence.
    region_class[
        overlap
    ] = IGNORE_INDEX

    region_valid = (
        native
        & ~overlap
    ).astype(
        np.uint8
    )

    srf_boundary = (
        inner_boundary_1px(
            srf
        )
    )

    irf_boundary = (
        inner_boundary_1px(
            irf
        )
    )

    region_boundary = np.stack(
        [
            srf_boundary,
            irf_boundary,
        ],
        axis=0,
    ).astype(
        np.uint8
    )

    # Native FOV is valid. In addition, annotation-supported boundary
    # pixels just outside native FOV are valid for boundary supervision.
    boundary_positive_any = (
        region_boundary.max(
            axis=0
        )
        > 0
    )

    region_boundary_valid = (
        native
        | boundary_positive_any
    ).astype(
        np.uint8
    )

    return {
        "srf_mask":
            srf,

        "irf_mask":
            irf,

        "region_class":
            region_class,

        "region_valid":
            region_valid,

        "region_boundary":
            region_boundary,

        "region_boundary_valid":
            region_boundary_valid,

        "_counts":
            {
                "SRF":
                    n_srf,

                "IRF":
                    n_irf,

                "overlap_pixels":
                    int(
                        overlap.sum()
                    ),
            },
    }


# =============================================================================
# CURVE TARGETS
# =============================================================================

def sample_linestrip_columns(
    points: List[
        Tuple[
            float,
            float,
        ]
    ],
) -> Dict[
    int,
    List[
        float
    ],
]:
    """
    Segment-wise interpolation at integer NATIVE x-columns.

    Handles fragmented and rare non-monotonic linestrips without assuming
    exactly one global y value per x.
    """

    samples: Dict[
        int,
        List[
            float
        ],
    ] = defaultdict(
        list
    )

    arr = np.asarray(
        points,
        dtype=np.float64,
    )

    for i in range(
        len(
            arr
        )
        - 1
    ):

        x1, y1 = arr[
            i
        ]

        x2, y2 = arr[
            i + 1
        ]

        dx = (
            x2
            - x1
        )

        # Near-vertical segment.
        if abs(
            dx
        ) < 1e-12:

            col = int(
                round(
                    (
                        x1
                        + x2
                    )
                    / 2.0
                )
            )

            if (
                0
                <= col
                < NATIVE_W
            ):

                samples[
                    col
                ].append(
                    float(
                        (
                            y1
                            + y2
                        )
                        / 2.0
                    )
                )

            continue

        lo = max(
            0,
            int(
                math.ceil(
                    min(
                        x1,
                        x2,
                    )
                )
            ),
        )

        hi = min(
            NATIVE_W - 1,
            int(
                math.floor(
                    max(
                        x1,
                        x2,
                    )
                )
            ),
        )

        if hi < lo:

            continue

        for col in range(
            lo,
            hi + 1
        ):

            t = (
                float(
                    col
                )
                - x1
            ) / dx

            if (
                t < -1e-9
                or t > 1.0 + 1e-9
            ):

                continue

            y = (
                y1
                + t
                * (
                    y2
                    - y1
                )
            )

            # All valid ELM/EZ coordinates were shown by prior audit
            # to remain inside the native FOV.
            if (
                -1e-6
                <= y
                <= NATIVE_H
            ):

                samples[
                    col
                ].append(
                    float(
                        y
                    )
                )

    return samples


def gaussian_vector(
    length: int,
    centers: List[
        float
    ],
    sigma: float,
) -> np.ndarray:
    """
    Pixelwise maximum of Gaussian peaks.
    """

    if not centers:

        return np.zeros(
            (
                length,
            ),
            dtype=np.float32,
        )

    yy = np.arange(
        length,
        dtype=np.float32,
    )

    out = np.zeros(
        (
            length,
        ),
        dtype=np.float32,
    )

    denom = (
        2.0
        * sigma
        * sigma
    )

    for center in centers:

        g = np.exp(
            -(
                (
                    yy
                    - float(
                        center
                    )
                )
                ** 2
            )
            / denom
        ).astype(
            np.float32
        )

        out = np.maximum(
            out,
            g,
        )

    return out


def build_curve_targets(
    shapes: List[
        Dict[str, Any]
    ],
):
    """
    Returns:
      curve_centerline [2,H,W]
      curve_distance   [2,H,W]
      curve_presence   [2,W]
    """

    centerline = np.zeros(
        (
            2,
            MODEL_H,
            MODEL_W,
        ),
        dtype=np.float32,
    )

    distance = np.zeros(
        (
            2,
            MODEL_H,
            MODEL_W,
        ),
        dtype=np.float32,
    )

    presence = np.zeros(
        (
            2,
            MODEL_W,
        ),
        dtype=np.uint8,
    )

    valid_counts = {
        "ELM":
            0,

        "EZ":
            0,
    }

    yy_native = np.arange(
        NATIVE_H,
        dtype=np.float32,
    )

    for channel, label in enumerate(
        CURVE_LABELS
    ):

        by_column: Dict[
            int,
            List[
                float
            ],
        ] = defaultdict(
            list
        )

        for shape in shapes:

            if not valid_curve_shape(
                shape,
                label,
            ):

                continue

            valid_counts[
                label
            ] += 1

            points = finite_native_points(
                shape
            )

            assert points is not None

            sampled = sample_linestrip_columns(
                points
            )

            for native_x, ys in sampled.items():

                by_column[
                    native_x
                ].extend(
                    ys
                )

        for native_x, ys in by_column.items():

            model_x = (
                native_x
                + PAD_LEFT
            )

            # Keep all rare multiple levels.
            model_y = [
                y
                + PAD_TOP
                for y in ys
            ]

            presence[
                channel,
                model_x,
            ] = 1

            # Full model-canvas vertical Gaussian.
            centerline[
                channel,
                :,
                model_x,
            ] = gaussian_vector(
                MODEL_H,
                model_y,
                CURVE_SIGMA_PX,
            )

            # Native-FOV vertical distance to nearest curve level.
            # The output rows in padding remain zero and are masked later.
            ys_native = np.asarray(
                ys,
                dtype=np.float32,
            )

            dist_native = np.min(
                np.abs(
                    yy_native[
                        :,
                        None
                    ]
                    - ys_native[
                        None,
                        :
                    ]
                ),
                axis=1,
            )

            dist_native = np.clip(
                dist_native
                / CURVE_DISTANCE_TRUNC_PX,
                0.0,
                1.0,
            )

            distance[
                channel,
                PAD_TOP:
                PAD_TOP + NATIVE_H,
                model_x,
            ] = dist_native

    return {
        "curve_centerline":
            centerline.astype(
                np.float16
            ),

        "curve_distance":
            distance.astype(
                np.float16
            ),

        "curve_presence":
            presence,

        "_counts":
            valid_counts,
    }


# =============================================================================
# HF TARGETS
# =============================================================================

def add_gaussian_2d_max(
    heatmap: np.ndarray,
    cx: float,
    cy: float,
    sigma: float,
):
    """
    Continuous-center Gaussian, cropped to about +/-4 sigma.
    """

    radius = max(
        1,
        int(
            math.ceil(
                4.0
                * sigma
            )
        ),
    )

    x0 = max(
        0,
        int(
            math.floor(
                cx
            )
        )
        - radius,
    )

    x1 = min(
        heatmap.shape[
            1
        ],
        int(
            math.floor(
                cx
            )
        )
        + radius
        + 2,
    )

    y0 = max(
        0,
        int(
            math.floor(
                cy
            )
        )
        - radius,
    )

    y1 = min(
        heatmap.shape[
            0
        ],
        int(
            math.floor(
                cy
            )
        )
        + radius
        + 2,
    )

    if (
        x1 <= x0
        or y1 <= y0
    ):

        return

    xs = np.arange(
        x0,
        x1,
        dtype=np.float32,
    )

    ys = np.arange(
        y0,
        y1,
        dtype=np.float32,
    )

    xx, yy = np.meshgrid(
        xs,
        ys,
    )

    g = np.exp(
        -(
            (
                xx
                - float(
                    cx
                )
            )
            ** 2
            + (
                yy
                - float(
                    cy
                )
            )
            ** 2
        )
        / (
            2.0
            * sigma
            * sigma
        )
    ).astype(
        np.float32
    )

    patch = heatmap[
        y0:y1,
        x0:x1,
    ]

    np.maximum(
        patch,
        g,
        out=patch,
    )


def build_hf_targets(
    shapes: List[
        Dict[str, Any]
    ],
):
    heatmap = np.zeros(
        (
            HF_H,
            HF_W,
        ),
        dtype=np.float32,
    )

    offset = np.zeros(
        (
            2,
            HF_H,
            HF_W,
        ),
        dtype=np.float32,
    )

    offset_mask = np.zeros(
        (
            HF_H,
            HF_W,
        ),
        dtype=np.uint8,
    )

    valid_count = 0
    collisions = 0

    for shape in shapes:

        if not valid_hf_shape(
            shape
        ):

            continue

        points = finite_native_points(
            shape
        )

        assert (
            points is not None
            and len(
                points
            ) == 1
        )

        x_native, y_native = (
            points[
                0
            ]
        )

        # Previous audit proved valid HF centers are inside native FOV.
        if not (
            0.0
            <= x_native
            < NATIVE_W
            and 0.0
            <= y_native
            < NATIVE_H
        ):

            raise RuntimeError(
                "Valid HF point unexpectedly outside native FOV."
            )

        x_model = (
            x_native
            + PAD_LEFT
        )

        y_model = (
            y_native
            + PAD_TOP
        )

        x_map = (
            x_model
            / HF_STRIDE
        )

        y_map = (
            y_model
            / HF_STRIDE
        )

        cx = int(
            math.floor(
                x_map
            )
        )

        cy = int(
            math.floor(
                y_map
            )
        )

        if not (
            0
            <= cx
            < HF_W
            and 0
            <= cy
            < HF_H
        ):

            raise RuntimeError(
                "HF center outside stride-2 heatmap."
            )

        if (
            offset_mask[
                cy,
                cx,
            ]
            > 0
        ):

            collisions += 1

        offset_mask[
            cy,
            cx,
        ] = 1

        offset[
            0,
            cy,
            cx,
        ] = (
            x_map
            - cx
        )

        offset[
            1,
            cy,
            cx,
        ] = (
            y_map
            - cy
        )

        add_gaussian_2d_max(
            heatmap,
            x_map,
            y_map,
            HF_SIGMA_CELLS,
        )

        valid_count += 1

    return {
        "hf_heatmap":
            heatmap.astype(
                np.float16
            ),

        "hf_offset":
            offset.astype(
                np.float16
            ),

        "hf_offset_mask":
            offset_mask,

        "_counts":
            {
                "HF":
                    valid_count,

                "collisions":
                    collisions,
            },
    }


# =============================================================================
# PER-IMAGE TARGET BUILD
# =============================================================================

def build_one(
    stem: str,
    json_path: Path,
    native_fov: np.ndarray,
    curve_column_valid: np.ndarray,
    hf_valid: np.ndarray,
):

    with json_path.open(
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(
            f
        )

    shapes = data.get(
        "shapes",
        [],
    )

    region = build_region_targets(
        shapes,
        native_fov,
    )

    curve = build_curve_targets(
        shapes
    )

    hf = build_hf_targets(
        shapes
    )

    arrays = {
        "region_class":
            region[
                "region_class"
            ],

        "region_valid":
            region[
                "region_valid"
            ],

        "srf_mask":
            region[
                "srf_mask"
            ],

        "irf_mask":
            region[
                "irf_mask"
            ],

        "region_boundary":
            region[
                "region_boundary"
            ],

        "region_boundary_valid":
            region[
                "region_boundary_valid"
            ],

        "curve_centerline":
            curve[
                "curve_centerline"
            ],

        "curve_distance":
            curve[
                "curve_distance"
            ],

        "curve_presence":
            curve[
                "curve_presence"
            ],

        "curve_column_valid":
            curve_column_valid,

        "hf_heatmap":
            hf[
                "hf_heatmap"
            ],

        "hf_offset":
            hf[
                "hf_offset"
            ],

        "hf_offset_mask":
            hf[
                "hf_offset_mask"
            ],

        "hf_valid":
            hf_valid,
    }

    counts = {
        "IRF":
            int(
                region[
                    "_counts"
                ][
                    "IRF"
                ]
            ),

        "SRF":
            int(
                region[
                    "_counts"
                ][
                    "SRF"
                ]
            ),

        "ELM":
            int(
                curve[
                    "_counts"
                ][
                    "ELM"
                ]
            ),

        "EZ":
            int(
                curve[
                    "_counts"
                ][
                    "EZ"
                ]
            ),

        "HF":
            int(
                hf[
                    "_counts"
                ][
                    "HF"
                ]
            ),

        "hf_collisions":
            int(
                hf[
                    "_counts"
                ][
                    "collisions"
                ]
            ),

        "region_overlap_pixels":
            int(
                region[
                    "_counts"
                ][
                    "overlap_pixels"
                ]
            ),
    }

    return (
        arrays,
        counts,
    )


# =============================================================================
# ARRAY VALIDATION
# =============================================================================

def validate_arrays(
    stem: str,
    arrays: Dict[str, np.ndarray],
    counts: Dict[str, int],
):

    expected_shapes = {
        "region_class":
            (
                MODEL_H,
                MODEL_W,
            ),

        "region_valid":
            (
                MODEL_H,
                MODEL_W,
            ),

        "srf_mask":
            (
                MODEL_H,
                MODEL_W,
            ),

        "irf_mask":
            (
                MODEL_H,
                MODEL_W,
            ),

        "region_boundary":
            (
                2,
                MODEL_H,
                MODEL_W,
            ),

        "region_boundary_valid":
            (
                MODEL_H,
                MODEL_W,
            ),

        "curve_centerline":
            (
                2,
                MODEL_H,
                MODEL_W,
            ),

        "curve_distance":
            (
                2,
                MODEL_H,
                MODEL_W,
            ),

        "curve_presence":
            (
                2,
                MODEL_W,
            ),

        "curve_column_valid":
            (
                MODEL_W,
            ),

        "hf_heatmap":
            (
                HF_H,
                HF_W,
            ),

        "hf_offset":
            (
                2,
                HF_H,
                HF_W,
            ),

        "hf_offset_mask":
            (
                HF_H,
                HF_W,
            ),

        "hf_valid":
            (
                HF_H,
                HF_W,
            ),
    }

    for key, shape in expected_shapes.items():

        if (
            key
            not in arrays
        ):

            raise RuntimeError(
                f"{stem}: missing array {key}"
            )

        if (
            arrays[
                key
            ].shape
            != shape
        ):

            raise RuntimeError(
                f"{stem}: {key} shape mismatch. "
                f"Expected {shape}, found {arrays[key].shape}"
            )

    # Region class values.
    values = set(
        int(
            v
        )
        for v in np.unique(
            arrays[
                "region_class"
            ]
        )
    )

    if not values.issubset(
        {
            0,
            1,
            2,
            IGNORE_INDEX,
        }
    ):

        raise RuntimeError(
            f"{stem}: invalid region_class values {values}"
        )

    # Raw overlap must NEVER remain region-valid.
    raw_overlap = (
        (
            arrays[
                "srf_mask"
            ]
            > 0
        )
        & (
            arrays[
                "irf_mask"
            ]
            > 0
        )
    )

    if np.any(
        arrays[
            "region_valid"
        ][
            raw_overlap
        ]
        != 0
    ):

        raise RuntimeError(
            f"{stem}: overlap pixel remained region-valid."
        )

    if np.any(
        arrays[
            "region_class"
        ][
            raw_overlap
        ]
        != IGNORE_INDEX
    ):

        raise RuntimeError(
            f"{stem}: overlap pixel did not receive ignore index."
        )

    # Finite floating arrays.
    for key in (
        "curve_centerline",
        "curve_distance",
        "hf_heatmap",
        "hf_offset",
    ):

        if not np.isfinite(
            arrays[
                key
            ]
        ).all():

            raise RuntimeError(
                f"{stem}: non-finite values in {key}"
            )

    # Ranges.
    if (
        arrays[
            "curve_centerline"
        ].min()
        < 0
        or arrays[
            "curve_centerline"
        ].max()
        > 1.0001
    ):

        raise RuntimeError(
            f"{stem}: curve_centerline outside [0,1]"
        )

    if (
        arrays[
            "curve_distance"
        ].min()
        < 0
        or arrays[
            "curve_distance"
        ].max()
        > 1.0001
    ):

        raise RuntimeError(
            f"{stem}: curve_distance outside [0,1]"
        )

    if (
        arrays[
            "hf_heatmap"
        ].min()
        < 0
        or arrays[
            "hf_heatmap"
        ].max()
        > 1.0001
    ):

        raise RuntimeError(
            f"{stem}: hf_heatmap outside [0,1]"
        )

    if (
        arrays[
            "hf_offset"
        ].min()
        < -1e-5
        or arrays[
            "hf_offset"
        ].max()
        >= 1.0001
    ):

        raise RuntimeError(
            f"{stem}: hf_offset outside expected [0,1)"
        )

    # Step05E4C proved stride-2 HF must have zero collisions.
    if (
        counts[
            "hf_collisions"
        ]
        != 0
    ):

        raise RuntimeError(
            f"{stem}: unexpected HF stride-2 collision."
        )


# =============================================================================
# QC VISUALIZATION
# =============================================================================

def load_gray_native(
    path: Path,
) -> np.ndarray:

    image = cv2.imread(
        str(
            path
        ),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:

        raise RuntimeError(
            f"Could not read image: {path}"
        )

    if (
        image.shape
        != (
            NATIVE_H,
            NATIVE_W,
        )
    ):

        raise RuntimeError(
            f"Unexpected native image shape: {image.shape}"
        )

    return image


def make_qc_image(
    stem: str,
    image_path: Path,
    arrays: Dict[str, np.ndarray],
):

    gray = load_gray_native(
        image_path
    )

    padded = cv2.copyMakeBorder(
        gray,
        PAD_TOP,
        PAD_BOTTOM,
        PAD_LEFT,
        PAD_RIGHT,
        cv2.BORDER_CONSTANT,
        value=0,
    )

    rgb = cv2.cvtColor(
        padded,
        cv2.COLOR_GRAY2BGR,
    )

    # Region contours.
    srf_contours, _ = cv2.findContours(
        arrays[
            "srf_mask"
        ],
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    irf_contours, _ = cv2.findContours(
        arrays[
            "irf_mask"
        ],
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    cv2.drawContours(
        rgb,
        srf_contours,
        -1,
        (
            0,
            255,
            0,
        ),
        1,
    )

    cv2.drawContours(
        rgb,
        irf_contours,
        -1,
        (
            0,
            0,
            255,
        ),
        1,
    )

    # Curves from heatmap peaks.
    # ELM = cyan, EZ = magenta.
    curve_colors = [
        (
            255,
            255,
            0,
        ),
        (
            255,
            0,
            255,
        ),
    ]

    for c in range(
        2
    ):

        presence = arrays[
            "curve_presence"
        ][
            c
        ]

        heat = arrays[
            "curve_centerline"
        ][
            c
        ].astype(
            np.float32
        )

        for x in np.where(
            presence
            > 0
        )[0]:

            y = int(
                np.argmax(
                    heat[
                        :,
                        x
                    ]
                )
            )

            cv2.circle(
                rgb,
                (
                    int(
                        x
                    ),
                    y,
                ),
                1,
                curve_colors[
                    c
                ],
                -1,
            )

    # HF centers from offset mask.
    ys, xs = np.where(
        arrays[
            "hf_offset_mask"
        ]
        > 0
    )

    for cy, cx in zip(
        ys,
        xs,
    ):

        dx = float(
            arrays[
                "hf_offset"
            ][
                0,
                cy,
                cx,
            ]
        )

        dy = float(
            arrays[
                "hf_offset"
            ][
                1,
                cy,
                cx,
            ]
        )

        x_model = int(
            round(
                (
                    cx
                    + dx
                )
                * HF_STRIDE
            )
        )

        y_model = int(
            round(
                (
                    cy
                    + dy
                )
                * HF_STRIDE
            )
        )

        cv2.circle(
            rgb,
            (
                x_model,
                y_model,
            ),
            3,
            (
                255,
                255,
                255,
            ),
            1,
        )

    # Raw overlap ignore pixels = yellow.
    ignore_overlap = (
        (
            arrays[
                "srf_mask"
            ]
            > 0
        )
        & (
            arrays[
                "irf_mask"
            ]
            > 0
        )
    )

    oy, ox = np.where(
        ignore_overlap
    )

    for y, x in zip(
        oy,
        ox,
    ):

        rgb[
            y,
            x,
        ] = (
            0,
            255,
            255,
        )

    cv2.putText(
        rgb,
        (
            f"{stem} | SRF green | IRF red | "
            "ELM cyan | EZ magenta | HF white | overlap yellow"
        ),
        (
            8,
            18,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (
            255,
            255,
            255,
        ),
        1,
        cv2.LINE_AA,
    )

    QC_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(
            QC_DIR
            / f"{stem}_geometry_v2_qc.png"
        ),
        rgb,
    )


# =============================================================================
# TARGET SPEC
# =============================================================================

def build_target_spec(
    amendment: Dict[str, Any],
) -> Dict[str, Any]:

    spec = {
        "name":
            "RVO-ME geometry_v2",

        "status":
            "FROZEN_AFTER_SUCCESSFUL_BUILD",

        "scientific_parent":
            "GeoRCP-MIRAGE v1",

        "scientific_parent_sha256":
            EXPECTED_V1_MASTER_SHA,

        "target_validity_amendment":
            "GeoRCP-MIRAGE v1.1",

        "target_validity_amendment_sha256":
            amendment[
                "amendment_sha256"
            ],

        "source":
            "raw RVO_Lesion_Labelme float geometry",

        "canvas": {
            "native_wh":
                [
                    NATIVE_W,
                    NATIVE_H,
                ],

            "model_wh":
                [
                    MODEL_W,
                    MODEL_H,
                ],

            "padding_lrtb":
                [
                    PAD_LEFT,
                    PAD_RIGHT,
                    PAD_TOP,
                    PAD_BOTTOM,
                ],

            "resize":
                False,

            "near_native_protocol_sha256":
                EXPECTED_NEAR_NATIVE_SHA,
        },

        "region": {
            "channels/classes":
                [
                    "background",
                    "SRF",
                    "IRF",
                ],

            "ignore_index":
                IGNORE_INDEX,

            "raw_overlap_policy":
                (
                    "raw SRF/IRF overlap pixels ignored for "
                    "interior region classification; no precedence"
                ),

            "independent_binary_masks_saved":
                True,

            "class_specific_boundaries_saved":
                True,
        },

        "curve": {
            "channel_order":
                [
                    "ELM",
                    "EZ",
                ],

            "centerline_sigma_native_px":
                CURVE_SIGMA_PX,

            "distance_truncation_native_px":
                CURVE_DISTANCE_TRUNC_PX,

            "presence_saved":
                True,

            "y_distribution_storage":
                (
                    "derived on-the-fly by normalizing centerline "
                    "target over y on present columns"
                ),

            "global_connectivity_forced":
                False,
        },

        "HF": {
            "stride":
                HF_STRIDE,

            "heatmap_hw":
                [
                    HF_H,
                    HF_W,
                ],

            "gaussian_sigma_cells":
                HF_SIGMA_CELLS,

            "offset":
                (
                    "fractional coordinate inside floor(stride-2 cell)"
                ),

            "expected_cell_collisions":
                0,
        },

        "storage": {
            "format":
                "one compressed NPZ per image",

            "floating_dtype":
                "float16",

            "binary_dtype":
                "uint8",

            "master_hash":
                (
                    "deterministic array-content hashes, not NPZ file bytes"
                ),
        },

        "dependencies": {
            "geometry_v1_sha256":
                EXPECTED_GEOMETRY_V1_SHA,

            "patient_5fold_protocol_sha256":
                EXPECTED_PATIENT_PROTOCOL_SHA,
        },
    }

    spec[
        "target_spec_sha256"
    ] = stable_json_sha(
        spec
    )

    return spec


# =============================================================================
# PROJECT LOG
# =============================================================================

def append_project_log(
    master_sha: str,
    amendment_sha: str,
    output_size: str,
):
    """
    Idempotent append.
    """

    PROJECT_LOG.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if PROJECT_LOG.exists():

        existing = PROJECT_LOG.read_text(
            encoding="utf-8"
        )

    else:

        existing = ""

    if LOG_MARKER in existing:

        return False

    block = f"""

{LOG_MARKER}
## Step 06 — geometry_v2 target generation

Status: COMPLETE / FROZEN.

- GeoRCP-MIRAGE v1 scientific parent SHA: `{EXPECTED_V1_MASTER_SHA}`
- v1.1 target-validity amendment SHA: `{amendment_sha}`
- geometry_v2 master content SHA: `{master_sha}`
- Images: 3012
- Valid primary annotations: IRF 39534, SRF 1288, ELM 2384, EZ 4474, HF 9212
- IRF/SRF raw overlap: 165 pixels across 17 images; ignored for region-interior classification, no class precedence
- HF output stride: 2, zero center-cell collisions
- Canvas: 576x384 padding-only from native 570x380
- geometry_v2 storage size: {output_size}

Do not regenerate or modify geometry_v2 after this freeze. Any future target change must use a new version directory.
"""

    with PROJECT_LOG.open(
        "a",
        encoding="utf-8",
    ) as f:

        f.write(
            block
        )

    return True


# =============================================================================
# MAIN
# =============================================================================

def main():

    print(
        "\n"
        + "=" * 100
    )

    print(
        "RVO-ME STEP 06: CONSOLIDATED GEOMETRY_V2 TARGET BUILDER"
    )

    print(
        "=" * 100
    )

    # -------------------------------------------------------------------------
    # 1. Prerequisites
    # -------------------------------------------------------------------------

    print(
        "\n[1/8] Verifying frozen scientific design and overlap decision..."
    )

    (
        design,
        e5c,
    ) = verify_prerequisites()

    print(
        "    GeoRCP-MIRAGE v1 master SHA: PASS"
    )

    print(
        "    Step05E5C boundary-ambiguity decision: PASS"
    )

    # -------------------------------------------------------------------------
    # 2. Amendment
    # -------------------------------------------------------------------------

    print(
        "\n[2/8] Freezing GeoRCP-MIRAGE v1.1 target-validity amendment..."
    )

    amendment = create_v1_1_amendment(
        e5c
    )

    save_json(
        DESIGN_DIR
        / "TARGET_VALIDITY_AMENDMENT_v1_1.json",
        amendment,
    )

    print(
        "    Architecture change: NO"
    )

    print(
        "    Region overlap policy: IGNORE 165 ambiguous pixels"
    )

    print(
        "    SRF/IRF precedence: NONE"
    )

    print(
        "    Amendment SHA:"
    )

    print(
        "    "
        + amendment[
            "amendment_sha256"
        ]
    )

    # -------------------------------------------------------------------------
    # 3. Discover source data
    # -------------------------------------------------------------------------

    print(
        "\n[3/8] Discovering source images and LabelMe geometry..."
    )

    jsons = find_json_by_stem()

    images = find_images_by_stem(
        IMAGE_DIR
    )

    if len(
        jsons
    ) != EXPECTED_IMAGES:

        raise RuntimeError(
            f"Expected {EXPECTED_IMAGES} JSON files, found {len(jsons)}."
        )

    if len(
        images
    ) != EXPECTED_IMAGES:

        raise RuntimeError(
            f"Expected {EXPECTED_IMAGES} images, found {len(images)}."
        )

    if set(
        jsons
    ) != set(
        images
    ):

        raise RuntimeError(
            "Image/JSON filename stems do not match."
        )

    stems = sorted(
        jsons
    )

    print(
        f"    Images/JSON pairs: {len(stems)} PASS"
    )

    # -------------------------------------------------------------------------
    # 4. Prepare output
    # -------------------------------------------------------------------------

    print(
        "\n[4/8] Preparing geometry_v2 output..."
    )

    if OUTPUT_DIR.exists():

        complete_file = (
            OUTPUT_DIR
            / "GEOMETRY_V2_COMPLETE.json"
        )

        if complete_file.exists():

            existing = load_json(
                complete_file
            )

            if (
                existing.get(
                    "status"
                )
                == "COMPLETE"
            ):

                raise RuntimeError(
                    "A COMPLETE geometry_v2 already exists.\n"
                    f"{OUTPUT_DIR}\n"
                    "Refusing to overwrite a frozen target set."
                )

        # Incomplete previous attempt only: clean target/QC files,
        # preserving nothing as frozen.
        shutil.rmtree(
            OUTPUT_DIR
        )

    TARGET_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    QC_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    target_spec = build_target_spec(
        amendment
    )

    save_json(
        OUTPUT_DIR
        / "TARGET_SPEC.json",
        target_spec,
    )

    print(
        "    Fresh output directory: PASS"
    )

    # -------------------------------------------------------------------------
    # 5. Build
    # -------------------------------------------------------------------------

    print(
        "\n[5/8] Building all 3012 geometry_v2 target packs..."
    )

    native_fov = make_native_fov_mask()

    curve_column_valid = (
        make_curve_column_valid()
    )

    hf_valid = make_hf_valid(
        native_fov
    )

    annotation_totals = Counter()

    positive_images = Counter()

    overlap_images = 0
    overlap_pixels = 0

    hf_collisions = 0

    manifest_rows = []

    qc_candidates = []

    for index, stem in enumerate(
        stems,
        start=1,
    ):

        arrays, counts = build_one(
            stem,
            jsons[
                stem
            ],
            native_fov,
            curve_column_valid,
            hf_valid,
        )

        validate_arrays(
            stem,
            arrays,
            counts,
        )

        for label in (
            "IRF",
            "SRF",
            "ELM",
            "EZ",
            "HF",
        ):

            annotation_totals[
                label
            ] += counts[
                label
            ]

            if (
                counts[
                    label
                ]
                > 0
            ):

                positive_images[
                    label
                ] += 1

        if (
            counts[
                "region_overlap_pixels"
            ]
            > 0
        ):

            overlap_images += 1

            overlap_pixels += counts[
                "region_overlap_pixels"
            ]

        hf_collisions += counts[
            "hf_collisions"
        ]

        content_sha = array_content_sha(
            arrays
        )

        out_path = (
            TARGET_DIR
            / f"{stem}.npz"
        )

        np.savez_compressed(
            out_path,
            **arrays,
        )

        # Score for deterministic QC selection: prefer images with
        # heterogeneous geometry and some HF.
        qc_score = (
            3
            * int(
                counts[
                    "SRF"
                ]
                > 0
            )
            + 3
            * int(
                counts[
                    "IRF"
                ]
                > 0
            )
            + 2
            * int(
                counts[
                    "ELM"
                ]
                > 0
            )
            + 2
            * int(
                counts[
                    "EZ"
                ]
                > 0
            )
            + 2
            * min(
                counts[
                    "HF"
                ],
                5,
            )
            + 5
            * int(
                counts[
                    "region_overlap_pixels"
                ]
                > 0
            )
        )

        qc_candidates.append(
            (
                -qc_score,
                stem,
            )
        )

        manifest_rows.append({
            "stem":
                stem,

            "target_file":
                str(
                    out_path
                ),

            "content_sha256":
                content_sha,

            "IRF_annotations":
                counts[
                    "IRF"
                ],

            "SRF_annotations":
                counts[
                    "SRF"
                ],

            "ELM_annotations":
                counts[
                    "ELM"
                ],

            "EZ_annotations":
                counts[
                    "EZ"
                ],

            "HF_points":
                counts[
                    "HF"
                ],

            "region_overlap_ignore_pixels":
                counts[
                    "region_overlap_pixels"
                ],

            "hf_stride2_collisions":
                counts[
                    "hf_collisions"
                ],
        })

        if (
            index % 250 == 0
            or index == len(
                stems
            )
        ):

            print(
                f"    {index:4d}/{len(stems)} complete"
            )

    # -------------------------------------------------------------------------
    # 6. Full-dataset validation
    # -------------------------------------------------------------------------

    print(
        "\n[6/8] Validating global counts and scientific invariants..."
    )

    for label, expected in (
        EXPECTED_VALID.items()
    ):

        found = int(
            annotation_totals[
                label
            ]
        )

        if found != expected:

            raise RuntimeError(
                f"{label} valid count mismatch: "
                f"expected {expected}, found {found}"
            )

        print(
            f"    {label} valid annotations: "
            f"{found} PASS"
        )

    for label, expected in (
        EXPECTED_POSITIVE_IMAGES.items()
    ):

        found = int(
            positive_images[
                label
            ]
        )

        if found != expected:

            raise RuntimeError(
                f"{label} positive-image mismatch: "
                f"expected {expected}, found {found}"
            )

        print(
            f"    {label} positive images: "
            f"{found} PASS"
        )

    if (
        overlap_images
        != EXPECTED_OVERLAP_IMAGES
    ):

        raise RuntimeError(
            "IRF/SRF overlap-image count mismatch."
        )

    if (
        overlap_pixels
        != EXPECTED_OVERLAP_PIXELS
    ):

        raise RuntimeError(
            "IRF/SRF overlap-pixel count mismatch."
        )

    print(
        f"    Region overlap ignore: "
        f"{overlap_pixels} px / {overlap_images} images PASS"
    )

    if (
        hf_collisions
        != 0
    ):

        raise RuntimeError(
            "HF stride-2 collision invariant failed."
        )

    print(
        "    HF stride-2 center collisions: 0 PASS"
    )

    # -------------------------------------------------------------------------
    # 7. QC + deterministic manifest/master hash
    # -------------------------------------------------------------------------

    print(
        "\n[7/8] Generating QC and freezing deterministic content manifest..."
    )

    qc_candidates.sort()

    selected_qc = [
        stem
        for (
            _score,
            stem,
        ) in qc_candidates[
            :N_QC_IMAGES
        ]
    ]

    for stem in selected_qc:

        with np.load(
            TARGET_DIR
            / f"{stem}.npz"
        ) as data:

            arrays = {
                key:
                    data[
                        key
                    ]
                for key in data.files
            }

        make_qc_image(
            stem,
            images[
                stem
            ],
            arrays,
        )

    print(
        f"    QC images: {len(selected_qc)} PASS"
    )

    write_csv(
        OUTPUT_DIR
        / "manifest.csv",
        manifest_rows,
    )

    manifest_for_hash = [
        {
            "stem":
                row[
                    "stem"
                ],

            "content_sha256":
                row[
                    "content_sha256"
                ],
        }
        for row in manifest_rows
    ]

    geometry_v2_master = {
        "target_spec_sha256":
            target_spec[
                "target_spec_sha256"
            ],

        "amendment_sha256":
            amendment[
                "amendment_sha256"
            ],

        "images":
            EXPECTED_IMAGES,

        "manifest_content":
            manifest_for_hash,
    }

    geometry_v2_master_sha = (
        stable_json_sha(
            geometry_v2_master
        )
    )

    save_json(
        OUTPUT_DIR
        / "CONTENT_HASH_MANIFEST.json",
        {
            "master_content_sha256":
                geometry_v2_master_sha,

            "target_spec_sha256":
                target_spec[
                    "target_spec_sha256"
                ],

            "amendment_sha256":
                amendment[
                    "amendment_sha256"
                ],

            "files":
                manifest_for_hash,
        },
    )

    print(
        "    geometry_v2 master content SHA:"
    )

    print(
        "    "
        + geometry_v2_master_sha
    )

    # -------------------------------------------------------------------------
    # 8. Freeze / log
    # -------------------------------------------------------------------------

    print(
        "\n[8/8] Freezing geometry_v2 and updating PROJECT_LOG..."
    )

    output_size_bytes = (
        directory_size_bytes(
            OUTPUT_DIR
        )
    )

    output_size_text = (
        human_bytes(
            output_size_bytes
        )
    )

    complete = {
        "status":
            "COMPLETE",

        "step":
            "06",

        "target_version":
            "geometry_v2",

        "scientific_parent":
            "GeoRCP-MIRAGE v1",

        "scientific_parent_sha256":
            EXPECTED_V1_MASTER_SHA,

        "target_validity_amendment":
            "v1.1",

        "target_validity_amendment_sha256":
            amendment[
                "amendment_sha256"
            ],

        "target_spec_sha256":
            target_spec[
                "target_spec_sha256"
            ],

        "geometry_v2_master_content_sha256":
            geometry_v2_master_sha,

        "images":
            EXPECTED_IMAGES,

        "annotation_totals":
            {
                label:
                    int(
                        annotation_totals[
                            label
                        ]
                    )
                for label in (
                    "IRF",
                    "SRF",
                    "ELM",
                    "EZ",
                    "HF",
                )
            },

        "positive_images":
            {
                label:
                    int(
                        positive_images[
                            label
                        ]
                    )
                for label in (
                    "IRF",
                    "SRF",
                    "ELM",
                    "EZ",
                    "HF",
                )
            },

        "region_overlap_ignore_pixels":
            overlap_pixels,

        "region_overlap_images":
            overlap_images,

        "hf_stride2_collisions":
            hf_collisions,

        "qc_images":
            selected_qc,

        "output_size_bytes":
            output_size_bytes,

        "output_size_human":
            output_size_text,

        "frozen_policy":
            (
                "Do not modify or regenerate geometry_v2. "
                "Any future target change requires geometry_v3."
            ),
    }

    save_json(
        OUTPUT_DIR
        / "GEOMETRY_V2_COMPLETE.json",
        complete,
    )

    log_appended = append_project_log(
        geometry_v2_master_sha,
        amendment[
            "amendment_sha256"
        ],
        output_size_text,
    )

    print(
        f"    Output size: {output_size_text}"
    )

    print(
        "    PROJECT_LOG append: "
        + (
            "PASS"
            if log_appended
            else "already present"
        )
    )

    print(
        "\n"
        + "-" * 100
    )

    print(
        "GEOMETRY_V2 FROZEN"
    )

    print(
        "-" * 100
    )

    print(
        "Images: 3012"
    )

    print(
        "IRF: 39534 annotations / 2893 positive images"
    )

    print(
        "SRF: 1288 annotations / 1215 positive images"
    )

    print(
        "ELM: 2384 annotations / 1625 positive images"
    )

    print(
        "EZ: 4474 annotations / 2730 positive images"
    )

    print(
        "HF: 9212 points / 2286 positive images"
    )

    print(
        "IRF/SRF ambiguous overlap: "
        "165 pixels in 17 images -> IGNORE"
    )

    print(
        "HF stride-2 collisions: 0"
    )

    print(
        "\nTarget-validity amendment SHA-256:"
    )

    print(
        amendment[
            "amendment_sha256"
        ]
    )

    print(
        "\ngeometry_v2 master content SHA-256:"
    )

    print(
        geometry_v2_master_sha
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
        "STEP 06 COMPLETE"
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()
