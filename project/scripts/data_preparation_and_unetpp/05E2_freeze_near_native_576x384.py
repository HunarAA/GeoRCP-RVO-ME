# =============================================================================
# RVO-ME STEP 05E2 — CORRECTED
# Freeze and validate the near-native 570x380 -> 576x384 padding protocol.
#
# SCIENTIFIC PURPOSE
# ------------------
# Preserve the released OCT raster and annotation geometry WITHOUT resizing.
#
# Source canvas:
#     width  = 570
#     height = 380
#
# Model canvas:
#     width  = 576
#     height = 384
#
# Padding:
#     left   = 3
#     right  = 3
#     top    = 2
#     bottom = 2
#
# Coordinate transform:
#     x_model = x_native + 3
#     y_model = y_native + 2
#
# Inverse:
#     x_native = x_model - 3
#     y_native = y_model - 2
#
# Images and masks are NOT resampled.
# No interpolation is applied.
# No source files are modified.
# =============================================================================

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


# =============================================================================
# PATHS / FROZEN DATASET EXPECTATIONS
# =============================================================================

ROOT = Path(r"E:\Papers\RVO_ME")
DATASET_ROOT = ROOT / "RVO-Lesion"

IMAGE_DIR = DATASET_ROOT / "Image_Seg" / "images"
MASK_DIR = DATASET_ROOT / "Image_Seg" / "masks"
LABELME_ROOT = DATASET_ROOT / "RVO_Lesion_Labelme"

OUTPUT_DIR = (
    ROOT
    / "project"
    / "preprocessing"
    / "near_native_576x384_v1"
)

SOURCE_W = 570
SOURCE_H = 380

TARGET_W = 576
TARGET_H = 384

PAD_LEFT = 3
PAD_RIGHT = 3
PAD_TOP = 2
PAD_BOTTOM = 2

EXPECTED_IMAGES = 3012
EXPECTED_MASKS = 3012
EXPECTED_JSON = 3012
EXPECTED_EYES = 146

EXPECTED_GEOMETRY_V1_SHA256 = (
    "a4029d5ab4570237ae607d1cc24d551ce438845eab1814e8a82735a122daeb55"
)

EXPECTED_PATIENT_PROTOCOL_SHA256 = (
    "52ed516dcc21f1c51068c59fb662c9a0030b143e6440d6f4a0380df713c9bcaf"
)

SEED = 20260815
N_ARRAY_ROUNDTRIP_CHECKS = 128


# =============================================================================
# GENERIC HELPERS
# =============================================================================

def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )


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

    fields = list(rows[0].keys())

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


def stable_json_sha256(obj: Any) -> str:

    raw = json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(raw).hexdigest()


def image_files(folder: Path) -> Dict[str, Path]:

    if not folder.exists():
        raise FileNotFoundError(folder)

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
                    f"Duplicate image stem in {folder}: "
                    f"{p.stem}"
                )

            out[p.stem] = p

    return out


def find_json_files(root: Path) -> Dict[str, Path]:

    if not root.exists():
        raise FileNotFoundError(root)

    out: Dict[str, Path] = {}

    for p in root.rglob("*.json"):

        if p.stem in out:
            raise RuntimeError(
                "Duplicate JSON stem found:\n"
                f"{p.stem}\n"
                f"{out[p.stem]}\n"
                f"{p}"
            )

        out[p.stem] = p

    return out


def parse_eye_id(stem: str) -> int | None:

    m = re.match(
        r"^(\d+)_\d+$",
        stem,
    )

    if not m:
        return None

    return int(m.group(1))


# =============================================================================
# NEAR-NATIVE TRANSFORM
# =============================================================================

def pad_array(
    arr: np.ndarray,
    value: int = 0,
) -> np.ndarray:

    return cv2.copyMakeBorder(
        arr,
        PAD_TOP,
        PAD_BOTTOM,
        PAD_LEFT,
        PAD_RIGHT,
        borderType=cv2.BORDER_CONSTANT,
        value=value,
    )


def crop_back(
    arr: np.ndarray,
) -> np.ndarray:

    return arr[
        PAD_TOP : PAD_TOP + SOURCE_H,
        PAD_LEFT : PAD_LEFT + SOURCE_W,
        ...
    ]


def transform_xy(
    x: float,
    y: float,
) -> Tuple[float, float]:

    return (
        float(x) + PAD_LEFT,
        float(y) + PAD_TOP,
    )


def inverse_xy(
    x: float,
    y: float,
) -> Tuple[float, float]:

    return (
        float(x) - PAD_LEFT,
        float(y) - PAD_TOP,
    )


# =============================================================================
# STATIC PROTOCOL
# =============================================================================

def build_protocol() -> Dict[str, Any]:

    if (
        SOURCE_W
        + PAD_LEFT
        + PAD_RIGHT
        != TARGET_W
    ):
        raise RuntimeError(
            "Width padding does not produce target width."
        )

    if (
        SOURCE_H
        + PAD_TOP
        + PAD_BOTTOM
        != TARGET_H
    ):
        raise RuntimeError(
            "Height padding does not produce target height."
        )

    if (
        TARGET_W % 16 != 0
        or TARGET_H % 16 != 0
    ):
        raise RuntimeError(
            "Target dimensions must be divisible by 16."
        )

    protocol: Dict[str, Any] = {
        "name":
            "RVO-ME near-native padding protocol v1",

        "version":
            1,

        "scientific_status":
            "FROZEN_FOR_MODERN_BASELINES_AND_PROPOSED_MODEL",

        "source_raster": {
            "width": SOURCE_W,
            "height": SOURCE_H,
        },

        "model_raster": {
            "width": TARGET_W,
            "height": TARGET_H,
        },

        "transform_type":
            "constant_zero_padding_only",

        "resize":
            False,

        "interpolation":
            None,

        "padding": {
            "left": PAD_LEFT,
            "right": PAD_RIGHT,
            "top": PAD_TOP,
            "bottom": PAD_BOTTOM,
            "image_value": 0,
            "mask_value": 0,
        },

        "coordinate_forward": {
            "x_model":
                "x_native + 3",

            "y_model":
                "y_native + 2",
        },

        "coordinate_inverse": {
            "x_native":
                "x_model - 3",

            "y_native":
                "y_model - 2",
        },

        "geometry_policy": {
            "float_coordinates_remain_float":
                True,

            "no_coordinate_rounding":
                True,

            "no_coordinate_clipping_for_authoritative_geometry":
                True,

            "rasterization_clipping_only_if_required_by_target_builder":
                True,
        },

        "evaluation_policy": {
            "native_prediction_crop":
                "[top:top+380, left:left+570]",

            "primary_native_evaluation":
                (
                    "crop model prediction back to exact "
                    "570x380 before native-raster evaluation"
                ),

            "no_resize_back_to_native":
                True,
        },

        "model_compatibility": {
            "width_divisible_by_16":
                TARGET_W % 16 == 0,

            "height_divisible_by_16":
                TARGET_H % 16 == 0,

            "width_patch16_tokens":
                TARGET_W // 16,

            "height_patch16_tokens":
                TARGET_H // 16,

            "patch16_token_grid": [
                TARGET_H // 16,
                TARGET_W // 16,
            ],
        },

        "frozen_project_dependencies": {
            "geometry_v1_sha256":
                EXPECTED_GEOMETRY_V1_SHA256,

            "patient_5fold_protocol_sha256":
                EXPECTED_PATIENT_PROTOCOL_SHA256,
        },

        "important_note": (
            "This protocol preserves the released 570x380 "
            "raster exactly. The added border is outside "
            "the native field of view and is removed before "
            "native-raster evaluation."
        ),
    }

    protocol[
        "near_native_protocol_sha256"
    ] = stable_json_sha256(
        protocol
    )

    return protocol


# =============================================================================
# DATASET PAIRING AUDIT
# =============================================================================

def audit_dataset():

    images = image_files(
        IMAGE_DIR
    )

    masks = image_files(
        MASK_DIR
    )

    jsons = find_json_files(
        LABELME_ROOT
    )

    image_stems = set(
        images
    )

    mask_stems = set(
        masks
    )

    json_stems = set(
        jsons
    )

    if len(images) != EXPECTED_IMAGES:
        raise RuntimeError(
            f"Expected {EXPECTED_IMAGES} images, "
            f"found {len(images)}."
        )

    if len(masks) != EXPECTED_MASKS:
        raise RuntimeError(
            f"Expected {EXPECTED_MASKS} masks, "
            f"found {len(masks)}."
        )

    if len(jsons) != EXPECTED_JSON:
        raise RuntimeError(
            f"Expected {EXPECTED_JSON} JSON files, "
            f"found {len(jsons)}."
        )

    if image_stems != mask_stems:
        missing_masks = sorted(
            image_stems - mask_stems
        )[:20]

        extra_masks = sorted(
            mask_stems - image_stems
        )[:20]

        raise RuntimeError(
            "Image/mask stems do not match exactly.\n"
            f"Missing masks: {missing_masks}\n"
            f"Extra masks:   {extra_masks}"
        )

    if image_stems != json_stems:

        missing_json = sorted(
            image_stems - json_stems
        )[:20]

        extra_json = sorted(
            json_stems - image_stems
        )[:20]

        raise RuntimeError(
            "Image/JSON stems do not match exactly.\n"
            f"Missing JSON: {missing_json}\n"
            f"Extra JSON:   {extra_json}"
        )

    eye_ids = {
        parse_eye_id(stem)
        for stem in image_stems
    }

    if None in eye_ids:
        raise RuntimeError(
            "At least one filename could not be "
            "parsed as EyeID_Index."
        )

    if len(eye_ids) != EXPECTED_EYES:
        raise RuntimeError(
            f"Expected {EXPECTED_EYES} eyes, "
            f"found {len(eye_ids)}."
        )

    return (
        images,
        masks,
        jsons,
        eye_ids,
    )


# =============================================================================
# FULL RASTER VALIDATION
# =============================================================================

def validate_rasters(
    images: Dict[str, Path],
    masks: Dict[str, Path],
) -> List[Dict[str, Any]]:

    rows: List[Dict[str, Any]] = []

    image_dimension_failures = 0
    mask_dimension_failures = 0
    mask_value_failures = 0

    valid_mask_values = {
        0,
        1,
        2,
        3,
        4,
    }

    for stem in sorted(images):

        image = cv2.imread(
            str(images[stem]),
            cv2.IMREAD_GRAYSCALE,
        )

        mask = cv2.imread(
            str(masks[stem]),
            cv2.IMREAD_UNCHANGED,
        )

        if image is None:
            raise RuntimeError(
                f"Could not read image: "
                f"{images[stem]}"
            )

        if mask is None:
            raise RuntimeError(
                f"Could not read mask: "
                f"{masks[stem]}"
            )

        ih, iw = image.shape[:2]
        mh, mw = mask.shape[:2]

        if (
            iw,
            ih,
        ) != (
            SOURCE_W,
            SOURCE_H,
        ):
            image_dimension_failures += 1

        if (
            mw,
            mh,
        ) != (
            SOURCE_W,
            SOURCE_H,
        ):
            mask_dimension_failures += 1

        unique_mask = {
            int(v)
            for v in np.unique(mask)
        }

        if not unique_mask.issubset(
            valid_mask_values
        ):
            mask_value_failures += 1

        rows.append({
            "stem":
                stem,

            "eye_id":
                parse_eye_id(stem),

            "image_width":
                iw,

            "image_height":
                ih,

            "mask_width":
                mw,

            "mask_height":
                mh,

            "mask_values":
                ",".join(
                    str(x)
                    for x in sorted(
                        unique_mask
                    )
                ),
        })

    if image_dimension_failures:
        raise RuntimeError(
            f"{image_dimension_failures} image "
            "dimension failures."
        )

    if mask_dimension_failures:
        raise RuntimeError(
            f"{mask_dimension_failures} mask "
            "dimension failures."
        )

    if mask_value_failures:
        raise RuntimeError(
            f"{mask_value_failures} unexpected "
            "mask-value failures."
        )

    return rows


# =============================================================================
# EXACT ARRAY ROUND-TRIP TEST
# =============================================================================

def exact_array_roundtrip_test(
    images: Dict[str, Path],
    masks: Dict[str, Path],
) -> Dict[str, Any]:

    rng = random.Random(
        SEED
    )

    stems = sorted(
        images
    )

    if (
        len(stems)
        <= N_ARRAY_ROUNDTRIP_CHECKS
    ):
        selected = stems

    else:
        selected = sorted(
            rng.sample(
                stems,
                N_ARRAY_ROUNDTRIP_CHECKS,
            )
        )

    image_pass = 0
    mask_pass = 0

    padded_image_shapes = set()
    padded_mask_shapes = set()

    for stem in selected:

        image = cv2.imread(
            str(images[stem]),
            cv2.IMREAD_GRAYSCALE,
        )

        mask = cv2.imread(
            str(masks[stem]),
            cv2.IMREAD_UNCHANGED,
        )

        if image is None:
            raise RuntimeError(
                f"Could not read image: "
                f"{images[stem]}"
            )

        if mask is None:
            raise RuntimeError(
                f"Could not read mask: "
                f"{masks[stem]}"
            )

        padded_image = pad_array(
            image,
            0,
        )

        padded_mask = pad_array(
            mask,
            0,
        )

        padded_image_shapes.add(
            tuple(
                padded_image.shape[:2]
            )
        )

        padded_mask_shapes.add(
            tuple(
                padded_mask.shape[:2]
            )
        )

        restored_image = crop_back(
            padded_image
        )

        restored_mask = crop_back(
            padded_mask
        )

        if np.array_equal(
            image,
            restored_image,
        ):
            image_pass += 1

        else:
            raise RuntimeError(
                "Image padding/cropping changed "
                f"pixels: {stem}"
            )

        if np.array_equal(
            mask,
            restored_mask,
        ):
            mask_pass += 1

        else:
            raise RuntimeError(
                "Mask padding/cropping changed "
                f"pixels: {stem}"
            )

    expected_shape = (
        TARGET_H,
        TARGET_W,
    )

    if padded_image_shapes != {
        expected_shape
    }:
        raise RuntimeError(
            "Unexpected padded image shapes: "
            f"{padded_image_shapes}"
        )

    if padded_mask_shapes != {
        expected_shape
    }:
        raise RuntimeError(
            "Unexpected padded mask shapes: "
            f"{padded_mask_shapes}"
        )

    return {
        "tested_files":
            len(selected),

        "image_exact_roundtrip_pass":
            image_pass,

        "mask_exact_roundtrip_pass":
            mask_pass,

        "padded_shape_hw": [
            TARGET_H,
            TARGET_W,
        ],
    }


# =============================================================================
# LABELME FLOAT-COORDINATE AUDIT
# =============================================================================

def audit_labelme_coordinates(
    jsons: Dict[str, Path],
):

    total_shapes = 0
    total_points = 0
    malformed_points = 0

    max_roundtrip_abs_error = 0.0

    native_in_bounds_points = 0
    native_out_of_bounds_points = 0

    transformed_in_canvas_points = 0
    transformed_out_of_canvas_points = 0

    min_x = math.inf
    max_x = -math.inf
    min_y = math.inf
    max_y = -math.inf

    rows: List[
        Dict[str, Any]
    ] = []

    for stem in sorted(
        jsons
    ):

        path = jsons[
            stem
        ]

        with path.open(
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

        file_points = 0
        file_native_oob = 0
        file_model_oob = 0
        file_max_error = 0.0

        for shape in shapes:

            total_shapes += 1

            points = shape.get(
                "points",
                [],
            )

            for point in points:

                if (
                    not isinstance(
                        point,
                        (list, tuple),
                    )
                    or len(point) < 2
                ):
                    malformed_points += 1
                    continue

                try:
                    x = float(
                        point[0]
                    )

                    y = float(
                        point[1]
                    )

                except Exception:
                    malformed_points += 1
                    continue

                if not (
                    math.isfinite(x)
                    and math.isfinite(y)
                ):
                    malformed_points += 1
                    continue

                total_points += 1
                file_points += 1

                min_x = min(
                    min_x,
                    x,
                )

                max_x = max(
                    max_x,
                    x,
                )

                min_y = min(
                    min_y,
                    y,
                )

                max_y = max(
                    max_y,
                    y,
                )

                native_ok = (
                    0.0 <= x < SOURCE_W
                    and
                    0.0 <= y < SOURCE_H
                )

                if native_ok:
                    native_in_bounds_points += 1

                else:
                    native_out_of_bounds_points += 1
                    file_native_oob += 1

                tx, ty = transform_xy(
                    x,
                    y,
                )

                model_ok = (
                    0.0 <= tx < TARGET_W
                    and
                    0.0 <= ty < TARGET_H
                )

                if model_ok:
                    transformed_in_canvas_points += 1

                else:
                    transformed_out_of_canvas_points += 1
                    file_model_oob += 1

                rx, ry = inverse_xy(
                    tx,
                    ty,
                )

                err = max(
                    abs(
                        rx - x
                    ),
                    abs(
                        ry - y
                    ),
                )

                max_roundtrip_abs_error = max(
                    max_roundtrip_abs_error,
                    err,
                )

                file_max_error = max(
                    file_max_error,
                    err,
                )

        rows.append({
            "stem":
                stem,

            "eye_id":
                parse_eye_id(
                    stem
                ),

            "n_shapes":
                len(shapes),

            "n_valid_coordinate_points":
                file_points,

            "native_out_of_bounds_points":
                file_native_oob,

            "model_out_of_bounds_points":
                file_model_oob,

            "max_coordinate_roundtrip_abs_error":
                file_max_error,
        })

    if malformed_points:
        raise RuntimeError(
            "Malformed coordinate points found: "
            f"{malformed_points}"
        )

    if (
        max_roundtrip_abs_error
        > 1e-12
    ):
        raise RuntimeError(
            "Coordinate round-trip error "
            "exceeded tolerance: "
            f"{max_roundtrip_abs_error}"
        )

    return rows, {
        "total_shapes":
            total_shapes,

        "total_valid_coordinate_points":
            total_points,

        "malformed_coordinate_points":
            malformed_points,

        "native_in_bounds_points":
            native_in_bounds_points,

        "native_out_of_bounds_points":
            native_out_of_bounds_points,

        "transformed_in_canvas_points":
            transformed_in_canvas_points,

        "transformed_out_of_canvas_points":
            transformed_out_of_canvas_points,

        "raw_coordinate_extent": {
            "min_x":
                min_x,

            "max_x":
                max_x,

            "min_y":
                min_y,

            "max_y":
                max_y,
        },

        "max_coordinate_roundtrip_abs_error":
            max_roundtrip_abs_error,
    }


# =============================================================================
# HUMAN-READABLE ONLINE TRANSFORM REFERENCE
# =============================================================================

ONLINE_TRANSFORM_EXAMPLE = (
    "# Example training-time transform (padding only)\n"
    "#\n"
    "# image: numpy HxW or HxWxC\n"
    "# target masks: numpy HxW\n"
    "#\n"
    "# image_576x384 = cv2.copyMakeBorder(\n"
    "#     image,\n"
    "#     2, 2, 3, 3,\n"
    "#     cv2.BORDER_CONSTANT,\n"
    "#     value=0,\n"
    "# )\n"
    "#\n"
    "# mask_576x384 = cv2.copyMakeBorder(\n"
    "#     mask,\n"
    "#     2, 2, 3, 3,\n"
    "#     cv2.BORDER_CONSTANT,\n"
    "#     value=0,\n"
    "# )\n"
    "#\n"
    "# Float geometry:\n"
    "#     x_model = x_native + 3\n"
    "#     y_model = y_native + 2\n"
    "#\n"
    "# Native evaluation:\n"
    "#     pred_native = pred_model[2:382, 3:573]\n"
    "#\n"
    "# There is NO resize in either direction.\n"
)


# =============================================================================
# MAIN
# =============================================================================

def main():

    print(
        "\n"
        + "=" * 100
    )

    print(
        "RVO-ME STEP 05E2: "
        "NEAR-NATIVE 576x384 PADDING PROTOCOL"
    )

    print(
        "=" * 100
    )

    # -------------------------------------------------------------------------
    # 1
    # -------------------------------------------------------------------------

    print(
        "\n[1/6] Building frozen "
        "near-native protocol..."
    )

    protocol = build_protocol()

    print(
        "    Source raster: "
        f"{SOURCE_W}x{SOURCE_H}"
    )

    print(
        "    Model raster:  "
        f"{TARGET_W}x{TARGET_H}"
    )

    print(
        "    Padding: "
        f"L{PAD_LEFT} "
        f"R{PAD_RIGHT} "
        f"T{PAD_TOP} "
        f"B{PAD_BOTTOM}"
    )

    print(
        "    Resize/interpolation: NONE"
    )

    print(
        "    Patch-16 token grid: "
        f"{TARGET_H // 16} x "
        f"{TARGET_W // 16}"
    )

    # -------------------------------------------------------------------------
    # 2
    # -------------------------------------------------------------------------

    print(
        "\n[2/6] Verifying "
        "dataset pairing..."
    )

    (
        images,
        masks,
        jsons,
        eye_ids,
    ) = audit_dataset()

    print(
        f"    Images: {len(images)} PASS"
    )

    print(
        f"    Masks:  {len(masks)} PASS"
    )

    print(
        f"    JSON:   {len(jsons)} PASS"
    )

    print(
        f"    Eyes:   {len(eye_ids)} PASS"
    )

    print(
        "    Image/mask/JSON "
        "stem pairing: PASS"
    )

    # -------------------------------------------------------------------------
    # 3
    # -------------------------------------------------------------------------

    print(
        "\n[3/6] Validating all released "
        "image/mask rasters..."
    )

    manifest_rows = validate_rasters(
        images,
        masks,
    )

    print(
        f"    {len(manifest_rows)} "
        "image dimensions: PASS"
    )

    print(
        f"    {len(manifest_rows)} "
        "mask dimensions: PASS"
    )

    print(
        "    Mask values restricted "
        "to {0,1,2,3,4}: PASS"
    )

    # -------------------------------------------------------------------------
    # 4
    # -------------------------------------------------------------------------

    print(
        "\n[4/6] Testing exact array "
        "padding/crop round trip..."
    )

    roundtrip = exact_array_roundtrip_test(
        images,
        masks,
    )

    print(
        "    Files sampled: "
        f"{roundtrip['tested_files']}"
    )

    print(
        "    Images exact pixel round trip: "
        f"{roundtrip['image_exact_roundtrip_pass']} PASS"
    )

    print(
        "    Masks exact pixel round trip:  "
        f"{roundtrip['mask_exact_roundtrip_pass']} PASS"
    )

    print(
        "    Padded raster shape: "
        f"{TARGET_W}x{TARGET_H} PASS"
    )

    # -------------------------------------------------------------------------
    # 5
    # -------------------------------------------------------------------------

    print(
        "\n[5/6] Auditing every "
        "LabelMe float coordinate..."
    )

    (
        coord_rows,
        coord_summary,
    ) = audit_labelme_coordinates(
        jsons
    )

    print(
        "    Total shapes: "
        f"{coord_summary['total_shapes']}"
    )

    print(
        "    Coordinate points: "
        f"{coord_summary['total_valid_coordinate_points']}"
    )

    print(
        "    Malformed points: "
        f"{coord_summary['malformed_coordinate_points']}"
    )

    print(
        "    Native out-of-bounds points: "
        f"{coord_summary['native_out_of_bounds_points']}"
    )

    print(
        "    Model-canvas out-of-bounds points: "
        f"{coord_summary['transformed_out_of_canvas_points']}"
    )

    print(
        "    Max float coordinate round-trip error: "
        f"{coord_summary['max_coordinate_roundtrip_abs_error']:.3e}"
    )

    # -------------------------------------------------------------------------
    # 6
    # -------------------------------------------------------------------------

    print(
        "\n[6/6] Saving frozen protocol "
        "and audit artifacts..."
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        OUTPUT_DIR
        / "image_mask_manifest.csv",
        manifest_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "labelme_coordinate_audit.csv",
        coord_rows,
    )

    save_json(
        OUTPUT_DIR
        / "coordinate_summary.json",
        coord_summary,
    )

    save_json(
        OUTPUT_DIR
        / "array_roundtrip_summary.json",
        roundtrip,
    )

    save_json(
        OUTPUT_DIR
        / "NEAR_NATIVE_PROTOCOL_FROZEN.json",
        protocol,
    )

    with (
        OUTPUT_DIR
        / "ONLINE_TRANSFORM_REFERENCE.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            ONLINE_TRANSFORM_EXAMPLE
        )

    complete: Dict[str, Any] = {
        "status":
            "COMPLETE",

        "step":
            "05E2",

        "script_revision":
            "corrected_full_replacement",

        "dataset_images":
            len(images),

        "dataset_masks":
            len(masks),

        "dataset_json":
            len(jsons),

        "eyes":
            len(eye_ids),

        "source_raster_wh": [
            SOURCE_W,
            SOURCE_H,
        ],

        "model_raster_wh": [
            TARGET_W,
            TARGET_H,
        ],

        "padding_lrtb": [
            PAD_LEFT,
            PAD_RIGHT,
            PAD_TOP,
            PAD_BOTTOM,
        ],

        "resize_used":
            False,

        "interpolation_used":
            False,

        "array_roundtrip":
            roundtrip,

        "coordinate_summary":
            coord_summary,

        "near_native_protocol_sha256":
            protocol[
                "near_native_protocol_sha256"
            ],
    }

    complete[
        "step05e2_summary_sha256"
    ] = stable_json_sha256(
        complete
    )

    save_json(
        OUTPUT_DIR
        / "STEP05E2_COMPLETE.json",
        complete,
    )

    print(
        "\n"
        + "-" * 100
    )

    print(
        "FROZEN NEAR-NATIVE POLICY"
    )

    print(
        "-" * 100
    )

    print(
        "570x380 -> symmetric "
        "zero-padding -> 576x384"
    )

    print(
        "NO spatial resize. "
        "NO interpolation."
    )

    print(
        "Forward coordinates: "
        "x+3, y+2"
    )

    print(
        "Inverse coordinates: "
        "x-3, y-2"
    )

    print(
        "Native evaluation crop: "
        "rows 2:382, cols 3:573"
    )

    print(
        "\nNear-native protocol SHA-256:"
    )

    print(
        protocol[
            "near_native_protocol_sha256"
        ]
    )

    print(
        "\nStep05E2 summary SHA-256:"
    )

    print(
        complete[
            "step05e2_summary_sha256"
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
        "STEP 05E2 COMPLETE"
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()
