from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


# =====================================================================
# CONFIGURATION
# =====================================================================

DATASET_ROOT = Path(r"E:\Papers\RVO_ME\RVO-Lesion")
RAW_LABELME_DIR = DATASET_ROOT / "RVO_Lesion_Labelme"

PROJECT_ROOT = DATASET_ROOT.parent / "project"

SPLIT_ROOT = (
    PROJECT_ROOT
    / "splits"
    / "patient_5fold_v1"
)

SPLIT_MANIFEST = SPLIT_ROOT / "image_manifest.csv"
SPLIT_SUMMARY = SPLIT_ROOT / "split_summary.json"
FOLD_ASSIGNMENTS = SPLIT_ROOT / "fold_assignments.csv"

TARGET_PARENT = PROJECT_ROOT / "targets"

FINAL_OUTPUT_DIR = (
    TARGET_PARENT
    / "geometry_v1"
)

BUILD_OUTPUT_DIR = (
    TARGET_PARENT
    / "geometry_v1__building"
)

# Never replace a frozen target set accidentally.
FORCE_OVERWRITE = False


# =====================================================================
# DATASET VERSION EXPECTATIONS
# =====================================================================

EXPECTED_IMAGES = 3012
EXPECTED_WIDTH = 570
EXPECTED_HEIGHT = 380

EXPECTED_VALID_COUNTS = {
    "IRF": 39534,
    "SRF": 1288,
    "ELM": 2384,
    "EZ": 4474,
    "HF": 9212,
}

EXPECTED_VALID_TOTAL = 56892

EXPECTED_PRIMARY_EXCLUDED = 32
EXPECTED_NON_PRIMARY = 561

EXPECTED_SPLIT_SHA256 = (
    "7e7c2ddf51054bf998b2d750b50c74983b9bc5b751b762213bc96343ffe01346"
)

STRICT_CHECKS = True


# =====================================================================
# GEOMETRY DEFINITIONS
# =====================================================================

REGION_LABELS = ("IRF", "SRF")
CURVE_LABELS = ("ELM", "EZ")
POINT_LABELS = ("HF",)

PRIMARY_LABELS = set(
    REGION_LABELS
    + CURVE_LABELS
    + POINT_LABELS
)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


# =====================================================================
# GENERAL HELPERS
# =====================================================================

def safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def natural_key(text: str):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(
            r"(\d+)",
            str(text),
        )
    ]


def canonical_id(value: Any) -> str:

    if value is None:
        return ""

    if isinstance(value, int):
        return str(value)

    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return str(value)

    text = str(value).strip()

    try:
        number = float(text)

        if (
            number.is_integer()
            and "." in text
        ):
            return str(int(number))

    except ValueError:
        pass

    return text


def get_eye_id(filename: str) -> Optional[str]:

    stem = Path(filename).stem

    match = re.match(
        r"^([^_]+)_",
        stem,
    )

    if match:
        return canonical_id(
            match.group(1)
        )

    return None


def write_csv(
    path: Path,
    rows: List[Dict[str, Any]],
    fieldnames: Optional[List[str]] = None,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        path.write_text(
            "",
            encoding="utf-8",
        )
        return

    if fieldnames is None:
        fieldnames = list(
            rows[0].keys()
        )

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def save_json(
    path: Path,
    data: Any,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )


def sha256_file(path: Path) -> str:

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:

        while True:

            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def write_png(
    path: Path,
    image: np.ndarray,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    success = cv2.imwrite(
        str(path),
        image,
    )

    if not success:

        raise RuntimeError(
            f"Could not write PNG:\n{path}"
        )


# =====================================================================
# FROZEN SPLIT VERIFICATION
# =====================================================================

def load_and_verify_split():

    print(
        "\n[1/7] Verifying frozen patient split..."
    )

    required = [
        SPLIT_MANIFEST,
        SPLIT_SUMMARY,
        FOLD_ASSIGNMENTS,
    ]

    for path in required:

        if not path.exists():

            raise FileNotFoundError(
                f"Required frozen split file "
                f"is missing:\n{path}"
            )

    actual_sha = sha256_file(
        FOLD_ASSIGNMENTS
    )

    with SPLIT_SUMMARY.open(
        "r",
        encoding="utf-8",
    ) as f:

        summary = json.load(f)

    recorded_sha = summary.get(
        "fold_assignment_sha256",
        "",
    )

    if actual_sha != recorded_sha:

        raise RuntimeError(
            "Frozen fold_assignments.csv has "
            "changed since split creation.\n\n"
            f"Recorded: {recorded_sha}\n"
            f"Actual:   {actual_sha}"
        )

    if STRICT_CHECKS:

        if (
            actual_sha
            != EXPECTED_SPLIT_SHA256
        ):

            raise RuntimeError(
                "Frozen split fingerprint does not "
                "match the approved split.\n\n"
                f"Expected:\n{EXPECTED_SPLIT_SHA256}\n\n"
                f"Found:\n{actual_sha}"
            )

    manifest = {}

    with SPLIT_MANIFEST.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            filename = safe_text(
                row.get(
                    "image_filename"
                )
            )

            if not filename:

                raise RuntimeError(
                    "image_manifest.csv contains "
                    "a row without image_filename."
                )

            if filename in manifest:

                raise RuntimeError(
                    "Duplicate filename in frozen "
                    f"manifest: {filename}"
                )

            manifest[
                filename
            ] = row

    if (
        STRICT_CHECKS
        and len(manifest)
        != EXPECTED_IMAGES
    ):

        raise RuntimeError(
            "Expected "
            f"{EXPECTED_IMAGES} images in "
            "frozen split manifest, found "
            f"{len(manifest)}."
        )

    print(
        f"    Images in frozen manifest: "
        f"{len(manifest)}"
    )

    print(
        "    Fold assignment SHA-256:"
    )

    print(
        f"    {actual_sha}"
    )

    print(
        "    Frozen split verification: PASS"
    )

    return manifest, actual_sha


# =====================================================================
# SOURCE FILE LOOKUP
# =====================================================================

def build_source_lookup():

    print(
        "[2/7] Indexing original LabelMe dataset..."
    )

    json_lookup = {}
    image_lookup = {}

    for path in RAW_LABELME_DIR.rglob("*"):

        if not path.is_file():
            continue

        stem = path.stem

        if path.suffix.lower() == ".json":

            if stem in json_lookup:

                raise RuntimeError(
                    "Duplicate raw JSON stem: "
                    f"{stem}"
                )

            json_lookup[
                stem
            ] = path

        elif (
            path.suffix.lower()
            in IMAGE_EXTENSIONS
        ):

            if stem in image_lookup:

                raise RuntimeError(
                    "Duplicate raw image stem: "
                    f"{stem}"
                )

            image_lookup[
                stem
            ] = path

    print(
        f"    JSON files: {len(json_lookup)}"
    )

    print(
        f"    Images:     {len(image_lookup)}"
    )

    if STRICT_CHECKS:

        if (
            len(json_lookup)
            != EXPECTED_IMAGES
        ):

            raise RuntimeError(
                "Unexpected JSON count."
            )

        if (
            len(image_lookup)
            != EXPECTED_IMAGES
        ):

            raise RuntimeError(
                "Unexpected raw image count."
            )

        if (
            set(json_lookup)
            != set(image_lookup)
        ):

            raise RuntimeError(
                "Image/JSON stem sets do not match."
            )

    return (
        json_lookup,
        image_lookup,
    )


# =====================================================================
# GEOMETRY VALIDATION
# =====================================================================

def parse_points(
    raw_points: Any,
) -> Tuple[
    Optional[np.ndarray],
    Optional[str],
]:

    if not isinstance(
        raw_points,
        list,
    ):

        return (
            None,
            "points_not_list",
        )

    try:

        points = np.asarray(
            raw_points,
            dtype=np.float64,
        )

    except Exception:

        return (
            None,
            "points_not_numeric",
        )

    if (
        points.ndim != 2
        or points.shape[1] != 2
    ):

        return (
            None,
            "points_not_Nx2",
        )

    if not np.all(
        np.isfinite(points)
    ):

        return (
            None,
            "non_finite_coordinate",
        )

    return (
        points,
        None,
    )


def classify_shape(
    label: str,
    shape_type: str,
    n_points: int,
) -> Tuple[str, str]:

    # ---------------------------------------------------------
    # Regions
    # ---------------------------------------------------------

    if label in REGION_LABELS:

        if (
            shape_type
            not in {
                "polygon",
                "linestrip",
            }
        ):

            return (
                "EXCLUDE_PRIMARY",
                "invalid_region_shape_type",
            )

        if n_points < 3:

            return (
                "EXCLUDE_PRIMARY",
                "region_has_fewer_than_3_points",
            )

        return (
            "REGION",
            "valid_region",
        )

    # ---------------------------------------------------------
    # Curves
    # ---------------------------------------------------------

    if label in CURVE_LABELS:

        if (
            shape_type
            != "linestrip"
        ):

            return (
                "EXCLUDE_PRIMARY",
                "invalid_curve_shape_type",
            )

        if n_points < 2:

            return (
                "EXCLUDE_PRIMARY",
                "curve_has_fewer_than_2_points",
            )

        return (
            "CURVE",
            "valid_curve",
        )

    # ---------------------------------------------------------
    # Points
    # ---------------------------------------------------------

    if label in POINT_LABELS:

        if (
            shape_type
            != "point"
        ):

            return (
                "EXCLUDE_PRIMARY",
                "invalid_point_shape_type",
            )

        if n_points != 1:

            return (
                "EXCLUDE_PRIMARY",
                "point_does_not_have_exactly_1_coordinate",
            )

        return (
            "POINT",
            "valid_point",
        )

    # ---------------------------------------------------------
    # Non-primary annotations
    # ---------------------------------------------------------

    return (
        "NON_PRIMARY",
        "non_primary_label",
    )


def raster_points(
    points: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:

    result = np.rint(
        points
    ).astype(
        np.int32
    )

    result[:, 0] = np.clip(
        result[:, 0],
        0,
        width - 1,
    )

    result[:, 1] = np.clip(
        result[:, 1],
        0,
        height - 1,
    )

    return result


# =====================================================================
# OUTPUT DIRECTORY SETUP
# =====================================================================

def prepare_output_directory():

    TARGET_PARENT.mkdir(
        parents=True,
        exist_ok=True,
    )

    if FINAL_OUTPUT_DIR.exists():

        if not FORCE_OVERWRITE:

            raise RuntimeError(
                "\nFrozen geometry_v1 already exists:\n"
                f"{FINAL_OUTPUT_DIR}\n\n"
                "It will not be overwritten.\n"
                "If this is an intentional rebuild, "
                "set FORCE_OVERWRITE = True."
            )

        shutil.rmtree(
            FINAL_OUTPUT_DIR
        )

    # A stale partial build is safe to remove because
    # it has never been promoted to geometry_v1.
    if BUILD_OUTPUT_DIR.exists():

        shutil.rmtree(
            BUILD_OUTPUT_DIR
        )

    BUILD_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# =====================================================================
# PROCESS ONE IMAGE
# =====================================================================

def process_one_image(
    stem: str,
    json_path: Path,
    image_path: Path,
    split_row: Dict[str, Any],
):

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:

        raise RuntimeError(
            f"Cannot read source image:\n"
            f"{image_path}"
        )

    height, width = (
        image.shape[:2]
    )

    if STRICT_CHECKS:

        if (
            width != EXPECTED_WIDTH
            or height != EXPECTED_HEIGHT
        ):

            raise RuntimeError(
                f"Unexpected source image size "
                f"for {stem}: "
                f"{width}x{height}"
            )

    with json_path.open(
        "r",
        encoding="utf-8-sig",
    ) as f:

        raw_json = json.load(f)

    json_width = raw_json.get(
        "imageWidth"
    )

    json_height = raw_json.get(
        "imageHeight"
    )

    if (
        json_width is not None
        and int(json_width) != width
    ):

        raise RuntimeError(
            f"JSON/image width mismatch: {stem}"
        )

    if (
        json_height is not None
        and int(json_height) != height
    ):

        raise RuntimeError(
            f"JSON/image height mismatch: {stem}"
        )

    # ---------------------------------------------------------
    # Target arrays
    # ---------------------------------------------------------

    region_masks = {
        label: np.zeros(
            (height, width),
            dtype=np.uint8,
        )
        for label in REGION_LABELS
    }

    region_boundaries = {
        label: np.zeros(
            (height, width),
            dtype=np.uint8,
        )
        for label in REGION_LABELS
    }

    curve_masks = {
        label: np.zeros(
            (height, width),
            dtype=np.uint8,
        )
        for label in CURVE_LABELS
    }

    point_masks = {
        label: np.zeros(
            (height, width),
            dtype=np.uint8,
        )
        for label in POINT_LABELS
    }

    standardized = {
        "schema_version":
            "geometry_v1",

        "image_filename":
            image_path.name,

        "image_stem":
            stem,

        "width":
            width,

        "height":
            height,

        "patient_id":
            canonical_id(
                split_row[
                    "patient_id"
                ]
            ),

        "eye_id":
            canonical_id(
                split_row[
                    "eye_id"
                ]
            ),

        "outer_test_fold":
            int(
                split_row[
                    "outer_test_fold"
                ]
            ),

        "official_split":
            safe_text(
                split_row.get(
                    "official_split"
                )
            ),

        "regions": {
            "IRF": [],
            "SRF": [],
        },

        "curves": {
            "ELM": [],
            "EZ": [],
        },

        "points": {
            "HF": [],
        },
    }

    excluded_rows = []

    valid_counts = Counter()

    primary_excluded = 0
    non_primary = 0
    point_pixel_collisions = 0

    shapes = raw_json.get(
        "shapes",
        [],
    )

    if not isinstance(
        shapes,
        list,
    ):

        raise RuntimeError(
            f"'shapes' is not a list: "
            f"{json_path}"
        )

    for shape_index, shape in enumerate(
        shapes
    ):

        if not isinstance(
            shape,
            dict,
        ):

            raise RuntimeError(
                f"Shape #{shape_index} in "
                f"{json_path} is not a dict."
            )

        label = safe_text(
            shape.get(
                "label"
            )
        )

        shape_type = (
            safe_text(
                shape.get(
                    "shape_type"
                )
            )
            or "polygon"
        )

        points, point_error = (
            parse_points(
                shape.get(
                    "points",
                    []
                )
            )
        )

        n_points = (
            0
            if points is None
            else len(points)
        )

        category, reason = (
            classify_shape(
                label,
                shape_type,
                n_points,
            )
        )

        # -----------------------------------------------------
        # Invalid coordinate structure
        # -----------------------------------------------------

        if point_error is not None:

            if label in PRIMARY_LABELS:
                category = (
                    "EXCLUDE_PRIMARY"
                )

                reason = point_error

            else:
                category = (
                    "NON_PRIMARY"
                )

                reason = (
                    "non_primary_"
                    + point_error
                )

        # -----------------------------------------------------
        # Coordinate-boundary handling
        # -----------------------------------------------------
        #
        # Do NOT exclude an otherwise valid annotation merely
        # because a floating-point coordinate lies on or slightly
        # outside the raster pixel-center range.
        #
        # LabelMe geometry may legitimately contain coordinates
        # such as x == width or y == height at an image boundary.
        #
        # We preserve the ORIGINAL floating-point coordinates in
        # geometry_json and clip only the rasterized representation
        # later in raster_points().
        #
        # Truly malformed coordinates (NaN, infinity, wrong shape)
        # have already been rejected by parse_points().

        # -----------------------------------------------------
        # Exclusions
        # -----------------------------------------------------

        if category in {
            "EXCLUDE_PRIMARY",
            "NON_PRIMARY",
        }:

            if (
                category
                == "EXCLUDE_PRIMARY"
            ):

                primary_excluded += 1

            else:

                non_primary += 1

            excluded_rows.append({
                "image_filename":
                    image_path.name,

                "image_stem":
                    stem,

                "patient_id":
                    standardized[
                        "patient_id"
                    ],

                "eye_id":
                    standardized[
                        "eye_id"
                    ],

                "shape_index":
                    shape_index,

                "label":
                    label,

                "shape_type":
                    shape_type,

                "n_points":
                    n_points,

                "category":
                    category,

                "reason":
                    reason,

                "points_json":
                    json.dumps(
                        shape.get(
                            "points",
                            []
                        ),
                        ensure_ascii=False,
                    ),
            })

            continue

        # -----------------------------------------------------
        # At this point we have valid points
        # -----------------------------------------------------

        assert points is not None

        raw_points = [
            [
                float(x),
                float(y),
            ]
            for x, y in points
        ]

        integer_points = (
            raster_points(
                points,
                width,
                height,
            )
        )

        trace_record = {
            "source_shape_index":
                shape_index,

            "source_shape_type":
                shape_type,

            "points":
                raw_points,
        }

        # -----------------------------------------------------
        # Region target
        # -----------------------------------------------------

        if category == "REGION":

            standardized[
                "regions"
            ][label].append(
                trace_record
            )

            polygon = (
                integer_points
                .reshape(
                    (-1, 1, 2)
                )
            )

            cv2.fillPoly(
                region_masks[
                    label
                ],
                [polygon],
                255,
                lineType=cv2.LINE_8,
            )

            # The region is intrinsically closed even when
            # LabelMe stored the contour as "linestrip".
            cv2.polylines(
                region_boundaries[
                    label
                ],
                [polygon],
                isClosed=True,
                color=255,
                thickness=1,
                lineType=cv2.LINE_8,
            )

        # -----------------------------------------------------
        # Curve target
        # -----------------------------------------------------

        elif category == "CURVE":

            standardized[
                "curves"
            ][label].append(
                trace_record
            )

            polyline = (
                integer_points
                .reshape(
                    (-1, 1, 2)
                )
            )

            cv2.polylines(
                curve_masks[
                    label
                ],
                [polyline],
                isClosed=False,
                color=255,
                thickness=1,
                lineType=cv2.LINE_8,
            )

        # -----------------------------------------------------
        # Point target
        # -----------------------------------------------------

        elif category == "POINT":

            x_float = float(
                points[0, 0]
            )

            y_float = float(
                points[0, 1]
            )

            x_int = int(
                integer_points[
                    0, 0
                ]
            )

            y_int = int(
                integer_points[
                    0, 1
                ]
            )

            standardized[
                "points"
            ][label].append({
                "source_shape_index":
                    shape_index,

                "source_shape_type":
                    shape_type,

                "x":
                    x_float,

                "y":
                    y_float,

                "raster_x":
                    x_int,

                "raster_y":
                    y_int,
            })

            if (
                point_masks[
                    label
                ][
                    y_int,
                    x_int,
                ]
                != 0
            ):

                point_pixel_collisions += 1

            point_masks[
                label
            ][
                y_int,
                x_int,
            ] = 255

        valid_counts[
            label
        ] += 1

    # ---------------------------------------------------------
    # Write standardized geometry JSON
    # ---------------------------------------------------------

    geometry_json_path = (
        BUILD_OUTPUT_DIR
        / "geometry_json"
        / f"{stem}.json"
    )

    save_json(
        geometry_json_path,
        standardized,
    )

    # ---------------------------------------------------------
    # Write fixed raster primitives
    # ---------------------------------------------------------

    for label in REGION_LABELS:

        write_png(
            BUILD_OUTPUT_DIR
            / "regions"
            / label
            / f"{stem}.png",
            region_masks[
                label
            ],
        )

        write_png(
            BUILD_OUTPUT_DIR
            / "region_boundaries"
            / label
            / f"{stem}.png",
            region_boundaries[
                label
            ],
        )

    for label in CURVE_LABELS:

        write_png(
            BUILD_OUTPUT_DIR
            / "curves"
            / label
            / f"{stem}.png",
            curve_masks[
                label
            ],
        )

    for label in POINT_LABELS:

        write_png(
            BUILD_OUTPUT_DIR
            / "points"
            / label
            / f"{stem}.png",
            point_masks[
                label
            ],
        )

    # ---------------------------------------------------------
    # Image-level manifest row
    # ---------------------------------------------------------

    manifest_row = {
        "image_filename":
            image_path.name,

        "image_stem":
            stem,

        "patient_id":
            standardized[
                "patient_id"
            ],

        "eye_id":
            standardized[
                "eye_id"
            ],

        "outer_test_fold":
            standardized[
                "outer_test_fold"
            ],

        "official_split":
            standardized[
                "official_split"
            ],

        "width":
            width,

        "height":
            height,

        "primary_excluded":
            primary_excluded,

        "non_primary_excluded":
            non_primary,

        "HF_raster_collisions":
            point_pixel_collisions,
    }

    for label in PRIMARY_LABELS:

        manifest_row[
            f"{label}_valid_instances"
        ] = int(
            valid_counts[
                label
            ]
        )

    for label in REGION_LABELS:

        manifest_row[
            f"{label}_region_pixels"
        ] = int(
            np.count_nonzero(
                region_masks[
                    label
                ]
            )
        )

        manifest_row[
            f"{label}_boundary_pixels"
        ] = int(
            np.count_nonzero(
                region_boundaries[
                    label
                ]
            )
        )

    for label in CURVE_LABELS:

        manifest_row[
            f"{label}_curve_pixels"
        ] = int(
            np.count_nonzero(
                curve_masks[
                    label
                ]
            )
        )

    for label in POINT_LABELS:

        manifest_row[
            f"{label}_point_pixels"
        ] = int(
            np.count_nonzero(
                point_masks[
                    label
                ]
            )
        )

    return (
        manifest_row,
        excluded_rows,
        valid_counts,
        point_pixel_collisions,
    )


# =====================================================================
# PROCESS ALL IMAGES
# =====================================================================

def generate_geometry_targets(
    split_manifest,
    json_lookup,
    image_lookup,
):

    print(
        "[3/7] Generating geometry-aware primitives..."
    )

    geometry_manifest = []

    excluded_annotations = []

    global_valid_counts = Counter()

    total_point_collisions = 0

    stems = sorted(
        json_lookup.keys(),
        key=natural_key,
    )

    for index, stem in enumerate(
        stems,
        start=1,
    ):

        json_path = (
            json_lookup[
                stem
            ]
        )

        image_path = (
            image_lookup[
                stem
            ]
        )

        filename = (
            image_path.name
        )

        if filename not in split_manifest:

            raise RuntimeError(
                "Source image is not present "
                "in frozen split manifest:\n"
                f"{filename}"
            )

        split_row = (
            split_manifest[
                filename
            ]
        )

        source_eye = get_eye_id(
            filename
        )

        manifest_eye = canonical_id(
            split_row[
                "eye_id"
            ]
        )

        if (
            source_eye
            != manifest_eye
        ):

            raise RuntimeError(
                "Eye-ID mismatch between "
                "filename and frozen split:\n"
                f"{filename}\n"
                f"Filename Eye ID: {source_eye}\n"
                f"Manifest Eye ID: {manifest_eye}"
            )

        (
            manifest_row,
            excluded_rows,
            valid_counts,
            collisions,
        ) = process_one_image(
            stem,
            json_path,
            image_path,
            split_row,
        )

        geometry_manifest.append(
            manifest_row
        )

        excluded_annotations.extend(
            excluded_rows
        )

        global_valid_counts.update(
            valid_counts
        )

        total_point_collisions += (
            collisions
        )

        if index % 250 == 0:

            print(
                f"    Generated "
                f"{index}/"
                f"{len(stems)} images"
            )

    return {
        "manifest":
            geometry_manifest,

        "excluded":
            excluded_annotations,

        "valid_counts":
            global_valid_counts,

        "point_collisions":
            total_point_collisions,
    }


# =====================================================================
# VALIDATE GENERATED OUTPUT
# =====================================================================

def validate_generated(
    results,
):

    print(
        "[4/7] Validating generated geometry..."
    )

    manifest = results[
        "manifest"
    ]

    excluded = results[
        "excluded"
    ]

    valid_counts = results[
        "valid_counts"
    ]

    if len(manifest) != EXPECTED_IMAGES:

        raise RuntimeError(
            "Generated manifest contains "
            f"{len(manifest)} images; "
            f"expected {EXPECTED_IMAGES}."
        )

    primary_excluded = [
        row
        for row in excluded
        if row[
            "category"
        ] == "EXCLUDE_PRIMARY"
    ]

    non_primary = [
        row
        for row in excluded
        if row[
            "category"
        ] == "NON_PRIMARY"
    ]

    total_valid = sum(
        valid_counts.values()
    )

    if STRICT_CHECKS:

        for label, expected in (
            EXPECTED_VALID_COUNTS.items()
        ):

            observed = int(
                valid_counts[
                    label
                ]
            )

            if observed != expected:

                raise RuntimeError(
                    "Unexpected valid annotation "
                    f"count for {label}.\n"
                    f"Expected: {expected}\n"
                    f"Observed: {observed}"
                )

        if (
            total_valid
            != EXPECTED_VALID_TOTAL
        ):

            raise RuntimeError(
                "Unexpected total valid "
                "annotation count.\n"
                f"Expected: "
                f"{EXPECTED_VALID_TOTAL}\n"
                f"Observed: {total_valid}"
            )

        if (
            len(primary_excluded)
            != EXPECTED_PRIMARY_EXCLUDED
        ):

            raise RuntimeError(
                "Unexpected malformed-primary "
                "annotation count.\n"
                f"Expected: "
                f"{EXPECTED_PRIMARY_EXCLUDED}\n"
                f"Observed: "
                f"{len(primary_excluded)}"
            )

        if (
            len(non_primary)
            != EXPECTED_NON_PRIMARY
        ):

            raise RuntimeError(
                "Unexpected non-primary "
                "annotation count.\n"
                f"Expected: "
                f"{EXPECTED_NON_PRIMARY}\n"
                f"Observed: "
                f"{len(non_primary)}"
            )

    # ---------------------------------------------------------
    # Verify every raster folder contains exactly 3012 files
    # ---------------------------------------------------------

    raster_directories = []

    for label in REGION_LABELS:

        raster_directories.append(
            BUILD_OUTPUT_DIR
            / "regions"
            / label
        )

        raster_directories.append(
            BUILD_OUTPUT_DIR
            / "region_boundaries"
            / label
        )

    for label in CURVE_LABELS:

        raster_directories.append(
            BUILD_OUTPUT_DIR
            / "curves"
            / label
        )

    for label in POINT_LABELS:

        raster_directories.append(
            BUILD_OUTPUT_DIR
            / "points"
            / label
        )

    for directory in raster_directories:

        count = len(
            list(
                directory.glob(
                    "*.png"
                )
            )
        )

        if (
            count
            != EXPECTED_IMAGES
        ):

            raise RuntimeError(
                "Unexpected raster count:\n"
                f"{directory}\n"
                f"Found: {count}"
            )

    geometry_json_count = len(
        list(
            (
                BUILD_OUTPUT_DIR
                / "geometry_json"
            ).glob(
                "*.json"
            )
        )
    )

    if (
        geometry_json_count
        != EXPECTED_IMAGES
    ):

        raise RuntimeError(
            "Unexpected standardized "
            "geometry JSON count."
        )

    print(
        "    Valid geometry counts:"
    )

    for label in [
        "IRF",
        "SRF",
        "ELM",
        "EZ",
        "HF",
    ]:

        print(
            f"      {label:<4}: "
            f"{valid_counts[label]}"
        )

    print(
        f"    Valid total: "
        f"{total_valid}"
    )

    print(
        f"    Primary annotations excluded: "
        f"{len(primary_excluded)}"
    )

    print(
        f"    Non-primary annotations logged: "
        f"{len(non_primary)}"
    )

    print(
        f"    HF raster-coordinate collisions: "
        f"{results['point_collisions']}"
    )

    print(
        "    Geometry validation: PASS"
    )

    return (
        primary_excluded,
        non_primary,
    )


# =====================================================================
# SAVE REPORTS
# =====================================================================

def save_reports(
    results,
    split_sha,
    primary_excluded,
    non_primary,
):

    print(
        "[5/7] Saving geometry manifests..."
    )

    manifest_path = (
        BUILD_OUTPUT_DIR
        / "geometry_manifest.csv"
    )

    exclusions_path = (
        BUILD_OUTPUT_DIR
        / "excluded_annotations.csv"
    )

    write_csv(
        manifest_path,
        results[
            "manifest"
        ],
    )

    write_csv(
        exclusions_path,
        results[
            "excluded"
        ],
    )

    # ---------------------------------------------------------
    # Exclusion summary
    # ---------------------------------------------------------

    exclusion_reason_counts = Counter(
        (
            row["label"],
            row["shape_type"],
            row["category"],
            row["reason"],
        )
        for row in results[
            "excluded"
        ]
    )

    exclusion_summary_rows = []

    for (
        label,
        shape_type,
        category,
        reason,
    ), count in sorted(
        exclusion_reason_counts.items()
    ):

        exclusion_summary_rows.append({
            "label":
                label,

            "shape_type":
                shape_type,

            "category":
                category,

            "reason":
                reason,

            "count":
                count,
        })

    write_csv(
        BUILD_OUTPUT_DIR
        / "exclusion_summary.csv",
        exclusion_summary_rows,
    )

    # ---------------------------------------------------------
    # Image-level positive counts
    # ---------------------------------------------------------

    image_positive_counts = {}

    for label in [
        "IRF",
        "SRF",
        "ELM",
        "EZ",
        "HF",
    ]:

        image_positive_counts[
            label
        ] = sum(
            1
            for row in results[
                "manifest"
            ]
            if int(
                row[
                    f"{label}_valid_instances"
                ]
            ) > 0
        )

    # ---------------------------------------------------------
    # Processing configuration
    # ---------------------------------------------------------

    processing_config = {
        "schema_version":
            "geometry_v1",

        "source_dataset":
            str(
                DATASET_ROOT
            ),

        "frozen_split_sha256":
            split_sha,

        "image_dimensions":
            [
                EXPECTED_WIDTH,
                EXPECTED_HEIGHT,
            ],

        "primary_geometry": {
            "IRF":
                "2D region",

            "SRF":
                "2D region",

            "ELM":
                "1D open curve",

            "EZ":
                "1D open curve",

            "HF":
                "0D point",
        },

        "region_rasterization": {
            "fill":
                "cv2.fillPoly",

            "boundary":
                "closed 1-pixel cv2.polylines",

            "pixel_value":
                255,
        },

        "curve_rasterization": {
            "method":
                "open 1-pixel cv2.polylines",

            "pixel_value":
                255,
        },

        "point_rasterization": {
            "method":
                "nearest integer pixel",

            "pixel_value":
                255,

            "important":
                (
                    "Original floating-point coordinates "
                    "remain authoritative in geometry_json."
                ),
        },

        "soft_targets_not_precomputed": {
            "HF_gaussian_heatmaps":
                (
                    "Generated later on-the-fly "
                    "to permit sigma ablation."
                ),

            "ELM_EZ_distance_fields":
                (
                    "Generated later on-the-fly "
                    "to permit distance-bandwidth "
                    "and loss ablation."
                ),
        },

        "annotation_policy": {
            "IRF_SRF":
                (
                    "polygon or linestrip with >=3 "
                    "points is interpreted as a "
                    "closed region contour."
                ),

            "ELM_EZ":
                (
                    "only linestrip with >=2 "
                    "points is accepted."
                ),

            "HF":
                (
                    "only point with exactly one "
                    "coordinate is accepted."
                ),

            "malformed_primary":
                (
                    "individual annotation excluded; "
                    "the image itself remains."
                ),

            "non_primary":
                (
                    "ERM/HRD/PED and any other "
                    "non-primary labels are logged "
                    "but not used as geometry_v1 targets."
                ),
        },

        "valid_annotation_counts": {
            label:
                int(
                    results[
                        "valid_counts"
                    ][label]
                )
            for label in [
                "IRF",
                "SRF",
                "ELM",
                "EZ",
                "HF",
            ]
        },

        "positive_image_counts":
            image_positive_counts,

        "primary_excluded_count":
            len(
                primary_excluded
            ),

        "non_primary_count":
            len(
                non_primary
            ),

        "HF_raster_collisions":
            int(
                results[
                    "point_collisions"
                ]
            ),
    }

    config_path = (
        BUILD_OUTPUT_DIR
        / "processing_config.json"
    )

    save_json(
        config_path,
        processing_config,
    )

    return (
        manifest_path,
        exclusions_path,
        config_path,
        processing_config,
    )


# =====================================================================
# GEOMETRY FINGERPRINT
# =====================================================================

def calculate_geometry_fingerprint(
    manifest_path: Path,
    exclusions_path: Path,
    config_path: Path,
):

    print(
        "[6/7] Calculating geometry fingerprint..."
    )

    digest = hashlib.sha256()

    files = []

    files.extend(
        sorted(
            (
                BUILD_OUTPUT_DIR
                / "geometry_json"
            ).glob(
                "*.json"
            ),
            key=lambda p:
                natural_key(
                    p.name
                ),
        )
    )

    files.extend([
        manifest_path,
        exclusions_path,
        config_path,
    ])

    for path in files:

        relative = path.relative_to(
            BUILD_OUTPUT_DIR
        )

        digest.update(
            str(relative)
            .replace(
                "\\",
                "/",
            )
            .encode(
                "utf-8"
            )
        )

        digest.update(
            b"\0"
        )

        with path.open(
            "rb"
        ) as f:

            while True:

                block = f.read(
                    1024 * 1024
                )

                if not block:
                    break

                digest.update(
                    block
                )

        digest.update(
            b"\0"
        )

    fingerprint = (
        digest.hexdigest()
    )

    (
        BUILD_OUTPUT_DIR
        / "geometry_fingerprint.txt"
    ).write_text(
        fingerprint + "\n",
        encoding="utf-8",
    )

    print(
        "    Geometry SHA-256:"
    )

    print(
        f"    {fingerprint}"
    )

    return fingerprint


# =====================================================================
# FINAL SUMMARY AND PROMOTION
# =====================================================================

def finalize_output(
    results,
    split_sha,
    geometry_fingerprint,
    processing_config,
):

    print(
        "[7/7] Finalizing frozen geometry_v1..."
    )

    summary = {
        "status":
            "PASS",

        "geometry_version":
            "geometry_v1",

        "number_of_images":
            len(
                results[
                    "manifest"
                ]
            ),

        "source_split_sha256":
            split_sha,

        "geometry_sha256":
            geometry_fingerprint,

        "valid_annotation_counts":
            processing_config[
                "valid_annotation_counts"
            ],

        "positive_image_counts":
            processing_config[
                "positive_image_counts"
            ],

        "primary_excluded_count":
            processing_config[
                "primary_excluded_count"
            ],

        "non_primary_count":
            processing_config[
                "non_primary_count"
            ],

        "HF_raster_collisions":
            processing_config[
                "HF_raster_collisions"
            ],

        "output_structure": {
            "geometry_json":
                (
                    "Lossless standardized "
                    "floating-point geometry."
                ),

            "regions":
                "Binary IRF/SRF region masks.",

            "region_boundaries":
                (
                    "1-pixel IRF/SRF "
                    "boundary masks."
                ),

            "curves":
                (
                    "1-pixel ELM/EZ "
                    "curve masks."
                ),

            "points":
                (
                    "Sparse HF raster masks; "
                    "floating coordinates remain "
                    "in geometry_json."
                ),
        },
    }

    save_json(
        BUILD_OUTPUT_DIR
        / "geometry_summary.json",
        summary,
    )

    # Promote only after every validation has passed.
    BUILD_OUTPUT_DIR.rename(
        FINAL_OUTPUT_DIR
    )

    return summary


# =====================================================================
# TERMINAL REPORT
# =====================================================================

def print_final_report(
    summary,
):

    print("\n")
    print("=" * 90)

    print(
        "RVO-ME GEOMETRY_V1 "
        "TARGET GENERATION COMPLETE"
    )

    print("=" * 90)

    print(
        f"\nImages processed: "
        f"{summary['number_of_images']}"
    )

    print(
        "\nValid geometry annotations:"
    )

    for label, count in (
        summary[
            "valid_annotation_counts"
        ].items()
    ):

        print(
            f"  {label:<4}: {count}"
        )

    print(
        "\nImages containing valid target:"
    )

    for label, count in (
        summary[
            "positive_image_counts"
        ].items()
    ):

        print(
            f"  {label:<4}: {count}"
        )

    print(
        "\nExcluded malformed primary annotations: "
        f"{summary['primary_excluded_count']}"
    )

    print(
        "Logged non-primary annotations:          "
        f"{summary['non_primary_count']}"
    )

    print(
        "HF same-pixel raster collisions:         "
        f"{summary['HF_raster_collisions']}"
    )

    print(
        "\nFrozen patient split SHA-256:"
    )

    print(
        summary[
            "source_split_sha256"
        ]
    )

    print(
        "\nGeometry-v1 SHA-256:"
    )

    print(
        summary[
            "geometry_sha256"
        ]
    )

    print(
        "\nFrozen geometry saved to:"
    )

    print(
        FINAL_OUTPUT_DIR
    )

    print(
        "\nDo NOT regenerate geometry_v1 "
        "after model experiments begin."
    )

    print(
        "Any future target formulation must "
        "use a new version such as geometry_v2."
    )

    print("\n")
    print("=" * 90)


# =====================================================================
# MAIN
# =====================================================================

def main():

    print("=" * 90)

    print(
        "RVO-ME STEP 02: "
        "GENERATE GEOMETRY-AWARE TARGET PRIMITIVES"
    )

    print("=" * 90)

    print(
        f"\nDataset root:\n"
        f"{DATASET_ROOT}"
    )

    if not RAW_LABELME_DIR.exists():

        raise FileNotFoundError(
            RAW_LABELME_DIR
        )

    split_manifest, split_sha = (
        load_and_verify_split()
    )

    (
        json_lookup,
        image_lookup,
    ) = build_source_lookup()

    prepare_output_directory()

    try:

        results = (
            generate_geometry_targets(
                split_manifest,
                json_lookup,
                image_lookup,
            )
        )

        (
            primary_excluded,
            non_primary,
        ) = validate_generated(
            results
        )

        (
            manifest_path,
            exclusions_path,
            config_path,
            processing_config,
        ) = save_reports(
            results,
            split_sha,
            primary_excluded,
            non_primary,
        )

        geometry_fingerprint = (
            calculate_geometry_fingerprint(
                manifest_path,
                exclusions_path,
                config_path,
            )
        )

        summary = finalize_output(
            results,
            split_sha,
            geometry_fingerprint,
            processing_config,
        )

    except Exception:

        print(
            "\nERROR: Geometry generation failed."
        )

        print(
            "The final frozen geometry_v1 "
            "directory was NOT created."
        )

        print(
            "A partial working directory may remain at:"
        )

        print(
            BUILD_OUTPUT_DIR
        )

        raise

    print_final_report(
        summary
    )


if __name__ == "__main__":
    main()