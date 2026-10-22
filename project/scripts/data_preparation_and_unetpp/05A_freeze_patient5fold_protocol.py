from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# =====================================================================
# RVO-ME STEP 05A
# FREEZE CANONICAL PATIENT-DISJOINT 5-FOLD BASELINE PROTOCOL
#
# NO MODEL TRAINING.
#
# Purpose
# -------
# Convert the already-frozen patient_5fold_v1 assignment into one
# canonical, model-independent train/validation/test protocol that every
# conventional baseline and the later proposed model can consume.
#
# The script NEVER creates a new patient split and NEVER re-stratifies.
# It only reads the frozen split artifacts, detects their stored role
# representation, validates it, and writes a canonical copy.
# =====================================================================


# =====================================================================
# PATHS
# =====================================================================

DATASET_ROOT = Path(
    r"E:\Papers\RVO_ME\RVO-Lesion"
)

PROJECT_ROOT = (
    DATASET_ROOT.parent
    / "project"
)

SPLIT_ROOT = (
    PROJECT_ROOT
    / "splits"
    / "patient_5fold_v1"
)

FOLD_ASSIGNMENTS_CSV = (
    SPLIT_ROOT
    / "fold_assignments.csv"
)

IMAGE_MANIFEST_CSV = (
    SPLIT_ROOT
    / "image_manifest.csv"
)

PATIENT_MANIFEST_CSV = (
    SPLIT_ROOT
    / "patient_manifest.csv"
)

SPLIT_SUMMARY_JSON = (
    SPLIT_ROOT
    / "split_summary.json"
)

GEOMETRY_ROOT = (
    PROJECT_ROOT
    / "targets"
    / "geometry_v1"
)

FINAL_OUTPUT_DIR = (
    PROJECT_ROOT
    / "baselines"
    / "segmentation"
    / "patient5fold_protocol_v1"
)

BUILD_OUTPUT_DIR = (
    PROJECT_ROOT
    / "baselines"
    / "segmentation"
    / "patient5fold_protocol_v1__building"
)


# =====================================================================
# FROZEN IDENTIFIERS
# =====================================================================

EXPECTED_SPLIT_SHA256 = (
    "7e7c2ddf51054bf998b2d750b50c74983b9bc5b751b762213bc96343ffe01346"
)

EXPECTED_GEOMETRY_SHA256 = (
    "a4029d5ab4570237ae607d1cc24d551ce438845eab1814e8a82735a122daeb55"
)


# =====================================================================
# EXPECTED FROZEN SPLIT FACTS
# =====================================================================

EXPECTED_PATIENTS = 130
EXPECTED_IMAGES = 3012
EXPECTED_FOLDS = [1, 2, 3, 4, 5]

EXPECTED_PATIENT_ROLE_COUNTS = {
    1: {
        "train": 91,
        "val": 13,
        "test": 26,
    },
    2: {
        "train": 91,
        "val": 13,
        "test": 26,
    },
    3: {
        "train": 90,
        "val": 14,
        "test": 26,
    },
    4: {
        "train": 89,
        "val": 15,
        "test": 26,
    },
    5: {
        "train": 89,
        "val": 15,
        "test": 26,
    },
}

EXPECTED_TEST_IMAGE_COUNTS = {
    1: 599,
    2: 606,
    3: 598,
    4: 631,
    5: 578,
}


# =====================================================================
# ROLE NORMALIZATION
# =====================================================================

ROLE_ALIASES = {
    "train":
        "train",

    "training":
        "train",

    "tr":
        "train",

    "val":
        "val",

    "valid":
        "val",

    "validation":
        "val",

    "dev":
        "val",

    "development":
        "val",

    "test":
        "test",

    "testing":
        "test",

    "te":
        "test",
}


# =====================================================================
# GENERAL HELPERS
# =====================================================================

def safe_text(
    value: Any,
) -> str:

    if value is None:
        return ""

    return str(
        value
    ).strip()


def canonical_id(
    value: Any,
) -> str:

    if value is None:
        return ""

    if isinstance(
        value,
        int,
    ):
        return str(
            value
        )

    if isinstance(
        value,
        float,
    ):

        if value.is_integer():

            return str(
                int(
                    value
                )
            )

        return str(
            value
        )

    text = str(
        value
    ).strip()

    try:

        number = float(
            text
        )

        if (
            number.is_integer()
            and "." in text
        ):

            return str(
                int(
                    number
                )
            )

    except ValueError:
        pass

    return text


def natural_key(
    text: str,
):

    return [
        int(
            part
        )
        if part.isdigit()
        else part.lower()
        for part
        in re.split(
            r"(\d+)",
            str(
                text
            ),
        )
    ]


def normalize_header(
    text: str,
) -> str:

    value = safe_text(
        text
    ).lower()

    value = re.sub(
        r"[^a-z0-9]+",
        "_",
        value,
    )

    return value.strip(
        "_"
    )


def normalize_role(
    value: Any,
) -> Optional[str]:

    text = safe_text(
        value
    ).lower()

    text = re.sub(
        r"[^a-z]+",
        "",
        text,
    )

    return ROLE_ALIASES.get(
        text
    )


def parse_fold_number(
    value: Any,
) -> Optional[int]:

    text = safe_text(
        value
    )

    try:

        number = int(
            float(
                text
            )
        )

    except (
        TypeError,
        ValueError,
    ):

        match = re.search(
            r"([1-5])",
            text,
        )

        if not match:
            return None

        number = int(
            match.group(
                1
            )
        )

    if number in EXPECTED_FOLDS:
        return number

    return None


def sha256_file(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:

        while True:

            block = f.read(
                1024
                * 1024
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


def sha256_lines(
    lines: Sequence[str],
) -> str:

    digest = hashlib.sha256()

    for line in lines:

        digest.update(
            (
                str(
                    line
                )
                + "\n"
            ).encode(
                "utf-8"
            )
        )

    return digest.hexdigest()


def load_json(
    path: Path,
) -> Any:

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(
            f
        )


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
            default=str,
        )


def read_csv_rows(
    path: Path,
) -> Tuple[
    List[str],
    List[
        Dict[
            str,
            str,
        ]
    ],
]:

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        reader = csv.DictReader(
            f
        )

        fieldnames = (
            list(
                reader.fieldnames
                or []
            )
        )

        rows = [
            {
                key:
                    safe_text(
                        value
                    )
                for key, value
                in row.items()
            }
            for row
            in reader
        ]

    return (
        fieldnames,
        rows,
    )


def write_csv(
    path: Path,
    rows: List[
        Dict[
            str,
            Any,
        ]
    ],
    fieldnames: Optional[
        List[
            str
        ]
    ] = None,
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
            rows[
                0
            ].keys()
        )

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=
                fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def write_txt(
    path: Path,
    values: Sequence[Any],
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    text = "\n".join(
        str(
            value
        )
        for value
        in values
    )

    if text:
        text += "\n"

    path.write_text(
        text,
        encoding="utf-8",
    )


def find_column(
    fieldnames: Sequence[str],
    candidates: Sequence[str],
) -> Optional[str]:

    normalized = {
        normalize_header(
            field
        ):
            field
        for field
        in fieldnames
    }

    for candidate in candidates:

        key = normalize_header(
            candidate
        )

        if key in normalized:
            return normalized[
                key
            ]

    return None


# =====================================================================
# ASSET VERIFICATION
# =====================================================================

def verify_frozen_assets():

    print(
        "\n[1/7] Verifying frozen project assets..."
    )

    required = [
        FOLD_ASSIGNMENTS_CSV,
        IMAGE_MANIFEST_CSV,
        PATIENT_MANIFEST_CSV,
    ]

    for path in required:

        if not path.exists():

            raise FileNotFoundError(
                f"Missing frozen split asset:\n"
                f"{path}"
            )

    split_sha = sha256_file(
        FOLD_ASSIGNMENTS_CSV
    )

    if (
        split_sha
        != EXPECTED_SPLIT_SHA256
    ):

        raise RuntimeError(
            "Frozen fold_assignments.csv SHA-256 mismatch.\n"
            f"Expected: {EXPECTED_SPLIT_SHA256}\n"
            f"Observed: {split_sha}\n"
            "Do not continue with a changed split."
        )

    geometry_sha = None

    geometry_summary_path = (
        GEOMETRY_ROOT
        / "geometry_summary.json"
    )

    if geometry_summary_path.exists():

        geometry_summary = load_json(
            geometry_summary_path
        )

        geometry_sha = safe_text(
            geometry_summary.get(
                "geometry_sha256"
            )
        )

        if (
            geometry_sha
            != EXPECTED_GEOMETRY_SHA256
        ):

            raise RuntimeError(
                "geometry_v1 SHA-256 mismatch.\n"
                f"Expected: {EXPECTED_GEOMETRY_SHA256}\n"
                f"Observed: {geometry_sha}"
            )

    print(
        "    patient_5fold_v1 SHA-256: PASS"
    )

    if geometry_sha is not None:

        print(
            "    geometry_v1 SHA-256:      PASS"
        )

    else:

        print(
            "    geometry_v1 check:         "
            "not required / summary not found"
        )

    return {
        "split_sha256":
            split_sha,

        "geometry_sha256":
            geometry_sha,
    }


# =====================================================================
# MANIFEST LOADING
# =====================================================================

def load_image_manifest():

    print(
        "[2/7] Loading frozen image/patient manifests..."
    )

    (
        image_fields,
        image_rows,
    ) = read_csv_rows(
        IMAGE_MANIFEST_CSV
    )

    (
        patient_fields,
        patient_rows,
    ) = read_csv_rows(
        PATIENT_MANIFEST_CSV
    )

    filename_col = find_column(
        image_fields,
        [
            "image_filename",
            "filename",
            "image",
        ],
    )

    patient_col = find_column(
        image_fields,
        [
            "patient_id",
            "patient",
        ],
    )

    eye_col = find_column(
        image_fields,
        [
            "eye_id",
            "eye",
        ],
    )

    outer_test_col = find_column(
        image_fields,
        [
            "outer_test_fold",
            "test_fold",
            "outer_fold",
        ],
    )

    if (
        filename_col is None
        or patient_col is None
        or outer_test_col is None
    ):

        raise RuntimeError(
            "image_manifest.csv is missing required columns.\n"
            f"Observed headers: {image_fields}"
        )

    normalized_images = []

    filenames_seen = set()

    for row in image_rows:

        filename = safe_text(
            row[
                filename_col
            ]
        )

        patient_id = canonical_id(
            row[
                patient_col
            ]
        )

        eye_id = (
            canonical_id(
                row[
                    eye_col
                ]
            )
            if eye_col
            else ""
        )

        outer_test_fold = (
            parse_fold_number(
                row[
                    outer_test_col
                ]
            )
        )

        if not filename:

            raise RuntimeError(
                "Blank image filename found in image_manifest.csv."
            )

        if not patient_id:

            raise RuntimeError(
                f"Blank patient ID for image {filename}."
            )

        if (
            outer_test_fold
            is None
        ):

            raise RuntimeError(
                f"Invalid outer_test_fold for image "
                f"{filename}: "
                f"{row[outer_test_col]}"
            )

        if filename in filenames_seen:

            raise RuntimeError(
                f"Duplicate image filename in manifest: "
                f"{filename}"
            )

        filenames_seen.add(
            filename
        )

        normalized_images.append({
            "image_filename":
                filename,

            "patient_id":
                patient_id,

            "eye_id":
                eye_id,

            "outer_test_fold":
                outer_test_fold,
        })

    patient_ids = {
        canonical_id(
            row.get(
                find_column(
                    patient_fields,
                    [
                        "patient_id",
                        "patient",
                    ],
                )
                or ""
            )
        )
        for row
        in patient_rows
    }

    patient_ids.discard(
        ""
    )

    manifest_patient_ids = {
        row[
            "patient_id"
        ]
        for row
        in normalized_images
    }

    if (
        len(
            normalized_images
        )
        != EXPECTED_IMAGES
    ):

        raise RuntimeError(
            f"Expected {EXPECTED_IMAGES} image-manifest rows, "
            f"found {len(normalized_images)}."
        )

    if (
        len(
            manifest_patient_ids
        )
        != EXPECTED_PATIENTS
    ):

        raise RuntimeError(
            f"Expected {EXPECTED_PATIENTS} unique patients in "
            f"image_manifest.csv, found "
            f"{len(manifest_patient_ids)}."
        )

    if (
        patient_ids
        and patient_ids
        != manifest_patient_ids
    ):

        raise RuntimeError(
            "patient_manifest.csv and image_manifest.csv "
            "contain different patient ID sets."
        )

    # Check that image-level frozen outer-test folds already reproduce
    # the expected test-image counts.
    test_counts = Counter(
        row[
            "outer_test_fold"
        ]
        for row
        in normalized_images
    )

    for fold in EXPECTED_FOLDS:

        observed = int(
            test_counts[
                fold
            ]
        )

        expected = int(
            EXPECTED_TEST_IMAGE_COUNTS[
                fold
            ]
        )

        if observed != expected:

            raise RuntimeError(
                f"Frozen image manifest test-count mismatch "
                f"for Fold {fold}: "
                f"expected {expected}, observed {observed}."
            )

    print(
        f"    Images:   {len(normalized_images)}"
    )

    print(
        f"    Patients: {len(manifest_patient_ids)}"
    )

    print(
        "    outer_test_fold image counts: PASS"
    )

    return (
        normalized_images,
        sorted(
            manifest_patient_ids,
            key=natural_key,
        ),
    )


# =====================================================================
# ASSIGNMENT DETECTION
# =====================================================================

def role_value_fraction(
    rows: Sequence[
        Dict[
            str,
            str,
        ]
    ],
    column: str,
) -> float:

    values = [
        safe_text(
            row.get(
                column
            )
        )
        for row
        in rows
    ]

    values = [
        value
        for value
        in values
        if value
    ]

    if not values:
        return 0.0

    good = sum(
        1
        for value
        in values
        if normalize_role(
            value
        )
        is not None
    )

    return (
        good
        / len(
            values
        )
    )


def fold_value_fraction(
    rows: Sequence[
        Dict[
            str,
            str,
        ]
    ],
    column: str,
) -> float:

    values = [
        safe_text(
            row.get(
                column
            )
        )
        for row
        in rows
    ]

    values = [
        value
        for value
        in values
        if value
    ]

    if not values:
        return 0.0

    good = sum(
        1
        for value
        in values
        if parse_fold_number(
            value
        )
        is not None
    )

    return (
        good
        / len(
            values
        )
    )


def validate_patient_roles_candidate(
    roles: Dict[
        int,
        Dict[
            str,
            str,
        ]
    ],
    patient_ids: Sequence[str],
) -> Tuple[
    bool,
    str,
]:

    expected_patients = set(
        patient_ids
    )

    for fold in EXPECTED_FOLDS:

        if fold not in roles:

            return (
                False,
                f"Fold {fold} missing.",
            )

        fold_roles = roles[
            fold
        ]

        if (
            set(
                fold_roles.keys()
            )
            != expected_patients
        ):

            return (
                False,
                f"Fold {fold} does not assign every patient exactly once.",
            )

        counts = Counter(
            fold_roles.values()
        )

        expected_counts = (
            EXPECTED_PATIENT_ROLE_COUNTS[
                fold
            ]
        )

        for role in [
            "train",
            "val",
            "test",
        ]:

            if (
                int(
                    counts[
                        role
                    ]
                )
                != int(
                    expected_counts[
                        role
                    ]
                )
            ):

                return (
                    False,
                    f"Fold {fold} role count mismatch for "
                    f"{role}: expected "
                    f"{expected_counts[role]}, "
                    f"observed {counts[role]}.",
                )

    # Each patient must be outer-test exactly once.
    test_appearances = Counter()

    for fold in EXPECTED_FOLDS:

        for patient_id, role in (
            roles[
                fold
            ].items()
        ):

            if role == "test":

                test_appearances[
                    patient_id
                ] += 1

    bad = {
        patient_id:
            count
        for patient_id, count
        in test_appearances.items()
        if count != 1
    }

    missing = (
        expected_patients
        - set(
            test_appearances.keys()
        )
    )

    if (
        bad
        or missing
    ):

        return (
            False,
            "Each patient is not assigned to test exactly once.",
        )

    return (
        True,
        "PASS",
    )


def detect_long_form_assignments(
    fieldnames: Sequence[str],
    rows: Sequence[
        Dict[
            str,
            str,
        ]
    ],
    patient_ids: Sequence[str],
) -> Optional[
    Tuple[
        Dict[
            int,
            Dict[
                str,
                str,
            ]
        ],
        Dict[
            str,
            Any,
        ],
    ]
]:

    patient_col = find_column(
        fieldnames,
        [
            "patient_id",
            "patient",
            "subject_id",
            "subject",
        ],
    )

    if patient_col is None:
        return None

    role_candidates = [
        field
        for field
        in fieldnames
        if (
            role_value_fraction(
                rows,
                field,
            )
            >= 0.95
        )
    ]

    fold_candidates = [
        field
        for field
        in fieldnames
        if (
            fold_value_fraction(
                rows,
                field,
            )
            >= 0.95
        )
    ]

    for role_col in role_candidates:

        for fold_col in fold_candidates:

            if (
                role_col
                == fold_col
            ):
                continue

            roles = {
                fold:
                    {}
                for fold
                in EXPECTED_FOLDS
            }

            valid_rows = 0
            failed = False

            for row in rows:

                patient_id = canonical_id(
                    row.get(
                        patient_col
                    )
                )

                fold = parse_fold_number(
                    row.get(
                        fold_col
                    )
                )

                role = normalize_role(
                    row.get(
                        role_col
                    )
                )

                if (
                    not patient_id
                    or fold is None
                    or role is None
                ):
                    continue

                if (
                    patient_id
                    in roles[
                        fold
                    ]
                ):

                    failed = True
                    break

                roles[
                    fold
                ][
                    patient_id
                ] = role

                valid_rows += 1

            if failed:
                continue

            if valid_rows < (
                EXPECTED_PATIENTS
                * len(
                    EXPECTED_FOLDS
                )
            ):
                continue

            (
                ok,
                reason,
            ) = (
                validate_patient_roles_candidate(
                    roles,
                    patient_ids,
                )
            )

            if ok:

                return (
                    roles,
                    {
                        "mode":
                            "fold_assignments_long_form",

                        "patient_column":
                            patient_col,

                        "fold_column":
                            fold_col,

                        "role_column":
                            role_col,
                    },
                )

    return None


def extract_fold_from_column_name(
    field: str,
) -> Optional[int]:

    normalized = normalize_header(
        field
    )

    patterns = [
        r"(?:^|_)fold_?([1-5])(?:_|$)",
        r"(?:^|_)f_?([1-5])(?:_|$)",
        r"(?:^|_)role_?([1-5])(?:_|$)",
        r"(?:^|_)([1-5])_?role(?:_|$)",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            normalized,
        )

        if match:

            return int(
                match.group(
                    1
                )
            )

    # Last-resort: if the column contains "role" and exactly one
    # standalone digit 1..5.
    if "role" in normalized:

        digits = re.findall(
            r"(?<!\d)([1-5])(?!\d)",
            normalized,
        )

        if len(
            digits
        ) == 1:

            return int(
                digits[
                    0
                ]
            )

    return None


def detect_wide_form_assignments(
    fieldnames: Sequence[str],
    rows: Sequence[
        Dict[
            str,
            str,
        ]
    ],
    patient_ids: Sequence[str],
) -> Optional[
    Tuple[
        Dict[
            int,
            Dict[
                str,
                str,
            ]
        ],
        Dict[
            str,
            Any,
        ],
    ]
]:

    patient_col = find_column(
        fieldnames,
        [
            "patient_id",
            "patient",
            "subject_id",
            "subject",
        ],
    )

    if patient_col is None:
        return None

    fold_to_columns = defaultdict(
        list
    )

    for field in fieldnames:

        if (
            role_value_fraction(
                rows,
                field,
            )
            < 0.95
        ):
            continue

        fold = extract_fold_from_column_name(
            field
        )

        if fold is not None:

            fold_to_columns[
                fold
            ].append(
                field
            )

    if not all(
        fold
        in fold_to_columns
        for fold
        in EXPECTED_FOLDS
    ):

        return None

    # Try every single-column combination where practical. In normal
    # frozen files there should be exactly one role column per fold.
    chosen = {}

    for fold in EXPECTED_FOLDS:

        if (
            len(
                fold_to_columns[
                    fold
                ]
            )
            != 1
        ):

            # Prefer names explicitly containing "role".
            role_named = [
                field
                for field
                in fold_to_columns[
                    fold
                ]
                if "role" in normalize_header(
                    field
                )
            ]

            if len(
                role_named
            ) == 1:

                chosen[
                    fold
                ] = role_named[
                    0
                ]

            else:

                return None

        else:

            chosen[
                fold
            ] = (
                fold_to_columns[
                    fold
                ][
                    0
                ]
            )

    roles = {
        fold:
            {}
        for fold
        in EXPECTED_FOLDS
    }

    for row in rows:

        patient_id = canonical_id(
            row.get(
                patient_col
            )
        )

        if not patient_id:
            continue

        for fold in EXPECTED_FOLDS:

            role = normalize_role(
                row.get(
                    chosen[
                        fold
                    ]
                )
            )

            if role is None:

                return None

            if (
                patient_id
                in roles[
                    fold
                ]
            ):

                return None

            roles[
                fold
            ][
                patient_id
            ] = role

    (
        ok,
        reason,
    ) = (
        validate_patient_roles_candidate(
            roles,
            patient_ids,
        )
    )

    if not ok:
        return None

    return (
        roles,
        {
            "mode":
                "fold_assignments_wide_form",

            "patient_column":
                patient_col,

            "role_columns":
                {
                    str(
                        fold
                    ):
                        chosen[
                            fold
                        ]
                    for fold
                    in EXPECTED_FOLDS
                },
        },
    )


def detect_from_fold_specific_files(
    patient_ids: Sequence[str],
    images: Sequence[
        Dict[
            str,
            Any,
        ]
    ],
) -> Optional[
    Tuple[
        Dict[
            int,
            Dict[
                str,
                str,
            ]
        ],
        Dict[
            str,
            Any,
        ],
    ]
]:

    # Fallback only.
    #
    # The frozen fold_assignments.csv stores the OUTER test fold.
    # Train/validation role files may live in nested directories such as:
    #
    #   fold_1/val_patients.txt
    #   fold_1/train_images.txt
    #
    # Therefore detection MUST inspect the entire relative path rather
    # than only path.stem. We also support either patient-ID lists or
    # image-filename lists and map images back to their frozen patient ID.

    files = [
        path
        for path
        in SPLIT_ROOT.rglob(
            "*"
        )
        if (
            path.is_file()
            and path.suffix.lower()
            in {
                ".txt",
                ".list",
                ".csv",
            }
        )
    ]

    patient_set = set(
        patient_ids
    )

    image_to_patient = {}

    stem_to_patient = {}

    for row in images:

        filename = safe_text(
            row[
                "image_filename"
            ]
        )

        patient_id = canonical_id(
            row[
                "patient_id"
            ]
        )

        image_to_patient[
            filename
        ] = patient_id

        stem_to_patient[
            Path(
                filename
            ).stem
        ] = patient_id

    detected: Dict[
        int,
        Dict[
            str,
            set
        ]
    ] = {
        fold: {
            "train": set(),
            "val": set(),
            "test": set(),
        }
        for fold
        in EXPECTED_FOLDS
    }

    sources = []

    candidate_debug = []

    def detect_fold_from_path(
        relative_path: Path,
    ) -> Optional[int]:

        normalized = normalize_header(
            str(
                relative_path
            )
        )

        patterns = [
            r"(?:^|_)fold_?0?([1-5])(?:_|$)",
            r"(?:^|_)f_?0?([1-5])(?:_|$)",
        ]

        for pattern in patterns:

            match = re.search(
                pattern,
                normalized,
            )

            if match:

                return int(
                    match.group(
                        1
                    )
                )

        return None

    def detect_role_from_path(
        relative_path: Path,
    ) -> Optional[str]:

        normalized = normalize_header(
            str(
                relative_path
            )
        )

        # Prefer unambiguous canonical role words.
        role_patterns = [
            (
                "train",
                [
                    r"(?:^|_)train(?:ing)?(?:_|$)",
                ],
            ),
            (
                "val",
                [
                    r"(?:^|_)val(?:id(?:ation)?)?(?:_|$)",
                    r"(?:^|_)dev(?:elopment)?(?:_|$)",
                ],
            ),
            (
                "test",
                [
                    r"(?:^|_)test(?:ing)?(?:_|$)",
                ],
            ),
        ]

        found = []

        for role, patterns in role_patterns:

            if any(
                re.search(
                    pattern,
                    normalized,
                )
                for pattern
                in patterns
            ):

                found.append(
                    role
                )

        if len(
            found
        ) == 1:

            return found[
                0
            ]

        return None

    def token_to_patient(
        raw_value: Any,
    ) -> Optional[str]:

        value = safe_text(
            raw_value
        )

        if not value:
            return None

        # Direct patient ID.
        patient_id = canonical_id(
            value
        )

        if patient_id in patient_set:

            return patient_id

        # Exact image filename.
        name = Path(
            value
        ).name

        if name in image_to_patient:

            return image_to_patient[
                name
            ]

        # Image stem if extension differs or list stores stem only.
        stem = Path(
            name
        ).stem

        if stem in stem_to_patient:

            return stem_to_patient[
                stem
            ]

        return None

    def parse_candidate_file(
        path: Path,
    ) -> Tuple[
        set,
        str,
    ]:

        patients = set()

        # ---------------------------------------------------------
        # Plain text/list files
        # ---------------------------------------------------------

        if path.suffix.lower() in {
            ".txt",
            ".list",
        }:

            raw_lines = [
                safe_text(
                    line
                )
                for line
                in path.read_text(
                    encoding="utf-8-sig"
                ).splitlines()
            ]

            raw_lines = [
                line
                for line
                in raw_lines
                if line
            ]

            for value in raw_lines:

                patient_id = (
                    token_to_patient(
                        value
                    )
                )

                if (
                    patient_id
                    is not None
                ):

                    patients.add(
                        patient_id
                    )

            return (
                patients,
                "text_list",
            )

        # ---------------------------------------------------------
        # CSV files
        # ---------------------------------------------------------

        (
            fields,
            rows,
        ) = read_csv_rows(
            path
        )

        patient_col = find_column(
            fields,
            [
                "patient_id",
                "patient",
                "subject_id",
                "subject",
            ],
        )

        image_col = find_column(
            fields,
            [
                "image_filename",
                "filename",
                "image",
                "scan",
                "scan_filename",
            ],
        )

        if patient_col is not None:

            for row in rows:

                patient_id = (
                    token_to_patient(
                        row.get(
                            patient_col
                        )
                    )
                )

                if (
                    patient_id
                    is not None
                ):

                    patients.add(
                        patient_id
                    )

            return (
                patients,
                "csv_patient_column",
            )

        if image_col is not None:

            for row in rows:

                patient_id = (
                    token_to_patient(
                        row.get(
                            image_col
                        )
                    )
                )

                if (
                    patient_id
                    is not None
                ):

                    patients.add(
                        patient_id
                    )

            return (
                patients,
                "csv_image_column",
            )

        # Last-resort CSV scan:
        # only accept a column if every nonblank value maps to a known
        # frozen patient/image. This prevents accidental parsing of
        # unrelated summary CSVs.
        for field in fields:

            values = [
                safe_text(
                    row.get(
                        field
                    )
                )
                for row
                in rows
            ]

            values = [
                value
                for value
                in values
                if value
            ]

            if not values:
                continue

            mapped = [
                token_to_patient(
                    value
                )
                for value
                in values
            ]

            if (
                all(
                    value
                    is not None
                    for value
                    in mapped
                )
            ):

                return (
                    set(
                        mapped
                    ),
                    f"csv_auto_column:{field}",
                )

        return (
            set(),
            "unrecognized",
        )

    # -------------------------------------------------------------
    # Scan frozen split directory
    # -------------------------------------------------------------

    for path in files:

        relative = path.relative_to(
            SPLIT_ROOT
        )

        fold = detect_fold_from_path(
            relative
        )

        role = detect_role_from_path(
            relative
        )

        if (
            fold is None
            or role is None
        ):
            continue

        (
            parsed_patients,
            parse_mode,
        ) = parse_candidate_file(
            path
        )

        candidate_debug.append({
            "relative_path":
                str(
                    relative
                ),

            "fold":
                fold,

            "role":
                role,

            "parse_mode":
                parse_mode,

            "patients_recovered":
                len(
                    parsed_patients
                ),
        })

        if not parsed_patients:
            continue

        # Reject impossible foreign IDs before merging.
        unknown = (
            parsed_patients
            - patient_set
        )

        if unknown:

            continue

        detected[
            fold
        ][
            role
        ].update(
            parsed_patients
        )

        sources.append(
            str(
                relative
            )
        )

    # Save a diagnostic record even on success. This makes the exact
    # frozen role sources auditable later.
    save_json(
        PROJECT_ROOT
        / "baselines"
        / "segmentation"
        / "step05a_frozen_role_source_scan.json",
        {
            "candidate_files":
                candidate_debug,

            "accepted_source_files":
                sorted(
                    set(
                        sources
                    )
                ),

            "recovered_patient_counts":
                {
                    str(
                        fold
                    ): {
                        role:
                            len(
                                detected[
                                    fold
                                ][
                                    role
                                ]
                            )
                        for role
                        in [
                            "train",
                            "val",
                            "test",
                        ]
                    }
                    for fold
                    in EXPECTED_FOLDS
                },
        },
    )

    # -------------------------------------------------------------
    # Convert recovered sets to one role per patient/fold.
    # -------------------------------------------------------------

    roles = {
        fold:
            {}
        for fold
        in EXPECTED_FOLDS
    }

    for fold in EXPECTED_FOLDS:

        for role in [
            "train",
            "val",
            "test",
        ]:

            for patient_id in sorted(
                detected[
                    fold
                ][
                    role
                ],
                key=natural_key,
            ):

                if (
                    patient_id
                    in roles[
                        fold
                    ]
                ):

                    # Same patient cannot belong to two roles.
                    return None

                roles[
                    fold
                ][
                    patient_id
                ] = role

    (
        ok,
        reason,
    ) = (
        validate_patient_roles_candidate(
            roles,
            patient_ids,
        )
    )

    if not ok:

        print(
            "    Frozen nested-file scan did not yet form "
            "a complete valid role assignment."
        )

        print(
            "    Recovered counts:"
        )

        for fold in EXPECTED_FOLDS:

            print(
                f"      Fold {fold}: "
                f"train={len(detected[fold]['train'])}, "
                f"val={len(detected[fold]['val'])}, "
                f"test={len(detected[fold]['test'])}"
            )

        print(
            f"    Validation reason: {reason}"
        )

        return None

    return (
        roles,
        {
            "mode":
                "frozen_fold_specific_role_files",

            "source_files":
                sorted(
                    set(
                        sources
                    )
                ),

            "role_source_scan_json":
                str(
                    PROJECT_ROOT
                    / "baselines"
                    / "segmentation"
                    / "step05a_frozen_role_source_scan.json"
                ),
        },
    )


def detect_assignments(
    patient_ids: Sequence[str],
    images: Sequence[
        Dict[
            str,
            Any,
        ]
    ],
):

    print(
        "[3/7] Detecting the stored train/val/test representation..."
    )

    (
        fieldnames,
        rows,
    ) = read_csv_rows(
        FOLD_ASSIGNMENTS_CSV
    )

    print(
        "    fold_assignments.csv headers:"
    )

    print(
        "    "
        + ", ".join(
            fieldnames
        )
    )

    print(
        f"    Rows: {len(rows)}"
    )

    # Show up to three rows so the run log itself permanently records
    # the frozen assignment schema.
    for index, row in enumerate(
        rows[
            :3
        ],
        start=1,
    ):

        compact = {
            key:
                value
            for key, value
            in row.items()
            if safe_text(
                value
            )
        }

        print(
            f"    Example row {index}: "
            f"{compact}"
        )

    detectors = [
        detect_long_form_assignments,
        detect_wide_form_assignments,
    ]

    for detector in detectors:

        result = detector(
            fieldnames,
            rows,
            patient_ids,
        )

        if result is not None:

            (
                roles,
                detection,
            ) = result

            print(
                f"    Detected mode: "
                f"{detection['mode']}"
            )

            return (
                roles,
                detection,
            )

    result = (
        detect_from_fold_specific_files(
            patient_ids,
            images,
        )
    )

    if result is not None:

        (
            roles,
            detection,
        ) = result

        print(
            f"    Detected mode: "
            f"{detection['mode']}"
        )

        return (
            roles,
            detection,
        )

    diagnostic = {
        "headers":
            fieldnames,

        "row_count":
            len(
                rows
            ),

        "example_rows":
            rows[
                :10
            ],

        "files_in_split_root":
            sorted(
                [
                    str(
                        path.relative_to(
                            SPLIT_ROOT
                        )
                    )
                    for path
                    in SPLIT_ROOT.rglob(
                        "*"
                    )
                    if path.is_file()
                ]
            ),
    }

    save_json(
        PROJECT_ROOT
        / "baselines"
        / "segmentation"
        / "step05a_assignment_detection_failed.json",
        diagnostic,
    )

    print(
        "\n    Frozen split files:"
    )

    for relative_path in diagnostic[
        "files_in_split_root"
    ]:

        print(
            f"      {relative_path}"
        )

    raise RuntimeError(
        "\nCould not automatically identify frozen validation roles.\n"
        "NO SPLIT WAS REGENERATED and NO TRAINING WAS STARTED.\n"
        "The terminal now lists every file in patient_5fold_v1. "
        "Send that output if this corrected detector still cannot "
        "recover the frozen role files."
    )


# =====================================================================
# CROSS-CHECK WITH IMAGE-LEVEL OUTER TEST FOLD
# =====================================================================

def crosscheck_outer_test_fold(
    roles: Dict[
        int,
        Dict[
            str,
            str,
        ]
    ],
    images: Sequence[
        Dict[
            str,
            Any,
        ]
    ],
):

    print(
        "[4/7] Cross-checking patient roles against image-level "
        "outer_test_fold..."
    )

    patient_outer_fold = {}

    for row in images:

        patient_id = row[
            "patient_id"
        ]

        fold = int(
            row[
                "outer_test_fold"
            ]
        )

        if (
            patient_id
            in patient_outer_fold
            and patient_outer_fold[
                patient_id
            ]
            != fold
        ):

            raise RuntimeError(
                f"Patient {patient_id} has images assigned "
                "to multiple outer test folds."
            )

        patient_outer_fold[
            patient_id
        ] = fold

    for patient_id, outer_fold in (
        patient_outer_fold.items()
    ):

        for fold in EXPECTED_FOLDS:

            role = roles[
                fold
            ][
                patient_id
            ]

            if (
                fold
                == outer_fold
                and role
                != "test"
            ):

                raise RuntimeError(
                    f"Patient {patient_id}: image_manifest says "
                    f"outer test Fold {outer_fold}, but frozen role "
                    f"is {role}."
                )

            if (
                fold
                != outer_fold
                and role
                == "test"
            ):

                raise RuntimeError(
                    f"Patient {patient_id} is unexpectedly test in "
                    f"Fold {fold}; image_manifest outer test fold is "
                    f"{outer_fold}."
                )

    print(
        "    Patient outer-test consistency: PASS"
    )


# =====================================================================
# CANONICAL PROTOCOL CONSTRUCTION
# =====================================================================

def build_canonical_rows(
    roles: Dict[
        int,
        Dict[
            str,
            str,
        ]
    ],
    images: Sequence[
        Dict[
            str,
            Any,
        ]
    ],
):

    print(
        "[5/7] Building and validating canonical model-independent protocol..."
    )

    patient_rows = []

    image_rows = []

    for fold in EXPECTED_FOLDS:

        fold_roles = roles[
            fold
        ]

        for patient_id in sorted(
            fold_roles.keys(),
            key=natural_key,
        ):

            patient_rows.append({
                "fold":
                    fold,

                "patient_id":
                    patient_id,

                "role":
                    fold_roles[
                        patient_id
                    ],
            })

        for image in sorted(
            images,
            key=lambda row:
                natural_key(
                    row[
                        "image_filename"
                    ]
                ),
        ):

            patient_id = image[
                "patient_id"
            ]

            image_rows.append({
                "fold":
                    fold,

                "role":
                    fold_roles[
                        patient_id
                    ],

                "patient_id":
                    patient_id,

                "eye_id":
                    image[
                        "eye_id"
                    ],

                "image_filename":
                    image[
                        "image_filename"
                    ],

                "outer_test_fold":
                    image[
                        "outer_test_fold"
                    ],
            })

    # -------------------------------------------------------------
    # Integrity checks
    # -------------------------------------------------------------

    if (
        len(
            patient_rows
        )
        != (
            EXPECTED_PATIENTS
            * len(
                EXPECTED_FOLDS
            )
        )
    ):

        raise RuntimeError(
            "Unexpected canonical patient-role row count."
        )

    if (
        len(
            image_rows
        )
        != (
            EXPECTED_IMAGES
            * len(
                EXPECTED_FOLDS
            )
        )
    ):

        raise RuntimeError(
            "Unexpected canonical image-role row count."
        )

    summary_rows = []

    test_images_across_folds = set()

    for fold in EXPECTED_FOLDS:

        fold_patient_rows = [
            row
            for row
            in patient_rows
            if row[
                "fold"
            ] == fold
        ]

        fold_image_rows = [
            row
            for row
            in image_rows
            if row[
                "fold"
            ] == fold
        ]

        psets = {
            role: {
                row[
                    "patient_id"
                ]
                for row
                in fold_patient_rows
                if row[
                    "role"
                ] == role
            }
            for role
            in [
                "train",
                "val",
                "test",
            ]
        }

        isets = {
            role: {
                row[
                    "image_filename"
                ]
                for row
                in fold_image_rows
                if row[
                    "role"
                ] == role
            }
            for role
            in [
                "train",
                "val",
                "test",
            ]
        }

        # Patient disjointness.
        if (
            psets[
                "train"
            ]
            & psets[
                "val"
            ]
        ):

            raise RuntimeError(
                f"Fold {fold}: train/val patient leakage."
            )

        if (
            psets[
                "train"
            ]
            & psets[
                "test"
            ]
        ):

            raise RuntimeError(
                f"Fold {fold}: train/test patient leakage."
            )

        if (
            psets[
                "val"
            ]
            & psets[
                "test"
            ]
        ):

            raise RuntimeError(
                f"Fold {fold}: val/test patient leakage."
            )

        if (
            (
                psets[
                    "train"
                ]
                | psets[
                    "val"
                ]
                | psets[
                    "test"
                ]
            )
            != {
                row[
                    "patient_id"
                ]
                for row
                in fold_patient_rows
            }
        ):

            raise RuntimeError(
                f"Fold {fold}: patient partition incomplete."
            )

        # Image disjointness.
        if (
            isets[
                "train"
            ]
            & isets[
                "val"
            ]
        ):

            raise RuntimeError(
                f"Fold {fold}: train/val image leakage."
            )

        if (
            isets[
                "train"
            ]
            & isets[
                "test"
            ]
        ):

            raise RuntimeError(
                f"Fold {fold}: train/test image leakage."
            )

        if (
            isets[
                "val"
            ]
            & isets[
                "test"
            ]
        ):

            raise RuntimeError(
                f"Fold {fold}: val/test image leakage."
            )

        if (
            len(
                isets[
                    "train"
                ]
                | isets[
                    "val"
                ]
                | isets[
                    "test"
                ]
            )
            != EXPECTED_IMAGES
        ):

            raise RuntimeError(
                f"Fold {fold}: image partition does not cover "
                f"all {EXPECTED_IMAGES} images."
            )

        expected_counts = (
            EXPECTED_PATIENT_ROLE_COUNTS[
                fold
            ]
        )

        for role in [
            "train",
            "val",
            "test",
        ]:

            if (
                len(
                    psets[
                        role
                    ]
                )
                != expected_counts[
                    role
                ]
            ):

                raise RuntimeError(
                    f"Fold {fold}: unexpected {role} patient count."
                )

        if (
            len(
                isets[
                    "test"
                ]
            )
            != EXPECTED_TEST_IMAGE_COUNTS[
                fold
            ]
        ):

            raise RuntimeError(
                f"Fold {fold}: unexpected test-image count."
            )

        overlap = (
            test_images_across_folds
            & isets[
                "test"
            ]
        )

        if overlap:

            raise RuntimeError(
                "An image appears in the test role in more "
                "than one fold."
            )

        test_images_across_folds.update(
            isets[
                "test"
            ]
        )

        summary_rows.append({
            "fold":
                fold,

            "train_patients":
                len(
                    psets[
                        "train"
                    ]
                ),

            "val_patients":
                len(
                    psets[
                        "val"
                    ]
                ),

            "test_patients":
                len(
                    psets[
                        "test"
                    ]
                ),

            "train_images":
                len(
                    isets[
                        "train"
                    ]
                ),

            "val_images":
                len(
                    isets[
                        "val"
                    ]
                ),

            "test_images":
                len(
                    isets[
                        "test"
                    ]
                ),

            "patient_leakage":
                0,

            "image_leakage":
                0,
        })

    if (
        len(
            test_images_across_folds
        )
        != EXPECTED_IMAGES
    ):

        raise RuntimeError(
            "Across all five folds, every image must appear "
            "in test exactly once."
        )

    print(
        "    Five-fold patient disjointness: PASS"
    )

    print(
        "    Five-fold image disjointness:   PASS"
    )

    print(
        "    Every image tested once:        PASS"
    )

    return (
        patient_rows,
        image_rows,
        summary_rows,
    )


# =====================================================================
# WRITE FROZEN CANONICAL PROTOCOL
# =====================================================================

def write_protocol(
    assets,
    detection,
    patient_rows,
    image_rows,
    summary_rows,
):

    print(
        "[6/7] Freezing canonical Step 05 baseline protocol..."
    )

    if FINAL_OUTPUT_DIR.exists():

        marker = (
            FINAL_OUTPUT_DIR
            / "PROTOCOL_FROZEN.json"
        )

        patient_csv = (
            FINAL_OUTPUT_DIR
            / "canonical_patient_roles.csv"
        )

        image_csv = (
            FINAL_OUTPUT_DIR
            / "canonical_image_roles.csv"
        )

        if (
            marker.exists()
            and patient_csv.exists()
            and image_csv.exists()
        ):

            frozen = load_json(
                marker
            )

            observed_patient_sha = (
                sha256_file(
                    patient_csv
                )
            )

            observed_image_sha = (
                sha256_file(
                    image_csv
                )
            )

            if (
                observed_patient_sha
                != safe_text(
                    frozen.get(
                        "canonical_patient_roles_sha256"
                    )
                )
                or observed_image_sha
                != safe_text(
                    frozen.get(
                        "canonical_image_roles_sha256"
                    )
                )
            ):

                raise RuntimeError(
                    "Existing patient5fold_protocol_v1 files "
                    "do not match their frozen fingerprints."
                )

            print(
                "    Canonical protocol already exists and "
                "fingerprints PASS."
            )

            print(
                "    No files were overwritten."
            )

            return frozen

        raise RuntimeError(
            "patient5fold_protocol_v1 already exists but is "
            "incomplete. Do not overwrite it automatically."
        )

    if BUILD_OUTPUT_DIR.exists():

        shutil.rmtree(
            BUILD_OUTPUT_DIR
        )

    BUILD_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    patient_csv = (
        BUILD_OUTPUT_DIR
        / "canonical_patient_roles.csv"
    )

    image_csv = (
        BUILD_OUTPUT_DIR
        / "canonical_image_roles.csv"
    )

    summary_csv = (
        BUILD_OUTPUT_DIR
        / "fold_summary.csv"
    )

    write_csv(
        patient_csv,
        patient_rows,
        [
            "fold",
            "patient_id",
            "role",
        ],
    )

    write_csv(
        image_csv,
        image_rows,
        [
            "fold",
            "role",
            "patient_id",
            "eye_id",
            "image_filename",
            "outer_test_fold",
        ],
    )

    write_csv(
        summary_csv,
        summary_rows,
    )

    # -------------------------------------------------------------
    # Fold-specific human-readable lists.
    # -------------------------------------------------------------

    for fold in EXPECTED_FOLDS:

        fold_dir = (
            BUILD_OUTPUT_DIR
            / f"fold_{fold}"
        )

        for role in [
            "train",
            "val",
            "test",
        ]:

            patients = sorted(
                [
                    row[
                        "patient_id"
                    ]
                    for row
                    in patient_rows
                    if (
                        row[
                            "fold"
                        ] == fold
                        and row[
                            "role"
                        ] == role
                    )
                ],
                key=natural_key,
            )

            images = sorted(
                [
                    row[
                        "image_filename"
                    ]
                    for row
                    in image_rows
                    if (
                        row[
                            "fold"
                        ] == fold
                        and row[
                            "role"
                        ] == role
                    )
                ],
                key=natural_key,
            )

            write_txt(
                fold_dir
                / f"{role}_patients.txt",
                patients,
            )

            write_txt(
                fold_dir
                / f"{role}_images.txt",
                images,
            )

    patient_sha = sha256_file(
        patient_csv
    )

    image_sha = sha256_file(
        image_csv
    )

    canonical_lines = [
        (
            f"{row['fold']}|"
            f"{row['role']}|"
            f"{row['patient_id']}|"
            f"{row['eye_id']}|"
            f"{row['image_filename']}|"
            f"{row['outer_test_fold']}"
        )
        for row
        in image_rows
    ]

    canonical_protocol_sha = (
        sha256_lines(
            canonical_lines
        )
    )

    summary = {
        "protocol_name":
            "patient5fold_protocol_v1",

        "source_split":
            "patient_5fold_v1",

        "source_fold_assignments_sha256":
            assets[
                "split_sha256"
            ],

        "geometry_v1_sha256":
            assets[
                "geometry_sha256"
            ],

        "assignment_detection":
            detection,

        "patient_count":
            EXPECTED_PATIENTS,

        "image_count":
            EXPECTED_IMAGES,

        "folds":
            summary_rows,

        "canonical_patient_roles_sha256":
            patient_sha,

        "canonical_image_roles_sha256":
            image_sha,

        "canonical_protocol_sha256":
            canonical_protocol_sha,

        "rules": {
            "patient_disjoint_within_each_fold":
                True,

            "eye_and_image_follow_patient_role":
                True,

            "every_patient_tested_exactly_once":
                True,

            "every_image_tested_exactly_once":
                True,

            "validation_roles_are_read_from_frozen_split":
                True,

            "validation_roles_regenerated":
                False,

            "future_models_must_use_identical_role_files":
                True,
        },
    }

    save_json(
        BUILD_OUTPUT_DIR
        / "protocol_summary.json",
        summary,
    )

    markdown = []

    markdown.append(
        "# RVO-ME Patient-Disjoint Baseline Protocol v1"
    )

    markdown.append("")
    markdown.append(
        "This protocol is a canonical copy of the already frozen "
        "`patient_5fold_v1` split. It does not create or optimize a "
        "new patient split."
    )

    markdown.append("")
    markdown.append(
        "## Required experimental rule"
    )

    markdown.append("")
    markdown.append(
        "All conventional baselines and the proposed geometry-aware "
        "model must use these exact patient/image roles for the primary "
        "Track-B five-fold comparison."
    )

    markdown.append("")
    markdown.append(
        "## Fold counts"
    )

    markdown.append("")
    markdown.append(
        "| Fold | Train patients | Val patients | Test patients | "
        "Train images | Val images | Test images |"
    )

    markdown.append(
        "|---:|---:|---:|---:|---:|---:|---:|"
    )

    for row in summary_rows:

        markdown.append(
            f"| {row['fold']} | "
            f"{row['train_patients']} | "
            f"{row['val_patients']} | "
            f"{row['test_patients']} | "
            f"{row['train_images']} | "
            f"{row['val_images']} | "
            f"{row['test_images']} |"
        )

    markdown.append("")
    markdown.append(
        "## Fingerprints"
    )

    markdown.append("")
    markdown.append(
        f"- Frozen source split SHA-256: "
        f"`{assets['split_sha256']}`"
    )

    markdown.append(
        f"- Canonical protocol SHA-256: "
        f"`{canonical_protocol_sha}`"
    )

    markdown.append("")
    markdown.append(
        "Do not edit these files after model experiments begin. "
        "Any future split variant must use a new protocol version."
    )

    (
        BUILD_OUTPUT_DIR
        / "PROTOCOL.md"
    ).write_text(
        "\n".join(
            markdown
        )
        + "\n",
        encoding="utf-8",
    )

    save_json(
        BUILD_OUTPUT_DIR
        / "PROTOCOL_FROZEN.json",
        summary,
    )

    BUILD_OUTPUT_DIR.rename(
        FINAL_OUTPUT_DIR
    )

    print(
        "    Protocol frozen successfully."
    )

    print(
        f"    Canonical protocol SHA-256:\n"
        f"    {canonical_protocol_sha}"
    )

    return summary


# =====================================================================
# FINAL REPORT
# =====================================================================

def print_final_report(
    frozen,
):

    print(
        "\n[7/7] COMPLETE."
    )

    print("\n")
    print("=" * 100)

    print(
        "RVO-ME STEP 05A — "
        "PATIENT-DISJOINT BASELINE PROTOCOL FROZEN"
    )

    print("=" * 100)

    print(
        "\nNo model training was performed."
    )

    print(
        "\nFrozen source split:"
    )

    print(
        f"patient_5fold_v1"
    )

    print(
        "\nCanonical protocol:"
    )

    print(
        f"{FINAL_OUTPUT_DIR}"
    )

    print(
        "\nFold summary:"
    )

    print(
        "-" * 90
    )

    print(
        f"{'Fold':<7}"
        f"{'Train P':>10}"
        f"{'Val P':>10}"
        f"{'Test P':>10}"
        f"{'Train Img':>12}"
        f"{'Val Img':>11}"
        f"{'Test Img':>11}"
    )

    print(
        "-" * 90
    )

    for row in frozen[
        "folds"
    ]:

        print(
            f"{row['fold']:<7}"
            f"{row['train_patients']:>10}"
            f"{row['val_patients']:>10}"
            f"{row['test_patients']:>10}"
            f"{row['train_images']:>12}"
            f"{row['val_images']:>11}"
            f"{row['test_images']:>11}"
        )

    print(
        "-" * 90
    )

    print(
        "\nLeakage checks: PASS"
    )

    print(
        "Every patient tested exactly once: PASS"
    )

    print(
        "Every image tested exactly once:   PASS"
    )

    print(
        "\nCanonical protocol SHA-256:"
    )

    print(
        frozen[
            "canonical_protocol_sha256"
        ]
    )

    print(
        "\nNEXT: Step 05B will train U-Net++ using these "
        "exact five frozen folds."
    )

    print("\n")
    print("=" * 100)


# =====================================================================
# MAIN
# =====================================================================

def main():

    print("=" * 100)

    print(
        "RVO-ME STEP 05A: "
        "FREEZE PATIENT-DISJOINT 5-FOLD BASELINE PROTOCOL"
    )

    print("=" * 100)

    print(
        f"\nFrozen split root:\n"
        f"{SPLIT_ROOT}"
    )

    print(
        f"\nCanonical output:\n"
        f"{FINAL_OUTPUT_DIR}"
    )

    assets = verify_frozen_assets()

    (
        images,
        patient_ids,
    ) = load_image_manifest()

    (
        roles,
        detection,
    ) = detect_assignments(
        patient_ids,
        images,
    )

    crosscheck_outer_test_fold(
        roles,
        images,
    )

    (
        patient_rows,
        image_rows,
        summary_rows,
    ) = build_canonical_rows(
        roles,
        images,
    )

    frozen = write_protocol(
        assets,
        detection,
        patient_rows,
        image_rows,
        summary_rows,
    )

    print_final_report(
        frozen
    )


if __name__ == "__main__":
    main()
