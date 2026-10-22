from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# =====================================================================
# RVO-ME STEP 05B
# TRACK-B PATIENT-DISJOINT U-NET++ — FOLD 3
#
# Primary scientific baseline track.
#
# Key rule:
#   Train patients -> model fitting
#   Validation patients -> checkpoint selection
#   Test patients -> evaluated exactly once after selection
#
# NO retraining on train+validation after checkpoint selection.
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

IMAGE_SEG_ROOT = (
    DATASET_ROOT
    / "Image_Seg"
)

IMAGES_DIR = (
    IMAGE_SEG_ROOT
    / "images"
)

MASKS_DIR = (
    IMAGE_SEG_ROOT
    / "masks"
)

FROZEN_SPLIT_ROOT = (
    PROJECT_ROOT
    / "splits"
    / "patient_5fold_v1"
)

GEOMETRY_ROOT = (
    PROJECT_ROOT
    / "targets"
    / "geometry_v1"
)

CANONICAL_PROTOCOL_ROOT = (
    PROJECT_ROOT
    / "baselines"
    / "segmentation"
    / "patient5fold_protocol_v1"
)

CANONICAL_IMAGE_ROLES_CSV = (
    CANONICAL_PROTOCOL_ROOT
    / "canonical_image_roles.csv"
)

CANONICAL_PATIENT_ROLES_CSV = (
    CANONICAL_PROTOCOL_ROOT
    / "canonical_patient_roles.csv"
)

CANONICAL_PROTOCOL_JSON = (
    CANONICAL_PROTOCOL_ROOT
    / "PROTOCOL_FROZEN.json"
)

TRACK_ROOT = (
    PROJECT_ROOT
    / "baselines"
    / "segmentation"
    / "trackB_conventional_v1"
)

TRAINING_PROTOCOL_JSON = (
    TRACK_ROOT
    / "TRAINING_PROTOCOL_FROZEN.json"
)

MODEL_NAME = "unetpp"
FOLD = 3

OUTPUT_DIR = (
    TRACK_ROOT
    / MODEL_NAME
    / f"fold_{FOLD}"
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

EXPECTED_CANONICAL_PROTOCOL_SHA256 = (
    "52ed516dcc21f1c51068c59fb662c9a0030b143e6440d6f4a0380df713c9bcaf"
)


# =====================================================================
# FOLD 1 EXPECTATIONS
# =====================================================================

EXPECTED_TRAIN_PATIENTS = 90
EXPECTED_VAL_PATIENTS = 14
EXPECTED_TEST_PATIENTS = 26

EXPECTED_TRAIN_IMAGES = 2106
EXPECTED_VAL_IMAGES = 308
EXPECTED_TEST_IMAGES = 598

EXPECTED_TOTAL_IMAGES = 3012


# =====================================================================
# MODEL / DATA
# =====================================================================

NUM_CLASSES = 5

CLASS_NAMES = {
    0: "Background",
    1: "SRF",
    2: "IRF",
    3: "ELM",
    4: "EZ",
}

FOREGROUND_CLASS_IDS = [
    1,
    2,
    3,
    4,
]

FILTERS = [
    64,
    128,
    256,
    512,
    1024,
]

INPUT_HEIGHT = 256
INPUT_WIDTH = 256
INPUT_CHANNELS = 3

ORIGINAL_HEIGHT = 380
ORIGINAL_WIDTH = 570


# =====================================================================
# FROZEN CONVENTIONAL TRAINING PROTOCOL v1
# =====================================================================

# This protocol intentionally follows the stronger clean 04B/v1 recipe,
# rather than the weaker source-aligned 04F run.
#
# It is frozen before Track-B model comparison and must remain identical
# for the conventional segmentation architectures.

SEED = 20260814

MAX_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5

INITIAL_LR = 1e-3
WEIGHT_DECAY = 0.0

CE_WEIGHT = 1.0
DICE_WEIGHT = 1.0
DICE_SMOOTH = 1e-5

HORIZONTAL_FLIP_P = 0.5

# RTX 4060 Laptop 8 GB already passed physical batch 4 in Step 04F.
# If an OOM occurs, automatically fall back while preserving an
# effective batch size of 4 through gradient accumulation.
REQUESTED_BATCH_SIZE = 4
TARGET_EFFECTIVE_BATCH = 4

NUM_WORKERS = 0
PIN_MEMORY = True
USE_AMP = True

AUTO_RESUME = True
SAVE_LAST_CHECKPOINT = True

# Model selection is performed at the actual 256x256 working raster.
CHECKPOINT_SELECTION_METRIC = (
    "validation_mean_foreground_dice_native256"
)

# Final test reporting is performed at BOTH:
#   1) native 256x256 model working scale
#   2) original 570x380 released-mask scale
FINAL_EVALUATION_SCALES = [
    "native256",
    "original570x380",
]


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


def set_seed(
    seed: int,
):

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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


def stable_json_sha256(
    data: Any,
) -> str:

    payload = json.dumps(
        data,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        ensure_ascii=False,
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        payload
    ).hexdigest()


def load_json(
    path: Path,
):

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
) -> List[
    Dict[
        str,
        str,
    ]
]:

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        reader = csv.DictReader(
            f
        )

        return [
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


def write_csv(
    path: Path,
    rows: List[
        Dict[
            str,
            Any,
        ]
    ],
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

    fieldnames = []

    seen = set()

    for row in rows:

        for key in row.keys():

            if key not in seen:

                seen.add(
                    key
                )

                fieldnames.append(
                    key
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


# =====================================================================
# FROZEN PROTOCOL VERIFICATION
# =====================================================================

def build_training_protocol_dict():

    return {
        "protocol_name":
            "trackB_conventional_training_protocol_v1",

        "canonical_patient_protocol":
            "patient5fold_protocol_v1",

        "canonical_protocol_sha256":
            EXPECTED_CANONICAL_PROTOCOL_SHA256,

        "task":
            "5-class conventional semantic segmentation",

        "classes":
            CLASS_NAMES,

        "input": {
            "height":
                INPUT_HEIGHT,

            "width":
                INPUT_WIDTH,

            "channels":
                INPUT_CHANNELS,

            "source":
                "grayscale OCT duplicated to 3 channels",

            "scaling":
                "[0,1]",
        },

        "augmentation": {
            "horizontal_flip_probability":
                HORIZONTAL_FLIP_P,

            "rotation":
                False,

            "vertical_flip":
                False,

            "photometric":
                False,
        },

        "loss": {
            "name":
                "CrossEntropy + foreground soft multiclass Dice",

            "cross_entropy_weight":
                CE_WEIGHT,

            "dice_weight":
                DICE_WEIGHT,

            "dice_smooth":
                DICE_SMOOTH,

            "dice_classes":
                [
                    "SRF",
                    "IRF",
                    "ELM",
                    "EZ",
                ],
        },

        "optimizer": {
            "name":
                "Adam",

            "initial_lr":
                INITIAL_LR,

            "weight_decay":
                WEIGHT_DECAY,

            "scheduler":
                None,
        },

        "training": {
            "seed":
                SEED,

            "maximum_epochs":
                MAX_EPOCHS,

            "early_stopping_patience":
                EARLY_STOPPING_PATIENCE,

            "checkpoint_selection_metric":
                CHECKPOINT_SELECTION_METRIC,

            "requested_physical_batch_size":
                REQUESTED_BATCH_SIZE,

            "target_effective_batch_size":
                TARGET_EFFECTIVE_BATCH,

            "amp":
                USE_AMP,

            "num_workers":
                NUM_WORKERS,

            "retrain_train_plus_validation":
                False,
        },

        "evaluation": {
            "test_evaluated_only_after_model_selection":
                True,

            "scales":
                FINAL_EVALUATION_SCALES,

            "primary_segmentation_metrics":
                [
                    "Dice",
                    "IoU",
                ],

            "patient_level_confusion_saved":
                True,

            "per_image_metrics_saved":
                True,
        },

        "policy": {
            "same_protocol_for_all_conventional_models":
                True,

            "do_not_tune_after_test_results":
                True,

            "future_protocol_changes_require_new_version":
                True,
        },
    }


def verify_and_freeze_protocols():

    print(
        "\n[1/8] Verifying frozen Track-B assets..."
    )

    required = [
        CANONICAL_PROTOCOL_JSON,
        CANONICAL_IMAGE_ROLES_CSV,
        CANONICAL_PATIENT_ROLES_CSV,
        GEOMETRY_ROOT
        / "geometry_summary.json",
    ]

    for path in required:

        if not path.exists():

            raise FileNotFoundError(
                f"Missing required frozen asset:\n"
                f"{path}"
            )

    canonical = load_json(
        CANONICAL_PROTOCOL_JSON
    )

    canonical_sha = safe_text(
        canonical.get(
            "canonical_protocol_sha256"
        )
    )

    if (
        canonical_sha
        != EXPECTED_CANONICAL_PROTOCOL_SHA256
    ):

        raise RuntimeError(
            "Canonical patient5fold protocol fingerprint changed.\n"
            f"Expected: {EXPECTED_CANONICAL_PROTOCOL_SHA256}\n"
            f"Observed: {canonical_sha}"
        )

    source_split_sha = safe_text(
        canonical.get(
            "source_fold_assignments_sha256"
        )
    )

    if (
        source_split_sha
        != EXPECTED_SPLIT_SHA256
    ):

        raise RuntimeError(
            "Source patient_5fold_v1 fingerprint changed."
        )

    geometry_summary = load_json(
        GEOMETRY_ROOT
        / "geometry_summary.json"
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
            "geometry_v1 fingerprint changed."
        )

    proposed_protocol = (
        build_training_protocol_dict()
    )

    proposed_protocol[
        "training_protocol_sha256"
    ] = stable_json_sha256(
        proposed_protocol
    )

    TRACK_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    if TRAINING_PROTOCOL_JSON.exists():

        existing = load_json(
            TRAINING_PROTOCOL_JSON
        )

        # ---------------------------------------------------------
        # JSON-safe protocol comparison.
        #
        # JSON converts integer dictionary keys (e.g. CLASS_NAMES:
        # {0: "Background", ...}) into strings when saved. A raw
        # Python dict comparison therefore produces a false mismatch
        # on every later fold even when the protocol is identical.
        #
        # Normalize BOTH structures through JSON before comparing,
        # and independently require the already-frozen protocol SHA.
        # ---------------------------------------------------------

        existing_sha = safe_text(
            existing.get(
                "training_protocol_sha256"
            )
        )

        proposed_sha = safe_text(
            proposed_protocol.get(
                "training_protocol_sha256"
            )
        )

        if existing_sha != proposed_sha:

            raise RuntimeError(
                "A different Track-B conventional training protocol "
                "is already frozen.\n"
                f"Frozen SHA:   {existing_sha}\n"
                f"Proposed SHA: {proposed_sha}\n"
                "Do NOT overwrite it after experiments begin."
            )

        existing_normalized = json.loads(
            json.dumps(
                existing,
                sort_keys=True,
                ensure_ascii=False,
            )
        )

        proposed_normalized = json.loads(
            json.dumps(
                proposed_protocol,
                sort_keys=True,
                ensure_ascii=False,
            )
        )

        if existing_normalized != proposed_normalized:

            raise RuntimeError(
                "The frozen Track-B protocol content differs even "
                "though its protocol identifier was expected to match. "
                "Refusing to continue."
            )

        print(
            "    Canonical patient protocol SHA: PASS"
        )

        print(
            "    Conventional training protocol: already frozen / PASS"
        )

    else:

        save_json(
            TRAINING_PROTOCOL_JSON,
            proposed_protocol,
        )

        print(
            "    Canonical patient protocol SHA: PASS"
        )

        print(
            "    Conventional training protocol: FROZEN"
        )

    print(
        "    geometry_v1 SHA: PASS"
    )

    print(
        f"    Training protocol SHA-256:\n"
        f"    {proposed_protocol['training_protocol_sha256']}"
    )

    return (
        canonical,
        proposed_protocol,
    )


# =====================================================================
# FOLD-1 ROLE LOADING / LEAKAGE CHECK
# =====================================================================

def load_fold_protocol():

    print(
        "[2/8] Loading canonical Fold 3 train/val/test roles..."
    )

    rows = read_csv_rows(
        CANONICAL_IMAGE_ROLES_CSV
    )

    fold_rows = [
        row
        for row
        in rows
        if int(
            row[
                "fold"
            ]
        ) == FOLD
    ]

    if (
        len(
            fold_rows
        )
        != EXPECTED_TOTAL_IMAGES
    ):

        raise RuntimeError(
            f"Fold {FOLD} should contain "
            f"{EXPECTED_TOTAL_IMAGES} role rows, "
            f"found {len(fold_rows)}."
        )

    role_rows = {
        role: [
            row
            for row
            in fold_rows
            if row[
                "role"
            ] == role
        ]
        for role
        in [
            "train",
            "val",
            "test",
        ]
    }

    patient_sets = {
        role: {
            canonical_id(
                row[
                    "patient_id"
                ]
            )
            for row
            in role_rows[
                role
            ]
        }
        for role
        in [
            "train",
            "val",
            "test",
        ]
    }

    image_sets = {
        role: {
            row[
                "image_filename"
            ]
            for row
            in role_rows[
                role
            ]
        }
        for role
        in [
            "train",
            "val",
            "test",
        ]
    }

    expected_patient_counts = {
        "train":
            EXPECTED_TRAIN_PATIENTS,

        "val":
            EXPECTED_VAL_PATIENTS,

        "test":
            EXPECTED_TEST_PATIENTS,
    }

    expected_image_counts = {
        "train":
            EXPECTED_TRAIN_IMAGES,

        "val":
            EXPECTED_VAL_IMAGES,

        "test":
            EXPECTED_TEST_IMAGES,
    }

    for role in [
        "train",
        "val",
        "test",
    ]:

        if (
            len(
                patient_sets[
                    role
                ]
            )
            != expected_patient_counts[
                role
            ]
        ):

            raise RuntimeError(
                f"Fold 1 {role} patient count mismatch."
            )

        if (
            len(
                image_sets[
                    role
                ]
            )
            != expected_image_counts[
                role
            ]
        ):

            raise RuntimeError(
                f"Fold 1 {role} image count mismatch."
            )

    if (
        patient_sets[
            "train"
        ]
        & patient_sets[
            "val"
        ]
    ):

        raise RuntimeError(
            "Fold 3 train/val patient leakage."
        )

    if (
        patient_sets[
            "train"
        ]
        & patient_sets[
            "test"
        ]
    ):

        raise RuntimeError(
            "Fold 3 train/test patient leakage."
        )

    if (
        patient_sets[
            "val"
        ]
        & patient_sets[
            "test"
        ]
    ):

        raise RuntimeError(
            "Fold 3 val/test patient leakage."
        )

    if (
        image_sets[
            "train"
        ]
        & image_sets[
            "val"
        ]
        or image_sets[
            "train"
        ]
        & image_sets[
            "test"
        ]
        or image_sets[
            "val"
        ]
        & image_sets[
            "test"
        ]
    ):

        raise RuntimeError(
            "Fold 3 image leakage."
        )

    print(
        f"    Train: {EXPECTED_TRAIN_PATIENTS} patients, "
        f"{EXPECTED_TRAIN_IMAGES} images"
    )

    print(
        f"    Val:   {EXPECTED_VAL_PATIENTS} patients, "
        f"{EXPECTED_VAL_IMAGES} images"
    )

    print(
        f"    Test:  {EXPECTED_TEST_PATIENTS} patients, "
        f"{EXPECTED_TEST_IMAGES} images"
    )

    print(
        "    Patient leakage: 0"
    )

    print(
        "    Image leakage:   0"
    )

    # Save an exact copy inside the experiment.
    protocol_dir = (
        OUTPUT_DIR
        / "protocol"
    )

    protocol_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for role in [
        "train",
        "val",
        "test",
    ]:

        write_csv(
            protocol_dir
            / f"{role}_images.csv",
            role_rows[
                role
            ],
        )

    summary = {
        "fold":
            FOLD,

        "train_patients":
            EXPECTED_TRAIN_PATIENTS,

        "val_patients":
            EXPECTED_VAL_PATIENTS,

        "test_patients":
            EXPECTED_TEST_PATIENTS,

        "train_images":
            EXPECTED_TRAIN_IMAGES,

        "val_images":
            EXPECTED_VAL_IMAGES,

        "test_images":
            EXPECTED_TEST_IMAGES,

        "patient_leakage":
            0,

        "image_leakage":
            0,

        "canonical_protocol_sha256":
            EXPECTED_CANONICAL_PROTOCOL_SHA256,
    }

    save_json(
        protocol_dir
        / "fold_protocol_summary.json",
        summary,
    )

    return role_rows


# =====================================================================
# FILE LOOKUPS
# =====================================================================

def build_file_lookup(
    directory: Path,
) -> Dict[
    str,
    Path,
]:

    lookup = {}

    for path in directory.iterdir():

        if not path.is_file():
            continue

        stem = path.stem

        if stem in lookup:

            raise RuntimeError(
                f"Duplicate stem '{stem}' in {directory}"
            )

        lookup[
            stem
        ] = path

    return lookup


IMAGE_LOOKUP = None
MASK_LOOKUP = None


def initialize_file_lookups():

    global IMAGE_LOOKUP
    global MASK_LOOKUP

    IMAGE_LOOKUP = build_file_lookup(
        IMAGES_DIR
    )

    MASK_LOOKUP = build_file_lookup(
        MASKS_DIR
    )

    if (
        len(
            IMAGE_LOOKUP
        )
        != EXPECTED_TOTAL_IMAGES
    ):

        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL_IMAGES} images; "
            f"found {len(IMAGE_LOOKUP)}."
        )

    if (
        len(
            MASK_LOOKUP
        )
        != EXPECTED_TOTAL_IMAGES
    ):

        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL_IMAGES} masks; "
            f"found {len(MASK_LOOKUP)}."
        )


# =====================================================================
# DATASET
# =====================================================================

class RVOSegmentationDataset(
    Dataset
):

    def __init__(
        self,
        rows: Sequence[
            Dict[
                str,
                str,
            ]
        ],
        training: bool,
        evaluation: bool = False,
    ):

        self.rows = list(
            rows
        )

        self.training = bool(
            training
        )

        self.evaluation = bool(
            evaluation
        )

    def __len__(
        self,
    ):

        return len(
            self.rows
        )

    def __getitem__(
        self,
        index: int,
    ):

        row = self.rows[
            index
        ]

        filename = row[
            "image_filename"
        ]

        stem = Path(
            filename
        ).stem

        patient_id = canonical_id(
            row[
                "patient_id"
            ]
        )

        if (
            stem
            not in IMAGE_LOOKUP
            or stem
            not in MASK_LOOKUP
        ):

            raise RuntimeError(
                f"Image/mask lookup failed for {filename}."
            )

        image = cv2.imread(
            str(
                IMAGE_LOOKUP[
                    stem
                ]
            ),
            cv2.IMREAD_GRAYSCALE,
        )

        mask_original = cv2.imread(
            str(
                MASK_LOOKUP[
                    stem
                ]
            ),
            cv2.IMREAD_GRAYSCALE,
        )

        if (
            image is None
            or mask_original is None
        ):

            raise RuntimeError(
                f"Could not read {filename}."
            )

        if (
            image.shape
            != (
                ORIGINAL_HEIGHT,
                ORIGINAL_WIDTH,
            )
            or mask_original.shape
            != (
                ORIGINAL_HEIGHT,
                ORIGINAL_WIDTH,
            )
        ):

            raise RuntimeError(
                f"Unexpected image/mask shape for {filename}."
            )

        if self.training:

            if (
                random.random()
                < HORIZONTAL_FLIP_P
            ):

                image = cv2.flip(
                    image,
                    1,
                )

                mask_original = cv2.flip(
                    mask_original,
                    1,
                )

        image_256 = cv2.resize(
            image,
            (
                INPUT_WIDTH,
                INPUT_HEIGHT,
            ),
            interpolation=
                cv2.INTER_LINEAR,
        )

        mask_256 = cv2.resize(
            mask_original,
            (
                INPUT_WIDTH,
                INPUT_HEIGHT,
            ),
            interpolation=
                cv2.INTER_NEAREST,
        )

        image_float = (
            image_256.astype(
                np.float32
            )
            / 255.0
        )

        image_3ch = np.stack(
            [
                image_float,
                image_float,
                image_float,
            ],
            axis=0,
        )

        image_tensor = (
            torch.from_numpy(
                image_3ch
            )
        )

        mask_256_tensor = (
            torch.from_numpy(
                mask_256.astype(
                    np.int64
                )
            )
        )

        if not self.evaluation:

            return (
                image_tensor,
                mask_256_tensor,
            )

        mask_original_tensor = (
            torch.from_numpy(
                mask_original.astype(
                    np.int64
                )
            )
        )

        return (
            image_tensor,
            mask_256_tensor,
            mask_original_tensor,
            stem,
            patient_id,
        )


def make_loader(
    rows,
    batch_size: int,
    training: bool,
    evaluation: bool = False,
):

    dataset = RVOSegmentationDataset(
        rows=rows,
        training=training,
        evaluation=evaluation,
    )

    generator = torch.Generator()

    generator.manual_seed(
        SEED
        + (
            1
            if training
            else 2
        )
    )

    return DataLoader(
        dataset,
        batch_size=
            batch_size,
        shuffle=
            training,
        num_workers=
            NUM_WORKERS,
        pin_memory=(
            PIN_MEMORY
            and torch.cuda.is_available()
        ),
        drop_last=False,
        generator=
            generator,
    )


# =====================================================================
# U-NET++ ARCHITECTURE
# =====================================================================

class VGGBlock(
    nn.Module
):

    def __init__(
        self,
        in_channels: int,
        middle_channels: int,
        out_channels: int,
    ):

        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            middle_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(
            middle_channels
        )

        self.conv2 = nn.Conv2d(
            middle_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )

        self.bn2 = nn.BatchNorm2d(
            out_channels
        )

        self.relu = nn.ReLU(
            inplace=True
        )

    def forward(
        self,
        x,
    ):

        x = self.relu(
            self.bn1(
                self.conv1(
                    x
                )
            )
        )

        x = self.relu(
            self.bn2(
                self.conv2(
                    x
                )
            )
        )

        return x


class NestedUNet(
    nn.Module
):

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        input_channels: int = INPUT_CHANNELS,
        filters: Sequence[
            int
        ] = FILTERS,
    ):

        super().__init__()

        f = list(
            filters
        )

        self.pool = nn.MaxPool2d(
            2,
            2,
        )

        self.up = nn.Upsample(
            scale_factor=2,
            mode="bilinear",
            align_corners=True,
        )

        self.conv0_0 = VGGBlock(
            input_channels,
            f[0],
            f[0],
        )

        self.conv1_0 = VGGBlock(
            f[0],
            f[1],
            f[1],
        )

        self.conv2_0 = VGGBlock(
            f[1],
            f[2],
            f[2],
        )

        self.conv3_0 = VGGBlock(
            f[2],
            f[3],
            f[3],
        )

        self.conv4_0 = VGGBlock(
            f[3],
            f[4],
            f[4],
        )

        self.conv0_1 = VGGBlock(
            f[0]
            + f[1],
            f[0],
            f[0],
        )

        self.conv1_1 = VGGBlock(
            f[1]
            + f[2],
            f[1],
            f[1],
        )

        self.conv2_1 = VGGBlock(
            f[2]
            + f[3],
            f[2],
            f[2],
        )

        self.conv3_1 = VGGBlock(
            f[3]
            + f[4],
            f[3],
            f[3],
        )

        self.conv0_2 = VGGBlock(
            f[0]
            * 2
            + f[1],
            f[0],
            f[0],
        )

        self.conv1_2 = VGGBlock(
            f[1]
            * 2
            + f[2],
            f[1],
            f[1],
        )

        self.conv2_2 = VGGBlock(
            f[2]
            * 2
            + f[3],
            f[2],
            f[2],
        )

        self.conv0_3 = VGGBlock(
            f[0]
            * 3
            + f[1],
            f[0],
            f[0],
        )

        self.conv1_3 = VGGBlock(
            f[1]
            * 3
            + f[2],
            f[1],
            f[1],
        )

        self.conv0_4 = VGGBlock(
            f[0]
            * 4
            + f[1],
            f[0],
            f[0],
        )

        self.final = nn.Conv2d(
            f[0],
            num_classes,
            kernel_size=1,
        )

    def forward(
        self,
        x,
    ):

        x0_0 = self.conv0_0(
            x
        )

        x1_0 = self.conv1_0(
            self.pool(
                x0_0
            )
        )

        x0_1 = self.conv0_1(
            torch.cat(
                [
                    x0_0,
                    self.up(
                        x1_0
                    ),
                ],
                dim=1,
            )
        )

        x2_0 = self.conv2_0(
            self.pool(
                x1_0
            )
        )

        x1_1 = self.conv1_1(
            torch.cat(
                [
                    x1_0,
                    self.up(
                        x2_0
                    ),
                ],
                dim=1,
            )
        )

        x0_2 = self.conv0_2(
            torch.cat(
                [
                    x0_0,
                    x0_1,
                    self.up(
                        x1_1
                    ),
                ],
                dim=1,
            )
        )

        x3_0 = self.conv3_0(
            self.pool(
                x2_0
            )
        )

        x2_1 = self.conv2_1(
            torch.cat(
                [
                    x2_0,
                    self.up(
                        x3_0
                    ),
                ],
                dim=1,
            )
        )

        x1_2 = self.conv1_2(
            torch.cat(
                [
                    x1_0,
                    x1_1,
                    self.up(
                        x2_1
                    ),
                ],
                dim=1,
            )
        )

        x0_3 = self.conv0_3(
            torch.cat(
                [
                    x0_0,
                    x0_1,
                    x0_2,
                    self.up(
                        x1_2
                    ),
                ],
                dim=1,
            )
        )

        x4_0 = self.conv4_0(
            self.pool(
                x3_0
            )
        )

        x3_1 = self.conv3_1(
            torch.cat(
                [
                    x3_0,
                    self.up(
                        x4_0
                    ),
                ],
                dim=1,
            )
        )

        x2_2 = self.conv2_2(
            torch.cat(
                [
                    x2_0,
                    x2_1,
                    self.up(
                        x3_1
                    ),
                ],
                dim=1,
            )
        )

        x1_3 = self.conv1_3(
            torch.cat(
                [
                    x1_0,
                    x1_1,
                    x1_2,
                    self.up(
                        x2_2
                    ),
                ],
                dim=1,
            )
        )

        x0_4 = self.conv0_4(
            torch.cat(
                [
                    x0_0,
                    x0_1,
                    x0_2,
                    x0_3,
                    self.up(
                        x1_3
                    ),
                ],
                dim=1,
            )
        )

        return self.final(
            x0_4
        )


# =====================================================================
# LOSS
# =====================================================================

class CEDiceLoss(
    nn.Module
):

    def __init__(
        self,
        num_classes: int,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        smooth: float = 1e-5,
    ):

        super().__init__()

        self.num_classes = int(
            num_classes
        )

        self.ce_weight = float(
            ce_weight
        )

        self.dice_weight = float(
            dice_weight
        )

        self.smooth = float(
            smooth
        )

        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        logits,
        target,
    ):

        ce = self.ce(
            logits,
            target,
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        one_hot = F.one_hot(
            target,
            num_classes=
                self.num_classes,
        ).permute(
            0,
            3,
            1,
            2,
        ).float()

        dice_scores = []

        for class_id in (
            FOREGROUND_CLASS_IDS
        ):

            pred = probabilities[
                :,
                class_id,
                :,
                :
            ]

            gt = one_hot[
                :,
                class_id,
                :,
                :
            ]

            intersection = (
                pred
                * gt
            ).sum(
                dim=(
                    1,
                    2,
                )
            )

            denominator = (
                pred.sum(
                    dim=(
                        1,
                        2,
                    )
                )
                + gt.sum(
                    dim=(
                        1,
                        2,
                    )
                )
            )

            dice = (
                (
                    2.0
                    * intersection
                    + self.smooth
                )
                / (
                    denominator
                    + self.smooth
                )
            )

            dice_scores.append(
                dice
            )

        dice_mean = torch.stack(
            dice_scores,
            dim=1,
        ).mean()

        dice_loss = (
            1.0
            - dice_mean
        )

        total = (
            self.ce_weight
            * ce
            + self.dice_weight
            * dice_loss
        )

        return (
            total,
            ce.detach(),
            dice_loss.detach(),
        )


# =====================================================================
# CONFUSION / METRICS
# =====================================================================

def new_confusion():

    return np.zeros(
        (
            NUM_CLASSES,
            NUM_CLASSES,
        ),
        dtype=np.int64,
    )


def update_confusion(
    confusion: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
):

    target_flat = target.reshape(
        -1
    ).astype(
        np.int64
    )

    pred_flat = prediction.reshape(
        -1
    ).astype(
        np.int64
    )

    encoded = (
        target_flat
        * NUM_CLASSES
        + pred_flat
    )

    counts = np.bincount(
        encoded,
        minlength=
            NUM_CLASSES
            * NUM_CLASSES,
    ).reshape(
        NUM_CLASSES,
        NUM_CLASSES,
    )

    confusion += counts


def metrics_from_confusion(
    confusion: np.ndarray,
):

    classes = {}

    foreground_dice = []
    foreground_iou = []

    for class_id in range(
        NUM_CLASSES
    ):

        tp = int(
            confusion[
                class_id,
                class_id,
            ]
        )

        fp = int(
            confusion[
                :,
                class_id,
            ].sum()
            - tp
        )

        fn = int(
            confusion[
                class_id,
                :,
            ].sum()
            - tp
        )

        gt_pixels = int(
            confusion[
                class_id,
                :
            ].sum()
        )

        pred_pixels = int(
            confusion[
                :,
                class_id
            ].sum()
        )

        dice_den = (
            2
            * tp
            + fp
            + fn
        )

        iou_den = (
            tp
            + fp
            + fn
        )

        dice = (
            1.0
            if dice_den == 0
            else (
                2.0
                * tp
                / dice_den
            )
        )

        iou = (
            1.0
            if iou_den == 0
            else (
                tp
                / iou_den
            )
        )

        name = CLASS_NAMES[
            class_id
        ]

        classes[
            name
        ] = {
            "class_id":
                class_id,

            "dice":
                float(
                    dice
                ),

            "iou":
                float(
                    iou
                ),

            "tp":
                tp,

            "fp":
                fp,

            "fn":
                fn,

            "gt_pixels":
                gt_pixels,

            "pred_pixels":
                pred_pixels,
        }

        if class_id in (
            FOREGROUND_CLASS_IDS
        ):

            foreground_dice.append(
                float(
                    dice
                )
            )

            foreground_iou.append(
                float(
                    iou
                )
            )

    return {
        "classes":
            classes,

        "mean_foreground_dice":
            float(
                np.mean(
                    foreground_dice
                )
            ),

        "mean_foreground_iou":
            float(
                np.mean(
                    foreground_iou
                )
            ),

        "confusion_matrix":
            confusion.tolist(),
    }


def single_image_class_counts(
    target: np.ndarray,
    prediction: np.ndarray,
    prefix: str,
):

    row = {}

    dice_values = []
    iou_values = []

    for class_id in (
        FOREGROUND_CLASS_IDS
    ):

        name = CLASS_NAMES[
            class_id
        ]

        gt = (
            target
            == class_id
        )

        pred = (
            prediction
            == class_id
        )

        tp = int(
            np.logical_and(
                gt,
                pred,
            ).sum()
        )

        fp = int(
            np.logical_and(
                ~gt,
                pred,
            ).sum()
        )

        fn = int(
            np.logical_and(
                gt,
                ~pred,
            ).sum()
        )

        dice_den = (
            2
            * tp
            + fp
            + fn
        )

        iou_den = (
            tp
            + fp
            + fn
        )

        if dice_den == 0:

            dice = None
            iou = None

        else:

            dice = (
                2.0
                * tp
                / dice_den
            )

            iou = (
                tp
                / iou_den
            )

            dice_values.append(
                dice
            )

            iou_values.append(
                iou
            )

        row[
            f"{prefix}_{name}_tp"
        ] = tp

        row[
            f"{prefix}_{name}_fp"
        ] = fp

        row[
            f"{prefix}_{name}_fn"
        ] = fn

        row[
            f"{prefix}_{name}_dice"
        ] = (
            ""
            if dice is None
            else float(
                dice
            )
        )

        row[
            f"{prefix}_{name}_iou"
        ] = (
            ""
            if iou is None
            else float(
                iou
            )
        )

    row[
        f"{prefix}_mean_present_class_dice"
    ] = (
        ""
        if not dice_values
        else float(
            np.mean(
                dice_values
            )
        )
    )

    row[
        f"{prefix}_mean_present_class_iou"
    ] = (
        ""
        if not iou_values
        else float(
            np.mean(
                iou_values
            )
        )
    )

    return row


def patient_rows_from_confusions(
    patient_confusions,
    scale_name: str,
):

    rows = []

    for patient_id in sorted(
        patient_confusions.keys(),
        key=natural_key,
    ):

        confusion = patient_confusions[
            patient_id
        ]

        metrics = metrics_from_confusion(
            confusion
        )

        row = {
            "patient_id":
                patient_id,

            "scale":
                scale_name,

            "mean_foreground_dice":
                metrics[
                    "mean_foreground_dice"
                ],

            "mean_foreground_iou":
                metrics[
                    "mean_foreground_iou"
                ],
        }

        for class_id in (
            FOREGROUND_CLASS_IDS
        ):

            name = CLASS_NAMES[
                class_id
            ]

            class_metrics = metrics[
                "classes"
            ][
                name
            ]

            for field in [
                "tp",
                "fp",
                "fn",
                "gt_pixels",
                "pred_pixels",
                "dice",
                "iou",
            ]:

                row[
                    f"{name}_{field}"
                ] = class_metrics[
                    field
                ]

        rows.append(
            row
        )

    return rows


# =====================================================================
# TRAINING / VALIDATION
# =====================================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    criterion,
    device,
    accumulation_steps,
):

    model.train()

    optimizer.zero_grad(
        set_to_none=True
    )

    total_loss = 0.0
    total_ce = 0.0
    total_dice_loss = 0.0
    total_samples = 0

    for batch_index, (
        images,
        targets,
    ) in enumerate(
        loader,
        start=1,
    ):

        images = images.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )

        with torch.amp.autocast(
            device_type="cuda",
            enabled=(
                USE_AMP
                and device.type
                == "cuda"
            ),
        ):

            logits = model(
                images
            )

            (
                loss,
                ce_component,
                dice_component,
            ) = criterion(
                logits,
                targets,
            )

            scaled_loss = (
                loss
                / accumulation_steps
            )

        scaler.scale(
            scaled_loss
        ).backward()

        should_step = (
            batch_index
            % accumulation_steps
            == 0
            or batch_index
            == len(
                loader
            )
        )

        if should_step:

            scaler.step(
                optimizer
            )

            scaler.update()

            optimizer.zero_grad(
                set_to_none=True
            )

        n = int(
            images.size(
                0
            )
        )

        total_loss += (
            float(
                loss.item()
            )
            * n
        )

        total_ce += (
            float(
                ce_component.item()
            )
            * n
        )

        total_dice_loss += (
            float(
                dice_component.item()
            )
            * n
        )

        total_samples += n

    return {
        "loss":
            total_loss
            / total_samples,

        "ce_loss":
            total_ce
            / total_samples,

        "dice_loss":
            total_dice_loss
            / total_samples,
    }


@torch.no_grad()
def evaluate_native256(
    model,
    loader,
    device,
    criterion=None,
):

    model.eval()

    confusion = new_confusion()

    total_loss = 0.0
    total_samples = 0

    for batch in loader:

        if len(
            batch
        ) == 2:

            (
                images,
                targets,
            ) = batch

        else:

            (
                images,
                targets,
                _,
                _,
                _,
            ) = batch

        images = images.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )

        with torch.amp.autocast(
            device_type="cuda",
            enabled=(
                USE_AMP
                and device.type
                == "cuda"
            ),
        ):

            logits = model(
                images
            )

            if criterion is not None:

                (
                    loss,
                    _,
                    _,
                ) = criterion(
                    logits,
                    targets,
                )

                total_loss += (
                    float(
                        loss.item()
                    )
                    * int(
                        images.size(
                            0
                        )
                    )
                )

        predictions = torch.argmax(
            logits,
            dim=1,
        ).detach().cpu().numpy()

        targets_np = (
            targets
            .detach()
            .cpu()
            .numpy()
        )

        for gt, pred in zip(
            targets_np,
            predictions,
        ):

            update_confusion(
                confusion,
                gt,
                pred,
            )

        total_samples += int(
            images.size(
                0
            )
        )

    metrics = metrics_from_confusion(
        confusion
    )

    metrics[
        "loss"
    ] = (
        None
        if criterion is None
        else (
            total_loss
            / total_samples
        )
    )

    return metrics


# =====================================================================
# FULL TEST EVALUATION AT 256 AND 570x380
# =====================================================================

@torch.no_grad()
def evaluate_test_both_scales(
    model,
    loader,
    device,
):

    model.eval()

    global_256 = new_confusion()
    global_570 = new_confusion()

    patient_256 = defaultdict(
        new_confusion
    )

    patient_570 = defaultdict(
        new_confusion
    )

    per_image_rows = []

    for (
        images,
        targets_256,
        targets_original,
        stems,
        patient_ids,
    ) in loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        with torch.amp.autocast(
            device_type="cuda",
            enabled=(
                USE_AMP
                and device.type
                == "cuda"
            ),
        ):

            logits = model(
                images
            )

        predictions_256 = (
            torch.argmax(
                logits,
                dim=1,
            )
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.uint8
            )
        )

        targets_256_np = (
            targets_256
            .numpy()
            .astype(
                np.uint8
            )
        )

        targets_original_np = (
            targets_original
            .numpy()
            .astype(
                np.uint8
            )
        )

        for (
            pred_256,
            gt_256,
            gt_570,
            stem,
            patient_id,
        ) in zip(
            predictions_256,
            targets_256_np,
            targets_original_np,
            stems,
            patient_ids,
        ):

            patient_id = canonical_id(
                patient_id
            )

            pred_570 = cv2.resize(
                pred_256,
                (
                    ORIGINAL_WIDTH,
                    ORIGINAL_HEIGHT,
                ),
                interpolation=
                    cv2.INTER_NEAREST,
            )

            update_confusion(
                global_256,
                gt_256,
                pred_256,
            )

            update_confusion(
                global_570,
                gt_570,
                pred_570,
            )

            update_confusion(
                patient_256[
                    patient_id
                ],
                gt_256,
                pred_256,
            )

            update_confusion(
                patient_570[
                    patient_id
                ],
                gt_570,
                pred_570,
            )

            row = {
                "image_stem":
                    stem,

                "patient_id":
                    patient_id,
            }

            row.update(
                single_image_class_counts(
                    gt_256,
                    pred_256,
                    "native256",
                )
            )

            row.update(
                single_image_class_counts(
                    gt_570,
                    pred_570,
                    "original570",
                )
            )

            per_image_rows.append(
                row
            )

    metrics_256 = metrics_from_confusion(
        global_256
    )

    metrics_570 = metrics_from_confusion(
        global_570
    )

    patient_rows_256 = (
        patient_rows_from_confusions(
            patient_256,
            "native256",
        )
    )

    patient_rows_570 = (
        patient_rows_from_confusions(
            patient_570,
            "original570x380",
        )
    )

    return {
        "native256":
            metrics_256,

        "original570x380":
            metrics_570,

        "per_image":
            per_image_rows,

        "per_patient_native256":
            patient_rows_256,

        "per_patient_original570":
            patient_rows_570,
    }


# =====================================================================
# BATCH-SIZE PROBE
# =====================================================================

def probe_batch_size(
    device,
):

    print(
        "[3/8] Probing GPU memory..."
    )

    candidates = []

    for value in [
        REQUESTED_BATCH_SIZE,
        2,
        1,
    ]:

        if value not in candidates:

            candidates.append(
                value
            )

    if device.type != "cuda":

        print(
            "    CUDA unavailable; using batch size 1."
        )

        return 1

    for batch_size in candidates:

        try:

            torch.cuda.empty_cache()

            model = NestedUNet().to(
                device
            )

            criterion = CEDiceLoss(
                NUM_CLASSES,
                CE_WEIGHT,
                DICE_WEIGHT,
                DICE_SMOOTH,
            ).to(
                device
            )

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=INITIAL_LR,
                weight_decay=
                    WEIGHT_DECAY,
            )

            scaler = torch.amp.GradScaler(
                "cuda",
                enabled=USE_AMP,
            )

            x = torch.randn(
                batch_size,
                INPUT_CHANNELS,
                INPUT_HEIGHT,
                INPUT_WIDTH,
                device=device,
            )

            y = torch.randint(
                0,
                NUM_CLASSES,
                (
                    batch_size,
                    INPUT_HEIGHT,
                    INPUT_WIDTH,
                ),
                device=device,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.amp.autocast(
                device_type="cuda",
                enabled=USE_AMP,
            ):

                logits = model(
                    x
                )

                (
                    loss,
                    _,
                    _,
                ) = criterion(
                    logits,
                    y,
                )

            scaler.scale(
                loss
            ).backward()

            scaler.step(
                optimizer
            )

            scaler.update()

            del (
                x,
                y,
                logits,
                loss,
                model,
                criterion,
                optimizer,
                scaler,
            )

            torch.cuda.empty_cache()

            print(
                f"    Batch size {batch_size}: PASS"
            )

            return batch_size

        except RuntimeError as exc:

            message = str(
                exc
            ).lower()

            if (
                "out of memory"
                not in message
            ):

                raise

            print(
                f"    Batch size {batch_size}: OOM"
            )

            torch.cuda.empty_cache()

    raise RuntimeError(
        "Even batch size 1 does not fit GPU memory."
    )


# =====================================================================
# CHECKPOINTING
# =====================================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    scaler,
    epoch,
    best_metric,
    best_epoch,
    epochs_without_improvement,
    history,
    batch_size,
    accumulation_steps,
    training_protocol_sha,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "scaler_state_dict":
                scaler.state_dict(),

            "epoch":
                int(
                    epoch
                ),

            "best_metric":
                float(
                    best_metric
                ),

            "best_epoch":
                int(
                    best_epoch
                ),

            "epochs_without_improvement":
                int(
                    epochs_without_improvement
                ),

            "history":
                history,

            "fold":
                FOLD,

            "model_name":
                MODEL_NAME,

            "batch_size":
                int(
                    batch_size
                ),

            "accumulation_steps":
                int(
                    accumulation_steps
                ),

            "canonical_protocol_sha256":
                EXPECTED_CANONICAL_PROTOCOL_SHA256,

            "training_protocol_sha256":
                training_protocol_sha,

            "seed":
                SEED,
        },
        path,
    )


def load_checkpoint(
    path,
    model,
    optimizer,
    scaler,
    device,
    training_protocol_sha,
):

    checkpoint = torch.load(
        path,
        map_location=device,
    )

    if (
        safe_text(
            checkpoint.get(
                "canonical_protocol_sha256"
            )
        )
        != EXPECTED_CANONICAL_PROTOCOL_SHA256
    ):

        raise RuntimeError(
            "Checkpoint canonical protocol fingerprint mismatch."
        )

    if (
        safe_text(
            checkpoint.get(
                "training_protocol_sha256"
            )
        )
        != training_protocol_sha
    ):

        raise RuntimeError(
            "Checkpoint training protocol fingerprint mismatch."
        )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    optimizer.load_state_dict(
        checkpoint[
            "optimizer_state_dict"
        ]
    )

    scaler.load_state_dict(
        checkpoint[
            "scaler_state_dict"
        ]
    )

    return checkpoint


# =====================================================================
# TRAIN FOLD 1
# =====================================================================

def train_fold1(
    role_rows,
    device,
    batch_size,
    accumulation_steps,
    training_protocol_sha,
):

    print(
        "\n[4/8] Training U-Net++ — Track B Fold 3..."
    )

    complete_path = (
        OUTPUT_DIR
        / "FOLD_COMPLETE.json"
    )

    if complete_path.exists():

        raise RuntimeError(
            "\nFold 3 is already marked COMPLETE.\n"
            "Refusing to overwrite a finished experiment.\n"
            f"{complete_path}"
        )

    checkpoint_dir = (
        OUTPUT_DIR
        / "checkpoints"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_path = (
        checkpoint_dir
        / "best_model.pth"
    )

    last_path = (
        checkpoint_dir
        / "last_checkpoint.pth"
    )

    history_path = (
        OUTPUT_DIR
        / "training_history.csv"
    )

    train_loader = make_loader(
        role_rows[
            "train"
        ],
        batch_size=
            batch_size,
        training=True,
        evaluation=False,
    )

    val_loader = make_loader(
        role_rows[
            "val"
        ],
        batch_size=
            batch_size,
        training=False,
        evaluation=False,
    )

    set_seed(
        SEED
    )

    model = NestedUNet().to(
        device
    )

    criterion = CEDiceLoss(
        NUM_CLASSES,
        CE_WEIGHT,
        DICE_WEIGHT,
        DICE_SMOOTH,
    ).to(
        device
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=INITIAL_LR,
        weight_decay=
            WEIGHT_DECAY,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            USE_AMP
            and device.type
            == "cuda"
        ),
    )

    start_epoch = 1
    best_metric = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    if (
        AUTO_RESUME
        and last_path.exists()
    ):

        checkpoint = load_checkpoint(
            last_path,
            model,
            optimizer,
            scaler,
            device,
            training_protocol_sha,
        )

        start_epoch = (
            int(
                checkpoint[
                    "epoch"
                ]
            )
            + 1
        )

        best_metric = float(
            checkpoint[
                "best_metric"
            ]
        )

        best_epoch = int(
            checkpoint[
                "best_epoch"
            ]
        )

        epochs_without_improvement = int(
            checkpoint[
                "epochs_without_improvement"
            ]
        )

        history = list(
            checkpoint.get(
                "history",
                [],
            )
        )

        print(
            f"    Resuming from epoch {start_epoch}."
        )

    print(
        f"    Train images: {len(role_rows['train'])}"
    )

    print(
        f"    Val images:   {len(role_rows['val'])}"
    )

    print(
        f"    Physical batch size:      {batch_size}"
    )

    print(
        f"    Gradient accumulation:    {accumulation_steps}"
    )

    print(
        f"    Effective batch size:     "
        f"{batch_size * accumulation_steps}"
    )

    print(
        f"    Checkpoint metric: native-256 mean foreground Dice"
    )

    for epoch in range(
        start_epoch,
        MAX_EPOCHS
        + 1,
    ):

        epoch_start = time.time()

        if device.type == "cuda":

            torch.cuda.reset_peak_memory_stats()

        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            criterion,
            device,
            accumulation_steps,
        )

        val_metrics = evaluate_native256(
            model,
            val_loader,
            device,
            criterion=
                criterion,
        )

        val_dice = float(
            val_metrics[
                "mean_foreground_dice"
            ]
        )

        val_iou = float(
            val_metrics[
                "mean_foreground_iou"
            ]
        )

        improved = (
            val_dice
            > best_metric
            + 1e-8
        )

        if improved:

            best_metric = val_dice
            best_epoch = epoch
            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        epoch_seconds = (
            time.time()
            - epoch_start
        )

        peak_memory_gb = None

        if device.type == "cuda":

            peak_memory_gb = (
                torch.cuda.max_memory_allocated()
                / (
                    1024
                    ** 3
                )
            )

        row = {
            "epoch":
                epoch,

            "train_loss":
                train_stats[
                    "loss"
                ],

            "train_ce_loss":
                train_stats[
                    "ce_loss"
                ],

            "train_dice_loss":
                train_stats[
                    "dice_loss"
                ],

            "val_loss":
                val_metrics[
                    "loss"
                ],

            "val_mean_foreground_dice":
                val_dice,

            "val_mean_foreground_iou":
                val_iou,

            "best_val_mean_foreground_dice":
                best_metric,

            "best_epoch":
                best_epoch,

            "epochs_without_improvement":
                epochs_without_improvement,

            "epoch_minutes":
                epoch_seconds
                / 60.0,

            "gpu_peak_memory_gb":
                peak_memory_gb,
        }

        for class_id in (
            FOREGROUND_CLASS_IDS
        ):

            name = CLASS_NAMES[
                class_id
            ]

            row[
                f"val_{name}_dice"
            ] = (
                val_metrics[
                    "classes"
                ][
                    name
                ][
                    "dice"
                ]
            )

            row[
                f"val_{name}_iou"
            ] = (
                val_metrics[
                    "classes"
                ][
                    name
                ][
                    "iou"
                ]
            )

        history.append(
            row
        )

        write_csv(
            history_path,
            history,
        )

        if improved:

            save_checkpoint(
                best_path,
                model,
                optimizer,
                scaler,
                epoch,
                best_metric,
                best_epoch,
                epochs_without_improvement,
                history,
                batch_size,
                accumulation_steps,
                training_protocol_sha,
            )

        if SAVE_LAST_CHECKPOINT:

            save_checkpoint(
                last_path,
                model,
                optimizer,
                scaler,
                epoch,
                best_metric,
                best_epoch,
                epochs_without_improvement,
                history,
                batch_size,
                accumulation_steps,
                training_protocol_sha,
            )

        print(
            f"    Epoch {epoch:02d}/{MAX_EPOCHS}: "
            f"train_loss={train_stats['loss']:.4f} | "
            f"val_Dice={val_dice * 100:.2f}% | "
            f"val_IoU={val_iou * 100:.2f}% | "
            f"best_epoch={best_epoch} | "
            f"time={epoch_seconds / 60:.1f} min"
        )

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print(
                "    Early stopping triggered."
            )

            break

    if not best_path.exists():

        raise RuntimeError(
            "No best checkpoint was saved."
        )

    summary = {
        "fold":
            FOLD,

        "best_epoch":
            best_epoch,

        "best_validation_mean_foreground_dice":
            best_metric,

        "epochs_completed":
            len(
                history
            ),

        "physical_batch_size":
            batch_size,

        "gradient_accumulation":
            accumulation_steps,

        "effective_batch_size":
            batch_size
            * accumulation_steps,

        "training_protocol_sha256":
            training_protocol_sha,
    }

    save_json(
        OUTPUT_DIR
        / "training_summary.json",
        summary,
    )

    print(
        f"\n    Selected epoch: {best_epoch}"
    )

    print(
        f"    Best validation mean Dice: "
        f"{best_metric * 100:.2f}%"
    )

    return (
        summary,
        best_path,
    )


# =====================================================================
# FINAL VALIDATION + TEST
# =====================================================================

def evaluate_best_checkpoint(
    role_rows,
    best_path,
    device,
    batch_size,
    training_protocol_sha,
):

    print(
        "\n[5/8] Loading selected checkpoint..."
    )

    checkpoint = torch.load(
        best_path,
        map_location=device,
    )

    if (
        safe_text(
            checkpoint.get(
                "training_protocol_sha256"
            )
        )
        != training_protocol_sha
    ):

        raise RuntimeError(
            "Best checkpoint protocol fingerprint mismatch."
        )

    model = NestedUNet().to(
        device
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    print(
        f"    Best epoch: {checkpoint['epoch']}"
    )

    # -------------------------------------------------------------
    # Final validation check.
    # -------------------------------------------------------------

    print(
        "[6/8] Final validation check at native 256..."
    )

    val_loader = make_loader(
        role_rows[
            "val"
        ],
        batch_size=
            batch_size,
        training=False,
        evaluation=False,
    )

    val_metrics = evaluate_native256(
        model,
        val_loader,
        device,
        criterion=None,
    )

    save_json(
        OUTPUT_DIR
        / "validation"
        / "best_validation_metrics_native256.json",
        val_metrics,
    )

    print(
        f"    Validation mean Dice: "
        f"{val_metrics['mean_foreground_dice'] * 100:.2f}%"
    )

    # -------------------------------------------------------------
    # ONE-TIME TEST evaluation.
    # -------------------------------------------------------------

    print(
        "[7/8] ONE-TIME evaluation on Fold 3 test patients..."
    )

    test_loader = make_loader(
        role_rows[
            "test"
        ],
        batch_size=
            batch_size,
        training=False,
        evaluation=True,
    )

    test_results = (
        evaluate_test_both_scales(
            model,
            test_loader,
            device,
        )
    )

    test_dir = (
        OUTPUT_DIR
        / "test"
    )

    save_json(
        test_dir
        / "global_metrics_native256.json",
        test_results[
            "native256"
        ],
    )

    save_json(
        test_dir
        / "global_metrics_original570x380.json",
        test_results[
            "original570x380"
        ],
    )

    write_csv(
        test_dir
        / "per_image_metrics.csv",
        test_results[
            "per_image"
        ],
    )

    write_csv(
        test_dir
        / "per_patient_metrics_native256.csv",
        test_results[
            "per_patient_native256"
        ],
    )

    write_csv(
        test_dir
        / "per_patient_metrics_original570x380.csv",
        test_results[
            "per_patient_original570"
        ],
    )

    return (
        val_metrics,
        test_results,
    )


# =====================================================================
# FINAL REPORT
# =====================================================================

def print_metric_table(
    title,
    metrics,
):

    print("\n")
    print(
        title
    )

    print(
        "-" * 72
    )

    print(
        f"{'Class':<12}"
        f"{'Dice':>14}"
        f"{'IoU':>14}"
    )

    print(
        "-" * 72
    )

    for class_id in (
        FOREGROUND_CLASS_IDS
    ):

        name = CLASS_NAMES[
            class_id
        ]

        class_metrics = (
            metrics[
                "classes"
            ][
                name
            ]
        )

        print(
            f"{name:<12}"
            f"{class_metrics['dice'] * 100:>13.2f}%"
            f"{class_metrics['iou'] * 100:>13.2f}%"
        )

    print(
        "-" * 72
    )

    print(
        f"{'Mean':<12}"
        f"{metrics['mean_foreground_dice'] * 100:>13.2f}%"
        f"{metrics['mean_foreground_iou'] * 100:>13.2f}%"
    )

    print(
        "-" * 72
    )


def finalize_experiment(
    canonical,
    training_protocol,
    train_summary,
    val_metrics,
    test_results,
):

    print(
        "[8/8] Saving frozen Fold 1 result summary..."
    )

    final_summary = {
        "status":
            "COMPLETE",

        "track":
            "B_patient_disjoint",

        "model":
            MODEL_NAME,

        "fold":
            FOLD,

        "canonical_protocol_sha256":
            EXPECTED_CANONICAL_PROTOCOL_SHA256,

        "training_protocol_sha256":
            training_protocol[
                "training_protocol_sha256"
            ],

        "selected_epoch":
            train_summary[
                "best_epoch"
            ],

        "validation_native256_mean_dice":
            val_metrics[
                "mean_foreground_dice"
            ],

        "test_native256_mean_dice":
            test_results[
                "native256"
            ][
                "mean_foreground_dice"
            ],

        "test_native256_mean_iou":
            test_results[
                "native256"
            ][
                "mean_foreground_iou"
            ],

        "test_original570x380_mean_dice":
            test_results[
                "original570x380"
            ][
                "mean_foreground_dice"
            ],

        "test_original570x380_mean_iou":
            test_results[
                "original570x380"
            ][
                "mean_foreground_iou"
            ],

        "test_patients":
            EXPECTED_TEST_PATIENTS,

        "test_images":
            EXPECTED_TEST_IMAGES,

        "test_was_used_for_checkpoint_selection":
            False,
    }

    save_json(
        OUTPUT_DIR
        / "FOLD_COMPLETE.json",
        final_summary,
    )

    print("\n")
    print("=" * 100)

    print(
        "RVO-ME STEP 05B — "
        "TRACK-B U-NET++ FOLD 3 COMPLETE"
    )

    print("=" * 100)

    print(
        f"\nSelected epoch: "
        f"{train_summary['best_epoch']}"
    )

    print(
        f"Best validation Dice (256): "
        f"{val_metrics['mean_foreground_dice'] * 100:.2f}%"
    )

    print_metric_table(
        "Fold 3 TEST — native 256x256",
        test_results[
            "native256"
        ],
    )

    print_metric_table(
        "Fold 3 TEST — original 570x380 released-mask scale",
        test_results[
            "original570x380"
        ],
    )

    print(
        "\nPatient leakage: 0"
    )

    print(
        "Test used for checkpoint selection: NO"
    )

    print(
        "\nResults saved to:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        "\nFold 3 complete."
    )

    print(
        "Keep this output unchanged and continue with the next "
        "frozen fold using the identical protocol."
    )

    print("\n")
    print("=" * 100)


# =====================================================================
# MAIN
# =====================================================================

def main():

    print("=" * 100)

    print(
        "RVO-ME STEP 05B: "
        "TRACK-B PATIENT-DISJOINT U-NET++ — FOLD 3"
    )

    print("=" * 100)

    print(
        f"\nOutput:\n"
        f"{OUTPUT_DIR}"
    )

    (
        canonical,
        training_protocol,
    ) = verify_and_freeze_protocols()

    role_rows = load_fold_protocol()

    initialize_file_lookups()

    if torch.cuda.is_available():

        device = torch.device(
            "cuda"
        )

        print(
            f"\nGPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    else:

        device = torch.device(
            "cpu"
        )

        print(
            "\nWARNING: CUDA unavailable; CPU training would be very slow."
        )

    batch_size = probe_batch_size(
        device
    )

    accumulation_steps = max(
        1,
        int(
            math.ceil(
                TARGET_EFFECTIVE_BATCH
                / batch_size
            )
        ),
    )

    model_probe = NestedUNet()

    trainable_parameters = sum(
        parameter.numel()
        for parameter
        in model_probe.parameters()
        if parameter.requires_grad
    )

    del model_probe

    print(
        f"\n    U-Net++ trainable parameters: "
        f"{trainable_parameters / 1e6:.3f} M"
    )

    print(
        f"    Physical batch size:          "
        f"{batch_size}"
    )

    print(
        f"    Effective batch size:         "
        f"{batch_size * accumulation_steps}"
    )

    train_summary, best_path = (
        train_fold1(
            role_rows,
            device,
            batch_size,
            accumulation_steps,
            training_protocol[
                "training_protocol_sha256"
            ],
        )
    )

    (
        val_metrics,
        test_results,
    ) = evaluate_best_checkpoint(
        role_rows,
        best_path,
        device,
        batch_size,
        training_protocol[
            "training_protocol_sha256"
        ],
    )

    finalize_experiment(
        canonical,
        training_protocol,
        train_summary,
        val_metrics,
        test_results,
    )


if __name__ == "__main__":
    main()
