from __future__ import annotations

# =============================================================================
# RVO-ME STEP 12
# PRESPECIFIED ATTRIBUTION / GO-NO-GO EVALUATION
#
# RUN 1
#   - verify frozen B2/A0 conventional MIRAGE and A2 no-interaction training
#   - calibrate A2 curve/HF thresholds on VALIDATION ONLY
#   - freeze a common comparison protocol
#   - EXIT WITHOUT RESOLVING ANY NEW OUTER TEST ROLE
#
# RUN 2 (same unchanged file)
#   - verify protocol + script unchanged
#   - resolve the five outer TEST folds
#   - evaluate B2/A0 conventional MIRAGE once in the common region/curve space
#   - evaluate A2 no-interaction once with validation-frozen thresholds
#   - reuse (DO NOT rerun) the already-frozen Full GeoRCP Step-09 held-out result
#   - aggregate pooled metrics and patient-level paired bootstrap comparisons
#
# Scientific purpose
# ------------------
# Decide whether additional large ablations are justified:
#
#   B2/A0 conventional MIRAGE
#       -> A2 geometry-specific R/C/P, no task interaction
#       -> Full GeoRCP geometry + interaction
#
# Regions (SRF/IRF):
#   all models are compared against the SAME geometry_v3 region targets.
#
# Curves (ELM/EZ):
#   all models are compared against the SAME geometry_v3 curve-presence/y
#   targets. Conventional semantic predictions are deterministically converted:
#       - class 3 = ELM, class 4 = EZ
#       - predicted column presence = any semantic pixels in that native column
#       - predicted y = arithmetic mean of predicted class pixels in the column
#       - no morphology, no gap fill, no connectivity, no tuned parameters
#
# HF:
#   B2/A0 has no HF detector, so HF attribution is A2 vs Full GeoRCP only.
#
# IMPORTANT
# ---------
# This script NEVER re-evaluates Full GeoRCP. It reads the already completed
# Step-09 held-out result only AFTER B2/A0 and A2 held-out evaluation has begun
# under this frozen protocol.
# =============================================================================

import csv
import gc
import gzip
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch


# =============================================================================
# OFFLINE / PATHS
# =============================================================================

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

SCRIPT_REVISION = "STEP12_ATTRIBUTION_EVAL_V1_20260905"

ROOT = Path(r"D:\RVO_ME")
DATASET = ROOT / "RVO-Lesion"
PROJECT = ROOT / "project"
SCRIPTS = PROJECT / "scripts"

STEP09_SCRIPT = SCRIPTS / "09_georcp_calibrate_and_evaluate.py"
STEP10_SCRIPT = SCRIPTS / "10_mirage_semantic_baseline_train_resume_safe_v3.py"
STEP11_SCRIPT = SCRIPTS / "11_georcp_A2_nointeraction_train_resume.py"

STEP07_SCRIPT = SCRIPTS / "07_georcp_preflight_v2_ampfix_py38compat.py"

MIRAGE_DIR = (
    PROJECT
    / "foundation_models"
    / "mirage_base_preflight_v1"
)
MIRAGE_WRAPPER = MIRAGE_DIR / "official_mirage_hf.py"
MIRAGE_HF_CACHE = MIRAGE_DIR / "hf_cache"

B2_ROOT = (
    PROJECT
    / "baselines"
    / "segmentation"
    / "mirage_semantic_384x576_v1"
)
B2_MANIFEST = B2_ROOT / "FIVEFOLD_TRAINING_MANIFEST.json"
B2_PROTOCOL = B2_ROOT / "B2_A0_PROTOCOL_FROZEN.json"

A2_ROOT = (
    PROJECT
    / "training"
    / "georcp_v1"
    / "ablation_A2_no_interaction_v1"
)
A2_MANIFEST = A2_ROOT / "FIVEFOLD_TRAINING_MANIFEST.json"
A2_PROTOCOL = A2_ROOT / "A2_NO_INTERACTION_PROTOCOL_FROZEN.json"

FULL_EVAL_ROOT = (
    PROJECT
    / "evaluation"
    / "georcp_v1"
    / "trackB_fivefold_v1"
)
FULL_EVAL_PROTOCOL = (
    FULL_EVAL_ROOT
    / "STEP09_EVALUATION_PROTOCOL_FROZEN.json"
)
FULL_HELDOUT_COMPLETE = (
    FULL_EVAL_ROOT
    / "STEP09_HELDOUT_RESULTS_COMPLETE.json"
)
FULL_PER_IMAGE_CSV = (
    FULL_EVAL_ROOT
    / "heldout_per_image_metrics.csv"
)

EVAL_ROOT = (
    PROJECT
    / "evaluation"
    / "attribution_v1"
)
CALIBRATION_ROOT = EVAL_ROOT / "calibration_A2"
PROTOCOL_FILE = EVAL_ROOT / "STEP12_ATTRIBUTION_PROTOCOL_FROZEN.json"
HELDOUT_STARTED_FILE = EVAL_ROOT / "HELDOUT_EVALUATION_STARTED.json"
HELDOUT_COMPLETE_FILE = EVAL_ROOT / "STEP12_ATTRIBUTION_RESULTS_COMPLETE.json"

B2_EVAL_ROOT = EVAL_ROOT / "B2_semantic_common_space"
A2_EVAL_ROOT = EVAL_ROOT / "A2_no_interaction"

B2_ALL_CSV = EVAL_ROOT / "B2_heldout_per_image_metrics.csv"
A2_ALL_CSV = EVAL_ROOT / "A2_heldout_per_image_metrics.csv"
SUMMARY_CSV = EVAL_ROOT / "attribution_summary.csv"
PAIRED_BOOTSTRAP_CSV = EVAL_ROOT / "paired_patient_bootstrap.csv"

PROJECT_LOG = PROJECT / "PROJECT_LOG.md"


# =============================================================================
# FROZEN IDENTIFIERS
# =============================================================================

EXPECTED_STEP09_SCRIPT_SHA = (
    "bcfb0f2dad703238c426c7d90e8e6a34738a0f2a40eaea8bb40d6776156bc83f"
)

EXPECTED_FULL_EVAL_PROTOCOL_SHA = (
    "dc3b45f10b24226425dc3ef94e59b69421192c04d37cb17831bfe23d7567f6ba"
)

EXPECTED_FULL_TRAIN_PROTOCOL_SHA = (
    "8640e7b4039cfc5f6fcd45269aaf9003a337a090f3af5a8aa787c9dcd9a0491e"
)

EXPECTED_B2_PROTOCOL_SHA = (
    "a9aeabf91e4eef9f82dfc1f9999301d737303f12b8713c3d9a24cf1bfd366c95"
)

EXPECTED_B2_SCIENTIFIC_SOURCE_SHA = (
    "c28137c519894d48c5f1ee1eb4b2fcd9b58f45e34f544698deb78a629fc2caae"
)

EXPECTED_A2_PROTOCOL_SHA = (
    "48790283a42311f666b09ad9e2d2f2d6eb03e99a58170ce8891e3e2cf43ca944"
)

EXPECTED_A2_SCIENTIFIC_SOURCE_SHA = (
    "0227d2b558e7b816c37fa0cac12fa5142a7d91127a1612287ead8ddbfb5b51e5"
)

EXPECTED_G3_SHA = (
    "521859acb4c5389eb4244298f755420f1339b7e4c4bedcfac7c64ddcfe98ce59"
)

EXPECTED_IMPLEMENTATION_SPEC_SHA = (
    "b5e59901eb67190d5924698ca53165d30d1bccf45028a7454d9ca98147149422"
)

EXPECTED_SOURCE_BUNDLE_SHA = (
    "04e7da4fe311b756ad375a6c241530d0572a3cf1b9c71a70af2e7e7e24524412"
)

EXPECTED_MIRAGE_WRAPPER_SHA = (
    "07fa63e2766b67dbc0c07f0d9d29db1562f707425ef2fbf90a688f62a94e6ede"
)

EXPECTED_TRACKB_SPLIT_SHA = (
    "7e7c2ddf51054bf998b2d750b50c74983b9bc5b751b762213bc96343ffe01346"
)

EXPECTED_IMAGES = 3012
EXPECTED_PATIENTS = 130
EXPECTED_TEST_PATIENTS_PER_FOLD = 26

EXPECTED_VAL_IMAGES = {
    1: 303,
    2: 270,
    3: 308,
    4: 342,
    5: 362,
}

EXPECTED_TEST_IMAGES = {
    1: 599,
    2: 606,
    3: 598,
    4: 631,
    5: 578,
}

BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260822

REGION_NAMES = {
    1: "SRF",
    2: "IRF",
}

CURVE_NAMES = {
    0: "ELM",
    1: "EZ",
}

B2_CURVE_CLASS = {
    0: 3,  # ELM
    1: 4,  # EZ
}

NH, NW = 380, 570
PT, PB, PL, PR = 2, 2, 3, 3
H, W = 384, 576
EPS = 1e-12


# =============================================================================
# HELPERS
# =============================================================================

def load_json(
    path: Path,
) -> Any:
    if not path.exists():
        raise FileNotFoundError(
            path
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(
            f
        )


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


def save_json_gz(
    path: Path,
    obj: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with gzip.open(
        path,
        "wt",
        encoding="utf-8",
    ) as f:
        json.dump(
            obj,
            f,
            ensure_ascii=False,
            separators=(",", ":"),
        )


def load_json_gz(
    path: Path,
) -> Any:
    with gzip.open(
        path,
        "rt",
        encoding="utf-8",
    ) as f:
        return json.load(
            f
        )


def sha_file(
    path: Path,
) -> str:
    h = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:
        for chunk in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(
                chunk
            )

    return h.hexdigest()


def stable_sha(
    obj: Any,
) -> str:
    raw = json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        raw
    ).hexdigest()


def current_script_sha() -> str:
    return sha_file(
        Path(
            __file__
        ).resolve()
    )


def hash_stems(
    stems: Sequence[str],
) -> str:
    return hashlib.sha256(
        "\n".join(
            sorted(
                stems
            )
        ).encode(
            "utf-8"
        )
    ).hexdigest()


def module_from_file(
    name: str,
    path: Path,
):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            "Could not import {}".format(
                path
            )
        )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[
        name
    ] = module

    spec.loader.exec_module(
        module
    )

    return module


def reset_cuda() -> None:
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def safe_div(
    a: float,
    b: float,
) -> float:
    if abs(
        float(
            b
        )
    ) <= EPS:
        return float(
            "nan"
        )

    return float(
        a
    ) / float(
        b
    )


def finite(
    x: Any,
) -> bool:
    try:
        return bool(
            np.isfinite(
                float(
                    x
                )
            )
        )
    except Exception:
        return False


def fv(
    row: Dict[str, Any],
    key: str,
    default: float = 0.0,
) -> float:
    value = row.get(
        key
    )

    if value in (
        None,
        "",
        "nan",
        "NaN",
    ):
        return float(
            "nan"
        )

    try:
        return float(
            value
        )
    except Exception:
        return float(
            default
        )


def f1_counts(
    tp: float,
    fp: float,
    fn: float,
) -> Dict[str, float]:
    p = safe_div(
        tp,
        tp + fp,
    )

    r = safe_div(
        tp,
        tp + fn,
    )

    if (
        not np.isfinite(
            p
        )
        or not np.isfinite(
            r
        )
        or p + r <= EPS
    ):
        f = 0.0
    else:
        f = (
            2.0
            * p
            * r
            / (
                p + r
            )
        )

    return {
        "precision": float(
            p
        ),
        "recall": float(
            r
        ),
        "f1": float(
            f
        ),
    }


def read_rows(
    path: Path,
) -> List[Dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        return list(
            csv.DictReader(
                f
            )
        )


def write_rows(
    path: Path,
    rows: Sequence[Dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        raise RuntimeError(
            "No rows to write: {}".format(
                path
            )
        )

    fields = sorted(
        {
            key
            for row in rows
            for key in row
        }
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                row
            )


def append_log_once(
    marker: str,
    text: str,
) -> None:
    if not PROJECT_LOG.exists():
        return

    old = PROJECT_LOG.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    if marker in old:
        return

    with PROJECT_LOG.open(
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            "\n\n{}\n{}\n".format(
                marker,
                text,
            )
        )


# =============================================================================
# FROZEN MODULE / MANIFEST VERIFICATION
# =============================================================================

def load_modules():
    required = (
        STEP09_SCRIPT,
        STEP10_SCRIPT,
        STEP11_SCRIPT,
        STEP07_SCRIPT,
        MIRAGE_WRAPPER,
        MIRAGE_HF_CACHE,
        B2_MANIFEST,
        B2_PROTOCOL,
        A2_MANIFEST,
        A2_PROTOCOL,
        FULL_EVAL_PROTOCOL,
    )

    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    if (
        sha_file(
            STEP09_SCRIPT
        )
        != EXPECTED_STEP09_SCRIPT_SHA
    ):
        raise RuntimeError(
            "Frozen Step-09 script SHA mismatch."
        )

    step09 = module_from_file(
        "rvo_step09_attribution",
        STEP09_SCRIPT,
    )

    step10 = module_from_file(
        "rvo_step10_attribution",
        STEP10_SCRIPT,
    )

    step11 = module_from_file(
        "rvo_step11_attribution",
        STEP11_SCRIPT,
    )

    step07, source_sha = step09.load_exact_step07()

    # Engineering-only desktop binding. No source bytes are modified.
    step07.ROOT = ROOT
    step07.DATASET = DATASET
    step07.PROJECT = PROJECT
    step07.IMAGE_DIR = (
        DATASET
        / "Image_Seg"
        / "images"
    )
    step07.MIRAGE_DIR = MIRAGE_DIR
    step07.MIRAGE_WRAPPER = MIRAGE_WRAPPER
    step07.HF_CACHE = MIRAGE_HF_CACHE

    if source_sha != EXPECTED_SOURCE_BUNDLE_SHA:
        raise RuntimeError(
            "Step-07 source bundle mismatch."
        )

    if (
        step07.implementation_spec().get(
            "implementation_spec_sha256"
        )
        != EXPECTED_IMPLEMENTATION_SPEC_SHA
    ):
        raise RuntimeError(
            "Step-07 implementation-spec mismatch."
        )

    if (
        sha_file(
            MIRAGE_WRAPPER
        )
        != EXPECTED_MIRAGE_WRAPPER_SHA
    ):
        raise RuntimeError(
            "MIRAGE wrapper SHA mismatch."
        )

    # Step-10 scientific implementation check, robust to resume-only file edits.
    if (
        getattr(
            step10,
            "EXPECTED_STEP10_SCIENTIFIC_SOURCE_SHA",
            None,
        )
        != EXPECTED_B2_SCIENTIFIC_SOURCE_SHA
    ):
        raise RuntimeError(
            "Step-10 expected scientific-source SHA mismatch."
        )

    step10_scientific_sha = (
        step10.named_source_bundle_sha(
            Path(
                step10.__file__
            ),
            step10.STEP10_SCIENTIFIC_SOURCE_NAMES,
        )
    )

    if (
        step10_scientific_sha
        != EXPECTED_B2_SCIENTIFIC_SOURCE_SHA
    ):
        raise RuntimeError(
            "Step-10 scientific source has changed."
        )

    a2_scientific_sha = (
        step11.ablation_scientific_source_sha()
    )

    if (
        a2_scientific_sha
        != EXPECTED_A2_SCIENTIFIC_SOURCE_SHA
    ):
        raise RuntimeError(
            "Step-11/A2 scientific source has changed."
        )

    return (
        step07,
        step09,
        step10,
        step11,
    )


def verify_environment() -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA required."
        )

    gpu = torch.cuda.get_device_name(
        0
    )

    if "RTX 4080" not in gpu:
        raise RuntimeError(
            "Validated RTX 4080 environment required; found {}.".format(
                gpu
            )
        )

    if not sys.version.startswith(
        "3.8"
    ):
        raise RuntimeError(
            "Validated Python 3.8.x required."
        )

    if (
        torch.__version__
        != "2.4.1+cu121"
    ):
        raise RuntimeError(
            "Expected PyTorch 2.4.1+cu121; found {}.".format(
                torch.__version__
            )
        )

    if str(
        torch.version.cuda
    ) != "12.1":
        raise RuntimeError(
            "Expected CUDA 12.1 build."
        )

    return {
        "gpu": gpu,
        "python": sys.version,
        "pytorch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "bf16_supported": bool(
            torch.cuda.is_bf16_supported()
        ),
    }


def verify_manifests():
    b2 = load_json(
        B2_MANIFEST
    )

    a2 = load_json(
        A2_MANIFEST
    )

    if (
        b2.get(
            "status"
        )
        != "ALL_FIVE_FOLDS_TRAINING_COMPLETE"
        or b2.get(
            "global_protocol_sha256"
        )
        != EXPECTED_B2_PROTOCOL_SHA
        or b2.get(
            "trackb_split_master_sha256"
        )
        != EXPECTED_TRACKB_SPLIT_SHA
        or b2.get(
            "outer_test_resolved"
        )
        or b2.get(
            "outer_test_evaluated"
        )
    ):
        raise RuntimeError(
            "B2/A0 training manifest integrity failure."
        )

    if (
        a2.get(
            "status"
        )
        != "ALL_FIVE_FOLDS_TRAINING_COMPLETE"
        or a2.get(
            "global_protocol_sha256"
        )
        != EXPECTED_A2_PROTOCOL_SHA
        or a2.get(
            "ablation_scientific_source_sha256"
        )
        != EXPECTED_A2_SCIENTIFIC_SOURCE_SHA
        or a2.get(
            "trackb_split_master_sha256"
        )
        != EXPECTED_TRACKB_SPLIT_SHA
        or a2.get(
            "outer_test_resolved"
        )
        or a2.get(
            "outer_test_evaluated"
        )
    ):
        raise RuntimeError(
            "A2 training manifest integrity failure."
        )

    for fold in range(
        1,
        6,
    ):
        for name, manifest in (
            (
                "B2",
                b2,
            ),
            (
                "A2",
                a2,
            ),
        ):
            entry = manifest[
                "folds"
            ][
                str(
                    fold
                )
            ]

            if (
                entry.get(
                    "status"
                )
                != "COMPLETE"
            ):
                raise RuntimeError(
                    "{} Fold {} incomplete.".format(
                        name,
                        fold,
                    )
                )

            checkpoint = Path(
                entry[
                    "best_checkpoint"
                ]
            )

            report = Path(
                entry[
                    "completion_report"
                ]
            )

            if (
                not checkpoint.exists()
                or not report.exists()
            ):
                raise RuntimeError(
                    "{} Fold {} checkpoint/report missing.".format(
                        name,
                        fold,
                    )
                )

    # Protocol files themselves must match the manifest-level frozen SHAs.
    if (
        load_json(
            B2_PROTOCOL
        ).get(
            "global_protocol_sha256"
        )
        != EXPECTED_B2_PROTOCOL_SHA
    ):
        raise RuntimeError(
            "B2 frozen protocol SHA mismatch."
        )

    if (
        load_json(
            A2_PROTOCOL
        ).get(
            "protocol_sha256"
        )
        != EXPECTED_A2_PROTOCOL_SHA
    ):
        raise RuntimeError(
            "A2 frozen protocol SHA mismatch."
        )

    full_protocol = load_json(
        FULL_EVAL_PROTOCOL
    )

    if (
        full_protocol.get(
            "status"
        )
        != "FROZEN_BEFORE_HELDOUT_TEST"
        or full_protocol.get(
            "evaluation_protocol_sha256"
        )
        != EXPECTED_FULL_EVAL_PROTOCOL_SHA
        or full_protocol.get(
            "heldout_test_used_for_calibration"
        )
    ):
        raise RuntimeError(
            "Full GeoRCP frozen Step-09 evaluation protocol mismatch."
        )

    # Full held-out files are required for later comparison, but are deliberately
    # not opened/read during Run 1.
    if (
        not FULL_HELDOUT_COMPLETE.exists()
        or not FULL_PER_IMAGE_CSV.exists()
    ):
        raise FileNotFoundError(
            "Existing Full GeoRCP Step-09 held-out result is missing."
        )

    return (
        b2,
        a2,
        full_protocol,
    )


# =============================================================================
# MODEL BUILD / CHECKPOINT INTEGRITY
# =============================================================================

def torch_load_cpu(
    path: Path,
):
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location="cpu",
        )


def build_b2_model(
    fold: int,
    checkpoint: Path,
    step07,
    step10,
    device: torch.device,
):
    reset_cuda()

    model = step10.build_model(
        step07,
        device,
    )

    state = torch_load_cpu(
        checkpoint
    )

    if (
        state.get(
            "status"
        )
        != "B2_A0_MIRAGE_SEMANTIC"
        or state.get(
            "global_protocol_sha256"
        )
        != EXPECTED_B2_PROTOCOL_SHA
        or state.get(
            "trackb_split_master_sha256"
        )
        != EXPECTED_TRACKB_SPLIT_SHA
        or state.get(
            "mirage_wrapper_sha256"
        )
        != EXPECTED_MIRAGE_WRAPPER_SHA
        or int(
            state.get(
                "fold",
                -1,
            )
        )
        != fold
    ):
        raise RuntimeError(
            "B2 Fold {} checkpoint integrity mismatch.".format(
                fold
            )
        )

    model.load_state_dict(
        state[
            "model"
        ],
        strict=True,
    )

    del state

    model.eval()

    return model


def build_a2_model(
    fold: int,
    checkpoint: Path,
    step07,
    step11,
    device: torch.device,
):
    reset_cuda()

    model, loss_module = (
        step11.build_no_interaction_model(
            step07,
            device,
        )
    )

    state = torch_load_cpu(
        checkpoint
    )

    if (
        state.get(
            "status"
        )
        != "A2_NOINTERACTION_NOAUG"
        or state.get(
            "global_protocol_sha256"
        )
        != EXPECTED_A2_PROTOCOL_SHA
        or state.get(
            "ablation_scientific_source_sha256"
        )
        != EXPECTED_A2_SCIENTIFIC_SOURCE_SHA
        or state.get(
            "active_target_master_sha256"
        )
        != EXPECTED_G3_SHA
        or state.get(
            "parent_implementation_spec_sha256"
        )
        != EXPECTED_IMPLEMENTATION_SPEC_SHA
        or state.get(
            "source_bundle_sha256"
        )
        != EXPECTED_SOURCE_BUNDLE_SHA
        or int(
            state.get(
                "fold",
                -1,
            )
        )
        != fold
    ):
        raise RuntimeError(
            "A2 Fold {} checkpoint integrity mismatch.".format(
                fold
            )
        )

    model.load_state_dict(
        state[
            "model"
        ],
        strict=True,
    )

    del state
    del loss_module

    model.eval()

    return model


# =============================================================================
# SHARED DATA / CURVE CONVERSION
# =============================================================================

def make_dataset(
    stems: Sequence[str],
    images: Dict[str, Path],
    step07,
    step09,
):
    return step09.GeoRCPDataset(
        stems,
        images,
        step07,
    )


def b2_curve_from_semantic(
    pred_native: np.ndarray,
    class_id: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Deterministic common-space conversion with no tunable parameters.

    presence[x] = any pixels of semantic class in column x
    y[x]        = arithmetic mean of row indices of those pixels
    """
    mask = (
        pred_native
        == int(
            class_id
        )
    )

    presence = mask.any(
        axis=0
    )

    y = np.zeros(
        NW,
        dtype=np.float32,
    )

    for x in np.where(
        presence
    )[0]:
        ys = np.where(
            mask[
                :,
                x
            ]
        )[0]

        y[
            x
        ] = float(
            ys.mean()
        )

    return (
        presence,
        y,
    )


def infer_a2_records(
    fold: int,
    stems: Sequence[str],
    checkpoint: Path,
    step07,
    step09,
    step11,
    images: Dict[str, Path],
    device: torch.device,
    include_regions: bool,
):
    model = build_a2_model(
        fold,
        checkpoint,
        step07,
        step11,
        device,
    )

    ds = make_dataset(
        stems,
        images,
        step07,
        step09,
    )

    loader = step09.loader(
        ds
    )

    records = []

    with torch.no_grad():
        for batch in loader:
            image = batch[
                "image"
            ].to(
                device
            )

            with torch.amp.autocast(
                "cuda",
                dtype=torch.bfloat16,
            ):
                out = model(
                    image
                )

            region_pred = out[
                "region_logits"
            ].argmax(
                dim=1
            ).cpu().numpy()

            curve_prob = torch.sigmoid(
                out[
                    "curve_presence_logits"
                ]
            )[
                :,
                :,
                PL:PL + NW,
            ].float().cpu().numpy()

            for i, stem in enumerate(
                batch[
                    "_stems"
                ]
            ):
                sample = {
                    key: batch[
                        key
                    ][
                        i
                    ].cpu()
                    for key in batch
                    if key != "_stems"
                }

                record = {
                    "stem": stem,
                    "curve_prob": curve_prob[
                        i
                    ],
                    "curve_gt": (
                        sample[
                            "curve_presence"
                        ].numpy()[
                            :,
                            PL:PL + NW,
                        ]
                        > 0.5
                    ),
                    "curve_valid": (
                        sample[
                            "curve_column_valid"
                        ].numpy()[
                            PL:PL + NW
                        ]
                        > 0.5
                    ),
                    "curve_py": step09.pred_curve_y(
                        out[
                            "curve_centerline_logits"
                        ][
                            i
                        ]
                    ),
                    "curve_gy": step09.gt_curve_y(
                        sample[
                            "curve_centerline"
                        ].numpy()
                    ),
                    "hf_pred": step09.decode_hf(
                        out[
                            "hf_heatmap_logits"
                        ][
                            i
                        ],
                        out[
                            "hf_offset"
                        ][
                            i
                        ],
                    ),
                    "hf_gt": step09.gt_hf(
                        sample
                    ),
                }

                if include_regions:
                    record.update(
                        {
                            "region_pred": region_pred[
                                i,
                                PT:PT + NH,
                                PL:PL + NW,
                            ],
                            "region_gt": (
                                sample[
                                    "region_class"
                                ].numpy()[
                                    PT:PT + NH,
                                    PL:PL + NW,
                                ]
                            ),
                            "region_valid": (
                                sample[
                                    "region_valid"
                                ].numpy()[
                                    PT:PT + NH,
                                    PL:PL + NW,
                                ]
                                > 0.5
                            ),
                        }
                    )

                records.append(
                    record
                )

    del loader
    del ds
    del model

    reset_cuda()

    return records


def infer_b2_records(
    fold: int,
    stems: Sequence[str],
    checkpoint: Path,
    step07,
    step09,
    step10,
    images: Dict[str, Path],
    device: torch.device,
):
    model = build_b2_model(
        fold,
        checkpoint,
        step07,
        step10,
        device,
    )

    ds = make_dataset(
        stems,
        images,
        step07,
        step09,
    )

    loader = step09.loader(
        ds
    )

    records = []

    with torch.no_grad():
        for batch in loader:
            image = batch[
                "image"
            ].to(
                device
            )

            with torch.amp.autocast(
                "cuda",
                dtype=torch.bfloat16,
            ):
                logits = model(
                    image
                )

            pred = logits.argmax(
                dim=1
            ).cpu().numpy()

            for i, stem in enumerate(
                batch[
                    "_stems"
                ]
            ):
                sample = {
                    key: batch[
                        key
                    ][
                        i
                    ].cpu()
                    for key in batch
                    if key != "_stems"
                }

                native = pred[
                    i,
                    PT:PT + NH,
                    PL:PL + NW,
                ]

                pred_presence = np.zeros(
                    (
                        2,
                        NW,
                    ),
                    dtype=bool,
                )

                pred_y = np.zeros(
                    (
                        2,
                        NW,
                    ),
                    dtype=np.float32,
                )

                for ch in (
                    0,
                    1,
                ):
                    (
                        pred_presence[
                            ch
                        ],
                        pred_y[
                            ch
                        ],
                    ) = b2_curve_from_semantic(
                        native,
                        B2_CURVE_CLASS[
                            ch
                        ],
                    )

                records.append(
                    {
                        "stem": stem,
                        "region_pred": native,
                        "region_gt": (
                            sample[
                                "region_class"
                            ].numpy()[
                                PT:PT + NH,
                                PL:PL + NW,
                            ]
                        ),
                        "region_valid": (
                            sample[
                                "region_valid"
                            ].numpy()[
                                PT:PT + NH,
                                PL:PL + NW,
                            ]
                            > 0.5
                        ),
                        "curve_pred_presence": pred_presence,
                        "curve_py": pred_y,
                        "curve_gt": (
                            sample[
                                "curve_presence"
                            ].numpy()[
                                :,
                                PL:PL + NW,
                            ]
                            > 0.5
                        ),
                        "curve_valid": (
                            sample[
                                "curve_column_valid"
                            ].numpy()[
                                PL:PL + NW
                            ]
                            > 0.5
                        ),
                        "curve_gy": step09.gt_curve_y(
                            sample[
                                "curve_centerline"
                            ].numpy()
                        ),
                    }
                )

    del loader
    del ds
    del model

    reset_cuda()

    return records


# =============================================================================
# RUN 1 — A2 VALIDATION CALIBRATION
# =============================================================================

def run_a2_calibration(
    manifest: Dict[str, Any],
    step07,
    step09,
    step11,
    images: Dict[str, Path],
    available: Sequence[str],
    device: torch.device,
):
    CALIBRATION_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    out = {}

    for fold in range(
        1,
        6,
    ):
        print(
            "\nA2 CALIBRATION FOLD {}".format(
                fold
            )
        )

        stems = step09.resolve_split_role(
            fold,
            "val",
            available,
        )

        if (
            len(
                stems
            )
            != EXPECTED_VAL_IMAGES[
                fold
            ]
        ):
            raise RuntimeError(
                "Fold {} validation count mismatch.".format(
                    fold
                )
            )

        checkpoint = Path(
            manifest[
                "folds"
            ][
                str(
                    fold
                )
            ][
                "best_checkpoint"
            ]
        )

        records = infer_a2_records(
            fold,
            stems,
            checkpoint,
            step07,
            step09,
            step11,
            images,
            device,
            include_regions=False,
        )

        curve = step09.calibrate_curve(
            records
        )

        hf = step09.calibrate_hf(
            records
        )

        result = {
            "fold": fold,
            "validation_images": len(
                stems
            ),
            "validation_stem_sha256": hash_stems(
                stems
            ),
            "curve": curve,
            "hf": hf,
        }

        save_json(
            CALIBRATION_ROOT
            / "fold_{}_calibration.json".format(
                fold
            ),
            result,
        )

        out[
            str(
                fold
            )
        ] = result

        print(
            "  curve threshold {:.2f}, score {:.5f}".format(
                curve[
                    "selected_threshold"
                ],
                curve[
                    "selected_score"
                ],
            )
        )

        print(
            "  HF threshold {:.2f}, score {:.5f}".format(
                hf[
                    "selected_threshold"
                ],
                hf[
                    "selected_score"
                ],
            )
        )

        del records

    return out


def make_protocol(
    calibration: Dict[str, Any],
    environment: Dict[str, Any],
) -> Dict[str, Any]:
    protocol = {
        "status": "FROZEN_BEFORE_B2_A2_HELDOUT_TEST",
        "script_revision": SCRIPT_REVISION,
        "script_sha256": current_script_sha(),
        "purpose": (
            "Attribution go/no-go: B2 conventional MIRAGE -> "
            "A2 geometry-specific no interaction -> Full GeoRCP"
        ),
        "environment": environment,
        "trackb_split_master_sha256": EXPECTED_TRACKB_SPLIT_SHA,
        "active_target_master_sha256": EXPECTED_G3_SHA,
        "implementation_spec_sha256": EXPECTED_IMPLEMENTATION_SPEC_SHA,
        "source_bundle_sha256": EXPECTED_SOURCE_BUNDLE_SHA,
        "mirage_wrapper_sha256": EXPECTED_MIRAGE_WRAPPER_SHA,
        "B2": {
            "training_protocol_sha256": EXPECTED_B2_PROTOCOL_SHA,
            "scientific_source_sha256": EXPECTED_B2_SCIENTIFIC_SOURCE_SHA,
            "test_status_at_freeze": "unresolved/unevaluated",
            "HF": "not applicable; B2 has no HF detector",
        },
        "A2": {
            "training_protocol_sha256": EXPECTED_A2_PROTOCOL_SHA,
            "scientific_source_sha256": EXPECTED_A2_SCIENTIFIC_SOURCE_SHA,
            "test_status_at_freeze": "unresolved/unevaluated",
            "threshold_calibration": calibration,
            "calibration_scope": "fold-specific validation only",
            "heldout_test_used_for_calibration": False,
        },
        "Full_GeoRCP": {
            "training_protocol_sha256": EXPECTED_FULL_TRAIN_PROTOCOL_SHA,
            "existing_evaluation_protocol_sha256": (
                EXPECTED_FULL_EVAL_PROTOCOL_SHA
            ),
            "policy": (
                "reuse existing Step-09 held-out result; NEVER re-evaluate"
            ),
        },
        "common_region_evaluation": {
            "targets": [
                "SRF",
                "IRF",
            ],
            "ground_truth": "geometry_v3 region_class",
            "decode": "native 570x380 crop",
            "overlap": "exclude region_valid=0",
            "metrics": [
                "Dice",
                "IoU",
                "boundary F1 @ 2 native pixels",
                "symmetric boundary HD95",
            ],
        },
        "common_curve_evaluation": {
            "targets": [
                "ELM",
                "EZ",
            ],
            "ground_truth": (
                "geometry_v3 curve_presence + centerline expected-y"
            ),
            "B2_conversion": {
                "semantic_classes": {
                    "ELM": 3,
                    "EZ": 4,
                },
                "presence": (
                    "any predicted class pixel in native column"
                ),
                "y": (
                    "arithmetic mean native row of predicted class "
                    "pixels in that column"
                ),
                "tuned_parameters": False,
                "morphology": False,
                "gap_fill": False,
                "connectivity_postprocessing": False,
            },
            "A2_decode": (
                "same Step-09 geometry decode; validation-frozen "
                "presence threshold; expected-y"
            ),
            "Full_decode": (
                "existing frozen Step-09 result; no re-evaluation"
            ),
            "metrics": [
                "column presence precision/recall/F1",
                "mean/median localization error where both present",
                "symmetric curve HD95",
                "internal-gap false bridge rate",
                "secondary 1-pixel centerline Dice",
            ],
        },
        "HF_evaluation": {
            "B2": "not applicable",
            "A2_vs_Full": (
                "same Step-09 Hungarian F1@5/10px, localization, "
                "count MAE, FROC and AP@0.5 convention"
            ),
        },
        "paired_patient_statistics": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "unit": "patient",
            "ci": "95% percentile",
            "comparisons": [
                "A2_vs_B2",
                "Full_vs_A2",
                "Full_vs_B2",
            ],
            "orientation": (
                "positive oriented delta always favors first-named model"
            ),
        },
        "test_policy": (
            "one frozen pass for B2 and A2; no test-derived tuning; "
            "Full GeoRCP reused from existing Step-09 result only"
        ),
    }

    protocol[
        "evaluation_protocol_sha256"
    ] = stable_sha(
        protocol
    )

    return protocol


# =============================================================================
# TEST ROW CONSTRUCTION
# =============================================================================

def evaluate_b2_fold(
    fold: int,
    stems: Sequence[str],
    checkpoint: Path,
    protocol_sha: str,
    step07,
    step09,
    step10,
    images: Dict[str, Path],
    eye2patient: Dict[str, str],
    device: torch.device,
):
    folder = (
        B2_EVAL_ROOT
        / "heldout"
        / "fold_{}".format(
            fold
        )
    )

    folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    complete_file = (
        folder
        / "TEST_EVALUATION_COMPLETE.json"
    )

    csv_file = (
        folder
        / "per_image_metrics.csv"
    )

    array_file = (
        folder
        / "curve_distance_arrays.npz"
    )

    if complete_file.exists():
        old = load_json(
            complete_file
        )

        if (
            old.get(
                "evaluation_protocol_sha256"
            )
            != protocol_sha
        ):
            raise RuntimeError(
                "Existing B2 Fold {} result protocol mismatch.".format(
                    fold
                )
            )

        print(
            "  B2 Fold {} already COMPLETE — skipping".format(
                fold
            )
        )

        return old

    records = infer_b2_records(
        fold,
        stems,
        checkpoint,
        step07,
        step09,
        step10,
        images,
        device,
    )

    rows = []
    curve_errors = {
        "ELM": [],
        "EZ": [],
    }

    for record in records:
        stem = record[
            "stem"
        ]

        patient = step09.patient_for_stem(
            stem,
            eye2patient,
        )

        row = {
            "model": "B2",
            "fold": fold,
            "stem": stem,
            "patient_id": patient,
        }

        for class_id, name in REGION_NAMES.items():
            metrics = step09.eval_region(
                record[
                    "region_pred"
                ],
                record[
                    "region_gt"
                ],
                record[
                    "region_valid"
                ],
                class_id,
            )

            for key, value in metrics.items():
                row[
                    "region_{}_{}".format(
                        name.lower(),
                        key,
                    )
                ] = value

        for channel, name in CURVE_NAMES.items():
            # Fixed binary semantic presence; threshold 0.5 is merely the
            # binary interface into the identical Step-09 evaluator.
            probability = record[
                "curve_pred_presence"
            ][
                channel
            ].astype(
                np.float32
            )

            metrics, errors = step09.eval_curve(
                probability,
                record[
                    "curve_py"
                ][
                    channel
                ],
                record[
                    "curve_gt"
                ][
                    channel
                ],
                record[
                    "curve_gy"
                ][
                    channel
                ],
                record[
                    "curve_valid"
                ],
                0.5,
            )

            for key, value in metrics.items():
                row[
                    "curve_{}_{}".format(
                        name.lower(),
                        key,
                    )
                ] = value

            if len(
                errors
            ):
                curve_errors[
                    name
                ].append(
                    errors
                )

        rows.append(
            row
        )

    write_rows(
        csv_file,
        rows,
    )

    arrays = {
        "curve_elm_errors": (
            np.concatenate(
                curve_errors[
                    "ELM"
                ]
            )
            if curve_errors[
                "ELM"
            ]
            else np.zeros(
                0,
                np.float32,
            )
        ),
        "curve_ez_errors": (
            np.concatenate(
                curve_errors[
                    "EZ"
                ]
            )
            if curve_errors[
                "EZ"
            ]
            else np.zeros(
                0,
                np.float32,
            )
        ),
    }

    np.savez_compressed(
        array_file,
        **arrays
    )

    patients = {
        row[
            "patient_id"
        ]
        for row in rows
    }

    if (
        len(
            patients
        )
        != EXPECTED_TEST_PATIENTS_PER_FOLD
    ):
        raise RuntimeError(
            "B2 Fold {} patient count mismatch.".format(
                fold
            )
        )

    result = {
        "status": "COMPLETE",
        "model": "B2",
        "fold": fold,
        "test_images": len(
            stems
        ),
        "test_patients": len(
            patients
        ),
        "test_stem_sha256": hash_stems(
            stems
        ),
        "evaluation_protocol_sha256": protocol_sha,
        "semantic_curve_conversion": (
            "any class pixel per column; y=mean predicted class row"
        ),
        "per_image_csv": str(
            csv_file
        ),
        "distance_arrays": str(
            array_file
        ),
        "outer_test_evaluated": True,
        "post_test_tuning_allowed": False,
    }

    save_json(
        complete_file,
        result,
    )

    del records

    return result


def evaluate_a2_fold(
    fold: int,
    stems: Sequence[str],
    checkpoint: Path,
    thresholds: Dict[str, Any],
    protocol_sha: str,
    step07,
    step09,
    step11,
    images: Dict[str, Path],
    eye2patient: Dict[str, str],
    device: torch.device,
):
    folder = (
        A2_EVAL_ROOT
        / "heldout"
        / "fold_{}".format(
            fold
        )
    )

    folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    complete_file = (
        folder
        / "TEST_EVALUATION_COMPLETE.json"
    )

    csv_file = (
        folder
        / "per_image_metrics.csv"
    )

    hf_file = (
        folder
        / "hf_payload.json.gz"
    )

    array_file = (
        folder
        / "matched_distance_arrays.npz"
    )

    if complete_file.exists():
        old = load_json(
            complete_file
        )

        if (
            old.get(
                "evaluation_protocol_sha256"
            )
            != protocol_sha
        ):
            raise RuntimeError(
                "Existing A2 Fold {} result protocol mismatch.".format(
                    fold
                )
            )

        print(
            "  A2 Fold {} already COMPLETE — skipping".format(
                fold
            )
        )

        return old

    records = infer_a2_records(
        fold,
        stems,
        checkpoint,
        step07,
        step09,
        step11,
        images,
        device,
        include_regions=True,
    )

    curve_threshold = float(
        thresholds[
            "curve"
        ][
            "selected_threshold"
        ]
    )

    hf_threshold = float(
        thresholds[
            "hf"
        ][
            "selected_threshold"
        ]
    )

    rows = []
    curve_errors = {
        "ELM": [],
        "EZ": [],
    }
    hf_errors = {
        "5": [],
        "10": [],
    }
    hf_payload = []

    for record in records:
        stem = record[
            "stem"
        ]

        patient = step09.patient_for_stem(
            stem,
            eye2patient,
        )

        row = {
            "model": "A2",
            "fold": fold,
            "stem": stem,
            "patient_id": patient,
        }

        for class_id, name in REGION_NAMES.items():
            metrics = step09.eval_region(
                record[
                    "region_pred"
                ],
                record[
                    "region_gt"
                ],
                record[
                    "region_valid"
                ],
                class_id,
            )

            for key, value in metrics.items():
                row[
                    "region_{}_{}".format(
                        name.lower(),
                        key,
                    )
                ] = value

        for channel, name in CURVE_NAMES.items():
            metrics, errors = step09.eval_curve(
                record[
                    "curve_prob"
                ][
                    channel
                ],
                record[
                    "curve_py"
                ][
                    channel
                ],
                record[
                    "curve_gt"
                ][
                    channel
                ],
                record[
                    "curve_gy"
                ][
                    channel
                ],
                record[
                    "curve_valid"
                ],
                curve_threshold,
            )

            for key, value in metrics.items():
                row[
                    "curve_{}_{}".format(
                        name.lower(),
                        key,
                    )
                ] = value

            if len(
                errors
            ):
                curve_errors[
                    name
                ].append(
                    errors
                )

        metrics, errors_by_radius = step09.eval_hf(
            record[
                "hf_pred"
            ],
            record[
                "hf_gt"
            ],
            hf_threshold,
        )

        for key, value in metrics.items():
            row[
                "hf_{}".format(
                    key
                )
            ] = value

        for key, errors in errors_by_radius.items():
            if len(
                errors
            ):
                hf_errors[
                    key
                ].append(
                    errors
                )

        hf_payload.append(
            {
                "stem": stem,
                "patient_id": patient,
                "pred": record[
                    "hf_pred"
                ],
                "gt": record[
                    "hf_gt"
                ].tolist(),
            }
        )

        rows.append(
            row
        )

    write_rows(
        csv_file,
        rows,
    )

    save_json_gz(
        hf_file,
        hf_payload,
    )

    arrays = {
        "curve_elm_errors": (
            np.concatenate(
                curve_errors[
                    "ELM"
                ]
            )
            if curve_errors[
                "ELM"
            ]
            else np.zeros(
                0,
                np.float32,
            )
        ),
        "curve_ez_errors": (
            np.concatenate(
                curve_errors[
                    "EZ"
                ]
            )
            if curve_errors[
                "EZ"
            ]
            else np.zeros(
                0,
                np.float32,
            )
        ),
        "hf_5_errors": (
            np.concatenate(
                hf_errors[
                    "5"
                ]
            )
            if hf_errors[
                "5"
            ]
            else np.zeros(
                0,
                np.float32,
            )
        ),
        "hf_10_errors": (
            np.concatenate(
                hf_errors[
                    "10"
                ]
            )
            if hf_errors[
                "10"
            ]
            else np.zeros(
                0,
                np.float32,
            )
        ),
    }

    np.savez_compressed(
        array_file,
        **arrays
    )

    patients = {
        row[
            "patient_id"
        ]
        for row in rows
    }

    if (
        len(
            patients
        )
        != EXPECTED_TEST_PATIENTS_PER_FOLD
    ):
        raise RuntimeError(
            "A2 Fold {} patient count mismatch.".format(
                fold
            )
        )

    result = {
        "status": "COMPLETE",
        "model": "A2",
        "fold": fold,
        "test_images": len(
            stems
        ),
        "test_patients": len(
            patients
        ),
        "test_stem_sha256": hash_stems(
            stems
        ),
        "evaluation_protocol_sha256": protocol_sha,
        "curve_presence_threshold": curve_threshold,
        "hf_score_threshold": hf_threshold,
        "per_image_csv": str(
            csv_file
        ),
        "hf_payload": str(
            hf_file
        ),
        "distance_arrays": str(
            array_file
        ),
        "outer_test_evaluated": True,
        "post_test_tuning_allowed": False,
    }

    save_json(
        complete_file,
        result,
    )

    del records

    return result


# =============================================================================
# AGGREGATION
# =============================================================================

def collect_b2(
    step09,
):
    rows = []
    curve_errors = {
        "ELM": [],
        "EZ": [],
    }
    folds = {}

    for fold in range(
        1,
        6,
    ):
        folder = (
            B2_EVAL_ROOT
            / "heldout"
            / "fold_{}".format(
                fold
            )
        )

        complete = load_json(
            folder
            / "TEST_EVALUATION_COMPLETE.json"
        )

        fold_rows = read_rows(
            Path(
                complete[
                    "per_image_csv"
                ]
            )
        )

        rows.extend(
            fold_rows
        )

        arrays = np.load(
            Path(
                complete[
                    "distance_arrays"
                ]
            )
        )

        curve_errors[
            "ELM"
        ].append(
            arrays[
                "curve_elm_errors"
            ]
        )

        curve_errors[
            "EZ"
        ].append(
            arrays[
                "curve_ez_errors"
            ]
        )

        folds[
            str(
                fold
            )
        ] = {
            "region": {
                name: step09.agg_region(
                    fold_rows,
                    name,
                )
                for name in (
                    "SRF",
                    "IRF",
                )
            },
            "curve": {
                name: step09.agg_curve(
                    fold_rows,
                    name,
                    arrays[
                        "curve_{}_errors".format(
                            name.lower()
                        )
                    ],
                )
                for name in (
                    "ELM",
                    "EZ",
                )
            },
        }

    if (
        len(
            rows
        )
        != EXPECTED_IMAGES
        or len(
            {
                row[
                    "stem"
                ]
                for row in rows
            }
        )
        != EXPECTED_IMAGES
    ):
        raise RuntimeError(
            "B2 did not evaluate every image exactly once."
        )

    combined_errors = {
        name: np.concatenate(
            values
        )
        if values
        else np.zeros(
            0,
            np.float32,
        )
        for name, values in curve_errors.items()
    }

    pooled = {
        "region": {
            name: step09.agg_region(
                rows,
                name,
            )
            for name in (
                "SRF",
                "IRF",
            )
        },
        "curve": {
            name: step09.agg_curve(
                rows,
                name,
                combined_errors[
                    name
                ],
            )
            for name in (
                "ELM",
                "EZ",
            )
        },
        "hf": "NOT_APPLICABLE",
    }

    write_rows(
        B2_ALL_CSV,
        rows,
    )

    return (
        rows,
        folds,
        pooled,
    )


def collect_a2(
    step09,
):
    rows = []
    curve_errors = {
        "ELM": [],
        "EZ": [],
    }
    hf_errors = {
        "5": [],
        "10": [],
    }
    hf_payload = []
    folds = {}

    for fold in range(
        1,
        6,
    ):
        folder = (
            A2_EVAL_ROOT
            / "heldout"
            / "fold_{}".format(
                fold
            )
        )

        complete = load_json(
            folder
            / "TEST_EVALUATION_COMPLETE.json"
        )

        fold_rows = read_rows(
            Path(
                complete[
                    "per_image_csv"
                ]
            )
        )

        rows.extend(
            fold_rows
        )

        hf_payload.extend(
            load_json_gz(
                Path(
                    complete[
                        "hf_payload"
                    ]
                )
            )
        )

        arrays = np.load(
            Path(
                complete[
                    "distance_arrays"
                ]
            )
        )

        curve_errors[
            "ELM"
        ].append(
            arrays[
                "curve_elm_errors"
            ]
        )

        curve_errors[
            "EZ"
        ].append(
            arrays[
                "curve_ez_errors"
            ]
        )

        hf_errors[
            "5"
        ].append(
            arrays[
                "hf_5_errors"
            ]
        )

        hf_errors[
            "10"
        ].append(
            arrays[
                "hf_10_errors"
            ]
        )

        folds[
            str(
                fold
            )
        ] = {
            "region": {
                name: step09.agg_region(
                    fold_rows,
                    name,
                )
                for name in (
                    "SRF",
                    "IRF",
                )
            },
            "curve": {
                name: step09.agg_curve(
                    fold_rows,
                    name,
                    arrays[
                        "curve_{}_errors".format(
                            name.lower()
                        )
                    ],
                )
                for name in (
                    "ELM",
                    "EZ",
                )
            },
            "hf": step09.agg_hf(
                fold_rows,
                arrays[
                    "hf_5_errors"
                ],
                arrays[
                    "hf_10_errors"
                ],
            ),
        }

    if (
        len(
            rows
        )
        != EXPECTED_IMAGES
        or len(
            {
                row[
                    "stem"
                ]
                for row in rows
            }
        )
        != EXPECTED_IMAGES
    ):
        raise RuntimeError(
            "A2 did not evaluate every image exactly once."
        )

    combined_curve = {
        name: np.concatenate(
            values
        )
        if values
        else np.zeros(
            0,
            np.float32,
        )
        for name, values in curve_errors.items()
    }

    combined_hf = {
        name: np.concatenate(
            values
        )
        if values
        else np.zeros(
            0,
            np.float32,
        )
        for name, values in hf_errors.items()
    }

    pooled = {
        "region": {
            name: step09.agg_region(
                rows,
                name,
            )
            for name in (
                "SRF",
                "IRF",
            )
        },
        "curve": {
            name: step09.agg_curve(
                rows,
                name,
                combined_curve[
                    name
                ],
            )
            for name in (
                "ELM",
                "EZ",
            )
        },
        "hf": step09.agg_hf(
            rows,
            combined_hf[
                "5"
            ],
            combined_hf[
                "10"
            ],
        ),
    }

    pooled[
        "hf"
    ][
        "ap50_10x10_boxes"
    ] = step09.ap50(
        hf_payload
    )

    pooled[
        "hf"
    ][
        "froc"
    ] = step09.froc(
        hf_payload
    )

    write_rows(
        A2_ALL_CSV,
        rows,
    )

    return (
        rows,
        folds,
        pooled,
    )


# =============================================================================
# PATIENT-PAIRED ATTRIBUTION STATISTICS
# =============================================================================

def by_patient(
    rows: Sequence[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    grouped = {}

    for row in rows:
        grouped.setdefault(
            str(
                row[
                    "patient_id"
                ]
            ),
            [],
        ).append(
            row
        )

    return grouped


def patient_metric_set(
    rows: Sequence[Dict[str, Any]],
    include_hf: bool,
) -> Dict[str, float]:
    result = {}

    for name in (
        "SRF",
        "IRF",
    ):
        prefix = "region_{}_".format(
            name.lower()
        )

        inter = sum(
            fv(
                row,
                prefix + "inter",
            )
            for row in rows
        )

        pred = sum(
            fv(
                row,
                prefix + "pred",
            )
            for row in rows
        )

        gt = sum(
            fv(
                row,
                prefix + "gt",
            )
            for row in rows
        )

        result[
            "{}_dice".format(
                name
            )
        ] = (
            safe_div(
                2.0 * inter,
                pred + gt,
            )
            if pred + gt > 0
            else float(
                "nan"
            )
        )

    for name in (
        "ELM",
        "EZ",
    ):
        prefix = "curve_{}_".format(
            name.lower()
        )

        tp = sum(
            fv(
                row,
                prefix + "tp",
            )
            for row in rows
        )

        fp = sum(
            fv(
                row,
                prefix + "fp",
            )
            for row in rows
        )

        fn = sum(
            fv(
                row,
                prefix + "fn",
            )
            for row in rows
        )

        result[
            "{}_presence_f1".format(
                name
            )
        ] = f1_counts(
            tp,
            fp,
            fn,
        )[
            "f1"
        ]

        loc_sum = sum(
            fv(
                row,
                prefix + "loc_sum",
            )
            for row in rows
        )

        loc_count = sum(
            fv(
                row,
                prefix + "loc_count",
            )
            for row in rows
        )

        result[
            "{}_mean_loc_px".format(
                name
            )
        ] = (
            safe_div(
                loc_sum,
                loc_count,
            )
            if loc_count > 0
            else float(
                "nan"
            )
        )

    if include_hf:
        for tag in (
            "5",
            "10",
        ):
            tp = sum(
                fv(
                    row,
                    "hf_tp_{}".format(
                        tag
                    ),
                )
                for row in rows
            )

            fp = sum(
                fv(
                    row,
                    "hf_fp_{}".format(
                        tag
                    ),
                )
                for row in rows
            )

            fn = sum(
                fv(
                    row,
                    "hf_fn_{}".format(
                        tag
                    ),
                )
                for row in rows
            )

            result[
                "HF_f1_{}px".format(
                    tag
                )
            ] = f1_counts(
                tp,
                fp,
                fn,
            )[
                "f1"
            ]

        count_errors = [
            fv(
                row,
                "hf_count_abs_error",
            )
            for row in rows
        ]

        count_errors = [
            value
            for value in count_errors
            if np.isfinite(
                value
            )
        ]

        result[
            "HF_count_mae"
        ] = (
            float(
                np.mean(
                    count_errors
                )
            )
            if count_errors
            else float(
                "nan"
            )
        )

    return result


HIGHER_BETTER = {
    "SRF_dice": True,
    "IRF_dice": True,
    "ELM_presence_f1": True,
    "EZ_presence_f1": True,
    "ELM_mean_loc_px": False,
    "EZ_mean_loc_px": False,
    "HF_f1_5px": True,
    "HF_f1_10px": True,
    "HF_count_mae": False,
}


def paired_bootstrap(
    candidate_name: str,
    candidate_rows: Sequence[Dict[str, Any]],
    reference_name: str,
    reference_rows: Sequence[Dict[str, Any]],
    include_hf: bool,
) -> List[Dict[str, Any]]:
    candidate_group = by_patient(
        candidate_rows
    )

    reference_group = by_patient(
        reference_rows
    )

    candidate_patients = set(
        candidate_group
    )

    reference_patients = set(
        reference_group
    )

    if (
        candidate_patients
        != reference_patients
        or len(
            candidate_patients
        )
        != EXPECTED_PATIENTS
    ):
        raise RuntimeError(
            "{} vs {} patient sets do not match.".format(
                candidate_name,
                reference_name,
            )
        )

    patients = sorted(
        candidate_patients
    )

    candidate_metrics = {
        patient: patient_metric_set(
            candidate_group[
                patient
            ],
            include_hf=include_hf,
        )
        for patient in patients
    }

    reference_metrics = {
        patient: patient_metric_set(
            reference_group[
                patient
            ],
            include_hf=include_hf,
        )
        for patient in patients
    }

    metric_names = [
        name
        for name in HIGHER_BETTER
        if (
            include_hf
            or not name.startswith(
                "HF_"
            )
        )
    ]

    rng = np.random.RandomState(
        BOOTSTRAP_SEED
    )

    rows = []

    for metric in metric_names:
        patient_deltas = []

        for patient in patients:
            candidate_value = candidate_metrics[
                patient
            ][
                metric
            ]

            reference_value = reference_metrics[
                patient
            ][
                metric
            ]

            if (
                not finite(
                    candidate_value
                )
                or not finite(
                    reference_value
                )
            ):
                continue

            raw_delta = (
                float(
                    candidate_value
                )
                - float(
                    reference_value
                )
            )

            oriented = (
                raw_delta
                if HIGHER_BETTER[
                    metric
                ]
                else -raw_delta
            )

            patient_deltas.append(
                (
                    patient,
                    raw_delta,
                    oriented,
                )
            )

        if not patient_deltas:
            continue

        raw = np.asarray(
            [
                item[
                    1
                ]
                for item in patient_deltas
            ],
            dtype=np.float64,
        )

        oriented = np.asarray(
            [
                item[
                    2
                ]
                for item in patient_deltas
            ],
            dtype=np.float64,
        )

        boot_raw = []
        boot_oriented = []

        n = len(
            patient_deltas
        )

        for _ in range(
            BOOTSTRAP_REPLICATES
        ):
            idx = rng.randint(
                0,
                n,
                size=n,
            )

            boot_raw.append(
                float(
                    raw[
                        idx
                    ].mean()
                )
            )

            boot_oriented.append(
                float(
                    oriented[
                        idx
                    ].mean()
                )
            )

        boot_raw = np.asarray(
            boot_raw,
            dtype=np.float64,
        )

        boot_oriented = np.asarray(
            boot_oriented,
            dtype=np.float64,
        )

        rows.append(
            {
                "comparison": (
                    "{}_vs_{}".format(
                        candidate_name,
                        reference_name,
                    )
                ),
                "candidate": candidate_name,
                "reference": reference_name,
                "metric": metric,
                "higher_is_better": bool(
                    HIGHER_BETTER[
                        metric
                    ]
                ),
                "paired_patients": int(
                    n
                ),
                "raw_delta_candidate_minus_reference": float(
                    raw.mean()
                ),
                "raw_ci95_low": float(
                    np.percentile(
                        boot_raw,
                        2.5,
                    )
                ),
                "raw_ci95_high": float(
                    np.percentile(
                        boot_raw,
                        97.5,
                    )
                ),
                "oriented_improvement": float(
                    oriented.mean()
                ),
                "oriented_ci95_low": float(
                    np.percentile(
                        boot_oriented,
                        2.5,
                    )
                ),
                "oriented_ci95_high": float(
                    np.percentile(
                        boot_oriented,
                        97.5,
                    )
                ),
                "bootstrap_probability_improvement": float(
                    np.mean(
                        boot_oriented
                        > 0
                    )
                ),
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            }
        )

    return rows


# =============================================================================
# SUMMARY OUTPUT
# =============================================================================

def flatten_pooled(
    model: str,
    pooled: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rows = []

    for family in (
        "region",
        "curve",
    ):
        for target, metrics in pooled[
            family
        ].items():
            for metric, value in metrics.items():
                rows.append(
                    {
                        "model": model,
                        "family": family,
                        "target": target,
                        "metric": metric,
                        "value": value,
                    }
                )

    hf = pooled.get(
        "hf"
    )

    if isinstance(
        hf,
        dict,
    ):
        for target, value in hf.items():
            if target == "froc":
                for op, sensitivity in value[
                    "operating_points"
                ].items():
                    rows.append(
                        {
                            "model": model,
                            "family": "hf",
                            "target": "FROC",
                            "metric": (
                                "sensitivity_at_{}_fp_per_image".format(
                                    op
                                )
                            ),
                            "value": sensitivity,
                        }
                    )
            elif isinstance(
                value,
                dict,
            ):
                for metric, metric_value in value.items():
                    rows.append(
                        {
                            "model": model,
                            "family": "hf",
                            "target": target,
                            "metric": metric,
                            "value": metric_value,
                        }
                    )
            else:
                rows.append(
                    {
                        "model": model,
                        "family": "hf",
                        "target": "HF",
                        "metric": target,
                        "value": value,
                    }
                )

    return rows


# =============================================================================
# MAIN
# =============================================================================

def main():
    print(
        "\n"
        + "=" * 110
    )

    print(
        "RVO-ME STEP 12: ATTRIBUTION / GO-NO-GO EVALUATION"
    )

    print(
        "SCRIPT REVISION: {}".format(
            SCRIPT_REVISION
        )
    )

    print(
        "=" * 110
    )

    EVAL_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    if HELDOUT_COMPLETE_FILE.exists():
        complete = load_json(
            HELDOUT_COMPLETE_FILE
        )

        print(
            "\nSTEP 12 already COMPLETE."
        )

        print(
            "Protocol SHA: {}".format(
                complete[
                    "evaluation_protocol_sha256"
                ]
            )
        )

        print(
            "No re-evaluation performed."
        )

        return

    print(
        "\n[1/6] Verifying modules, environment, manifests and frozen protocols..."
    )

    (
        step07,
        step09,
        step10,
        step11,
    ) = load_modules()

    environment = verify_environment()

    (
        b2_manifest,
        a2_manifest,
        full_protocol,
    ) = verify_manifests()

    images = step07.images_by_stem()

    available = sorted(
        images
    )

    if len(
        available
    ) != EXPECTED_IMAGES:
        raise RuntimeError(
            "Expected {} images; found {}.".format(
                EXPECTED_IMAGES,
                len(
                    available
                ),
            )
        )

    device = torch.device(
        "cuda"
    )

    print(
        "  B2 five-fold training: PASS; outer test untouched"
    )

    print(
        "  A2 five-fold training: PASS; outer test untouched"
    )

    print(
        "  Full GeoRCP frozen Step-09 protocol: PASS"
    )

    print(
        "  Full GeoRCP held-out artifact exists and will NOT be opened in Run 1"
    )

    if not PROTOCOL_FILE.exists():
        print(
            "\n[2/6] RUN 1 — A2 VALIDATION-ONLY calibration."
        )

        print(
            "  New TEST roles will NOT be resolved."
        )

        calibration = run_a2_calibration(
            a2_manifest,
            step07,
            step09,
            step11,
            images,
            available,
            device,
        )

        protocol = make_protocol(
            calibration,
            environment,
        )

        save_json(
            PROTOCOL_FILE,
            protocol,
        )

        append_log_once(
            "<!-- STEP12_ATTRIBUTION_PROTOCOL_FROZEN -->",
            (
                "Step12 attribution protocol frozen after A2 "
                "validation-only calibration. B2/A2 outer tests "
                "remained unresolved. Protocol SHA {}."
            ).format(
                protocol[
                    "evaluation_protocol_sha256"
                ]
            ),
        )

        print(
            "\n"
            + "=" * 110
        )

        print(
            "STEP 12 RUN 1 COMPLETE — ATTRIBUTION PROTOCOL FROZEN"
        )

        print(
            "=" * 110
        )

        print(
            "New B2/A2 held-out tests resolved/evaluated: NO / NO"
        )

        print(
            "Protocol:"
        )

        print(
            PROTOCOL_FILE
        )

        print(
            "Protocol SHA:"
        )

        print(
            protocol[
                "evaluation_protocol_sha256"
            ]
        )

        print(
            "\nRun THIS SAME UNCHANGED SCRIPT a second time "
            "for the one-time B2/A2 held-out evaluation."
        )

        return

    print(
        "\n[2/6] RUN 2 — verifying frozen attribution protocol..."
    )

    protocol = load_json(
        PROTOCOL_FILE
    )

    if (
        protocol.get(
            "status"
        )
        != "FROZEN_BEFORE_B2_A2_HELDOUT_TEST"
    ):
        raise RuntimeError(
            "Attribution protocol has unexpected status."
        )

    if (
        protocol.get(
            "script_sha256"
        )
        != current_script_sha()
    ):
        raise RuntimeError(
            "Step-12 script changed after validation calibration."
        )

    stored_sha = protocol.get(
        "evaluation_protocol_sha256"
    )

    check = dict(
        protocol
    )

    check.pop(
        "evaluation_protocol_sha256",
        None,
    )

    if stable_sha(
        check
    ) != stored_sha:
        raise RuntimeError(
            "Attribution protocol content SHA mismatch."
        )

    print(
        "  Frozen attribution protocol: PASS"
    )

    print(
        "  Same unchanged script: PASS"
    )

    print(
        "\n[3/6] Resolving B2/A2 TEST roles under the frozen protocol..."
    )

    eye2patient = step09.load_eye_to_patient()

    tests = {}
    seen_stems = set()
    seen_patients = set()

    for fold in range(
        1,
        6,
    ):
        stems = step09.resolve_split_role(
            fold,
            "test",
            available,
        )

        if (
            len(
                stems
            )
            != EXPECTED_TEST_IMAGES[
                fold
            ]
        ):
            raise RuntimeError(
                "Fold {} test image-count mismatch.".format(
                    fold
                )
            )

        patients = {
            step09.patient_for_stem(
                stem,
                eye2patient,
            )
            for stem in stems
        }

        if (
            len(
                patients
            )
            != EXPECTED_TEST_PATIENTS_PER_FOLD
            or seen_stems
            & set(
                stems
            )
            or seen_patients
            & patients
        ):
            raise RuntimeError(
                "Fold {} test leakage/coverage failure.".format(
                    fold
                )
            )

        seen_stems.update(
            stems
        )

        seen_patients.update(
            patients
        )

        tests[
            fold
        ] = stems

        print(
            "  Fold {}: {} images / {} patients PASS".format(
                fold,
                len(
                    stems
                ),
                len(
                    patients
                ),
            )
        )

    if (
        len(
            seen_stems
        )
        != EXPECTED_IMAGES
        or len(
            seen_patients
        )
        != EXPECTED_PATIENTS
    ):
        raise RuntimeError(
            "Five test folds do not cover all images/patients exactly once."
        )

    started = {
        "status": "STARTED",
        "evaluation_protocol_sha256": stored_sha,
        "script_sha256": current_script_sha(),
        "test_images": len(
            seen_stems
        ),
        "test_patients": len(
            seen_patients
        ),
        "B2_test_started": True,
        "A2_test_started": True,
        "Full_GeoRCP_re_evaluation": False,
        "post_test_tuning_allowed": False,
    }

    if HELDOUT_STARTED_FILE.exists():
        old_started = load_json(
            HELDOUT_STARTED_FILE
        )

        if (
            old_started.get(
                "evaluation_protocol_sha256"
            )
            != stored_sha
        ):
            raise RuntimeError(
                "Held-out start marker protocol mismatch."
            )
    else:
        save_json(
            HELDOUT_STARTED_FILE,
            started,
        )

    print(
        "\n[4/6] One-time held-out evaluation: B2 conventional MIRAGE..."
    )

    for fold in range(
        1,
        6,
    ):
        checkpoint = Path(
            b2_manifest[
                "folds"
            ][
                str(
                    fold
                )
            ][
                "best_checkpoint"
            ]
        )

        result = evaluate_b2_fold(
            fold,
            tests[
                fold
            ],
            checkpoint,
            stored_sha,
            step07,
            step09,
            step10,
            images,
            eye2patient,
            device,
        )

        print(
            "  B2 Fold {}: {}".format(
                fold,
                result[
                    "status"
                ],
            )
        )

    print(
        "\n[5/6] One-time held-out evaluation: A2 no-interaction..."
    )

    for fold in range(
        1,
        6,
    ):
        checkpoint = Path(
            a2_manifest[
                "folds"
            ][
                str(
                    fold
                )
            ][
                "best_checkpoint"
            ]
        )

        thresholds = protocol[
            "A2"
        ][
            "threshold_calibration"
        ][
            str(
                fold
            )
        ]

        result = evaluate_a2_fold(
            fold,
            tests[
                fold
            ],
            checkpoint,
            thresholds,
            stored_sha,
            step07,
            step09,
            step11,
            images,
            eye2patient,
            device,
        )

        print(
            "  A2 Fold {}: {}".format(
                fold,
                result[
                    "status"
                ],
            )
        )

    print(
        "\n[6/6] Aggregating and opening existing Full GeoRCP result for comparison..."
    )

    (
        b2_rows,
        b2_folds,
        b2_pooled,
    ) = collect_b2(
        step09
    )

    (
        a2_rows,
        a2_folds,
        a2_pooled,
    ) = collect_a2(
        step09
    )

    # Only now, after B2/A2 held-out evaluation is underway under the frozen
    # protocol, read the already-completed Full GeoRCP result.
    full_complete = load_json(
        FULL_HELDOUT_COMPLETE
    )

    if (
        full_complete.get(
            "status"
        )
        != "COMPLETE"
        or full_complete.get(
            "evaluation_protocol_sha256"
        )
        != EXPECTED_FULL_EVAL_PROTOCOL_SHA
        or full_complete.get(
            "test_images"
        )
        != EXPECTED_IMAGES
        or full_complete.get(
            "test_patients"
        )
        != EXPECTED_PATIENTS
        or full_complete.get(
            "post_test_tuning_allowed"
        )
    ):
        raise RuntimeError(
            "Existing Full GeoRCP held-out result integrity mismatch."
        )

    full_rows = read_rows(
        FULL_PER_IMAGE_CSV
    )

    if (
        len(
            full_rows
        )
        != EXPECTED_IMAGES
        or len(
            {
                row[
                    "stem"
                ]
                for row in full_rows
            }
        )
        != EXPECTED_IMAGES
    ):
        raise RuntimeError(
            "Existing Full GeoRCP per-image result coverage mismatch."
        )

    # Exact paired stem identity across all three methods.
    b2_stems = {
        row[
            "stem"
        ]
        for row in b2_rows
    }

    a2_stems = {
        row[
            "stem"
        ]
        for row in a2_rows
    }

    full_stems = {
        row[
            "stem"
        ]
        for row in full_rows
    }

    if not (
        b2_stems
        == a2_stems
        == full_stems
    ):
        raise RuntimeError(
            "Paired image sets differ across B2/A2/Full."
        )

    paired_rows = []

    paired_rows.extend(
        paired_bootstrap(
            "A2",
            a2_rows,
            "B2",
            b2_rows,
            include_hf=False,
        )
    )

    paired_rows.extend(
        paired_bootstrap(
            "Full",
            full_rows,
            "A2",
            a2_rows,
            include_hf=True,
        )
    )

    paired_rows.extend(
        paired_bootstrap(
            "Full",
            full_rows,
            "B2",
            b2_rows,
            include_hf=False,
        )
    )

    write_rows(
        PAIRED_BOOTSTRAP_CSV,
        paired_rows,
    )

    summary_rows = []

    summary_rows.extend(
        flatten_pooled(
            "B2",
            b2_pooled,
        )
    )

    summary_rows.extend(
        flatten_pooled(
            "A2",
            a2_pooled,
        )
    )

    summary_rows.extend(
        flatten_pooled(
            "Full",
            full_complete[
                "pooled"
            ],
        )
    )

    write_rows(
        SUMMARY_CSV,
        summary_rows,
    )

    result = {
        "status": "COMPLETE",
        "scientific_status": (
            "ATTRIBUTION_HELDOUT_COMPARISON_COMPLETE_NO_POST_TEST_TUNING"
        ),
        "evaluation_protocol_sha256": stored_sha,
        "script_revision": SCRIPT_REVISION,
        "script_sha256": current_script_sha(),
        "test_images": EXPECTED_IMAGES,
        "test_patients": EXPECTED_PATIENTS,
        "each_image_paired_across_models": True,
        "B2": {
            "pooled": b2_pooled,
            "folds": b2_folds,
            "per_image_csv": str(
                B2_ALL_CSV
            ),
            "HF": "NOT_APPLICABLE",
        },
        "A2": {
            "pooled": a2_pooled,
            "folds": a2_folds,
            "per_image_csv": str(
                A2_ALL_CSV
            ),
        },
        "Full_GeoRCP": {
            "source": (
                "existing frozen Step-09 held-out result; not re-evaluated"
            ),
            "evaluation_protocol_sha256": (
                EXPECTED_FULL_EVAL_PROTOCOL_SHA
            ),
            "pooled": full_complete[
                "pooled"
            ],
            "per_image_csv": str(
                FULL_PER_IMAGE_CSV
            ),
        },
        "paired_patient_bootstrap": paired_rows,
        "attribution_summary_csv": str(
            SUMMARY_CSV
        ),
        "paired_bootstrap_csv": str(
            PAIRED_BOOTSTRAP_CSV
        ),
        "post_test_tuning_allowed": False,
        "next_action": (
            "Interpret attribution evidence before authorizing any "
            "additional large training ablation."
        ),
    }

    save_json(
        HELDOUT_COMPLETE_FILE,
        result,
    )

    append_log_once(
        "<!-- STEP12_ATTRIBUTION_HELDOUT_COMPLETE -->",
        (
            "Step12 attribution comparison complete under frozen protocol "
            "SHA {}. B2 and A2 held-out folds evaluated once; Full GeoRCP "
            "Step-09 result reused without re-evaluation; no post-test "
            "tuning permitted."
        ).format(
            stored_sha
        ),
    )

    print(
        "\n"
        + "=" * 110
    )

    print(
        "STEP 12 ATTRIBUTION HELD-OUT COMPARISON COMPLETE"
    )

    print(
        "=" * 110
    )

    print(
        "B2/A2 outer test coverage: 3012 images / 130 patients exactly once"
    )

    print(
        "Full GeoRCP re-evaluated: NO"
    )

    print(
        "Post-test tuning allowed: NO"
    )

    print(
        "\nResults:"
    )

    print(
        HELDOUT_COMPLETE_FILE
    )

    print(
        "\nSummary CSV:"
    )

    print(
        SUMMARY_CSV
    )

    print(
        "\nPaired patient bootstrap:"
    )

    print(
        PAIRED_BOOTSTRAP_CSV
    )


if __name__ == "__main__":
    main()
