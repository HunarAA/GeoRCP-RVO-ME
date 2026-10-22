from __future__ import annotations

# =============================================================================
# RVO-ME STEP 10
# MIRAGE-BASE + STANDARD CONVNEXT SEMANTIC DECODER
# CONSOLIDATED TRACK-B FIVE-FOLD TRAINING
#
# Scientific purpose
# ------------------
# This is the prespecified conventional MIRAGE semantic-segmentation control
# (B2 / A0) from the frozen Step-05E5A ablation plan. It asks whether the
# retinal-OCT foundation encoder + near-native input alone explains performance,
# before geometry-specific Region/Curve/Point modeling is credited.
#
# IMPORTANT SAFETY RULES
# ----------------------
# - TRAIN + VAL only.
# - This file cannot resolve or evaluate outer TEST roles.
# - Uses released Image_Seg masks, as frozen for conventional semantic controls.
# - Uses native 570x380 content zero-padded to 576x384; NO resize.
# - Uses the same MIRAGE downstream per-image min-max normalization as GeoRCP.
# - Full MIRAGE fine-tuning.
# - No augmentation, matching the definitive GeoRCP training condition.
# - Best checkpoint selected only from validation mean foreground Dice.
# - Resume is deterministic at epoch boundaries.
#
# The outer-test evaluation must be a separate later step after all five
# training folds complete and the evaluation protocol is frozen.
# =============================================================================

import ast
import csv
import copy
import gc
import hashlib
import importlib.util
import json
import math
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# IDENTITY / PATHS
# =============================================================================

SCRIPT_REVISION = "STEP10_MIRAGE_SEMANTIC_B2_A0_V6_RESUME_SAFE_RNGCPU_20260830"

ROOT = Path(r"D:\RVO_ME")
DATASET = ROOT / "RVO-Lesion"
PROJECT = ROOT / "project"

IMAGE_DIR = DATASET / "Image_Seg" / "images"
MASK_DIR = DATASET / "Image_Seg" / "masks"

SPLIT_ROOT = PROJECT / "splits" / "patient_5fold_v1"

STEP07_SCRIPT = (
    PROJECT / "scripts" / "07_georcp_preflight_v2_ampfix_py38compat.py"
)

DESIGN_DIR = PROJECT / "design" / "geometry_aware_rcp_v1"
ABLATION_PLAN = DESIGN_DIR / "ABLATION_PLAN_FROZEN.json"

MIRAGE_WRAPPER = (
    PROJECT
    / "foundation_models"
    / "mirage_base_preflight_v1"
    / "official_mirage_hf.py"
)

MIRAGE_DIR = MIRAGE_WRAPPER.parent
MIRAGE_HF_CACHE = MIRAGE_DIR / "hf_cache"

RUN_ROOT = (
    PROJECT
    / "baselines"
    / "segmentation"
    / "mirage_semantic_384x576_v1"
)

GLOBAL_PROTOCOL_FILE = RUN_ROOT / "B2_A0_PROTOCOL_FROZEN.json"
PREFLIGHT_FILE = RUN_ROOT / "B2_A0_PREFLIGHT_COMPLETE.json"
FIVEFOLD_MANIFEST = RUN_ROOT / "FIVEFOLD_TRAINING_MANIFEST.json"

PROJECT_LOG = PROJECT / "PROJECT_LOG.md"


# =============================================================================
# FROZEN IDENTIFIERS
# =============================================================================

EXPECTED_DESIGN_SHA = (
    "9f1586d102523009443e2b88dca54a9ca8fa5c983f08fdde837184d436a95801"
)

EXPECTED_ABLATION_PLAN_SHA = (
    "e9c292325f9ad9413e11a2f805b5854e9779c733b7ed9dc221bf79575a158233"
)

EXPECTED_MIRAGE_WRAPPER_SHA = (
    "07fa63e2766b67dbc0c07f0d9d29db1562f707425ef2fbf90a688f62a94e6ede"
)

EXPECTED_SOURCE_BUNDLE_SHA = (
    "04e7da4fe311b756ad375a6c241530d0572a3cf1b9c71a70af2e7e7e24524412"
)

CANONICAL_TRACKB_SPLIT_SHA = (
    "7e7c2ddf51054bf998b2d750b50c74983b9bc5b751b762213bc96343ffe01346"
)

EXPECTED_IMAGES = 3012

EXPECTED_ROLE_COUNTS = {
    1: {"train": 2110, "val": 303, "test": 599},
    2: {"train": 2136, "val": 270, "test": 606},
    3: {"train": 2106, "val": 308, "test": 598},
    4: {"train": 2039, "val": 342, "test": 631},
    5: {"train": 2072, "val": 362, "test": 578},
}

FOLDS_TO_TRAIN = (1, 2, 3, 4, 5)

SOURCE_BUNDLE_NAMES = [
    "build_mirage_class",
    "load_mirage",
    "LN2d",
    "CNBlock",
    "LocalEncoder",
    "GlobalFusion",
    "TaskInteraction",
    "Reassemble",
    "TaskFusion",
    "FullTrunk",
    "RegionHead",
    "CurveHead",
    "PointHead",
    "GeoRCP",
    "masked_mean",
    "binary_dice",
    "foreground_dice",
    "centernet_focal",
    "GeoRCPLoss",
    "mirage_norm",
    "load_sample",
    "optimizer_for",
    "build_model",
]

# Step-10 scientific implementation guard. This deliberately excludes
# protocol/resume/reporting helpers so line endings or resume-only engineering
# edits cannot invalidate scientifically identical training.
STEP10_SCIENTIFIC_SOURCE_NAMES = [
    "mirage_norm_worker_safe",
    "MIRAGESemanticDataset",
    "MIRAGESemanticBaseline",
    "build_model",
    "foreground_soft_dice_loss",
    "semantic_loss",
    "build_optimizer",
    "lr_factor_for_epoch",
    "apply_epoch_lrs",
    "train_one_epoch",
    "validate_one_epoch",
]

# Filled below after constructing this version; excludes this constant itself.
EXPECTED_STEP10_SCIENTIFIC_SOURCE_SHA = "c28137c519894d48c5f1ee1eb4b2fcd9b58f45e34f544698deb78a629fc2caae"


# =============================================================================
# INPUT / TASK
# =============================================================================

NW, NH = 570, 380
W, H = 576, 384
PL, PR, PT, PB = 3, 3, 2, 2

PATCH = 32
PH, PW = H // PATCH, W // PATCH
NTOK = PH * PW

IGNORE = 255

NUM_CLASSES = 5
CLASS_NAMES = {
    0: "Background",
    1: "SRF",
    2: "IRF",
    3: "ELM",
    4: "EZ",
}
FOREGROUND_CLASS_IDS = (1, 2, 3, 4)


# =============================================================================
# PRESPECIFIED ENGINEERING PROTOCOL
# =============================================================================

SEED = 20260815

PRECISION = "BF16"
PHYSICAL_BATCH = 7
VAL_BATCH = 7

OPTIMIZER_NAME = "AdamW"
ENCODER_LR = 1e-5
DECODER_LR = 1e-4
WEIGHT_DECAY = 1e-2

MAX_EPOCHS = 60
WARMUP_EPOCHS = 5
MIN_LR_FACTOR = 0.01

EARLY_STOP_PATIENCE = 12
EARLY_STOP_MIN_DELTA = 1e-3
MIN_EPOCHS_BEFORE_STOP = 20

CHECKPOINT_METRIC = "validation_mean_foreground_dice_native_fov"
CHECKPOINT_MODE = "max"

CE_WEIGHT = 1.0
DICE_WEIGHT = 1.0
DICE_SMOOTH = 1e-5

AUGMENTATION_POLICY = "NONE"

# Engineering-only data-pipeline acceleration for the RTX 4080 desktop.
# These settings do NOT change model/data/targets/loss/optimizer/batch/LR/
# checkpoint rule/augmentation/epochs.
NUM_WORKERS = 4
PIN_MEMORY = True
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 2
NON_BLOCKING_TRANSFER = True
LOG_EVERY_STEPS = 50


# =============================================================================
# GENERIC HELPERS
# =============================================================================

def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def stable_sha(obj: Any) -> str:
    raw = json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def current_script_sha() -> str:
    return sha_file(Path(__file__).resolve())


def hash_stems(stems: Sequence[str]) -> str:
    return hashlib.sha256(
        "\n".join(sorted(stems)).encode("utf-8")
    ).hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reset_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def cuda_memory() -> Dict[str, float]:
    g = float(1024 ** 3)
    return {
        "allocated_gb": float(torch.cuda.memory_allocated() / g),
        "reserved_gb": float(torch.cuda.memory_reserved() / g),
        "peak_allocated_gb": float(torch.cuda.max_memory_allocated() / g),
        "peak_reserved_gb": float(torch.cuda.max_memory_reserved() / g),
    }


def append_project_log_once(marker: str, text: str) -> None:
    if not PROJECT_LOG.exists():
        return
    current = PROJECT_LOG.read_text(
        encoding="utf-8",
        errors="ignore",
    )
    if marker in current:
        return
    with PROJECT_LOG.open("a", encoding="utf-8") as f:
        f.write("\n\n{}\n{}\n".format(marker, text))


def module_from_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not import {}".format(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def source_bundle_sha(path: Path) -> str:
    text = path.read_text(
        encoding="utf-8"
    ).replace("\r\n", "\n")

    tree = ast.parse(
        text,
        filename=str(path),
    )
    lines = text.splitlines()

    wanted = set(SOURCE_BUNDLE_NAMES)
    found = []
    chunks = []

    for node in tree.body:
        if not isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
            ),
        ):
            continue

        if node.name not in wanted:
            continue

        if not hasattr(node, "end_lineno"):
            raise RuntimeError(
                "Python AST lacks end_lineno."
            )

        segment = "\n".join(
            lines[node.lineno - 1:node.end_lineno]
        )
        chunks.append(
            "## {}\n{}\n".format(
                node.name,
                segment,
            )
        )
        found.append(node.name)

    missing = sorted(
        wanted - set(found)
    )

    if missing:
        raise RuntimeError(
            "Missing frozen Step-07 definitions: "
            + ", ".join(missing)
        )

    return hashlib.sha256(
        "\n".join(chunks).encode("utf-8")
    ).hexdigest()


def named_source_bundle_sha(
    path: Path,
    names: Sequence[str],
) -> str:
    """
    Stable SHA over selected top-level function/class definitions.
    Line endings are normalized. Used to prove that resume-only changes do not
    alter the Step-10 scientific model/data/loss/optimizer/training behavior.
    """
    source = path.read_text(
        encoding="utf-8"
    ).replace("\r\n", "\n")

    tree = ast.parse(
        source,
        filename=str(path),
    )
    lines = source.splitlines()

    wanted = set(names)
    found = []
    chunks = []

    for node in tree.body:
        if not isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
            ),
        ):
            continue

        if node.name not in wanted:
            continue

        if not hasattr(node, "end_lineno"):
            raise RuntimeError(
                "Python AST lacks end_lineno."
            )

        segment = "\n".join(
            lines[node.lineno - 1:node.end_lineno]
        )

        chunks.append(
            "## {}\n{}\n".format(
                node.name,
                segment,
            )
        )
        found.append(
            node.name
        )

    missing = sorted(
        wanted - set(found)
    )

    if missing:
        raise RuntimeError(
            "Missing Step-10 scientific definitions: "
            + ", ".join(missing)
        )

    return hashlib.sha256(
        "\n".join(chunks).encode("utf-8")
    ).hexdigest()


def protocol_scientific_view(
    obj: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compare frozen protocol content while excluding script-byte identity only.

    JSON object keys are always strings on disk. The in-memory protocol uses
    integer keys for CLASS_NAMES, so normalize through a JSON round-trip before
    comparison. This changes no scientific value; it only makes in-memory and
    on-disk JSON representations comparable.
    """
    view = copy.deepcopy(obj)

    for key in (
        "script_revision",
        "script_sha256",
        "global_protocol_sha256",
    ):
        view.pop(
            key,
            None,
        )

    return json.loads(
        json.dumps(
            view,
            sort_keys=True,
            ensure_ascii=False,
        )
    )


def first_difference(
    a: Any,
    b: Any,
    path: str = "root",
) -> Optional[str]:
    if type(a) != type(b):
        return "{}: type {} != {}".format(
            path,
            type(a).__name__,
            type(b).__name__,
        )

    if isinstance(a, dict):
        ak = set(a)
        bk = set(b)

        if ak != bk:
            return "{}: keys differ; only_existing={} only_current={}".format(
                path,
                sorted(ak - bk),
                sorted(bk - ak),
            )

        for key in sorted(ak):
            diff = first_difference(
                a[key],
                b[key],
                "{}.{}".format(
                    path,
                    key,
                ),
            )
            if diff is not None:
                return diff

        return None

    if isinstance(a, list):
        if len(a) != len(b):
            return "{}: list length {} != {}".format(
                path,
                len(a),
                len(b),
            )

        for i, (
            av,
            bv,
        ) in enumerate(
            zip(
                a,
                b,
            )
        ):
            diff = first_difference(
                av,
                bv,
                "{}[{}]".format(
                    path,
                    i,
                ),
            )
            if diff is not None:
                return diff

        return None

    if a != b:
        return "{}: {!r} != {!r}".format(
            path,
            a,
            b,
        )

    return None


# =============================================================================
# LOAD / VERIFY FROZEN STEP-07 + PREPLANNED B2/A0 CONTROL
# =============================================================================

def load_exact_step07():
    if not STEP07_SCRIPT.exists():
        raise FileNotFoundError(STEP07_SCRIPT)

    source_sha = source_bundle_sha(
        STEP07_SCRIPT
    )

    if source_sha != EXPECTED_SOURCE_BUNDLE_SHA:
        raise RuntimeError(
            "GeoRCP/Step-07 source-bundle SHA mismatch.\n"
            "Expected: {}\nFound:    {}".format(
                EXPECTED_SOURCE_BUNDLE_SHA,
                source_sha,
            )
        )

    mod = module_from_file(
        "georcp_step07_for_mirage_semantic_b2",
        STEP07_SCRIPT,
    )

    for name in (
        "load_mirage",
        "mirage_norm",
        "LN2d",
        "CNBlock",
    ):
        if not hasattr(mod, name):
            raise RuntimeError(
                "Step-07 implementation missing {}.".format(
                    name
                )
            )

    return mod, source_sha


def verify_preplanned_control() -> Dict[str, Any]:
    plan = load_json(ABLATION_PLAN)

    if (
        plan.get("ablation_plan_sha256")
        != EXPECTED_ABLATION_PLAN_SHA
    ):
        raise RuntimeError(
            "Frozen ablation-plan SHA mismatch."
        )

    if (
        str(plan.get("status", ""))
        != "FROZEN_BEFORE_PROPOSED_MODEL_TESTING"
    ):
        raise RuntimeError(
            "Ablation plan is not marked as frozen before "
            "proposed-model testing."
        )

    baselines = plan.get(
        "modern_baselines",
        [],
    )

    b2 = next(
        (
            item
            for item in baselines
            if str(item.get("id", "")) == "B2"
        ),
        None,
    )

    if b2 is None:
        raise RuntimeError(
            "Frozen ablation plan does not contain B2."
        )

    b2_model = str(
        b2.get("model", "")
    ).lower()

    b2_input = str(
        b2.get("input", "")
    ).lower()

    if (
        "mirage-base" not in b2_model
        or "semantic" not in b2_model
        or "convnext" not in b2_model
    ):
        raise RuntimeError(
            "Frozen B2 definition is not the expected "
            "MIRAGE semantic ConvNeXt control."
        )

    if (
        "384" not in b2_input
        or "576" not in b2_input
    ):
        raise RuntimeError(
            "Frozen B2 input is not 384x576."
        )

    seq = plan.get(
        "proposed_ablation_sequence",
        [],
    )

    a0 = next(
        (
            item
            for item in seq
            if str(item.get("id", "")) == "A0"
        ),
        None,
    )

    if a0 is None:
        raise RuntimeError(
            "Frozen ablation plan does not contain A0."
        )

    if "mirage" not in str(
        a0.get("configuration", "")
    ).lower():
        raise RuntimeError(
            "Frozen A0 is not MIRAGE semantic segmentation."
        )

    return {
        "status": "PASS",
        "ablation_plan_sha256": EXPECTED_ABLATION_PLAN_SHA,
        "B2": b2,
        "A0": a0,
    }


def verify_environment(
    step07,
    source_sha: str,
) -> Dict[str, Any]:

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required."
        )

    gpu = torch.cuda.get_device_name(0)
    vram = (
        torch.cuda.get_device_properties(0).total_memory
        / (1024 ** 3)
    )

    if "RTX 4080" not in gpu:
        raise RuntimeError(
            "This definitive training control is reserved "
            "for the validated RTX 4080 environment; found {}.".format(
                gpu
            )
        )

    if not sys.version.startswith("3.8"):
        raise RuntimeError(
            "Expected validated Python 3.8.x environment."
        )

    if torch.__version__ != "2.4.1+cu121":
        raise RuntimeError(
            "Expected PyTorch 2.4.1+cu121; found {}.".format(
                torch.__version__
            )
        )

    if str(torch.version.cuda) != "12.1":
        raise RuntimeError(
            "Expected CUDA build 12.1; found {}.".format(
                torch.version.cuda
            )
        )

    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "BF16 is not supported."
        )

    if not MIRAGE_WRAPPER.exists():
        raise FileNotFoundError(
            MIRAGE_WRAPPER
        )

    if (
        sha_file(MIRAGE_WRAPPER)
        != EXPECTED_MIRAGE_WRAPPER_SHA
    ):
        raise RuntimeError(
            "Frozen MIRAGE wrapper SHA mismatch."
        )

    # Engineering-only desktop path binding:
    # the frozen Step-07 source can contain a laptop-rooted global path even
    # though its function/class source bundle is byte-for-byte the verified
    # implementation. Rebind only its MIRAGE file/cache globals in memory to
    # this desktop project's already-verified D:\\RVO_ME assets. The Step-07
    # source file itself is NOT edited, and the wrapper SHA remains mandatory.
    step07.MIRAGE_DIR = MIRAGE_DIR
    step07.MIRAGE_WRAPPER = MIRAGE_WRAPPER
    step07.HF_CACHE = MIRAGE_HF_CACHE

    if not step07.MIRAGE_WRAPPER.exists():
        raise FileNotFoundError(step07.MIRAGE_WRAPPER)

    if sha_file(step07.MIRAGE_WRAPPER) != EXPECTED_MIRAGE_WRAPPER_SHA:
        raise RuntimeError(
            "Rebound Step-07 MIRAGE wrapper SHA mismatch."
        )

    if not step07.HF_CACHE.exists():
        raise FileNotFoundError(
            "Desktop MIRAGE cache is missing: {}".format(
                step07.HF_CACHE
            )
        )

    if source_sha != EXPECTED_SOURCE_BUNDLE_SHA:
        raise RuntimeError(
            "Frozen source bundle mismatch."
        )

    return {
        "python": sys.version,
        "pytorch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": gpu,
        "vram_gb": float(vram),
        "bf16_supported": True,
        "mirage_wrapper_sha256": EXPECTED_MIRAGE_WRAPPER_SHA,
        "source_bundle_sha256": source_sha,
    }


# =============================================================================
# DATA DISCOVERY / RELEASED CONVENTIONAL MASK AUDIT
# =============================================================================

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


def indexed_files(
    folder: Path,
) -> Dict[str, Path]:

    if not folder.exists():
        raise FileNotFoundError(folder)

    out = {}

    for path in sorted(folder.iterdir()):
        if (
            not path.is_file()
            or path.suffix.lower()
            not in IMAGE_EXTENSIONS
        ):
            continue

        if path.stem in out:
            raise RuntimeError(
                "Duplicate stem in {}: {}".format(
                    folder,
                    path.stem,
                )
            )

        out[path.stem] = path

    return out


def discover_image_mask_pairs():
    images = indexed_files(IMAGE_DIR)
    masks = indexed_files(MASK_DIR)

    if len(images) != EXPECTED_IMAGES:
        raise RuntimeError(
            "Expected {} released images, found {}.".format(
                EXPECTED_IMAGES,
                len(images),
            )
        )

    if len(masks) != EXPECTED_IMAGES:
        raise RuntimeError(
            "Expected {} released masks, found {}.".format(
                EXPECTED_IMAGES,
                len(masks),
            )
        )

    image_stems = set(images)
    mask_stems = set(masks)

    if image_stems != mask_stems:
        raise RuntimeError(
            "Released Image_Seg image/mask stem mismatch."
        )

    return images, masks


def audit_released_masks(
    masks: Dict[str, Path],
) -> Dict[str, Any]:

    allowed = set(range(NUM_CLASSES))
    pixel_counts = {
        str(i): 0
        for i in range(NUM_CLASSES)
    }

    bad_shape = []
    bad_values = []

    for i, stem in enumerate(sorted(masks), start=1):
        m = cv2.imread(
            str(masks[stem]),
            cv2.IMREAD_GRAYSCALE,
        )

        if m is None:
            raise RuntimeError(
                "Could not read released mask {}.".format(
                    stem
                )
            )

        if m.shape != (NH, NW):
            bad_shape.append(
                {
                    "stem": stem,
                    "shape": list(m.shape),
                }
            )
            continue

        values, counts = np.unique(
            m,
            return_counts=True,
        )

        found = {
            int(v)
            for v in values.tolist()
        }

        if not found.issubset(allowed):
            bad_values.append(
                {
                    "stem": stem,
                    "values": sorted(found),
                }
            )
            continue

        for value, count in zip(
            values.tolist(),
            counts.tolist(),
        ):
            pixel_counts[str(int(value))] += int(count)

        if (
            i == 1
            or i % 500 == 0
            or i == len(masks)
        ):
            print(
                "    released-mask audit: {}/{}".format(
                    i,
                    len(masks),
                )
            )

    if bad_shape:
        raise RuntimeError(
            "Released masks with bad shape: {}".format(
                bad_shape[:5]
            )
        )

    if bad_values:
        raise RuntimeError(
            "Released masks with invalid class values: {}".format(
                bad_values[:5]
            )
        )

    return {
        "status": "PASS",
        "images": len(masks),
        "native_height": NH,
        "native_width": NW,
        "allowed_values": sorted(allowed),
        "class_names": CLASS_NAMES,
        "pixel_counts": pixel_counts,
    }


# =============================================================================
# SPLIT RESOLUTION — TRAIN / VAL ONLY
# =============================================================================

STEM_COLUMNS = {
    "stem",
    "image",
    "image_id",
    "filename",
    "file",
    "name",
}

ROLE_COLUMNS = {
    "role",
    "split",
    "subset",
    "set",
}

FOLD_COLUMNS = {
    "fold",
    "fold_id",
    "outer_fold",
}


def normalize_fold_value(
    value: Any,
) -> Optional[int]:

    m = re.search(
        r"(\d+)",
        str(value).strip().lower(),
    )

    if m is None:
        return None

    try:
        return int(m.group(1))
    except Exception:
        return None


def resolve_split_role(
    fold: int,
    role: str,
    available_stems: Sequence[str],
) -> Tuple[List[str], Dict[str, Any]]:

    role = role.lower()

    # Deliberate safety gate. There is no TEST path in this training script.
    if role not in {"train", "val"}:
        raise RuntimeError(
            "This training script may resolve TRAIN/VAL only."
        )

    available = set(available_stems)
    found = set()
    evidence = []

    files = sorted(
        p
        for p in SPLIT_ROOT.rglob("*")
        if (
            p.is_file()
            and p.suffix.lower()
            in {".txt", ".csv", ".json"}
        )
    )

    fold_tokens = {
        "fold_{}".format(fold),
        "fold{}".format(fold),
        "/{}/".format(fold),
    }

    # TXT
    for path in files:
        if path.suffix.lower() != ".txt":
            continue

        low = str(path).lower().replace("\\", "/")

        if not any(
            token in low
            for token in fold_tokens
        ):
            continue

        if role not in path.name.lower():
            continue

        before = len(found)

        content = path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        for token in re.split(
            r"[\s,;]+",
            content,
        ):
            stem = Path(
                token.strip("\"'")
            ).stem

            if stem in available:
                found.add(stem)

        if len(found) > before:
            evidence.append(str(path))

    # CSV
    for path in files:
        if path.suffix.lower() != ".csv":
            continue

        low = str(path).lower().replace("\\", "/")

        path_fold = (
            fold
            if any(
                token in low
                for token in fold_tokens
            )
            else None
        )

        try:
            with path.open(
                "r",
                encoding="utf-8-sig",
                newline="",
            ) as f:
                reader = csv.DictReader(f)

                if reader.fieldnames is None:
                    continue

                cmap = {
                    c.lower(): c
                    for c in reader.fieldnames
                }

                stem_col = next(
                    (
                        cmap[c]
                        for c in STEM_COLUMNS
                        if c in cmap
                    ),
                    None,
                )

                role_col = next(
                    (
                        cmap[c]
                        for c in ROLE_COLUMNS
                        if c in cmap
                    ),
                    None,
                )

                fold_col = next(
                    (
                        cmap[c]
                        for c in FOLD_COLUMNS
                        if c in cmap
                    ),
                    None,
                )

                if stem_col is None:
                    continue

                before = len(found)

                for row in reader:
                    row_role = (
                        str(
                            row.get(
                                role_col,
                                "",
                            )
                        ).strip().lower()
                        if role_col
                        else (
                            role
                            if role in path.name.lower()
                            else ""
                        )
                    )

                    if row_role != role:
                        continue

                    row_fold = (
                        normalize_fold_value(
                            row.get(
                                fold_col,
                                "",
                            )
                        )
                        if fold_col
                        else path_fold
                    )

                    if row_fold != fold:
                        continue

                    stem = Path(
                        str(
                            row.get(
                                stem_col,
                                "",
                            )
                        )
                    ).stem

                    if stem in available:
                        found.add(stem)

                if len(found) > before:
                    evidence.append(str(path))

        except Exception:
            continue

    # JSON
    def walk(
        obj: Any,
        inherited_fold: Optional[int] = None,
        inherited_role: Optional[str] = None,
    ) -> None:

        if isinstance(obj, dict):
            lower = {
                str(k).lower(): v
                for k, v in obj.items()
            }

            local_fold = inherited_fold
            local_role = inherited_role

            for key in FOLD_COLUMNS:
                if key in lower:
                    local_fold = normalize_fold_value(
                        lower[key]
                    )

            for key in ROLE_COLUMNS:
                if key in lower:
                    local_role = str(
                        lower[key]
                    ).strip().lower()

            stem_value = None

            for key in STEM_COLUMNS:
                if key in lower:
                    stem_value = lower[key]
                    break

            if (
                local_fold == fold
                and local_role == role
                and stem_value is not None
            ):
                stem = Path(
                    str(stem_value)
                ).stem

                if stem in available:
                    found.add(stem)

            for value in obj.values():
                walk(
                    value,
                    local_fold,
                    local_role,
                )

        elif isinstance(obj, list):
            for value in obj:
                walk(
                    value,
                    inherited_fold,
                    inherited_role,
                )

    for path in files:
        if path.suffix.lower() != ".json":
            continue

        before = len(found)

        try:
            walk(
                load_json(path)
            )
        except Exception:
            continue

        if len(found) > before:
            evidence.append(str(path))

    stems = sorted(found)
    expected = EXPECTED_ROLE_COUNTS[fold][role]

    if len(stems) != expected:
        raise RuntimeError(
            "Fold {} {} split mismatch: expected {}, found {}.".format(
                fold,
                role,
                expected,
                len(stems),
            )
        )

    return stems, {
        "status": "PASS",
        "fold": fold,
        "role": role,
        "expected": expected,
        "found": len(stems),
        "stem_sha256": hash_stems(stems),
        "evidence_files": evidence[:30],
    }


# =============================================================================
# DATASET
# =============================================================================

def mirage_norm_worker_safe(gray: np.ndarray) -> np.ndarray:
    """Exact worker-safe equivalent of frozen Step-07 MIRAGE min-max normalization."""
    x = gray.astype(np.float32)
    lo = float(x.min())
    hi = float(x.max())
    if hi <= lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


class MIRAGESemanticDataset(Dataset):
    def __init__(
        self,
        stems: Sequence[str],
        images: Dict[str, Path],
        masks: Dict[str, Path],
    ):
        self.stems = list(stems)
        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.stems)

    def __getitem__(
        self,
        index: int,
    ):
        stem = self.stems[index]

        image = cv2.imread(
            str(self.images[stem]),
            cv2.IMREAD_GRAYSCALE,
        )

        mask = cv2.imread(
            str(self.masks[stem]),
            cv2.IMREAD_GRAYSCALE,
        )

        if image is None or mask is None:
            raise RuntimeError(
                "Could not read image/mask for {}.".format(
                    stem
                )
            )

        if image.shape != (NH, NW):
            raise RuntimeError(
                "Bad image shape for {}: {}.".format(
                    stem,
                    image.shape,
                )
            )

        if mask.shape != (NH, NW):
            raise RuntimeError(
                "Bad mask shape for {}: {}.".format(
                    stem,
                    mask.shape,
                )
            )

        if int(mask.max()) >= NUM_CLASSES:
            raise RuntimeError(
                "Unexpected mask class for {}.".format(
                    stem
                )
            )

        image = mirage_norm_worker_safe(
            image
        )

        image = cv2.copyMakeBorder(
            image,
            PT,
            PB,
            PL,
            PR,
            cv2.BORDER_CONSTANT,
            value=0,
        )

        # Padding is IGNORE, not background supervision.
        target = np.full(
            (H, W),
            IGNORE,
            dtype=np.int64,
        )

        target[
            PT:PT + NH,
            PL:PL + NW,
        ] = mask.astype(
            np.int64
        )

        return {
            "image": torch.from_numpy(
                image[None]
            ).float(),
            "target": torch.from_numpy(
                target
            ).long(),
            "_stem": stem,
        }


def collate_fn(
    items: List[Dict[str, Any]],
):
    return {
        "image": torch.stack(
            [
                item["image"]
                for item in items
            ],
            dim=0,
        ),
        "target": torch.stack(
            [
                item["target"]
                for item in items
            ],
            dim=0,
        ),
        "_stems": [
            item["_stem"]
            for item in items
        ],
    }


def move_batch(
    batch: Dict[str, Any],
    device: torch.device,
):
    return {
        key: (
            value.to(
                device,
                non_blocking=NON_BLOCKING_TRANSFER,
            )
            if torch.is_tensor(value)
            else value
        )
        for key, value in batch.items()
    }


def training_loader(
    dataset: MIRAGESemanticDataset,
) -> DataLoader:

    # Reused across epochs so Windows worker processes stay alive.
    # The same generator is explicitly re-seeded each epoch, preserving the
    # original deterministic SEED + epoch shuffle rule.
    generator = torch.Generator()
    generator.manual_seed(
        SEED + 1
    )

    kwargs = {
        "dataset": dataset,
        "batch_size": PHYSICAL_BATCH,
        "shuffle": True,
        "generator": generator,
        "num_workers": NUM_WORKERS,
        "pin_memory": PIN_MEMORY,
        "drop_last": False,
        "collate_fn": collate_fn,
    }
    if NUM_WORKERS > 0:
        kwargs["persistent_workers"] = PERSISTENT_WORKERS
        kwargs["prefetch_factor"] = PREFETCH_FACTOR

    loader = DataLoader(**kwargs)
    loader._rvo_epoch_generator = generator
    return loader


def reseed_training_loader(
    loader: DataLoader,
    epoch: int,
) -> None:
    loader._rvo_epoch_generator.manual_seed(
        SEED + int(epoch)
    )


def validation_loader(
    dataset: MIRAGESemanticDataset,
) -> DataLoader:

    kwargs = {
        "dataset": dataset,
        "batch_size": VAL_BATCH,
        "shuffle": False,
        "num_workers": NUM_WORKERS,
        "pin_memory": PIN_MEMORY,
        "drop_last": False,
        "collate_fn": collate_fn,
    }
    if NUM_WORKERS > 0:
        kwargs["persistent_workers"] = PERSISTENT_WORKERS
        kwargs["prefetch_factor"] = PREFETCH_FACTOR
    return DataLoader(**kwargs)


# =============================================================================
# B2 / A0 MODEL
# =============================================================================

class MIRAGESemanticBaseline(nn.Module):
    """
    Prespecified B2/A0 conventional control.

    Encoder:
      - frozen official MIRAGE-Base initialization
      - full fine-tuning
      - final MIRAGE transformer block only

    Decoder:
      - token projection 768 -> 256
      - learned 8x8 patch reassembly to stride-4, 64 channels
      - ConvNeXt-style refinement at stride-4
      - bilinear upsample to the full 384x576 canvas
      - ConvNeXt-style full-resolution refinement
      - ordinary 5-class semantic softmax logits

    No local-detail path, no geometry-specific heads, no task interaction.
    """

    def __init__(
        self,
        encoder,
        step07,
    ):
        super().__init__()

        self.encoder = encoder

        self.token_proj = nn.Sequential(
            nn.Linear(
                768,
                256,
            ),
            nn.LayerNorm(
                256
            ),
            nn.GELU(),
        )

        self.reassemble = nn.Linear(
            256,
            64 * 8 * 8,
        )

        self.reasm_norm = step07.LN2d(
            64
        )

        self.reasm_refine = nn.Sequential(
            step07.CNBlock(
                64
            ),
            step07.CNBlock(
                64
            ),
        )

        self.full_refine = nn.Sequential(
            nn.Conv2d(
                64,
                32,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                32,
            ),
            nn.GELU(),
            step07.CNBlock(
                32
            ),
        )

        self.classifier = nn.Conv2d(
            32,
            NUM_CLASSES,
            kernel_size=1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        if x.shape[1:] != (
            1,
            H,
            W,
        ):
            raise RuntimeError(
                "Expected Bx1x{}x{}; got {}.".format(
                    H,
                    W,
                    tuple(x.shape),
                )
            )

        layers = self.encoder.model(
            {
                "bscan": x
            },
            return_all_layers=True,
        )

        if (
            not isinstance(
                layers,
                list,
            )
            or len(layers) != 12
        ):
            raise RuntimeError(
                "MIRAGE did not return 12 layer outputs."
            )

        final = layers[-1]

        if (
            final.ndim != 3
            or final.shape[1] != NTOK + 1
            or final.shape[2] != 768
        ):
            raise RuntimeError(
                "Unexpected final MIRAGE tensor shape: {}.".format(
                    tuple(final.shape)
                )
            )

        # Official wrapper appends its global token after image tokens.
        tok = final[:, :NTOK, :]

        tok = self.token_proj(
            tok
        )

        b = tok.shape[0]

        feat = self.reassemble(
            tok
        ).view(
            b,
            PH,
            PW,
            64,
            8,
            8,
        )

        feat = feat.permute(
            0,
            3,
            1,
            4,
            2,
            5,
        ).reshape(
            b,
            64,
            H // 4,
            W // 4,
        )

        feat = self.reasm_refine(
            self.reasm_norm(
                feat
            )
        )

        feat = F.interpolate(
            feat,
            size=(
                H,
                W,
            ),
            mode="bilinear",
            align_corners=False,
        )

        feat = self.full_refine(
            feat
        )

        return self.classifier(
            feat
        )


def build_model(
    step07,
    device: torch.device,
):
    encoder = step07.load_mirage()

    model = MIRAGESemanticBaseline(
        encoder,
        step07,
    ).to(device)

    return model


def model_parameter_report(
    model: nn.Module,
) -> Dict[str, int]:

    encoder_ids = {
        id(p)
        for p in model.encoder.parameters()
    }

    encoder_total = sum(
        p.numel()
        for p in model.encoder.parameters()
    )

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    decoder_total = sum(
        p.numel()
        for p in model.parameters()
        if id(p) not in encoder_ids
    )

    return {
        "total": int(total),
        "encoder": int(encoder_total),
        "decoder": int(decoder_total),
        "trainable": int(
            sum(
                p.numel()
                for p in model.parameters()
                if p.requires_grad
            )
        ),
    }


# =============================================================================
# LOSS / METRICS
# =============================================================================

def foreground_soft_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    eps: float = DICE_SMOOTH,
) -> torch.Tensor:

    valid = (
        target != IGNORE
    )

    safe_target = target.clone()
    safe_target[
        ~valid
    ] = 0

    probs = F.softmax(
        logits,
        dim=1,
    ).float()[:, 1:]

    onehot = F.one_hot(
        safe_target.long(),
        NUM_CLASSES,
    ).permute(
        0,
        3,
        1,
        2,
    ).float()[:, 1:]

    v = valid.float().unsqueeze(1)

    probs = probs * v
    onehot = onehot * v

    inter = (
        probs * onehot
    ).sum(
        dim=(0, 2, 3)
    )

    den = probs.sum(
        dim=(0, 2, 3)
    ) + onehot.sum(
        dim=(0, 2, 3)
    )

    dice = (
        2.0 * inter + eps
    ) / (
        den + eps
    )

    return 1.0 - dice.mean()


def semantic_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
):
    ce = F.cross_entropy(
        logits,
        target,
        ignore_index=IGNORE,
    )

    dice = foreground_soft_dice_loss(
        logits,
        target,
    )

    total = (
        CE_WEIGHT * ce
        + DICE_WEIGHT * dice
    )

    return total, {
        "total": total,
        "cross_entropy": ce,
        "foreground_dice_loss": dice,
    }


def finite_gradients(
    model: nn.Module,
) -> bool:

    found = False

    for p in model.parameters():
        if p.grad is None:
            continue

        found = True

        if not torch.isfinite(
            p.grad
        ).all():
            return False

    return found


def float_parts(
    parts: Dict[str, torch.Tensor],
) -> Dict[str, float]:

    return {
        key: float(
            value.detach()
            .float()
            .cpu()
        )
        for key, value in parts.items()
    }


class RunningMean:
    def __init__(self):
        self.n = 0
        self.sums = {}

    def update(
        self,
        parts: Dict[str, float],
        n: int,
    ) -> None:

        self.n += int(n)

        for key, value in parts.items():
            self.sums[key] = (
                self.sums.get(
                    key,
                    0.0,
                )
                + float(value) * float(n)
            )

    def means(
        self,
    ) -> Dict[str, float]:

        if self.n <= 0:
            raise RuntimeError(
                "RunningMean is empty."
            )

        return {
            key: value / float(self.n)
            for key, value in self.sums.items()
        }


class SegmentationCounts:
    def __init__(self):
        self.tp = {
            c: 0
            for c in FOREGROUND_CLASS_IDS
        }
        self.fp = {
            c: 0
            for c in FOREGROUND_CLASS_IDS
        }
        self.fn = {
            c: 0
            for c in FOREGROUND_CLASS_IDS
        }

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> None:

        # Native FOV only.
        pred = pred[
            :,
            PT:PT + NH,
            PL:PL + NW,
        ]

        target = target[
            :,
            PT:PT + NH,
            PL:PL + NW,
        ]

        pred = pred.detach().cpu()
        target = target.detach().cpu()

        for class_id in FOREGROUND_CLASS_IDS:
            p = (
                pred == class_id
            )

            g = (
                target == class_id
            )

            self.tp[class_id] += int(
                (p & g).sum().item()
            )

            self.fp[class_id] += int(
                (p & (~g)).sum().item()
            )

            self.fn[class_id] += int(
                ((~p) & g).sum().item()
            )

    def report(
        self,
    ) -> Dict[str, Any]:

        classes = {}
        dices = []
        ious = []

        for class_id in FOREGROUND_CLASS_IDS:
            tp = self.tp[class_id]
            fp = self.fp[class_id]
            fn = self.fn[class_id]

            dice_den = (
                2 * tp + fp + fn
            )

            iou_den = (
                tp + fp + fn
            )

            dice = (
                float(
                    2 * tp
                    / dice_den
                )
                if dice_den > 0
                else 1.0
            )

            iou = (
                float(
                    tp
                    / iou_den
                )
                if iou_den > 0
                else 1.0
            )

            name = CLASS_NAMES[
                class_id
            ]

            classes[name] = {
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "dice": dice,
                "iou": iou,
            }

            dices.append(dice)
            ious.append(iou)

        return {
            "classes": classes,
            "mean_foreground_dice": float(
                np.mean(dices)
            ),
            "mean_foreground_iou": float(
                np.mean(ious)
            ),
        }


# =============================================================================
# OPTIMIZER / SCHEDULER
# =============================================================================

def build_optimizer(
    model: MIRAGESemanticBaseline,
):

    encoder_params = [
        p
        for p in model.encoder.parameters()
        if p.requires_grad
    ]

    encoder_ids = {
        id(p)
        for p in encoder_params
    }

    decoder_params = [
        p
        for p in model.parameters()
        if (
            p.requires_grad
            and id(p) not in encoder_ids
        )
    ]

    return torch.optim.AdamW(
        [
            {
                "params": encoder_params,
                "lr": ENCODER_LR,
                "weight_decay": WEIGHT_DECAY,
                "name": "encoder",
            },
            {
                "params": decoder_params,
                "lr": DECODER_LR,
                "weight_decay": WEIGHT_DECAY,
                "name": "decoder",
            },
        ]
    )


def lr_factor_for_epoch(
    epoch: int,
) -> float:

    if epoch <= WARMUP_EPOCHS:
        return float(epoch) / float(
            WARMUP_EPOCHS
        )

    denom = max(
        1,
        MAX_EPOCHS - WARMUP_EPOCHS,
    )

    progress = (
        float(
            epoch - WARMUP_EPOCHS
        )
        / float(denom)
    )

    progress = min(
        1.0,
        max(
            0.0,
            progress,
        ),
    )

    cosine = 0.5 * (
        1.0
        + math.cos(
            math.pi * progress
        )
    )

    return (
        MIN_LR_FACTOR
        + (
            1.0 - MIN_LR_FACTOR
        ) * cosine
    )


def apply_epoch_lrs(
    optimizer,
    epoch: int,
) -> Dict[str, float]:

    factor = lr_factor_for_epoch(
        epoch
    )

    base = {
        "encoder": ENCODER_LR,
        "decoder": DECODER_LR,
    }

    out = {}

    for group in optimizer.param_groups:
        name = group["name"]
        lr = base[name] * factor
        group["lr"] = lr
        out[name] = float(lr)

    return out


# =============================================================================
# TRAIN / VALIDATE
# =============================================================================

def train_one_epoch(
    model,
    optimizer,
    loader: DataLoader,
    device: torch.device,
    fold: int,
    epoch: int,
) -> Dict[str, Any]:

    model.train()

    running = RunningMean()

    start = time.perf_counter()

    for step, batch in enumerate(
        loader,
        start=1,
    ):
        batch_n = int(
            batch["image"].shape[0]
        )

        batch = move_batch(
            batch,
            device,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.amp.autocast(
            "cuda",
            dtype=torch.bfloat16,
        ):
            logits = model(
                batch["image"]
            )

            total, parts = semantic_loss(
                logits,
                batch["target"],
            )

        if not torch.isfinite(total):
            raise RuntimeError(
                "Non-finite loss at Fold {}, epoch {}, step {}.".format(
                    fold,
                    epoch,
                    step,
                )
            )

        total.backward()

        if not finite_gradients(model):
            raise RuntimeError(
                "Non-finite gradients at Fold {}, epoch {}, step {}.".format(
                    fold,
                    epoch,
                    step,
                )
            )

        optimizer.step()

        fp = float_parts(
            parts
        )

        running.update(
            fp,
            batch_n,
        )

        if (
            step == 1
            or step % LOG_EVERY_STEPS == 0
            or step == len(loader)
        ):
            elapsed = (
                time.perf_counter()
                - start
            )

            print(
                "      step {:4d}/{:4d} | "
                "loss={:.4f} CE={:.4f} DiceLoss={:.4f} | "
                "{:.2f} img/s".format(
                    step,
                    len(loader),
                    fp["total"],
                    fp["cross_entropy"],
                    fp["foreground_dice_loss"],
                    running.n / elapsed,
                )
            )

    torch.cuda.synchronize()

    elapsed = (
        time.perf_counter()
        - start
    )

    return {
        "images": running.n,
        "optimizer_steps": len(loader),
        "elapsed_seconds": float(elapsed),
        "images_per_second": float(
            running.n / elapsed
        ),
        "mean_losses": running.means(),
        "memory": cuda_memory(),
    }


def validate_one_epoch(
    model,
    loader: DataLoader,
    device: torch.device,
    fold: int,
) -> Dict[str, Any]:

    model.eval()

    running = RunningMean()
    counts = SegmentationCounts()

    start = time.perf_counter()

    with torch.no_grad():
        for batch in loader:
            batch_n = int(
                batch["image"].shape[0]
            )

            batch = move_batch(
                batch,
                device,
            )

            with torch.amp.autocast(
                "cuda",
                dtype=torch.bfloat16,
            ):
                logits = model(
                    batch["image"]
                )

                total, parts = semantic_loss(
                    logits,
                    batch["target"],
                )

            if not torch.isfinite(total):
                raise RuntimeError(
                    "Non-finite validation loss in Fold {}.".format(
                        fold
                    )
                )

            running.update(
                float_parts(parts),
                batch_n,
            )

            pred = logits.argmax(
                dim=1
            )

            counts.update(
                pred,
                batch["target"],
            )

    torch.cuda.synchronize()

    elapsed = (
        time.perf_counter()
        - start
    )

    metrics = counts.report()

    return {
        "images": running.n,
        "batches": len(loader),
        "elapsed_seconds": float(elapsed),
        "images_per_second": float(
            running.n / elapsed
        ),
        "mean_losses": running.means(),
        "native_fov_metrics": metrics,
    }


# =============================================================================
# PROTOCOL FREEZE
# =============================================================================

def global_protocol_object(
    preplanned: Dict[str, Any],
    environment: Dict[str, Any],
) -> Dict[str, Any]:

    obj = {
        "status": "FROZEN_BEFORE_B2_OUTER_TEST",
        "name": (
            "RVO-ME MIRAGE-Base conventional semantic "
            "segmentation B2/A0 v1"
        ),
        "script_revision": SCRIPT_REVISION,
        "script_sha256": current_script_sha(),
        "scientific_design_master_sha256": EXPECTED_DESIGN_SHA,
        "ablation_plan_sha256": EXPECTED_ABLATION_PLAN_SHA,
        "preplanned_control_ids": [
            "B2",
            "A0",
        ],
        "preplanned_control_evidence": preplanned,
        "trackb_split_master_sha256": CANONICAL_TRACKB_SPLIT_SHA,
        "mirage_wrapper_sha256": EXPECTED_MIRAGE_WRAPPER_SHA,
        "step07_source_bundle_sha256": EXPECTED_SOURCE_BUNDLE_SHA,
        "data": {
            "image_source": (
                "released Image_Seg/images"
            ),
            "target_source": (
                "released Image_Seg/masks retained by frozen "
                "scientific design for conventional semantic "
                "segmentation controls"
            ),
            "classes": CLASS_NAMES,
            "native_hw": [
                NH,
                NW,
            ],
            "model_hw": [
                H,
                W,
            ],
            "padding_lrtb": [
                PL,
                PR,
                PT,
                PB,
            ],
            "resize": False,
            "padding_target_value": IGNORE,
            "padding_supervised": False,
            "normalization": (
                "official MIRAGE downstream per-image min-max [0,1]"
            ),
        },
        "architecture": {
            "encoder": "MIRAGE-Base",
            "encoder_initialization": (
                "official pretrained retinal OCT weights"
            ),
            "encoder_training": "full fine-tuning",
            "encoder_feature": (
                "final MIRAGE transformer block only"
            ),
            "image_tokens": NTOK,
            "embedding_dim": 768,
            "decoder": (
                "standard semantic ConvNeXt-style decoder"
            ),
            "decoder_exact": [
                "Linear 768->256 + LayerNorm + GELU",
                (
                    "Linear 256->(64x8x8), token reassembly "
                    "to stride-4 64-channel feature map"
                ),
                "two 64-channel ConvNeXt blocks",
                "bilinear upsample to 384x576",
                (
                    "3x3 64->32 + GroupNorm + GELU + "
                    "one 32-channel ConvNeXt block"
                ),
                "1x1 32->5 semantic logits",
            ],
            "local_detail_encoder": False,
            "multilevel_mirage_fusion": False,
            "geometry_specific_heads": False,
            "task_interaction": False,
        },
        "loss": {
            "name": (
                "CrossEntropy + foreground soft multiclass Dice"
            ),
            "cross_entropy_weight": CE_WEIGHT,
            "foreground_dice_weight": DICE_WEIGHT,
            "dice_classes": [
                "SRF",
                "IRF",
                "ELM",
                "EZ",
            ],
            "ignore_padding": True,
        },
        "training": {
            "seed": SEED,
            "precision": PRECISION,
            "physical_batch_size": PHYSICAL_BATCH,
            "validation_batch_size": VAL_BATCH,
            "optimizer": OPTIMIZER_NAME,
            "encoder_lr": ENCODER_LR,
            "decoder_lr": DECODER_LR,
            "weight_decay": WEIGHT_DECAY,
            "scheduler": (
                "5-epoch linear warmup then cosine decay"
            ),
            "warmup_epochs": WARMUP_EPOCHS,
            "minimum_lr_factor": MIN_LR_FACTOR,
            "maximum_epochs": MAX_EPOCHS,
            "early_stopping_patience": EARLY_STOP_PATIENCE,
            "early_stopping_min_delta": EARLY_STOP_MIN_DELTA,
            "minimum_epochs_before_early_stop": MIN_EPOCHS_BEFORE_STOP,
            "checkpoint_selection_metric": CHECKPOINT_METRIC,
            "checkpoint_mode": CHECKPOINT_MODE,
            "augmentation": AUGMENTATION_POLICY,
            "data_pipeline_engineering": {
                "num_workers": NUM_WORKERS,
                "pin_memory": PIN_MEMORY,
                "persistent_workers": PERSISTENT_WORKERS,
                "prefetch_factor": PREFETCH_FACTOR,
                "non_blocking_transfer": NON_BLOCKING_TRANSFER,
                "scientific_effect": "NONE; engineering-only throughput optimization",
                "desktop_path_binding": (
                    "Step-07 MIRAGE_WRAPPER/MIRAGE_DIR/HF_CACHE globals rebound "
                    "in memory to verified D:\\RVO_ME assets; Step-07 source unchanged"
                ),
            },
            "outer_test_policy": (
                "never resolve/evaluate during training or "
                "checkpoint selection"
            ),
        },
        "environment": environment,
    }

    obj[
        "global_protocol_sha256"
    ] = stable_sha(
        obj
    )

    return obj


def freeze_or_verify_global_protocol(
    protocol: Dict[str, Any],
) -> Dict[str, Any]:

    RUN_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    if GLOBAL_PROTOCOL_FILE.exists():
        existing = load_json(
            GLOBAL_PROTOCOL_FILE
        )

        # Do not rewrite the original frozen protocol. Verify that every
        # scientific/engineering setting remains identical, ignoring only
        # script revision/byte SHA and the SHA derived from those identity
        # fields. This is what makes interrupted Fold-3 training resumable
        # even if Spyder/Windows changed file line endings or this resume-only
        # driver has a new script revision.
        existing_view = protocol_scientific_view(
            existing
        )
        current_view = protocol_scientific_view(
            protocol
        )

        diff = first_difference(
            existing_view,
            current_view,
        )

        if diff is not None:
            raise RuntimeError(
                "Existing B2/A0 protocol differs scientifically/engineering-wise. "
                "Resume refused. First difference: {}".format(
                    diff
                )
            )

        if existing.get(
            "status"
        ) != "FROZEN_BEFORE_B2_OUTER_TEST":
            raise RuntimeError(
                "Existing B2/A0 protocol has unexpected status."
            )

        print(
            "    B2/A0 global protocol: existing frozen protocol reused / PASS"
        )
        print(
            "    Resume identity difference allowed: script revision/SHA only"
        )

        authoritative = existing

    else:
        save_json(
            GLOBAL_PROTOCOL_FILE,
            protocol,
        )

        print(
            "    B2/A0 global protocol: FROZEN"
        )

        authoritative = protocol

    print(
        "    Global protocol SHA: {}".format(
            authoritative[
                "global_protocol_sha256"
            ]
        )
    )

    return authoritative


def fold_protocol_object(
    fold: int,
    global_protocol: Dict[str, Any],
    train_split: Dict[str, Any],
    val_split: Dict[str, Any],
) -> Dict[str, Any]:

    obj = {
        "status": "FROZEN",
        "fold": int(fold),
        "global_protocol_sha256": global_protocol[
            "global_protocol_sha256"
        ],
        "trackb_split_master_sha256": CANONICAL_TRACKB_SPLIT_SHA,
        "train_images": train_split[
            "found"
        ],
        "val_images": val_split[
            "found"
        ],
        "train_stem_sha256": train_split[
            "stem_sha256"
        ],
        "val_stem_sha256": val_split[
            "stem_sha256"
        ],
        "outer_test_resolved": False,
        "outer_test_evaluated": False,
    }

    obj[
        "fold_protocol_sha256"
    ] = stable_sha(
        obj
    )

    return obj


# =============================================================================
# SMOKE / MEMORY PREFLIGHT — TRAIN DATA ONLY
# =============================================================================

def run_preflight(
    step07,
    images: Dict[str, Path],
    masks: Dict[str, Path],
    fold1_train: Sequence[str],
    device: torch.device,
    global_protocol: Dict[str, Any],
) -> Dict[str, Any]:

    if PREFLIGHT_FILE.exists():
        existing = load_json(
            PREFLIGHT_FILE
        )

        if (
            existing.get("status") == "PASS"
            and existing.get(
                "global_protocol_sha256"
            )
            == global_protocol[
                "global_protocol_sha256"
            ]
            and int(
                existing.get(
                    "physical_batch_size",
                    -1,
                )
            ) == PHYSICAL_BATCH
            and str(
                existing.get(
                    "precision",
                    "",
                )
            ).upper() == PRECISION
            and not bool(
                existing.get(
                    "outer_test_resolved",
                    True,
                )
            )
            and not bool(
                existing.get(
                    "outer_test_evaluated",
                    True,
                )
            )
        ):
            print(
                "    B2/A0 batch-7 preflight: existing PASS reused"
            )
            return existing

        raise RuntimeError(
            "Existing B2/A0 preflight is incompatible with "
            "the frozen protocol or test-safety rules."
        )

    print(
        "\n[PRE-FLIGHT] Batch-7 BF16 forward/backward "
        "on Fold-1 TRAIN only..."
    )

    stems = list(
        fold1_train[:PHYSICAL_BATCH]
    )

    if len(stems) != PHYSICAL_BATCH:
        raise RuntimeError(
            "Not enough Fold-1 train images for preflight."
        )

    ds = MIRAGESemanticDataset(
        stems,
        images,
        masks,
    )

    loader = DataLoader(
        ds,
        batch_size=PHYSICAL_BATCH,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
        collate_fn=collate_fn,
    )

    batch = next(
        iter(loader)
    )

    set_seed(SEED)
    reset_cuda()

    model = build_model(
        step07,
        device,
    )

    params = model_parameter_report(
        model
    )

    optimizer = build_optimizer(
        model
    )

    batch = move_batch(
        batch,
        device,
    )

    model.train()
    optimizer.zero_grad(
        set_to_none=True
    )

    with torch.amp.autocast(
        "cuda",
        dtype=torch.bfloat16,
    ):
        logits = model(
            batch["image"]
        )

        total, parts = semantic_loss(
            logits,
            batch["target"],
        )

    if list(logits.shape) != [
        PHYSICAL_BATCH,
        NUM_CLASSES,
        H,
        W,
    ]:
        raise RuntimeError(
            "B2/A0 output shape mismatch: {}.".format(
                list(logits.shape)
            )
        )

    if not torch.isfinite(total):
        raise RuntimeError(
            "B2/A0 preflight produced non-finite loss."
        )

    total.backward()

    if not finite_gradients(model):
        raise RuntimeError(
            "B2/A0 preflight produced non-finite gradients."
        )

    torch.cuda.synchronize()

    report = {
        "status": "PASS",
        "script_revision": SCRIPT_REVISION,
        "script_sha256": current_script_sha(),
        "global_protocol_sha256": global_protocol[
            "global_protocol_sha256"
        ],
        "fold": 1,
        "role": "train",
        "images": PHYSICAL_BATCH,
        "precision": PRECISION,
        "physical_batch_size": PHYSICAL_BATCH,
        "output_shape": list(
            logits.shape
        ),
        "losses": float_parts(
            parts
        ),
        "parameters": params,
        "memory": cuda_memory(),
        "outer_test_resolved": False,
        "outer_test_evaluated": False,
    }

    save_json(
        PREFLIGHT_FILE,
        report,
    )

    print(
        "    B2/A0 preflight: PASS"
    )
    print(
        "    total params: {:.3f} M | encoder: {:.3f} M | decoder: {:.3f} M".format(
            params["total"] / 1e6,
            params["encoder"] / 1e6,
            params["decoder"] / 1e6,
        )
    )
    print(
        "    peak allocated: {:.3f} GB".format(
            report[
                "memory"
            ][
                "peak_allocated_gb"
            ]
        )
    )

    del logits
    del batch
    del optimizer
    del model
    del loader
    del ds

    reset_cuda()

    return report


# =============================================================================
# CHECKPOINT / RESUME
# =============================================================================

def torch_load_checkpoint(
    path: Path,
    device: torch.device,
):
    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location=device,
        )


def rng_state() -> Dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(
    state: Dict[str, Any],
) -> None:
    """
    Restore RNG state after a CUDA-mapped checkpoint load.

    torch.load(..., map_location=device) moves tensor-valued RNG states to
    CUDA along with model/optimizer tensors. PyTorch's RNG setters require
    CPU torch.ByteTensor states, so move/cast only these RNG-state tensors
    back to CPU before restoring them. This is resume engineering only and
    does not change the saved random state or the scientific training rule.
    """
    random.setstate(
        state["python"]
    )

    np.random.set_state(
        state["numpy"]
    )

    cpu_state = state["torch_cpu"]

    if not torch.is_tensor(
        cpu_state
    ):
        cpu_state = torch.as_tensor(
            cpu_state,
            dtype=torch.uint8,
        )

    cpu_state = cpu_state.detach().to(
        device="cpu",
        dtype=torch.uint8,
    ).contiguous()

    torch.set_rng_state(
        cpu_state
    )

    cuda_states = []

    for cuda_state in state["torch_cuda"]:
        if not torch.is_tensor(
            cuda_state
        ):
            cuda_state = torch.as_tensor(
                cuda_state,
                dtype=torch.uint8,
            )

        cuda_states.append(
            cuda_state.detach().to(
                device="cpu",
                dtype=torch.uint8,
            ).contiguous()
        )

    torch.cuda.set_rng_state_all(
        cuda_states
    )


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    fold: int,
    epoch: int,
    best_epoch: int,
    best_metric: float,
    epochs_without_improvement: int,
    global_protocol_sha: str,
    fold_protocol_sha: str,
) -> None:

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    state = {
        "status": "B2_A0_MIRAGE_SEMANTIC",
        "script_revision": SCRIPT_REVISION,
        "script_sha256": current_script_sha(),
        "resume_driver_scientific_source_sha256": (
            EXPECTED_STEP10_SCIENTIFIC_SOURCE_SHA
        ),
        "global_protocol_sha256": global_protocol_sha,
        "fold_protocol_sha256": fold_protocol_sha,
        "trackb_split_master_sha256": CANONICAL_TRACKB_SPLIT_SHA,
        "mirage_wrapper_sha256": EXPECTED_MIRAGE_WRAPPER_SHA,
        "fold": int(fold),
        "epoch": int(epoch),
        "best_epoch": int(best_epoch),
        "best_metric": float(best_metric),
        "checkpoint_metric": CHECKPOINT_METRIC,
        "epochs_without_improvement": int(
            epochs_without_improvement
        ),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng": rng_state(),
    }

    torch.save(
        state,
        tmp,
    )

    if path.exists():
        path.unlink()

    tmp.replace(
        path
    )


def load_resume(
    path: Path,
    model,
    optimizer,
    device: torch.device,
    fold: int,
    global_protocol_sha: str,
    fold_protocol_sha: str,
) -> Dict[str, Any]:

    if not path.exists():
        return {
            "start_epoch": 1,
            "best_epoch": 0,
            "best_metric": -float("inf"),
            "epochs_without_improvement": 0,
            "historical_script_sha256": None,
            "resumed": False,
        }

    state = torch_load_checkpoint(
        path,
        device,
    )

    checks = {
        "global_protocol_sha256": global_protocol_sha,
        "fold_protocol_sha256": fold_protocol_sha,
        "trackb_split_master_sha256": CANONICAL_TRACKB_SPLIT_SHA,
        "mirage_wrapper_sha256": EXPECTED_MIRAGE_WRAPPER_SHA,
    }

    for key, expected in checks.items():
        if state.get(key) != expected:
            raise RuntimeError(
                "Fold {} resume mismatch: {}.".format(
                    fold,
                    key,
                )
            )

    # Script-byte identity is provenance, not a scientific hyperparameter.
    # Exact scientific definitions are guarded independently at program start.
    historical_script_sha = state.get(
        "script_sha256"
    )

    if not historical_script_sha:
        raise RuntimeError(
            "Fold {} checkpoint lacks script provenance.".format(
                fold
            )
        )

    if int(
        state.get(
            "fold",
            -1,
        )
    ) != fold:
        raise RuntimeError(
            "Resume checkpoint belongs to another fold."
        )

    model.load_state_dict(
        state["model"],
        strict=True,
    )

    optimizer.load_state_dict(
        state["optimizer"]
    )

    restore_rng_state(
        state["rng"]
    )

    return {
        "start_epoch": int(
            state["epoch"]
        ) + 1,
        "best_epoch": int(
            state["best_epoch"]
        ),
        "best_metric": float(
            state["best_metric"]
        ),
        "epochs_without_improvement": int(
            state[
                "epochs_without_improvement"
            ]
        ),
        "historical_script_sha256": historical_script_sha,
        "resumed": True,
    }


# =============================================================================
# HISTORY
# =============================================================================

HISTORY_FIELDS = [
    "fold",
    "epoch",
    "encoder_lr",
    "decoder_lr",
    "train_total",
    "train_ce",
    "train_dice_loss",
    "val_total",
    "val_ce",
    "val_dice_loss",
    "val_mean_fg_dice",
    "val_mean_fg_iou",
    "val_SRF_dice",
    "val_IRF_dice",
    "val_ELM_dice",
    "val_EZ_dice",
    "train_seconds",
    "val_seconds",
    "peak_allocated_gb",
    "improved",
]


def upsert_history_row(
    history_csv: Path,
    row: Dict[str, Any],
) -> None:

    rows = {}

    if history_csv.exists():
        with history_csv.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as f:
            reader = csv.DictReader(f)

            for old in reader:
                try:
                    epoch = int(
                        old["epoch"]
                    )
                except Exception:
                    continue

                rows[epoch] = old

    rows[
        int(
            row["epoch"]
        )
    ] = row

    tmp = history_csv.with_suffix(
        ".csv.tmp"
    )

    with tmp.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=HISTORY_FIELDS,
        )

        writer.writeheader()

        for epoch in sorted(rows):
            writer.writerow(
                rows[epoch]
            )

    if history_csv.exists():
        history_csv.unlink()

    tmp.replace(
        history_csv
    )


# =============================================================================
# FIVE-FOLD MANIFEST
# =============================================================================

def update_manifest(
    global_protocol_sha: str,
) -> Dict[str, Any]:

    folds = {}

    for fold in FOLDS_TO_TRAIN:
        fold_dir = (
            RUN_ROOT
            / "fold_{}".format(
                fold
            )
        )

        complete_path = (
            fold_dir
            / "FOLD{}_TRAINING_COMPLETE.json".format(
                fold
            )
        )

        best_path = (
            fold_dir
            / "best_checkpoint.pt"
        )

        if complete_path.exists():
            report = load_json(
                complete_path
            )
            status = report.get(
                "status",
                "UNKNOWN",
            )
        else:
            report = {}
            status = "PENDING"

        folds[str(fold)] = {
            "status": status,
            "best_epoch": report.get(
                "best_epoch"
            ),
            "best_validation_mean_foreground_dice": report.get(
                "best_validation_mean_foreground_dice"
            ),
            "run_dir": str(
                fold_dir
            ),
            "completion_report": str(
                complete_path
            ),
            "best_checkpoint": str(
                best_path
            ),
        }

    all_complete = all(
        folds[
            str(fold)
        ][
            "status"
        ] == "COMPLETE"
        for fold in FOLDS_TO_TRAIN
    )

    manifest = {
        "status": (
            "ALL_FIVE_FOLDS_TRAINING_COMPLETE"
            if all_complete
            else "TRAINING_IN_PROGRESS"
        ),
        "model": (
            "MIRAGE-Base + standard semantic ConvNeXt decoder B2/A0"
        ),
        "global_protocol_sha256": global_protocol_sha,
        "trackb_split_master_sha256": CANONICAL_TRACKB_SPLIT_SHA,
        "outer_test_resolved": False,
        "outer_test_evaluated": False,
        "folds": folds,
    }

    save_json(
        FIVEFOLD_MANIFEST,
        manifest,
    )

    return manifest


# =============================================================================
# TRAIN ONE FOLD
# =============================================================================

def train_fold(
    fold: int,
    step07,
    images: Dict[str, Path],
    masks: Dict[str, Path],
    available_stems: Sequence[str],
    device: torch.device,
    global_protocol: Dict[str, Any],
) -> Dict[str, Any]:

    run_dir = (
        RUN_ROOT
        / "fold_{}".format(
            fold
        )
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_ckpt = (
        run_dir
        / "best_checkpoint.pt"
    )

    last_ckpt = (
        run_dir
        / "last_checkpoint.pt"
    )

    history_csv = (
        run_dir
        / "epoch_history.csv"
    )

    running_json = (
        run_dir
        / "TRAINING_RUNNING.json"
    )

    protocol_snapshot = (
        run_dir
        / "fold_protocol_snapshot.json"
    )

    complete_json = (
        run_dir
        / "FOLD{}_TRAINING_COMPLETE.json".format(
            fold
        )
    )

    if complete_json.exists():
        complete = load_json(
            complete_json
        )

        if complete.get(
            "status"
        ) != "COMPLETE":
            raise RuntimeError(
                "Fold {} completion report is not COMPLETE.".format(
                    fold
                )
            )

        if (
            complete.get(
                "global_protocol_sha256"
            )
            != global_protocol[
                "global_protocol_sha256"
            ]
        ):
            raise RuntimeError(
                "Fold {} was completed under another protocol.".format(
                    fold
                )
            )

        print(
            "\nFold {} already COMPLETE — skipping.".format(
                fold
            )
        )

        print(
            "    Best epoch: {}".format(
                complete.get(
                    "best_epoch"
                )
            )
        )

        print(
            "    Best val mean foreground Dice: {:.6f}".format(
                float(
                    complete.get(
                        "best_validation_mean_foreground_dice"
                    )
                )
            )
        )

        return complete

    print(
        "\n"
        + "=" * 110
    )
    print(
        "TRACK-B FOLD {} — MIRAGE SEMANTIC B2/A0 TRAINING".format(
            fold
        )
    )
    print(
        "=" * 110
    )

    train_stems, train_split = resolve_split_role(
        fold,
        "train",
        available_stems,
    )

    val_stems, val_split = resolve_split_role(
        fold,
        "val",
        available_stems,
    )

    if set(
        train_stems
    ) & set(
        val_stems
    ):
        raise RuntimeError(
            "Fold {} TRAIN/VAL filename overlap detected.".format(
                fold
            )
        )

    print(
        "    Train images: {} PASS".format(
            len(train_stems)
        )
    )

    print(
        "    Val images:   {} PASS".format(
            len(val_stems)
        )
    )

    print(
        "    Train/val overlap: 0 PASS"
    )

    print(
        "    Outer test: NOT RESOLVED / NOT LOADED"
    )

    fold_protocol = fold_protocol_object(
        fold,
        global_protocol,
        train_split,
        val_split,
    )

    if protocol_snapshot.exists():
        existing = load_json(
            protocol_snapshot
        )

        if (
            existing.get(
                "fold_protocol_sha256"
            )
            != fold_protocol[
                "fold_protocol_sha256"
            ]
        ):
            raise RuntimeError(
                "Fold {} protocol snapshot mismatch.".format(
                    fold
                )
            )
    else:
        save_json(
            protocol_snapshot,
            fold_protocol,
        )

    print(
        "    Fold protocol SHA: {}".format(
            fold_protocol[
                "fold_protocol_sha256"
            ]
        )
    )

    set_seed(
        SEED
    )

    reset_cuda()

    model = build_model(
        step07,
        device,
    )

    optimizer = build_optimizer(
        model
    )

    resume = load_resume(
        last_ckpt,
        model,
        optimizer,
        device,
        fold,
        global_protocol[
            "global_protocol_sha256"
        ],
        fold_protocol[
            "fold_protocol_sha256"
        ],
    )

    start_epoch = resume[
        "start_epoch"
    ]

    best_epoch = resume[
        "best_epoch"
    ]

    best_metric = resume[
        "best_metric"
    ]

    epochs_without_improvement = resume[
        "epochs_without_improvement"
    ]

    print(
        "    Resume: {}".format(
            "YES"
            if resume["resumed"]
            else "NO"
        )
    )

    print(
        "    Start epoch: {}".format(
            start_epoch
        )
    )

    if resume["resumed"]:
        print(
            "    Historical checkpoint script SHA: {}".format(
                resume[
                    "historical_script_sha256"
                ]
            )
        )
        print(
            "    Resume driver scientific-source SHA: {}".format(
                EXPECTED_STEP10_SCIENTIFIC_SOURCE_SHA
            )
        )

    if start_epoch > MAX_EPOCHS:
        raise RuntimeError(
            "Fold {} checkpoint exceeds MAX_EPOCHS, "
            "but completion report is missing.".format(
                fold
            )
        )

    train_dataset = MIRAGESemanticDataset(
        train_stems,
        images,
        masks,
    )

    val_dataset = MIRAGESemanticDataset(
        val_stems,
        images,
        masks,
    )

    tloader = training_loader(
        train_dataset
    )

    vloader = validation_loader(
        val_dataset
    )

    invocation_start = (
        time.perf_counter()
    )

    stop_reason = "MAX_EPOCHS"
    last_completed_epoch = (
        start_epoch - 1
    )

    for epoch in range(
        start_epoch,
        MAX_EPOCHS + 1,
    ):
        print(
            "\n"
            + "-" * 110
        )

        print(
            "FOLD {} | EPOCH {:02d}/{:02d}".format(
                fold,
                epoch,
                MAX_EPOCHS,
            )
        )

        lrs = apply_epoch_lrs(
            optimizer,
            epoch,
        )

        print(
            "    LR encoder={:.3e} | decoder={:.3e}".format(
                lrs["encoder"],
                lrs["decoder"],
            )
        )

        reseed_training_loader(
            tloader,
            epoch,
        )

        reset_cuda()

        train_report = train_one_epoch(
            model,
            optimizer,
            tloader,
            device,
            fold,
            epoch,
        )

        val_report = validate_one_epoch(
            model,
            vloader,
            device,
            fold,
        )

        tr = train_report[
            "mean_losses"
        ]

        va = val_report[
            "mean_losses"
        ]

        vm = val_report[
            "native_fov_metrics"
        ]

        metric = float(
            vm[
                "mean_foreground_dice"
            ]
        )

        improved = (
            metric
            > best_metric
            + EARLY_STOP_MIN_DELTA
        )

        if improved:
            best_metric = metric
            best_epoch = epoch
            epochs_without_improvement = 0

            save_checkpoint(
                best_ckpt,
                model,
                optimizer,
                fold,
                epoch,
                best_epoch,
                best_metric,
                epochs_without_improvement,
                global_protocol[
                    "global_protocol_sha256"
                ],
                fold_protocol[
                    "fold_protocol_sha256"
                ],
            )
        else:
            epochs_without_improvement += 1

        cls = vm["classes"]

        row = {
            "fold": fold,
            "epoch": epoch,
            "encoder_lr": lrs[
                "encoder"
            ],
            "decoder_lr": lrs[
                "decoder"
            ],
            "train_total": tr[
                "total"
            ],
            "train_ce": tr[
                "cross_entropy"
            ],
            "train_dice_loss": tr[
                "foreground_dice_loss"
            ],
            "val_total": va[
                "total"
            ],
            "val_ce": va[
                "cross_entropy"
            ],
            "val_dice_loss": va[
                "foreground_dice_loss"
            ],
            "val_mean_fg_dice": vm[
                "mean_foreground_dice"
            ],
            "val_mean_fg_iou": vm[
                "mean_foreground_iou"
            ],
            "val_SRF_dice": cls[
                "SRF"
            ][
                "dice"
            ],
            "val_IRF_dice": cls[
                "IRF"
            ][
                "dice"
            ],
            "val_ELM_dice": cls[
                "ELM"
            ][
                "dice"
            ],
            "val_EZ_dice": cls[
                "EZ"
            ][
                "dice"
            ],
            "train_seconds": train_report[
                "elapsed_seconds"
            ],
            "val_seconds": val_report[
                "elapsed_seconds"
            ],
            "peak_allocated_gb": train_report[
                "memory"
            ][
                "peak_allocated_gb"
            ],
            "improved": int(
                improved
            ),
        }

        upsert_history_row(
            history_csv,
            row,
        )

        save_checkpoint(
            last_ckpt,
            model,
            optimizer,
            fold,
            epoch,
            best_epoch,
            best_metric,
            epochs_without_improvement,
            global_protocol[
                "global_protocol_sha256"
            ],
            fold_protocol[
                "fold_protocol_sha256"
            ],
        )

        last_completed_epoch = epoch

        save_json(
            running_json,
            {
                "status": "RUNNING",
                "fold": fold,
                "epoch_completed": epoch,
                "best_epoch": best_epoch,
                "best_validation_mean_foreground_dice": best_metric,
                "epochs_without_improvement": epochs_without_improvement,
                "global_protocol_sha256": global_protocol[
                    "global_protocol_sha256"
                ],
                "fold_protocol_sha256": fold_protocol[
                    "fold_protocol_sha256"
                ],
                "outer_test_resolved": False,
                "outer_test_evaluated": False,
                "last_epoch": {
                    "train": train_report,
                    "validation": val_report,
                },
            },
        )

        print(
            "\n    Fold {} epoch {:02d} summary".format(
                fold,
                epoch,
            )
        )

        print(
            "      train loss={:.5f} | CE={:.5f} | DiceLoss={:.5f}".format(
                tr["total"],
                tr["cross_entropy"],
                tr[
                    "foreground_dice_loss"
                ],
            )
        )

        print(
            "      val   loss={:.5f} | mean Dice={:.2f}% | mean IoU={:.2f}%".format(
                va["total"],
                100.0 * vm[
                    "mean_foreground_dice"
                ],
                100.0 * vm[
                    "mean_foreground_iou"
                ],
            )
        )

        print(
            "      SRF={:.2f}% IRF={:.2f}% ELM={:.2f}% EZ={:.2f}%".format(
                100.0 * cls[
                    "SRF"
                ][
                    "dice"
                ],
                100.0 * cls[
                    "IRF"
                ][
                    "dice"
                ],
                100.0 * cls[
                    "ELM"
                ][
                    "dice"
                ],
                100.0 * cls[
                    "EZ"
                ][
                    "dice"
                ],
            )
        )

        print(
            "      best epoch={} | best val mean Dice={:.2f}% | {}".format(
                best_epoch,
                100.0 * best_metric,
                (
                    "IMPROVED"
                    if improved
                    else "no improvement {}/{}".format(
                        epochs_without_improvement,
                        EARLY_STOP_PATIENCE,
                    )
                ),
            )
        )

        if (
            epoch >= MIN_EPOCHS_BEFORE_STOP
            and epochs_without_improvement
            >= EARLY_STOP_PATIENCE
        ):
            stop_reason = (
                "EARLY_STOP_PATIENCE"
            )

            print(
                "\n    Early stopping: "
                "no validation mean-Dice improvement for "
                "{} epochs.".format(
                    EARLY_STOP_PATIENCE
                )
            )

            break

    elapsed = (
        time.perf_counter()
        - invocation_start
    )

    if not best_ckpt.exists():
        raise RuntimeError(
            "Fold {} finished without a best checkpoint.".format(
                fold
            )
        )

    best_state = torch_load_checkpoint(
        best_ckpt,
        device,
    )

    if int(
        best_state[
            "best_epoch"
        ]
    ) != int(
        best_epoch
    ):
        raise RuntimeError(
            "Fold {} best-checkpoint epoch mismatch.".format(
                fold
            )
        )

    complete = {
        "status": "COMPLETE",
        "model": (
            "MIRAGE-Base + standard semantic ConvNeXt decoder B2/A0"
        ),
        "script_revision": SCRIPT_REVISION,
        "script_sha256": current_script_sha(),
        "resume_driver_scientific_source_sha256": (
            EXPECTED_STEP10_SCIENTIFIC_SOURCE_SHA
        ),
        "global_protocol_sha256": global_protocol[
            "global_protocol_sha256"
        ],
        "fold_protocol_sha256": fold_protocol[
            "fold_protocol_sha256"
        ],
        "scientific_design_sha256": EXPECTED_DESIGN_SHA,
        "ablation_plan_sha256": EXPECTED_ABLATION_PLAN_SHA,
        "mirage_wrapper_sha256": EXPECTED_MIRAGE_WRAPPER_SHA,
        "step07_source_bundle_sha256": EXPECTED_SOURCE_BUNDLE_SHA,
        "trackb_split_master_sha256": CANONICAL_TRACKB_SPLIT_SHA,
        "fold": fold,
        "train_images": len(
            train_stems
        ),
        "val_images": len(
            val_stems
        ),
        "train_stem_sha256": train_split[
            "stem_sha256"
        ],
        "val_stem_sha256": val_split[
            "stem_sha256"
        ],
        "precision": PRECISION,
        "physical_batch_size": PHYSICAL_BATCH,
        "augmentation": AUGMENTATION_POLICY,
        "checkpoint_metric": CHECKPOINT_METRIC,
        "best_epoch": int(
            best_state["epoch"]
        ),
        "best_validation_mean_foreground_dice": float(
            best_state[
                "best_metric"
            ]
        ),
        "last_completed_epoch": int(
            last_completed_epoch
        ),
        "stop_reason": stop_reason,
        "elapsed_seconds_current_invocation": float(
            elapsed
        ),
        "best_checkpoint": str(
            best_ckpt
        ),
        "last_checkpoint": str(
            last_ckpt
        ),
        "history_csv": str(
            history_csv
        ),
        "outer_test_resolved": False,
        "outer_test_evaluated": False,
        "best_weights_preserved": True,
    }

    save_json(
        complete_json,
        complete,
    )

    append_project_log_once(
        "<!-- STEP10_MIRAGE_SEMANTIC_B2_FOLD{}_COMPLETE -->".format(
            fold
        ),
        (
            "Prespecified MIRAGE semantic B2/A0 Fold {} training complete. "
            "Best epoch {}; best validation mean foreground Dice {:.4f}. "
            "Released Image_Seg masks, native 570x380 padded to 576x384, "
            "full MIRAGE fine-tuning, no augmentation. Outer test remained "
            "unresolved and unevaluated. Global protocol SHA {}."
        ).format(
            fold,
            complete[
                "best_epoch"
            ],
            complete[
                "best_validation_mean_foreground_dice"
            ],
            global_protocol[
                "global_protocol_sha256"
            ],
        ),
    )

    print(
        "\n"
        + "-" * 110
    )

    print(
        "FOLD {} TRAINING COMPLETE".format(
            fold
        )
    )

    print(
        "-" * 110
    )

    print(
        "Best epoch: {}".format(
            complete[
                "best_epoch"
            ]
        )
    )

    print(
        "Best validation mean foreground Dice: {:.2f}%".format(
            100.0 * complete[
                "best_validation_mean_foreground_dice"
            ]
        )
    )

    print(
        "Stop reason: {}".format(
            stop_reason
        )
    )

    print(
        "Outer test evaluated: NO"
    )

    print(
        "Best weights preserved: YES"
    )

    print(
        "Completion report:"
    )

    print(
        complete_json
    )

    del best_state
    del optimizer
    del model
    del tloader
    del vloader
    del val_dataset
    del train_dataset

    reset_cuda()

    return complete


# =============================================================================
# MAIN
# =============================================================================

def main():
    print(
        "\n"
        + "=" * 110
    )

    print(
        "RVO-ME STEP 10: MIRAGE SEMANTIC B2/A0 — FIVE-FOLD TRAINING"
    )

    print(
        "SCRIPT REVISION: {}".format(
            SCRIPT_REVISION
        )
    )

    print(
        "=" * 110
    )

    print(
        "\n[0/7] Verifying Step-10 scientific source bundle..."
    )

    current_scientific_source_sha = named_source_bundle_sha(
        Path(__file__).resolve(),
        STEP10_SCIENTIFIC_SOURCE_NAMES,
    )

    if (
        current_scientific_source_sha
        != EXPECTED_STEP10_SCIENTIFIC_SOURCE_SHA
    ):
        raise RuntimeError(
            "Step-10 scientific source-bundle SHA mismatch. "
            "Resume refused before training."
        )

    print(
        "    Step-10 scientific source SHA: PASS"
    )

    print(
        "\n[1/7] Verifying this control was prespecified before GeoRCP test..."
    )

    preplanned = verify_preplanned_control()

    print(
        "    Frozen ablation plan: PASS"
    )

    print(
        "    B2 / A0 prespecification: PASS"
    )

    print(
        "\n[2/7] Loading exact frozen Step-07 MIRAGE implementation..."
    )

    step07, source_sha = load_exact_step07()

    print(
        "    Step-07 source-bundle SHA: PASS"
    )

    print(
        "\n[3/7] Verifying RTX 4080 / Python / PyTorch / wrapper..."
    )

    environment = verify_environment(
        step07,
        source_sha,
    )

    print(
        "    GPU: {}".format(
            environment["gpu"]
        )
    )

    print(
        "    VRAM: {:.3f} GB".format(
            environment["vram_gb"]
        )
    )

    print(
        "    Python/PyTorch/CUDA/BF16: PASS"
    )

    print(
        "    MIRAGE wrapper SHA: PASS"
    )
    print(
        "    Step-07 desktop MIRAGE path binding: PASS"
    )

    print(
        "    Fast I/O: workers={} | pin_memory={} | persistent={} | prefetch={} | non_blocking={}".format(
            NUM_WORKERS,
            PIN_MEMORY,
            PERSISTENT_WORKERS,
            PREFETCH_FACTOR,
            NON_BLOCKING_TRANSFER,
        )
    )

    print(
        "\n[4/7] Auditing released Image_Seg pairs and masks..."
    )

    images, masks = discover_image_mask_pairs()

    print(
        "    Image/mask pairs: {} PASS".format(
            len(images)
        )
    )

    mask_audit = audit_released_masks(
        masks
    )

    save_json(
        RUN_ROOT
        / "RELEASED_MASK_AUDIT.json",
        mask_audit,
    )

    print(
        "    Released mask class values 0..4: PASS"
    )

    print(
        "\n[5/7] Freezing/verifying B2/A0 training protocol..."
    )

    global_protocol = global_protocol_object(
        preplanned,
        environment,
    )

    global_protocol = freeze_or_verify_global_protocol(
        global_protocol
    )

    available_stems = sorted(
        images
    )

    fold1_train, fold1_train_report = resolve_split_role(
        1,
        "train",
        available_stems,
    )

    print(
        "\n[6/7] Running/reusing train-only BF16 memory preflight..."
    )

    run_preflight(
        step07,
        images,
        masks,
        fold1_train,
        torch.device("cuda"),
        global_protocol,
    )

    print(
        "\n[7/7] Training all five patient-disjoint folds..."
    )

    print(
        "    Outer-test resolution path: DISABLED"
    )

    print(
        "    Folds queued: 1, 2, 3, 4, 5"
    )

    device = torch.device(
        "cuda"
    )

    results = {}

    for fold in FOLDS_TO_TRAIN:
        results[str(fold)] = train_fold(
            fold,
            step07,
            images,
            masks,
            available_stems,
            device,
            global_protocol,
        )

        update_manifest(
            global_protocol[
                "global_protocol_sha256"
            ]
        )

    manifest = update_manifest(
        global_protocol[
            "global_protocol_sha256"
        ]
    )

    if (
        manifest["status"]
        != "ALL_FIVE_FOLDS_TRAINING_COMPLETE"
    ):
        raise RuntimeError(
            "Five-fold B2/A0 training manifest is incomplete."
        )

    append_project_log_once(
        "<!-- STEP10_MIRAGE_SEMANTIC_B2_FIVEFOLD_COMPLETE -->",
        (
            "Prespecified MIRAGE semantic B2/A0 five-fold training complete. "
            "All five best checkpoints preserved. Outer tests remained "
            "unresolved/unevaluated. Global protocol SHA {}."
        ).format(
            global_protocol[
                "global_protocol_sha256"
            ]
        ),
    )

    print(
        "\n"
        + "=" * 110
    )

    print(
        "STEP 10 MIRAGE SEMANTIC B2/A0 TRAINING COMPLETE"
    )

    print(
        "=" * 110
    )

    for fold in FOLDS_TO_TRAIN:
        report = results[
            str(fold)
        ]

        print(
            "Fold {} | best epoch {:>2} | val mean Dice {:.2f}%".format(
                fold,
                int(
                    report[
                        "best_epoch"
                    ]
                ),
                100.0 * float(
                    report[
                        "best_validation_mean_foreground_dice"
                    ]
                ),
            )
        )

    print(
        "\nOuter test resolved/evaluated: NO / NO"
    )

    print(
        "Do NOT evaluate B2/A0 outer tests until its "
        "evaluation protocol is frozen."
    )

    print(
        "\nManifest:"
    )

    print(
        FIVEFOLD_MANIFEST
    )


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
