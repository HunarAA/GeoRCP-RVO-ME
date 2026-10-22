# =============================================================================
# RVO-ME STEP 08 DEFINITIVE
# GeoRCP-MIRAGE — FOLD-1 TRAINING
#
# This is the FIRST DEFINITIVE proposed-model training run.
#
# Scientific policy:
#   - Track-B patient-disjoint Fold 1
#   - TRAIN + VAL only during fitting/model selection
#   - outer TEST is NEVER resolved or evaluated in this script
#   - best checkpoint selected by minimum validation RAW group-sum loss
#   - best checkpoint is preserved for later one-time Step-09 evaluation
#   - exact geometry_v3 targets
#   - exact frozen GeoRCP-MIRAGE model/loss implementation
#   - RTX 4080, BF16, physical batch 7
#
# Engineering:
#   - AdamW
#   - encoder LR 1e-5
#   - GeoRCP/head + uncertainty LR 1e-4
#   - weight decay 0.01 (uncertainty parameters excluded)
#   - 5-epoch linear warmup
#   - cosine decay to 1% of base LR
#   - max 60 epochs
#   - early stopping patience 12
#   - minimum 20 epochs before early stop
#   - exact horizontal flip p=0.5 only
#   - epoch-boundary deterministic resume
#
# IMPORTANT:
#   Do not modify hyperparameters after this run begins. If a scientific/code
#   correction is required, stop and version the protocol rather than silently
#   changing an in-progress experiment.
# =============================================================================

from __future__ import annotations

import ast
import csv
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
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# IDENTITY / PATHS
# =============================================================================

SCRIPT_REVISION = "STEP08_DEFINITIVE_FOLD1_V3_NOAUG_20260819"

ROOT = Path(r"D:\RVO_ME")
PROJECT = ROOT / "project"

STEP07_SCRIPT = (
    PROJECT
    / "scripts"
    / "07_georcp_preflight_v2_ampfix_py38compat.py"
)

SPLIT_ROOT = (
    PROJECT
    / "splits"
    / "patient_5fold_v1"
)

TARGET_ROOT = (
    PROJECT
    / "targets"
    / "geometry_v3"
)

TARGET_DIR = TARGET_ROOT / "targets"
G3_COMPLETE = TARGET_ROOT / "GEOMETRY_V3_COMPLETE.json"

MIRAGE_WRAPPER = (
    PROJECT
    / "foundation_models"
    / "mirage_base_preflight_v1"
    / "official_mirage_hf.py"
)

RTX4080_PREFLIGHT = (
    PROJECT
    / "preflight"
    / "step07_georcp_v1"
    / "NVIDIA_GeForce_RTX_4080"
    / "STEP07_PREFLIGHT_COMPLETE.json"
)

RTX4080_REVALIDATION = (
    PROJECT
    / "training"
    / "georcp_v1"
    / "development_rtx4080_revalidation"
    / "STEP08_RTX4080_REVALIDATION_COMPLETE.json"
)

DESIGN_DIR = (
    PROJECT
    / "design"
    / "geometry_aware_rcp_v1"
)

ENGINEERING_PROTOCOL = (
    DESIGN_DIR
    / "RTX4080_ENGINEERING_FROZEN_v2.json"
)

RUN_DIR = (
    PROJECT
    / "training"
    / "georcp_v1"
    / "definitive_v2"
    / "fold_1"
)

PROJECT_LOG = PROJECT / "PROJECT_LOG.md"

BEST_CKPT = RUN_DIR / "best_checkpoint.pt"
LAST_CKPT = RUN_DIR / "last_checkpoint.pt"
HISTORY_CSV = RUN_DIR / "epoch_history.csv"
RUNNING_JSON = RUN_DIR / "TRAINING_RUNNING.json"
COMPLETE_JSON = RUN_DIR / "FOLD1_TRAINING_COMPLETE.json"
PROTOCOL_SNAPSHOT = RUN_DIR / "protocol_snapshot.json"


# =============================================================================
# FROZEN SCIENTIFIC IDS
# =============================================================================

EXPECTED_DESIGN_SHA = (
    "9f1586d102523009443e2b88dca54a9ca8fa5c983f08fdde837184d436a95801"
)

EXPECTED_V1_1_SHA = (
    "2a5e6e487ad7243be2cb54058c1515611cd22fad99494ddff3e7a135642b847c"
)

EXPECTED_G3_SHA = (
    "521859acb4c5389eb4244298f755420f1339b7e4c4bedcfac7c64ddcfe98ce59"
)

EXPECTED_IMPLEMENTATION_SPEC_SHA = (
    "b5e59901eb67190d5924698ca53165d30d1bccf45028a7454d9ca98147149422"
)

EXPECTED_MIRAGE_WRAPPER_SHA = (
    "07fa63e2766b67dbc0c07f0d9d29db1562f707425ef2fbf90a688f62a94e6ede"
)

# Semantic SHA of the exact model/loss/data/optimizer source definitions in
# 07_georcp_preflight_v2_ampfix_py38compat.py. This is stronger than the
# architecture/config implementation-spec SHA because it fingerprints the
# actual source definitions used for definitive training.
EXPECTED_SOURCE_BUNDLE_SHA = (
    "04e7da4fe311b756ad375a6c241530d0572a3cf1b9c71a70af2e7e7e24524412"
)

CANONICAL_TRACKB_SPLIT_SHA = (
    "7e7c2ddf51054bf998b2d750b50c74983b9bc5b751b762213bc96343ffe01346"
)

EXPECTED_IMAGES = 3012
EXPECTED_F1_TRAIN = 2110
EXPECTED_F1_VAL = 303


# =============================================================================
# DEFINITIVE TRAINING PROTOCOL — FROZEN BY THIS SCRIPT
# =============================================================================

SEED = 20260815

PRECISION = "BF16"
PHYSICAL_BATCH = 7
VAL_BATCH = 7
GRAD_ACCUM_STEPS = 1
EFFECTIVE_BATCH = PHYSICAL_BATCH * GRAD_ACCUM_STEPS

OPTIMIZER = "AdamW"
ENCODER_LR = 1e-5
HEAD_LR = 1e-4
UNCERTAINTY_LR = 1e-4
WEIGHT_DECAY = 1e-2

MAX_EPOCHS = 60
WARMUP_EPOCHS = 5
MIN_LR_FACTOR = 0.01

EARLY_STOP_PATIENCE = 12
EARLY_STOP_MIN_DELTA = 1e-3
MIN_EPOCHS_BEFORE_STOP = 20

CHECKPOINT_METRIC = "val_raw_group_sum"
CHECKPOINT_MODE = "min"

HFLIP_PROB = 0.0
AUGMENTATION_POLICY = (
    "NONE. No horizontal/vertical flip, resize, crop, rotation, or "
    "photometric augmentation. Horizontal flip was removed because the exact "
    "pixel-space transform can create stride-2 HF CenterNet cell collisions "
    "after floor-cell quantization, making some augmented point targets "
    "unrepresentable by the frozen one-offset-per-cell decoder."
)

NUM_WORKERS = 0
LOG_EVERY_STEPS = 50

# Native/padded geometry from the frozen design.
W = 576
H = 384
HF_STRIDE = 2
HH = H // HF_STRIDE
HW = W // HF_STRIDE
HF_SIGMA = 1.0

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
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def script_sha() -> str:
    return sha_file(
        Path(__file__).resolve()
    )


def reset_cuda() -> None:
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def cuda_memory() -> Dict[str, float]:
    g = float(1024 ** 3)

    return {
        "allocated_gb":
            float(
                torch.cuda.memory_allocated()
                / g
            ),

        "reserved_gb":
            float(
                torch.cuda.memory_reserved()
                / g
            ),

        "peak_allocated_gb":
            float(
                torch.cuda.max_memory_allocated()
                / g
            ),

        "peak_reserved_gb":
            float(
                torch.cuda.max_memory_reserved()
                / g
            ),
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
            "Could not import {}".format(path)
        )

    mod = importlib.util.module_from_spec(
        spec
    )

    sys.modules[name] = mod
    spec.loader.exec_module(mod)

    return mod


def source_bundle_sha(
    path: Path,
) -> str:
    """
    Hash only the source definitions that determine the exact MIRAGE loading,
    GeoRCP model, final loss, sample loading, and optimizer grouping.
    """

    text = path.read_text(
        encoding="utf-8"
    ).replace(
        "\r\n",
        "\n",
    )

    tree = ast.parse(
        text,
        filename=str(path),
    )

    lines = text.splitlines()
    wanted = set(
        SOURCE_BUNDLE_NAMES
    )

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

        if not hasattr(
            node,
            "end_lineno",
        ):
            raise RuntimeError(
                "Python AST does not provide end_lineno; cannot verify source bundle."
            )

        segment = "\n".join(
            lines[
                node.lineno - 1:
                node.end_lineno
            ]
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
        wanted
        - set(
            found
        )
    )

    if missing:
        raise RuntimeError(
            "Missing source definitions: "
            + ", ".join(
                missing
            )
        )

    bundle = "\n".join(
        chunks
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        bundle
    ).hexdigest()


def append_project_log_once(
    marker: str,
    text: str,
) -> None:
    if not PROJECT_LOG.exists():
        return

    current = PROJECT_LOG.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    if marker in current:
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
# EXACT STEP-07 IMPLEMENTATION
# =============================================================================

def load_exact_step07():
    if not STEP07_SCRIPT.exists():
        raise FileNotFoundError(
            STEP07_SCRIPT
        )

    actual_source_sha = source_bundle_sha(
        STEP07_SCRIPT
    )

    if actual_source_sha != EXPECTED_SOURCE_BUNDLE_SHA:
        raise RuntimeError(
            "GeoRCP source-bundle SHA mismatch.\n"
            "Expected: {}\n"
            "Found:    {}\n"
            "Refusing definitive training with changed model/loss source.".format(
                EXPECTED_SOURCE_BUNDLE_SHA,
                actual_source_sha,
            )
        )

    mod = module_from_file(
        "georcp_step07_definitive",
        STEP07_SCRIPT,
    )

    required = [
        "build_model",
        "load_sample",
        "images_by_stem",
        "implementation_spec",
    ]

    missing = [
        name
        for name in required
        if not hasattr(
            mod,
            name,
        )
    ]

    if missing:
        raise RuntimeError(
            "Step-07 implementation missing: "
            + ", ".join(
                missing
            )
        )

    spec = mod.implementation_spec()

    if (
        spec.get(
            "implementation_spec_sha256"
        )
        != EXPECTED_IMPLEMENTATION_SPEC_SHA
    ):
        raise RuntimeError(
            "Implementation-spec SHA mismatch."
        )

    return (
        mod,
        actual_source_sha,
    )


# =============================================================================
# FROZEN-STATE / HARDWARE VERIFICATION
# =============================================================================

def verify_environment_and_inputs(
    step07,
    source_sha: str,
) -> Dict[str, Any]:

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required."
        )

    gpu = torch.cuda.get_device_name(
        0
    )

    vram = (
        torch.cuda.get_device_properties(
            0
        ).total_memory
        / (
            1024 ** 3
        )
    )

    if "RTX 4080" not in gpu:
        raise RuntimeError(
            "Definitive Fold-1 protocol is frozen for RTX 4080; found {}.".format(
                gpu
            )
        )

    if not sys.version.startswith(
        "3.8"
    ):
        raise RuntimeError(
            "This frozen RTX-4080 run expects the validated Python 3.8 environment."
        )

    if torch.__version__ != "2.4.1+cu121":
        raise RuntimeError(
            "Expected the validated RTX-4080 PyTorch build "
            "2.4.1+cu121; found {}.".format(
                torch.__version__
            )
        )

    if str(torch.version.cuda) != "12.1":
        raise RuntimeError(
            "Expected the validated RTX-4080 CUDA build 12.1; found {}.".format(
                torch.version.cuda
            )
        )

    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "BF16 is not supported by this CUDA device/runtime."
        )

    if sha_file(
        MIRAGE_WRAPPER
    ) != EXPECTED_MIRAGE_WRAPPER_SHA:
        raise RuntimeError(
            "Frozen MIRAGE wrapper SHA mismatch."
        )

    g3 = load_json(
        G3_COMPLETE
    )

    if (
        g3.get(
            "geometry_v3_master_content_sha256"
        )
        != EXPECTED_G3_SHA
    ):
        raise RuntimeError(
            "geometry_v3 SHA mismatch."
        )

    target_files = list(
        TARGET_DIR.glob(
            "*.npz"
        )
    )

    if len(
        target_files
    ) != EXPECTED_IMAGES:
        raise RuntimeError(
            "Expected {} geometry_v3 target packs; found {}.".format(
                EXPECTED_IMAGES,
                len(
                    target_files
                ),
            )
        )

    pre = load_json(
        RTX4080_PREFLIGHT
    )

    if pre.get(
        "status"
    ) != "COMPLETE":
        raise RuntimeError(
            "RTX-4080 Step-07 preflight is not COMPLETE."
        )

    if (
        pre.get(
            "active_target_master_sha256"
        )
        != EXPECTED_G3_SHA
    ):
        raise RuntimeError(
            "RTX-4080 preflight target SHA mismatch."
        )

    if (
        pre.get(
            "implementation_spec_sha256"
        )
        != EXPECTED_IMPLEMENTATION_SPEC_SHA
    ):
        raise RuntimeError(
            "RTX-4080 preflight implementation SHA mismatch."
        )

    if (
        str(
            pre.get(
                "selected_preflight_precision",
                {},
            ).get(
                "precision",
                "",
            )
        ).lower()
        != "bf16"
    ):
        raise RuntimeError(
            "RTX-4080 preflight did not select BF16."
        )

    if int(
        pre.get(
            "largest_profiled_batch_pass",
            0,
        )
    ) < 8:
        raise RuntimeError(
            "RTX-4080 preflight did not pass physical batch 8."
        )

    if (
        pre.get(
            "mini_overfit",
            {},
        ).get(
            "status"
        )
        != "PASS"
    ):
        raise RuntimeError(
            "RTX-4080 mini-overfit did not PASS."
        )

    rev = load_json(
        RTX4080_REVALIDATION
    )

    if rev.get(
        "status"
    ) != "COMPLETE":
        raise RuntimeError(
            "RTX-4080 Step-08 revalidation is not COMPLETE."
        )

    if "RTX 4080" not in str(
        rev.get(
            "gpu",
            "",
        )
    ):
        raise RuntimeError(
            "Revalidation was not recorded on RTX 4080."
        )

    if str(
        rev.get(
            "precision",
            "",
        )
    ).upper() != "BF16":
        raise RuntimeError(
            "Revalidation precision mismatch."
        )

    if int(
        rev.get(
            "memory_probe",
            {},
        ).get(
            "selected_batch",
            -1,
        )
    ) != PHYSICAL_BATCH:
        raise RuntimeError(
            "Clean revalidation did not select physical batch {}.".format(
                PHYSICAL_BATCH
            )
        )

    if bool(
        rev.get(
            "outer_test_evaluated",
            True,
        )
    ):
        raise RuntimeError(
            "Revalidation unexpectedly reports outer-test evaluation."
        )

    if source_sha != EXPECTED_SOURCE_BUNDLE_SHA:
        raise RuntimeError(
            "Internal source SHA verification failure."
        )

    return {
        "python":
            sys.version,

        "pytorch":
            torch.__version__,

        "torch_cuda":
            torch.version.cuda,

        "gpu":
            gpu,

        "vram_gb":
            float(
                vram
            ),

        "bf16_supported":
            True,

        "mirage_wrapper_sha256":
            EXPECTED_MIRAGE_WRAPPER_SHA,

        "source_bundle_sha256":
            source_sha,
    }


# =============================================================================
# FOLD-1 SPLIT RESOLUTION — TRAIN / VAL ONLY
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


def resolve_fold1_role(
    role: str,
    available_stems: Sequence[str],
) -> List[str]:

    role = role.lower()

    if role not in {
        "train",
        "val",
    }:
        raise RuntimeError(
            "This definitive training script may resolve TRAIN/VAL only."
        )

    available = set(
        available_stems
    )

    found = set()

    files = sorted(
        p
        for p in SPLIT_ROOT.rglob("*")
        if p.is_file()
        and p.suffix.lower()
        in {
            ".txt",
            ".csv",
            ".json",
        }
    )

    fold_tokens = {
        "fold_1",
        "fold1",
        "/1/",
    }

    # TXT
    for path in files:
        low = str(
            path
        ).lower().replace(
            "\\",
            "/",
        )

        if not any(
            token in low
            for token in fold_tokens
        ):
            continue

        if (
            path.suffix.lower()
            == ".txt"
            and role
            in path.name.lower()
        ):
            text = path.read_text(
                encoding="utf-8",
                errors="ignore",
            )

            for token in re.split(
                r"[\s,;]+",
                text,
            ):
                stem = Path(
                    token.strip(
                        "\"'"
                    )
                ).stem

                if stem in available:
                    found.add(
                        stem
                    )

    # CSV
    for path in files:
        if path.suffix.lower() != ".csv":
            continue

        low = str(
            path
        ).lower().replace(
            "\\",
            "/",
        )

        if not any(
            token in low
            for token in fold_tokens
        ):
            continue

        try:
            with path.open(
                "r",
                encoding="utf-8-sig",
                newline="",
            ) as f:
                reader = csv.DictReader(
                    f
                )

                if reader.fieldnames is None:
                    continue

                cmap = {
                    c.lower():
                        c
                    for c in reader.fieldnames
                }

                stem_col = next(
                    (
                        cmap[
                            c
                        ]
                        for c in STEM_COLUMNS
                        if c in cmap
                    ),
                    None,
                )

                role_col = next(
                    (
                        cmap[
                            c
                        ]
                        for c in ROLE_COLUMNS
                        if c in cmap
                    ),
                    None,
                )

                if stem_col is None:
                    continue

                for row in reader:
                    if role_col is not None:
                        row_role = str(
                            row.get(
                                role_col,
                                "",
                            )
                        ).strip().lower()

                    else:
                        row_role = (
                            role
                            if role in path.name.lower()
                            else ""
                        )

                    if row_role != role:
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
                        found.add(
                            stem
                        )

        except Exception:
            continue

    # JSON
    def walk_json(
        obj: Any,
        inherited_fold: Any = None,
        inherited_role: Any = None,
    ) -> None:

        if isinstance(
            obj,
            dict,
        ):
            lower = {
                str(
                    key
                ).lower():
                    value
                for key, value in obj.items()
            }

            local_fold = inherited_fold
            local_role = inherited_role

            for key in (
                "fold",
                "fold_id",
                "outer_fold",
            ):
                if key in lower:
                    m = re.search(
                        r"(\d+)",
                        str(
                            lower[
                                key
                            ]
                        ),
                    )

                    local_fold = (
                        int(
                            m.group(
                                1
                            )
                        )
                        if m
                        else None
                    )

            for key in ROLE_COLUMNS:
                if key in lower:
                    local_role = str(
                        lower[
                            key
                        ]
                    ).strip().lower()

            stem_value = None

            for key in STEM_COLUMNS:
                if key in lower:
                    stem_value = lower[
                        key
                    ]
                    break

            if (
                local_fold == 1
                and local_role == role
                and stem_value is not None
            ):
                stem = Path(
                    str(
                        stem_value
                    )
                ).stem

                if stem in available:
                    found.add(
                        stem
                    )

            for value in obj.values():
                walk_json(
                    value,
                    local_fold,
                    local_role,
                )

        elif isinstance(
            obj,
            list,
        ):
            for value in obj:
                walk_json(
                    value,
                    inherited_fold,
                    inherited_role,
                )

    for path in files:
        if path.suffix.lower() != ".json":
            continue

        try:
            walk_json(
                load_json(
                    path
                )
            )

        except Exception:
            continue

    stems = sorted(
        found
    )

    expected = (
        EXPECTED_F1_TRAIN
        if role == "train"
        else EXPECTED_F1_VAL
    )

    if len(
        stems
    ) != expected:
        raise RuntimeError(
            "Fold-1 {} resolution mismatch: expected {}, found {}.".format(
                role,
                expected,
                len(
                    stems
                ),
            )
        )

    return stems


def hash_stem_list(
    stems: Sequence[str],
) -> str:
    raw = "\n".join(
        sorted(
            stems
        )
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        raw
    ).hexdigest()


# =============================================================================
# EXACT HORIZONTAL-FLIP AUGMENTATION
# =============================================================================

def deterministic_flip(
    stem: str,
    epoch: int,
) -> bool:
    payload = "{}|{}|{}".format(
        SEED,
        epoch,
        stem,
    ).encode(
        "utf-8"
    )

    value = int.from_bytes(
        hashlib.sha256(
            payload
        ).digest()[
            :8
        ],
        "big",
    )

    threshold = int(
        HFLIP_PROB
        * (
            2 ** 64
        )
    )

    return value < threshold


def rebuild_hf_heatmap(
    centers: List[Tuple[int, int]],
    dtype: torch.dtype,
) -> torch.Tensor:
    hm = torch.zeros(
        (
            HH,
            HW,
        ),
        dtype=torch.float32,
    )

    radius = max(
        1,
        int(
            math.ceil(
                4
                * HF_SIGMA
            )
        ),
    )

    for cy, cx in centers:
        x0 = max(
            0,
            cx - radius,
        )

        x1 = min(
            HW,
            cx + radius + 1,
        )

        y0 = max(
            0,
            cy - radius,
        )

        y1 = min(
            HH,
            cy + radius + 1,
        )

        xs = torch.arange(
            x0,
            x1,
            dtype=torch.float32,
        )

        ys = torch.arange(
            y0,
            y1,
            dtype=torch.float32,
        )

        yy, xx = torch.meshgrid(
            ys,
            xs,
            indexing="ij",
        )

        g = torch.exp(
            -(
                (
                    xx - float(
                        cx
                    )
                ) ** 2
                + (
                    yy - float(
                        cy
                    )
                ) ** 2
            )
            / (
                2
                * HF_SIGMA
                * HF_SIGMA
            )
        )

        # geometry_v3 stores heatmaps in float16; reproduce that quantization.
        g = g.to(
            torch.float16
        ).to(
            torch.float32
        )

        hm[
            y0:y1,
            x0:x1,
        ] = torch.maximum(
            hm[
                y0:y1,
                x0:x1,
            ],
            g,
        )

    return hm.to(
        dtype
    )


def horizontal_flip_sample(
    sample: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Exact left-right flip for every geometry_v3 target.

    HF requires more than tensor reversal because the CenterNet offset is
    defined relative to floor(full_coordinate / stride). We reconstruct the
    continuous x coordinate, mirror it in the padded 576-pixel frame, and
    recompute the floor cell + fractional offset.
    """

    out = {
        key:
            value.clone()
        for key, value in sample.items()
    }

    # All regular image/region/curve tensors: reverse width.
    for key in (
        "image",
        "region_class",
        "region_valid",
        "region_boundary",
        "region_boundary_valid",
        "curve_centerline",
        "curve_distance",
        "curve_presence",
        "curve_column_valid",
        "hf_valid",
    ):
        out[
            key
        ] = sample[
            key
        ].flip(
            -1
        )

    old_mask = sample[
        "hf_offset_mask"
    ]

    old_offset = sample[
        "hf_offset"
    ]

    ys, xs = torch.where(
        old_mask
        > 0
    )

    new_mask = torch.zeros_like(
        old_mask
    )

    new_offset = torch.zeros_like(
        old_offset
    )

    centers = []

    mirror_constant = (
        float(
            W - 1
        )
        / float(
            HF_STRIDE
        )
    )

    for cy_t, cx_t in zip(
        ys.tolist(),
        xs.tolist(),
    ):
        cy = int(
            cy_t
        )

        cx = int(
            cx_t
        )

        dx = float(
            old_offset[
                0,
                cy,
                cx,
            ]
        )

        dy = float(
            old_offset[
                1,
                cy,
                cx,
            ]
        )

        u = float(
            cx
        ) + dx

        u2 = (
            mirror_constant
            - u
        )

        cx2 = int(
            math.floor(
                u2
                + 1e-9
            )
        )

        dx2 = (
            u2
            - float(
                cx2
            )
        )

        cy2 = cy
        dy2 = dy

        if not (
            0 <= cx2 < HW
            and 0 <= cy2 < HH
        ):
            raise RuntimeError(
                "HF horizontal flip generated out-of-range cell."
            )

        if float(
            new_mask[
                cy2,
                cx2,
            ]
        ) > 0:
            raise RuntimeError(
                "HF horizontal flip generated a stride-2 collision."
            )

        new_mask[
            cy2,
            cx2,
        ] = 1.0

        new_offset[
            0,
            cy2,
            cx2,
        ] = dx2

        new_offset[
            1,
            cy2,
            cx2,
        ] = dy2

        centers.append(
            (
                cy2,
                cx2,
            )
        )

    out[
        "hf_offset_mask"
    ] = new_mask

    out[
        "hf_offset"
    ] = new_offset

    out[
        "hf_heatmap"
    ] = rebuild_hf_heatmap(
        centers,
        sample[
            "hf_heatmap"
        ].dtype,
    )

    if int(
        new_mask.sum().item()
    ) != int(
        old_mask.sum().item()
    ):
        raise RuntimeError(
            "HF count changed during horizontal flip."
        )

    if centers:
        cys = torch.tensor(
            [
                p[
                    0
                ]
                for p in centers
            ],
            dtype=torch.long,
        )

        cxs = torch.tensor(
            [
                p[
                    1
                ]
                for p in centers
            ],
            dtype=torch.long,
        )

        center_values = out[
            "hf_heatmap"
        ][
            cys,
            cxs,
        ]

        if not torch.allclose(
            center_values,
            torch.ones_like(
                center_values
            ),
            atol=1e-3,
            rtol=0.0,
        ):
            raise RuntimeError(
                "HF flipped heatmap center is not 1."
            )

    return out


def augmentation_self_check(
    step07,
    images: Dict[str, Path],
    train_stems: Sequence[str],
) -> Dict[str, Any]:

    selected = [
        train_stems[
            0
        ],
        train_stems[
            len(
                train_stems
            )
            // 2
        ],
        train_stems[
            -1
        ],
    ]

    results = []

    for stem in selected:
        sample = step07.load_sample(
            stem,
            images[
                stem
            ],
            TARGET_DIR,
        )

        flipped = horizontal_flip_sample(
            sample
        )

        restored = horizontal_flip_sample(
            flipped
        )

        exact_keys = [
            "region_class",
            "region_valid",
            "region_boundary",
            "region_boundary_valid",
            "curve_centerline",
            "curve_distance",
            "curve_presence",
            "curve_column_valid",
            "hf_offset_mask",
            "hf_valid",
        ]

        for key in exact_keys:
            if not torch.equal(
                sample[
                    key
                ],
                restored[
                    key
                ],
            ):
                raise RuntimeError(
                    "{}: double horizontal flip failed exact check for {}.".format(
                        stem,
                        key,
                    )
                )

        if not torch.allclose(
            sample[
                "image"
            ],
            restored[
                "image"
            ],
            atol=0.0,
            rtol=0.0,
        ):
            raise RuntimeError(
                "{}: image double horizontal flip mismatch.".format(
                    stem
                )
            )

        positive = sample[
            "hf_offset_mask"
        ] > 0

        if positive.any():
            if not torch.allclose(
                sample[
                    "hf_offset"
                ][
                    :,
                    positive,
                ],
                restored[
                    "hf_offset"
                ][
                    :,
                    positive,
                ],
                atol=1e-5,
                rtol=0.0,
            ):
                raise RuntimeError(
                    "{}: HF offsets failed double-flip check.".format(
                        stem
                    )
                )

        # Rebuilt heatmap may differ at tiny floating-point level; tolerance
        # corresponds to the geometry_v3 float16 target precision.
        if not torch.allclose(
            sample[
                "hf_heatmap"
            ],
            restored[
                "hf_heatmap"
            ],
            atol=1e-3,
            rtol=0.0,
        ):
            raise RuntimeError(
                "{}: HF heatmap failed double-flip check.".format(
                    stem
                )
            )

        results.append(
            {
                "stem":
                    stem,

                "hf_points":
                    int(
                        sample[
                            "hf_offset_mask"
                        ].sum().item()
                    ),
            }
        )

    return {
        "status":
            "PASS",

        "samples":
            results,
    }


# =============================================================================
# DATASET / DATALOADER
# =============================================================================

class GeoRCPDataset(
    Dataset
):
    def __init__(
        self,
        stems: Sequence[str],
        images: Dict[str, Path],
        step07,
        augment: bool,
    ):
        self.stems = list(
            stems
        )

        self.images = images

        self.step07 = step07
        self.augment = augment
        self.epoch = 0

    def set_epoch(
        self,
        epoch: int,
    ) -> None:
        self.epoch = int(
            epoch
        )

    def __len__(
        self,
    ) -> int:
        return len(
            self.stems
        )

    def __getitem__(
        self,
        index: int,
    ):
        stem = self.stems[
            index
        ]

        sample = self.step07.load_sample(
            stem,
            self.images[
                stem
            ],
            TARGET_DIR,
        )

        if (
            self.augment
            and deterministic_flip(
                stem,
                self.epoch,
            )
        ):
            sample = horizontal_flip_sample(
                sample
            )

        sample[
            "_stem"
        ] = stem

        return sample


def collate_fn(
    items: List[Dict[str, Any]],
):
    stems = [
        item[
            "_stem"
        ]
        for item in items
    ]

    keys = [
        key
        for key in items[
            0
        ].keys()
        if key != "_stem"
    ]

    batch = {
        key:
            torch.stack(
                [
                    item[
                        key
                    ]
                    for item in items
                ],
                dim=0,
            )
        for key in keys
    }

    batch[
        "_stems"
    ] = stems

    return batch


def move_batch(
    batch: Dict[str, Any],
    device: torch.device,
):
    return {
        key:
            (
                value.to(
                    device,
                    non_blocking=False,
                )
                if torch.is_tensor(
                    value
                )
                else value
            )
        for key, value in batch.items()
    }


def train_loader_for_epoch(
    dataset: GeoRCPDataset,
    epoch: int,
) -> DataLoader:
    dataset.set_epoch(
        epoch
    )

    generator = torch.Generator()

    generator.manual_seed(
        SEED
        + int(
            epoch
        )
    )

    return DataLoader(
        dataset,
        batch_size=PHYSICAL_BATCH,
        shuffle=True,
        generator=generator,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        drop_last=False,
        collate_fn=collate_fn,
    )


def val_loader(
    dataset: GeoRCPDataset,
) -> DataLoader:
    dataset.set_epoch(
        0
    )

    return DataLoader(
        dataset,
        batch_size=VAL_BATCH,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        drop_last=False,
        collate_fn=collate_fn,
    )


# =============================================================================
# OPTIMIZER / LR SCHEDULE
# =============================================================================

def build_optimizer(
    model,
    loss_module,
):
    enc = [
        p
        for p in model.encoder.parameters()
        if p.requires_grad
    ]

    enc_ids = {
        id(
            p
        )
        for p in enc
    }

    other = [
        p
        for p in model.parameters()
        if (
            p.requires_grad
            and id(
                p
            )
            not in enc_ids
        )
    ]

    return torch.optim.AdamW(
        [
            {
                "params":
                    enc,

                "lr":
                    ENCODER_LR,

                "weight_decay":
                    WEIGHT_DECAY,

                "name":
                    "encoder",
            },
            {
                "params":
                    other,

                "lr":
                    HEAD_LR,

                "weight_decay":
                    WEIGHT_DECAY,

                "name":
                    "georcp",
            },
            {
                "params":
                    list(
                        loss_module.parameters()
                    ),

                "lr":
                    UNCERTAINTY_LR,

                "weight_decay":
                    0.0,

                "name":
                    "uncertainty",
            },
        ]
    )


def lr_factor_for_epoch(
    epoch: int,
) -> float:
    if epoch <= WARMUP_EPOCHS:
        return float(
            epoch
        ) / float(
            WARMUP_EPOCHS
        )

    denom = max(
        1,
        MAX_EPOCHS
        - WARMUP_EPOCHS,
    )

    progress = (
        float(
            epoch
            - WARMUP_EPOCHS
        )
        / float(
            denom
        )
    )

    progress = min(
        1.0,
        max(
            0.0,
            progress,
        ),
    )

    cosine = (
        0.5
        * (
            1.0
            + math.cos(
                math.pi
                * progress
            )
        )
    )

    return (
        MIN_LR_FACTOR
        + (
            1.0
            - MIN_LR_FACTOR
        )
        * cosine
    )


def apply_epoch_lrs(
    optimizer,
    epoch: int,
) -> Dict[str, float]:
    factor = lr_factor_for_epoch(
        epoch
    )

    base = {
        "encoder":
            ENCODER_LR,

        "georcp":
            HEAD_LR,

        "uncertainty":
            UNCERTAINTY_LR,
    }

    out = {}

    for group in optimizer.param_groups:
        name = group[
            "name"
        ]

        lr = (
            base[
                name
            ]
            * factor
        )

        group[
            "lr"
        ] = lr

        out[
            name
        ] = float(
            lr
        )

    return out


# =============================================================================
# LOSS / METRIC ACCUMULATION
# =============================================================================

def finite_gradients(
    model,
    loss_module,
) -> bool:
    found = False

    for parameter in list(
        model.parameters()
    ) + list(
        loss_module.parameters()
    ):
        if parameter.grad is None:
            continue

        found = True

        if not torch.isfinite(
            parameter.grad
        ).all():
            return False

    return found


def float_parts(
    parts: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    return {
        key:
            float(
                value.detach()
                .float()
                .cpu()
            )
        for key, value in parts.items()
    }


class RunningMean:
    def __init__(
        self,
    ):
        self.n = 0
        self.sums = {}

    def update(
        self,
        parts: Dict[str, float],
        n: int,
    ):
        self.n += int(
            n
        )

        for key, value in parts.items():
            self.sums[
                key
            ] = (
                self.sums.get(
                    key,
                    0.0,
                )
                + float(
                    value
                )
                * float(
                    n
                )
            )

    def means(
        self,
    ) -> Dict[str, float]:
        if self.n <= 0:
            raise RuntimeError(
                "RunningMean is empty."
            )

        return {
            key:
                value
                / float(
                    self.n
                )
            for key, value in self.sums.items()
        }


# =============================================================================
# TRAIN / VALIDATE
# =============================================================================

def train_one_epoch(
    model,
    loss_module,
    optimizer,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
) -> Dict[str, Any]:

    model.train()
    loss_module.train()

    running = RunningMean()

    no_hf_batches = 0
    max_point = -float(
        "inf"
    )

    max_total = -float(
        "inf"
    )

    hf_points_seen = 0
    flipped_images = 0

    start = time.perf_counter()

    for step, batch in enumerate(
        loader,
        start=1,
    ):
        batch_n = int(
            batch[
                "image"
            ].shape[
                0
            ]
        )

        hf_points = int(
            batch[
                "hf_offset_mask"
            ].sum().item()
        )

        hf_points_seen += (
            hf_points
        )

        if hf_points == 0:
            no_hf_batches += 1

        for stem in batch[
            "_stems"
        ]:
            if deterministic_flip(
                stem,
                epoch,
            ):
                flipped_images += 1

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
            outputs = model(
                batch[
                    "image"
                ]
            )

            total, parts = loss_module(
                outputs,
                batch,
            )

        if not torch.isfinite(
            total
        ):
            raise RuntimeError(
                "Non-finite training loss at epoch {}, step {}.".format(
                    epoch,
                    step,
                )
            )

        total.backward()

        if not finite_gradients(
            model,
            loss_module,
        ):
            raise RuntimeError(
                "Non-finite gradients at epoch {}, step {}.".format(
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

        max_point = max(
            max_point,
            fp[
                "point_group"
            ],
        )

        max_total = max(
            max_total,
            fp[
                "total"
            ],
        )

        if (
            step == 1
            or step % LOG_EVERY_STEPS == 0
            or step == len(
                loader
            )
        ):
            elapsed = (
                time.perf_counter()
                - start
            )

            img_s = (
                running.n
                / elapsed
            )

            print(
                "      step {:4d}/{:4d} | "
                "total={:.4f} R={:.4f} C={:.4f} P={:.4f} | "
                "{:.2f} img/s".format(
                    step,
                    len(
                        loader
                    ),
                    fp[
                        "total"
                    ],
                    fp[
                        "region_group"
                    ],
                    fp[
                        "curve_group"
                    ],
                    fp[
                        "point_group"
                    ],
                    img_s,
                )
            )

    torch.cuda.synchronize()

    elapsed = (
        time.perf_counter()
        - start
    )

    means = running.means()

    log_vars = (
        loss_module.log_vars.detach()
        .float()
        .cpu()
        .tolist()
    )

    return {
        "images":
            running.n,

        "optimizer_steps":
            len(
                loader
            ),

        "elapsed_seconds":
            float(
                elapsed
            ),

        "images_per_second":
            float(
                running.n
                / elapsed
            ),

        "mean_losses":
            means,

        "max_point_group_loss":
            float(
                max_point
            ),

        "max_total_loss":
            float(
                max_total
            ),

        "no_hf_batches":
            int(
                no_hf_batches
            ),

        "hf_points_seen":
            int(
                hf_points_seen
            ),

        "flipped_images":
            int(
                flipped_images
            ),

        "flip_fraction":
            float(
                flipped_images
                / float(
                    running.n
                )
            ),

        "log_vars":
            {
                "region":
                    float(
                        log_vars[
                            0
                        ]
                    ),

                "curve":
                    float(
                        log_vars[
                            1
                        ]
                    ),

                "point":
                    float(
                        log_vars[
                            2
                        ]
                    ),
            },

        "memory":
            cuda_memory(),
    }


def validate_one_epoch(
    model,
    loss_module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:

    model.eval()
    loss_module.eval()

    running = RunningMean()

    no_hf_batches = 0
    hf_points_seen = 0

    start = time.perf_counter()

    with torch.no_grad():
        for batch in loader:
            batch_n = int(
                batch[
                    "image"
                ].shape[
                    0
                ]
            )

            hf_points = int(
                batch[
                    "hf_offset_mask"
                ].sum().item()
            )

            hf_points_seen += (
                hf_points
            )

            if hf_points == 0:
                no_hf_batches += 1

            batch = move_batch(
                batch,
                device,
            )

            with torch.amp.autocast(
                "cuda",
                dtype=torch.bfloat16,
            ):
                outputs = model(
                    batch[
                        "image"
                    ]
                )

                total, parts = loss_module(
                    outputs,
                    batch,
                )

            if not torch.isfinite(
                total
            ):
                raise RuntimeError(
                    "Non-finite validation loss."
                )

            running.update(
                float_parts(
                    parts
                ),
                batch_n,
            )

    torch.cuda.synchronize()

    elapsed = (
        time.perf_counter()
        - start
    )

    return {
        "images":
            running.n,

        "batches":
            len(
                loader
            ),

        "elapsed_seconds":
            float(
                elapsed
            ),

        "images_per_second":
            float(
                running.n
                / elapsed
            ),

        "mean_losses":
            running.means(),

        "no_hf_batches":
            int(
                no_hf_batches
            ),

        "hf_points_seen":
            int(
                hf_points_seen
            ),
    }


# =============================================================================
# PROTOCOL FREEZE
# =============================================================================

def protocol_object(
    environment: Dict[str, Any],
    train_sha: str,
    val_sha: str,
) -> Dict[str, Any]:

    obj = {
        "status":
            "FROZEN",

        "name":
            "GeoRCP-MIRAGE RTX4080 definitive training protocol v2 — no augmentation",

        "created_by_script_revision":
            SCRIPT_REVISION,

        "scientific_design_sha256":
            EXPECTED_DESIGN_SHA,

        "target_validity_v1_1_sha256":
            EXPECTED_V1_1_SHA,

        "active_target_version":
            "geometry_v3",

        "active_target_master_sha256":
            EXPECTED_G3_SHA,

        "implementation_spec_sha256":
            EXPECTED_IMPLEMENTATION_SPEC_SHA,

        "source_bundle_sha256":
            EXPECTED_SOURCE_BUNDLE_SHA,

        "mirage_wrapper_sha256":
            EXPECTED_MIRAGE_WRAPPER_SHA,

        "trackb_split_master_sha256":
            CANONICAL_TRACKB_SPLIT_SHA,

        "fold":
            1,

        "fold1_train_images":
            EXPECTED_F1_TRAIN,

        "fold1_val_images":
            EXPECTED_F1_VAL,

        "fold1_train_stem_sha256":
            train_sha,

        "fold1_val_stem_sha256":
            val_sha,

        "outer_test_policy":
            "never resolve/evaluate during training or checkpoint selection",

        "gpu":
            "NVIDIA GeForce RTX 4080 16 GB",

        "validated_python_family":
            "3.8.x",

        "validated_pytorch":
            "2.4.1+cu121",

        "validated_torch_cuda":
            "12.1",

        "precision":
            PRECISION,

        "physical_batch_size":
            PHYSICAL_BATCH,

        "validation_batch_size":
            VAL_BATCH,

        "gradient_accumulation_steps":
            GRAD_ACCUM_STEPS,

        "effective_batch_size":
            EFFECTIVE_BATCH,

        "optimizer":
            OPTIMIZER,

        "encoder_lr":
            ENCODER_LR,

        "head_lr":
            HEAD_LR,

        "uncertainty_lr":
            UNCERTAINTY_LR,

        "weight_decay":
            WEIGHT_DECAY,

        "scheduler":
            "5-epoch linear warmup then cosine decay",

        "warmup_epochs":
            WARMUP_EPOCHS,

        "minimum_lr_factor":
            MIN_LR_FACTOR,

        "max_epochs":
            MAX_EPOCHS,

        "early_stopping_patience":
            EARLY_STOP_PATIENCE,

        "early_stopping_min_delta":
            EARLY_STOP_MIN_DELTA,

        "minimum_epochs_before_early_stop":
            MIN_EPOCHS_BEFORE_STOP,

        "checkpoint_metric":
            CHECKPOINT_METRIC,

        "checkpoint_mode":
            CHECKPOINT_MODE,

        "augmentation_policy":
            AUGMENTATION_POLICY,

        "horizontal_flip_probability":
            HFLIP_PROB,

        "protocol_correction":
            (
                "v1 definitive attempt was aborted during epoch 1 before any "
                "epoch completed because exact horizontal flip produced an HF "
                "stride-2 CenterNet cell collision. v2 removes augmentation "
                "rather than dropping/merging points or changing the frozen "
                "point representation. This is an implementation-validity "
                "correction, not validation/test-based tuning."
            ),

        "supersedes_protocol_file":
            "RTX4080_ENGINEERING_FROZEN.json",

        "input_size_hw":
            [
                H,
                W,
            ],

        "resize":
            False,

        "num_workers":
            NUM_WORKERS,

        "seed":
            SEED,

        "gradient_clipping":
            False,

        "final_test_evaluation":
            "separate later step using preserved best checkpoint exactly once",

        "validated_environment":
            environment,
    }

    obj[
        "protocol_sha256"
    ] = stable_sha(
        obj
    )

    return obj


def freeze_or_verify_protocol(
    protocol: Dict[str, Any],
) -> None:

    DESIGN_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if ENGINEERING_PROTOCOL.exists():
        existing = load_json(
            ENGINEERING_PROTOCOL
        )

        if (
            existing.get(
                "protocol_sha256"
            )
            != protocol.get(
                "protocol_sha256"
            )
        ):
            raise RuntimeError(
                "A different RTX4080_ENGINEERING_FROZEN.json already exists.\n"
                "Refusing to overwrite a frozen protocol."
            )

        print(
            "    Existing engineering protocol: VERIFIED"
        )

    else:
        save_json(
            ENGINEERING_PROTOCOL,
            protocol,
        )

        print(
            "    Engineering protocol frozen: CREATED"
        )

    save_json(
        PROTOCOL_SNAPSHOT,
        protocol,
    )


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
        "python":
            random.getstate(),

        "numpy":
            np.random.get_state(),

        "torch_cpu":
            torch.get_rng_state(),

        "torch_cuda":
            torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(
    state: Dict[str, Any],
) -> None:
    random.setstate(
        state[
            "python"
        ]
    )

    np.random.set_state(
        state[
            "numpy"
        ]
    )

    torch.set_rng_state(
        state[
            "torch_cpu"
        ]
    )

    torch.cuda.set_rng_state_all(
        state[
            "torch_cuda"
        ]
    )


def save_checkpoint(
    path: Path,
    model,
    loss_module,
    optimizer,
    epoch: int,
    best_epoch: int,
    best_metric: float,
    epochs_without_improvement: int,
    protocol_sha: str,
) -> None:

    tmp = path.with_suffix(
        path.suffix
        + ".tmp"
    )

    state = {
        "status":
            "DEFINITIVE_FOLD1",

        "script_revision":
            SCRIPT_REVISION,

        "script_sha256":
            script_sha(),

        "protocol_sha256":
            protocol_sha,

        "active_target_master_sha256":
            EXPECTED_G3_SHA,

        "implementation_spec_sha256":
            EXPECTED_IMPLEMENTATION_SPEC_SHA,

        "source_bundle_sha256":
            EXPECTED_SOURCE_BUNDLE_SHA,

        "epoch":
            int(
                epoch
            ),

        "best_epoch":
            int(
                best_epoch
            ),

        "best_metric":
            float(
                best_metric
            ),

        "epochs_without_improvement":
            int(
                epochs_without_improvement
            ),

        "model":
            model.state_dict(),

        "loss_module":
            loss_module.state_dict(),

        "optimizer":
            optimizer.state_dict(),

        "rng":
            rng_state(),
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
    model,
    loss_module,
    optimizer,
    device: torch.device,
    protocol_sha: str,
):
    if not LAST_CKPT.exists():
        return {
            "start_epoch":
                1,

            "best_epoch":
                0,

            "best_metric":
                float(
                    "inf"
                ),

            "epochs_without_improvement":
                0,

            "resumed":
                False,
        }

    state = torch_load_checkpoint(
        LAST_CKPT,
        device,
    )

    checks = {
        "protocol_sha256":
            protocol_sha,

        "active_target_master_sha256":
            EXPECTED_G3_SHA,

        "implementation_spec_sha256":
            EXPECTED_IMPLEMENTATION_SPEC_SHA,

        "source_bundle_sha256":
            EXPECTED_SOURCE_BUNDLE_SHA,
    }

    for key, expected in checks.items():
        if state.get(
            key
        ) != expected:
            raise RuntimeError(
                "Resume checkpoint mismatch: {}.".format(
                    key
                )
            )

    model.load_state_dict(
        state[
            "model"
        ],
        strict=True,
    )

    loss_module.load_state_dict(
        state[
            "loss_module"
        ],
        strict=True,
    )

    optimizer.load_state_dict(
        state[
            "optimizer"
        ]
    )

    restore_rng_state(
        state[
            "rng"
        ]
    )

    return {
        "start_epoch":
            int(
                state[
                    "epoch"
                ]
            )
            + 1,

        "best_epoch":
            int(
                state[
                    "best_epoch"
                ]
            ),

        "best_metric":
            float(
                state[
                    "best_metric"
                ]
            ),

        "epochs_without_improvement":
            int(
                state[
                    "epochs_without_improvement"
                ]
            ),

        "resumed":
            True,
    }


# =============================================================================
# HISTORY
# =============================================================================

HISTORY_FIELDS = [
    "epoch",
    "encoder_lr",
    "head_lr",
    "uncertainty_lr",
    "train_total",
    "train_raw_group_sum",
    "train_region_group",
    "train_curve_group",
    "train_point_group",
    "val_total",
    "val_raw_group_sum",
    "val_region_group",
    "val_curve_group",
    "val_point_group",
    "log_var_region",
    "log_var_curve",
    "log_var_point",
    "train_seconds",
    "val_seconds",
    "peak_allocated_gb",
    "flipped_images",
    "flip_fraction",
    "improved",
]


def append_history(
    row: Dict[str, Any],
) -> None:
    exists = HISTORY_CSV.exists()

    with HISTORY_CSV.open(
        "a",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=HISTORY_FIELDS,
        )

        if not exists:
            writer.writeheader()

        writer.writerow(
            row
        )


# =============================================================================
# MAIN
# =============================================================================

def main():
    print(
        "\n"
        + "=" * 110
    )

    print(
        "RVO-ME STEP 08 DEFINITIVE: GeoRCP-MIRAGE FOLD-1 TRAINING"
    )

    print(
        "SCRIPT REVISION: {}".format(
            SCRIPT_REVISION
        )
    )

    print(
        "=" * 110
    )

    RUN_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if COMPLETE_JSON.exists():
        complete = load_json(
            COMPLETE_JSON
        )

        print(
            "\nFold-1 definitive training is already COMPLETE."
        )

        print(
            "Best epoch: {}".format(
                complete.get(
                    "best_epoch"
                )
            )
        )

        print(
            "Best val raw-group sum: {}".format(
                complete.get(
                    "best_val_raw_group_sum"
                )
            )
        )

        print(
            "Best checkpoint: {}".format(
                BEST_CKPT
            )
        )

        return

    print(
        "\n[1/8] Loading exact GeoRCP implementation..."
    )

    step07, source_sha = load_exact_step07()

    print(
        "    Implementation-spec SHA: PASS"
    )

    print(
        "    Exact source-bundle SHA:  PASS"
    )

    print(
        "\n[2/8] Verifying RTX-4080 frozen inputs and clean preflight..."
    )

    environment = verify_environment_and_inputs(
        step07,
        source_sha,
    )

    print(
        "    GPU: {}".format(
            environment[
                "gpu"
            ]
        )
    )

    print(
        "    VRAM: {:.3f} GB".format(
            environment[
                "vram_gb"
            ]
        )
    )

    print(
        "    BF16: PASS"
    )

    print(
        "    MIRAGE wrapper SHA: PASS"
    )

    print(
        "    geometry_v3 SHA: PASS"
    )

    print(
        "    RTX-4080 Step-07 preflight: PASS"
    )

    print(
        "    RTX-4080 clean revalidation: PASS"
    )

    print(
        "\n[3/8] Resolving Fold-1 TRAIN/VAL only..."
    )

    images = step07.images_by_stem()

    available = sorted(
        images
    )

    train_stems = resolve_fold1_role(
        "train",
        available,
    )

    val_stems = resolve_fold1_role(
        "val",
        available,
    )

    overlap = set(
        train_stems
    ) & set(
        val_stems
    )

    if overlap:
        raise RuntimeError(
            "Fold-1 TRAIN/VAL filename overlap detected."
        )

    train_sha = hash_stem_list(
        train_stems
    )

    val_sha = hash_stem_list(
        val_stems
    )

    print(
        "    Train images: {} PASS".format(
            len(
                train_stems
            )
        )
    )

    print(
        "    Val images:   {} PASS".format(
            len(
                val_stems
            )
        )
    )

    print(
        "    Train/val overlap: 0 PASS"
    )

    print(
        "    Outer test: NOT RESOLVED / NOT LOADED"
    )

    print(
        "\n[4/8] Freezing/verifying definitive RTX-4080 protocol..."
    )

    protocol = protocol_object(
        environment,
        train_sha,
        val_sha,
    )

    freeze_or_verify_protocol(
        protocol
    )

    append_project_log_once(
        "<!-- STEP08_PROTOCOL_V2_NOAUG_CORRECTION -->",
        (
            "Definitive training protocol v2 created before any completed "
            "definitive epoch. Horizontal flip was removed because exact "
            "pixel-space mirroring can create stride-2 HF CenterNet cell "
            "collisions after floor quantization. The failed v1 attempt "
            "completed no epoch; outer test remained untouched. No model, "
            "loss, target, LR, batch, scheduler, or checkpoint-selection "
            "setting was otherwise changed."
        ),
    )

    print(
        "    Protocol SHA: {}".format(
            protocol[
                "protocol_sha256"
            ]
        )
    )

    print(
        "    BF16 | batch {} | AdamW | enc LR {:.1e} | head LR {:.1e}".format(
            PHYSICAL_BATCH,
            ENCODER_LR,
            HEAD_LR,
        )
    )

    print(
        "    Warmup {} | cosine | max epochs {} | patience {}".format(
            WARMUP_EPOCHS,
            MAX_EPOCHS,
            EARLY_STOP_PATIENCE,
        )
    )

    print(
        "    Selection metric: {}".format(
            CHECKPOINT_METRIC
        )
    )

    print(
        "\n[5/8] Verifying definitive augmentation policy..."
    )

    aug_check = {
        "status": "PASS",
        "augmentation": "NONE",
        "horizontal_flip_probability": HFLIP_PROB,
        "reason": (
            "Exact horizontal flip is disabled because it can create "
            "stride-2 HF CenterNet cell collisions after quantization."
        ),
    }

    if HFLIP_PROB != 0.0:
        raise RuntimeError(
            "Definitive v2 protocol requires HFLIP_PROB == 0.0."
        )

    print(
        "    Augmentation policy: PASS"
    )

    print(
        "    Policy: {}".format(
            AUGMENTATION_POLICY
        )
    )

    print(
        "\n[6/8] Building model / optimizer / resume state..."
    )

    set_seed(
        SEED
    )

    device = torch.device(
        "cuda"
    )

    reset_cuda()

    model, loss_module = step07.build_model(
        device
    )

    optimizer = build_optimizer(
        model,
        loss_module,
    )

    resume = load_resume(
        model,
        loss_module,
        optimizer,
        device,
        protocol[
            "protocol_sha256"
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
            if resume[
                "resumed"
            ]
            else "NO"
        )
    )

    print(
        "    Start epoch: {}".format(
            start_epoch
        )
    )

    if start_epoch > MAX_EPOCHS:
        raise RuntimeError(
            "Last checkpoint already exceeds MAX_EPOCHS but COMPLETE file is missing."
        )

    train_dataset = GeoRCPDataset(
        train_stems,
        images,
        step07,
        augment=False,
    )

    val_dataset = GeoRCPDataset(
        val_stems,
        images,
        step07,
        augment=False,
    )

    vloader = val_loader(
        val_dataset
    )

    print(
        "\n[7/8] DEFINITIVE FOLD-1 TRAINING..."
    )

    overall_start = time.perf_counter()
    stop_reason = "MAX_EPOCHS"

    for epoch in range(
        start_epoch,
        MAX_EPOCHS + 1,
    ):
        print(
            "\n"
            + "-" * 110
        )

        print(
            "EPOCH {:02d}/{:02d}".format(
                epoch,
                MAX_EPOCHS,
            )
        )

        lrs = apply_epoch_lrs(
            optimizer,
            epoch,
        )

        print(
            "    LR encoder={:.3e} | head={:.3e} | uncertainty={:.3e}".format(
                lrs[
                    "encoder"
                ],
                lrs[
                    "georcp"
                ],
                lrs[
                    "uncertainty"
                ],
            )
        )

        tloader = train_loader_for_epoch(
            train_dataset,
            epoch,
        )

        reset_cuda()

        train_report = train_one_epoch(
            model,
            loss_module,
            optimizer,
            tloader,
            device,
            epoch,
        )

        val_report = validate_one_epoch(
            model,
            loss_module,
            vloader,
            device,
        )

        train_losses = train_report[
            "mean_losses"
        ]

        val_losses = val_report[
            "mean_losses"
        ]

        metric = float(
            val_losses[
                "raw_group_sum"
            ]
        )

        improved = (
            metric
            < (
                best_metric
                - EARLY_STOP_MIN_DELTA
            )
        )

        if improved:
            best_metric = metric
            best_epoch = epoch
            epochs_without_improvement = 0

            save_checkpoint(
                BEST_CKPT,
                model,
                loss_module,
                optimizer,
                epoch,
                best_epoch,
                best_metric,
                epochs_without_improvement,
                protocol[
                    "protocol_sha256"
                ],
            )

        else:
            epochs_without_improvement += 1

        save_checkpoint(
            LAST_CKPT,
            model,
            loss_module,
            optimizer,
            epoch,
            best_epoch,
            best_metric,
            epochs_without_improvement,
            protocol[
                "protocol_sha256"
            ],
        )

        row = {
            "epoch":
                epoch,

            "encoder_lr":
                lrs[
                    "encoder"
                ],

            "head_lr":
                lrs[
                    "georcp"
                ],

            "uncertainty_lr":
                lrs[
                    "uncertainty"
                ],

            "train_total":
                train_losses[
                    "total"
                ],

            "train_raw_group_sum":
                train_losses[
                    "raw_group_sum"
                ],

            "train_region_group":
                train_losses[
                    "region_group"
                ],

            "train_curve_group":
                train_losses[
                    "curve_group"
                ],

            "train_point_group":
                train_losses[
                    "point_group"
                ],

            "val_total":
                val_losses[
                    "total"
                ],

            "val_raw_group_sum":
                val_losses[
                    "raw_group_sum"
                ],

            "val_region_group":
                val_losses[
                    "region_group"
                ],

            "val_curve_group":
                val_losses[
                    "curve_group"
                ],

            "val_point_group":
                val_losses[
                    "point_group"
                ],

            "log_var_region":
                train_report[
                    "log_vars"
                ][
                    "region"
                ],

            "log_var_curve":
                train_report[
                    "log_vars"
                ][
                    "curve"
                ],

            "log_var_point":
                train_report[
                    "log_vars"
                ][
                    "point"
                ],

            "train_seconds":
                train_report[
                    "elapsed_seconds"
                ],

            "val_seconds":
                val_report[
                    "elapsed_seconds"
                ],

            "peak_allocated_gb":
                train_report[
                    "memory"
                ][
                    "peak_allocated_gb"
                ],

            "flipped_images":
                train_report[
                    "flipped_images"
                ],

            "flip_fraction":
                train_report[
                    "flip_fraction"
                ],

            "improved":
                int(
                    improved
                ),
        }

        append_history(
            row
        )

        running = {
            "status":
                "RUNNING",

            "epoch_completed":
                epoch,

            "best_epoch":
                best_epoch,

            "best_val_raw_group_sum":
                best_metric,

            "epochs_without_improvement":
                epochs_without_improvement,

            "protocol_sha256":
                protocol[
                    "protocol_sha256"
                ],

            "outer_test_evaluated":
                False,

            "last_epoch": {
                "train":
                    train_report,

                "validation":
                    val_report,
            },
        }

        save_json(
            RUNNING_JSON,
            running,
        )

        print(
            "\n    Epoch {:02d} summary".format(
                epoch
            )
        )

        print(
            "      train raw={:.5f} | R={:.5f} C={:.5f} P={:.5f}".format(
                train_losses[
                    "raw_group_sum"
                ],
                train_losses[
                    "region_group"
                ],
                train_losses[
                    "curve_group"
                ],
                train_losses[
                    "point_group"
                ],
            )
        )

        print(
            "      val   raw={:.5f} | R={:.5f} C={:.5f} P={:.5f}".format(
                val_losses[
                    "raw_group_sum"
                ],
                val_losses[
                    "region_group"
                ],
                val_losses[
                    "curve_group"
                ],
                val_losses[
                    "point_group"
                ],
            )
        )

        print(
            "      best epoch={} | best val raw={:.5f} | {}".format(
                best_epoch,
                best_metric,
                "IMPROVED"
                if improved
                else "no improvement ({}/{})".format(
                    epochs_without_improvement,
                    EARLY_STOP_PATIENCE,
                ),
            )
        )

        print(
            "      train {:.2f} min | val {:.2f} min | peak {:.3f} GB".format(
                train_report[
                    "elapsed_seconds"
                ]
                / 60.0,
                val_report[
                    "elapsed_seconds"
                ]
                / 60.0,
                train_report[
                    "memory"
                ][
                    "peak_allocated_gb"
                ],
            )
        )

        if (
            epoch >= MIN_EPOCHS_BEFORE_STOP
            and epochs_without_improvement
            >= EARLY_STOP_PATIENCE
        ):
            stop_reason = "EARLY_STOPPING"
            print(
                "\n    Early stopping criterion reached."
            )
            break

    overall_elapsed = (
        time.perf_counter()
        - overall_start
    )

    print(
        "\n[8/8] Finalizing definitive Fold-1 training record..."
    )

    if not BEST_CKPT.exists():
        raise RuntimeError(
            "Definitive training ended without a best checkpoint."
        )

    best_state = torch_load_checkpoint(
        BEST_CKPT,
        device,
    )

    if (
        best_state.get(
            "protocol_sha256"
        )
        != protocol[
            "protocol_sha256"
        ]
    ):
        raise RuntimeError(
            "Best-checkpoint protocol SHA mismatch."
        )

    complete = {
        "status":
            "COMPLETE",

        "scientific_status":
            "DEFINITIVE_FOLD1_V2_NOAUG_TRAINING_COMPLETE_OUTER_TEST_UNTOUCHED",

        "script_revision":
            SCRIPT_REVISION,

        "script_sha256":
            script_sha(),

        "protocol_sha256":
            protocol[
                "protocol_sha256"
            ],

        "scientific_design_sha256":
            EXPECTED_DESIGN_SHA,

        "active_target_master_sha256":
            EXPECTED_G3_SHA,

        "implementation_spec_sha256":
            EXPECTED_IMPLEMENTATION_SPEC_SHA,

        "source_bundle_sha256":
            EXPECTED_SOURCE_BUNDLE_SHA,

        "mirage_wrapper_sha256":
            EXPECTED_MIRAGE_WRAPPER_SHA,

        "trackb_split_master_sha256":
            CANONICAL_TRACKB_SPLIT_SHA,

        "fold":
            1,

        "train_images":
            len(
                train_stems
            ),

        "val_images":
            len(
                val_stems
            ),

        "train_stem_sha256":
            train_sha,

        "val_stem_sha256":
            val_sha,

        "precision":
            PRECISION,

        "physical_batch_size":
            PHYSICAL_BATCH,

        "effective_batch_size":
            EFFECTIVE_BATCH,

        "best_epoch":
            int(
                best_state[
                    "epoch"
                ]
            ),

        "best_val_raw_group_sum":
            float(
                best_state[
                    "best_metric"
                ]
            ),

        "stop_reason":
            stop_reason,

        "elapsed_seconds_current_invocation":
            float(
                overall_elapsed
            ),

        "best_checkpoint":
            str(
                BEST_CKPT
            ),

        "last_checkpoint":
            str(
                LAST_CKPT
            ),

        "history_csv":
            str(
                HISTORY_CSV
            ),

        "outer_test_resolved":
            False,

        "outer_test_evaluated":
            False,

        "best_weights_preserved":
            True,

        "next_step":
            (
                "Step 09: load best checkpoint and perform the frozen "
                "task-specific Fold-1 evaluation; outer test is evaluated "
                "only after training/model selection is complete."
            ),
    }

    save_json(
        COMPLETE_JSON,
        complete,
    )

    append_project_log_once(
        "<!-- STEP08_DEFINITIVE_FOLD1_V2_NOAUG_COMPLETE -->",
        (
            "Step 08 definitive Fold-1 GeoRCP-MIRAGE training complete. "
            "Protocol SHA: {}. Best epoch: {}. Best validation raw-group sum: "
            "{:.6f}. Outer test remained untouched. Best checkpoint preserved."
        ).format(
            protocol[
                "protocol_sha256"
            ],
            complete[
                "best_epoch"
            ],
            complete[
                "best_val_raw_group_sum"
            ],
        ),
    )

    print(
        "\n"
        + "=" * 110
    )

    print(
        "STEP 08 DEFINITIVE FOLD-1 TRAINING COMPLETE"
    )

    print(
        "=" * 110
    )

    print(
        "Best epoch: {}".format(
            complete[
                "best_epoch"
            ]
        )
    )

    print(
        "Best validation raw-group sum: {:.6f}".format(
            complete[
                "best_val_raw_group_sum"
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
        "\nBest checkpoint:"
    )

    print(
        BEST_CKPT
    )

    print(
        "\nCompletion report:"
    )

    print(
        COMPLETE_JSON
    )

    print(
        "\nNext: send FOLD1_TRAINING_COMPLETE.json before any outer-test evaluation."
    )


if __name__ == "__main__":
    main()
