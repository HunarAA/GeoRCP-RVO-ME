# =============================================================================
# RVO-ME STEP 08 FINAL
# GeoRCP-MIRAGE — CONSOLIDATED DEFINITIVE TRAINING FOR TRACK-B FOLDS 2–5
#
# This continues the already-frozen definitive NO-AUGMENTATION protocol used
# for Fold 1. The photometric Fold-1 run is retained as an exploratory
# regularization ablation; it does NOT change the final protocol.
#
# Folds 2–5:
#   - use TRAIN + VAL only
#   - NEVER resolve/evaluate the held-out TEST role
#   - use the same geometry_v3 targets and exact GeoRCP implementation
#   - use the same seed, batch, BF16, AdamW, LR schedule, early stopping,
#     checkpoint criterion, and NO augmentation as definitive Fold 1
#   - preserve the best checkpoint
#   - resume automatically from the last completed epoch if interrupted
#
# After all folds complete, send FIVEFOLD_TRAINING_MANIFEST.json.
# Do not run held-out evaluation before that.
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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# IDENTITY / PATHS
# =============================================================================

SCRIPT_REVISION = "STEP08_FINAL_FOLDS2TO5_NOAUG_V1_20260820"

ROOT = Path(r"D:\RVO_ME")
PROJECT = ROOT / "project"

STEP07_SCRIPT = (
    PROJECT / "scripts" / "07_georcp_preflight_v2_ampfix_py38compat.py"
)

SPLIT_ROOT = PROJECT / "splits" / "patient_5fold_v1"

TARGET_ROOT = PROJECT / "targets" / "geometry_v3"
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

DESIGN_DIR = PROJECT / "design" / "geometry_aware_rcp_v1"

NOAUG_PROTOCOL_FILE = (
    DESIGN_DIR / "RTX4080_ENGINEERING_FROZEN_v2.json"
)

RUN_ROOT = (
    PROJECT
    / "training"
    / "georcp_v1"
    / "definitive_v2"
)

FOLD1_COMPLETE = (
    RUN_ROOT / "fold_1" / "FOLD1_TRAINING_COMPLETE.json"
)

FOLD1_BEST = (
    RUN_ROOT / "fold_1" / "best_checkpoint.pt"
)

FIVEFOLD_MANIFEST = (
    RUN_ROOT / "FIVEFOLD_TRAINING_MANIFEST.json"
)

PROJECT_LOG = PROJECT / "PROJECT_LOG.md"


# =============================================================================
# FROZEN IDENTIFIERS
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

EXPECTED_SOURCE_BUNDLE_SHA = (
    "04e7da4fe311b756ad375a6c241530d0572a3cf1b9c71a70af2e7e7e24524412"
)

CANONICAL_TRACKB_SPLIT_SHA = (
    "7e7c2ddf51054bf998b2d750b50c74983b9bc5b751b762213bc96343ffe01346"
)

EXPECTED_NOAUG_PROTOCOL_SHA = (
    "8640e7b4039cfc5f6fcd45269aaf9003a337a090f3af5a8aa787c9dcd9a0491e"
)

EXPECTED_FOLD1_SCRIPT_SHA = (
    "4ea1ba358bd5b67f5a6addd0ab6902710b38377f9b91cffb9e159f1c8d180b28"
)

EXPECTED_IMAGES = 3012

EXPECTED_ROLE_COUNTS = {
    1: {"train": 2110, "val": 303, "test": 599},
    2: {"train": 2136, "val": 270, "test": 606},
    3: {"train": 2106, "val": 308, "test": 598},
    4: {"train": 2039, "val": 342, "test": 631},
    5: {"train": 2072, "val": 362, "test": 578},
}

FOLDS_TO_TRAIN = (2, 3, 4, 5)


# =============================================================================
# FINAL PROTOCOL — IDENTICAL TO DEFINITIVE FOLD 1
# =============================================================================

SEED = 20260815

PRECISION = "BF16"
PHYSICAL_BATCH = 7
VAL_BATCH = 7
GRAD_ACCUM_STEPS = 1
EFFECTIVE_BATCH = 7

OPTIMIZER_NAME = "AdamW"
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

AUGMENTATION_POLICY = "NONE"

NUM_WORKERS = 0
LOG_EVERY_STEPS = 50

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
# HELPERS
# =============================================================================

def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=True)


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
    current = PROJECT_LOG.read_text(encoding="utf-8", errors="ignore")
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
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    tree = ast.parse(text, filename=str(path))
    lines = text.splitlines()

    wanted = set(SOURCE_BUNDLE_NAMES)
    found = []
    chunks = []

    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if node.name not in wanted:
            continue
        if not hasattr(node, "end_lineno"):
            raise RuntimeError("Python AST lacks end_lineno.")
        segment = "\n".join(lines[node.lineno - 1:node.end_lineno])
        chunks.append("## {}\n{}\n".format(node.name, segment))
        found.append(node.name)

    missing = sorted(wanted - set(found))
    if missing:
        raise RuntimeError(
            "Missing source definitions: " + ", ".join(missing)
        )

    return hashlib.sha256(
        "\n".join(chunks).encode("utf-8")
    ).hexdigest()


# =============================================================================
# EXACT MODEL / LOSS
# =============================================================================

def load_exact_step07():
    if not STEP07_SCRIPT.exists():
        raise FileNotFoundError(STEP07_SCRIPT)

    source_sha = source_bundle_sha(STEP07_SCRIPT)
    if source_sha != EXPECTED_SOURCE_BUNDLE_SHA:
        raise RuntimeError(
            "GeoRCP source-bundle SHA mismatch.\n"
            "Expected: {}\nFound:    {}".format(
                EXPECTED_SOURCE_BUNDLE_SHA,
                source_sha,
            )
        )

    mod = module_from_file(
        "georcp_step07_final_folds",
        STEP07_SCRIPT,
    )

    for name in (
        "build_model",
        "load_sample",
        "images_by_stem",
        "implementation_spec",
    ):
        if not hasattr(mod, name):
            raise RuntimeError(
                "Step-07 implementation missing {}.".format(name)
            )

    spec = mod.implementation_spec()
    if (
        spec.get("implementation_spec_sha256")
        != EXPECTED_IMPLEMENTATION_SPEC_SHA
    ):
        raise RuntimeError("Implementation-spec SHA mismatch.")

    return mod, source_sha


# =============================================================================
# VERIFY FROZEN ENVIRONMENT / FOLD-1 PROTOCOL
# =============================================================================

def verify_frozen_state(source_sha: str) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")

    gpu = torch.cuda.get_device_name(0)
    vram = (
        torch.cuda.get_device_properties(0).total_memory
        / (1024 ** 3)
    )

    if "RTX 4080" not in gpu:
        raise RuntimeError(
            "Final protocol requires RTX 4080; found {}.".format(gpu)
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
        raise RuntimeError("BF16 is not supported.")

    if sha_file(MIRAGE_WRAPPER) != EXPECTED_MIRAGE_WRAPPER_SHA:
        raise RuntimeError("Frozen MIRAGE wrapper SHA mismatch.")

    if source_sha != EXPECTED_SOURCE_BUNDLE_SHA:
        raise RuntimeError("Source-bundle SHA mismatch.")

    g3 = load_json(G3_COMPLETE)
    if (
        g3.get("geometry_v3_master_content_sha256")
        != EXPECTED_G3_SHA
    ):
        raise RuntimeError("geometry_v3 SHA mismatch.")

    if len(list(TARGET_DIR.glob("*.npz"))) != EXPECTED_IMAGES:
        raise RuntimeError("geometry_v3 target-file count mismatch.")

    pre = load_json(RTX4080_PREFLIGHT)
    if pre.get("status") != "COMPLETE":
        raise RuntimeError("RTX-4080 Step-07 preflight is not COMPLETE.")
    if pre.get("active_target_master_sha256") != EXPECTED_G3_SHA:
        raise RuntimeError("RTX-4080 preflight target SHA mismatch.")
    if (
        pre.get("implementation_spec_sha256")
        != EXPECTED_IMPLEMENTATION_SPEC_SHA
    ):
        raise RuntimeError("RTX-4080 preflight implementation SHA mismatch.")
    if (
        str(
            pre.get("selected_preflight_precision", {}).get(
                "precision", ""
            )
        ).lower()
        != "bf16"
    ):
        raise RuntimeError("RTX-4080 preflight did not select BF16.")
    if int(pre.get("largest_profiled_batch_pass", 0)) < 8:
        raise RuntimeError("RTX-4080 preflight did not pass batch 8.")
    if pre.get("mini_overfit", {}).get("status") != "PASS":
        raise RuntimeError("RTX-4080 mini-overfit did not PASS.")

    rev = load_json(RTX4080_REVALIDATION)
    if rev.get("status") != "COMPLETE":
        raise RuntimeError("RTX-4080 revalidation is not COMPLETE.")
    if "RTX 4080" not in str(rev.get("gpu", "")):
        raise RuntimeError("RTX-4080 revalidation GPU mismatch.")
    if str(rev.get("precision", "")).upper() != "BF16":
        raise RuntimeError("RTX-4080 revalidation precision mismatch.")
    if int(rev.get("memory_probe", {}).get("selected_batch", -1)) != 7:
        raise RuntimeError(
            "RTX-4080 revalidation did not select physical batch 7."
        )
    if bool(rev.get("outer_test_evaluated", True)):
        raise RuntimeError("RTX-4080 revalidation touched outer test.")

    fold1 = load_json(FOLD1_COMPLETE)
    if fold1.get("status") != "COMPLETE":
        raise RuntimeError("Definitive no-augmentation Fold 1 is not COMPLETE.")
    if fold1.get("script_sha256") != EXPECTED_FOLD1_SCRIPT_SHA:
        raise RuntimeError("Definitive Fold-1 script SHA mismatch.")
    if fold1.get("protocol_sha256") != EXPECTED_NOAUG_PROTOCOL_SHA:
        raise RuntimeError("Definitive Fold-1 protocol SHA mismatch.")
    if fold1.get("active_target_master_sha256") != EXPECTED_G3_SHA:
        raise RuntimeError("Definitive Fold-1 target SHA mismatch.")
    if (
        fold1.get("implementation_spec_sha256")
        != EXPECTED_IMPLEMENTATION_SPEC_SHA
    ):
        raise RuntimeError("Definitive Fold-1 implementation SHA mismatch.")
    if fold1.get("source_bundle_sha256") != EXPECTED_SOURCE_BUNDLE_SHA:
        raise RuntimeError("Definitive Fold-1 source SHA mismatch.")
    if bool(fold1.get("outer_test_evaluated", True)):
        raise RuntimeError("Definitive Fold 1 touched outer test.")
    if int(fold1.get("physical_batch_size", -1)) != PHYSICAL_BATCH:
        raise RuntimeError("Definitive Fold-1 batch mismatch.")
    if str(fold1.get("precision", "")).upper() != PRECISION:
        raise RuntimeError("Definitive Fold-1 precision mismatch.")
    if not FOLD1_BEST.exists():
        raise RuntimeError("Definitive Fold-1 best checkpoint is missing.")

    protocol = load_json(NOAUG_PROTOCOL_FILE)
    if protocol.get("protocol_sha256") != EXPECTED_NOAUG_PROTOCOL_SHA:
        raise RuntimeError("Frozen no-augmentation protocol file mismatch.")
    if str(protocol.get("augmentation_policy", "")).upper() != "NONE. NO HORIZONTAL/VERTICAL FLIP, RESIZE, CROP, ROTATION, OR PHOTOMETRIC AUGMENTATION. HORIZONTAL FLIP WAS REMOVED BECAUSE THE EXACT PIXEL-SPACE TRANSFORM CAN CREATE STRIDE-2 HF CENTERNET CELL COLLISIONS AFTER FLOOR-CELL QUANTIZATION, MAKING SOME AUGMENTED POINT TARGETS UNREPRESENTABLE BY THE FROZEN ONE-OFFSET-PER-CELL DECODER.".upper():
        # Do not silently accept a changed protocol file.
        if float(protocol.get("horizontal_flip_probability", -1.0)) != 0.0:
            raise RuntimeError("Frozen protocol unexpectedly enables flipping.")

    return {
        "python": sys.version,
        "pytorch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": gpu,
        "vram_gb": float(vram),
        "bf16_supported": True,
        "source_bundle_sha256": source_sha,
        "fold1_protocol_sha256": EXPECTED_NOAUG_PROTOCOL_SHA,
    }


# =============================================================================
# SPLIT RESOLUTION — TRAIN / VAL ONLY
# =============================================================================

STEM_COLUMNS = {
    "stem", "image", "image_id", "filename", "file", "name",
}

ROLE_COLUMNS = {
    "role", "split", "subset", "set",
}

FOLD_COLUMNS = {
    "fold", "fold_id", "outer_fold",
}


def normalize_fold_value(value: Any) -> Optional[int]:
    m = re.search(r"(\d+)", str(value).strip().lower())
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

    # Deliberate safety gate: this training script cannot resolve TEST.
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
        if p.is_file()
        and p.suffix.lower() in {".txt", ".csv", ".json"}
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
        if not any(token in low for token in fold_tokens):
            continue
        if role not in path.name.lower():
            continue

        before = len(found)
        content = path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        for token in re.split(r"[\s,;]+", content):
            stem = Path(token.strip("\"'")).stem
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
            if any(token in low for token in fold_tokens)
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

                cmap = {c.lower(): c for c in reader.fieldnames}

                stem_col = next(
                    (cmap[c] for c in STEM_COLUMNS if c in cmap),
                    None,
                )
                role_col = next(
                    (cmap[c] for c in ROLE_COLUMNS if c in cmap),
                    None,
                )
                fold_col = next(
                    (cmap[c] for c in FOLD_COLUMNS if c in cmap),
                    None,
                )

                if stem_col is None:
                    continue

                before = len(found)

                for row in reader:
                    row_role = (
                        str(row.get(role_col, "")).strip().lower()
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
                        normalize_fold_value(row.get(fold_col, ""))
                        if fold_col
                        else path_fold
                    )

                    if row_fold != fold:
                        continue

                    stem = Path(
                        str(row.get(stem_col, ""))
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
                    local_fold = normalize_fold_value(lower[key])

            for key in ROLE_COLUMNS:
                if key in lower:
                    local_role = str(lower[key]).strip().lower()

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
                stem = Path(str(stem_value)).stem
                if stem in available:
                    found.add(stem)

            for value in obj.values():
                walk(value, local_fold, local_role)

        elif isinstance(obj, list):
            for value in obj:
                walk(value, inherited_fold, inherited_role)

    for path in files:
        if path.suffix.lower() != ".json":
            continue

        before = len(found)
        try:
            walk(load_json(path))
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

    report = {
        "fold": fold,
        "role": role,
        "expected": expected,
        "found": len(stems),
        "status": "PASS",
        "stem_sha256": hash_stems(stems),
        "evidence_files": evidence[:30],
    }

    return stems, report


# =============================================================================
# DATA
# =============================================================================

class GeoRCPDataset(Dataset):
    def __init__(
        self,
        stems: Sequence[str],
        images: Dict[str, Path],
        step07,
    ):
        self.stems = list(stems)
        self.images = images
        self.step07 = step07

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, index: int):
        stem = self.stems[index]
        sample = self.step07.load_sample(
            stem,
            self.images[stem],
            TARGET_DIR,
        )
        sample["_stem"] = stem
        return sample


def collate_fn(items: List[Dict[str, Any]]):
    stems = [item["_stem"] for item in items]
    keys = [k for k in items[0].keys() if k != "_stem"]

    batch = {
        key: torch.stack(
            [item[key] for item in items],
            dim=0,
        )
        for key in keys
    }

    batch["_stems"] = stems
    return batch


def move_batch(
    batch: Dict[str, Any],
    device: torch.device,
):
    return {
        key: (
            value.to(device, non_blocking=False)
            if torch.is_tensor(value)
            else value
        )
        for key, value in batch.items()
    }


def train_loader_for_epoch(
    dataset: GeoRCPDataset,
    epoch: int,
) -> DataLoader:
    # Exactly the same deterministic loader seed rule as definitive Fold 1.
    generator = torch.Generator()
    generator.manual_seed(SEED + int(epoch))

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


def validation_loader(
    dataset: GeoRCPDataset,
) -> DataLoader:
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
# OPTIMIZER / SCHEDULER
# =============================================================================

def build_optimizer(model, loss_module):
    enc = [
        p
        for p in model.encoder.parameters()
        if p.requires_grad
    ]
    enc_ids = {id(p) for p in enc}

    other = [
        p
        for p in model.parameters()
        if p.requires_grad and id(p) not in enc_ids
    ]

    return torch.optim.AdamW(
        [
            {
                "params": enc,
                "lr": ENCODER_LR,
                "weight_decay": WEIGHT_DECAY,
                "name": "encoder",
            },
            {
                "params": other,
                "lr": HEAD_LR,
                "weight_decay": WEIGHT_DECAY,
                "name": "georcp",
            },
            {
                "params": list(loss_module.parameters()),
                "lr": UNCERTAINTY_LR,
                "weight_decay": 0.0,
                "name": "uncertainty",
            },
        ]
    )


def lr_factor_for_epoch(epoch: int) -> float:
    if epoch <= WARMUP_EPOCHS:
        return float(epoch) / float(WARMUP_EPOCHS)

    denom = max(1, MAX_EPOCHS - WARMUP_EPOCHS)
    progress = (
        float(epoch - WARMUP_EPOCHS)
        / float(denom)
    )
    progress = min(1.0, max(0.0, progress))

    cosine = 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )

    return (
        MIN_LR_FACTOR
        + (1.0 - MIN_LR_FACTOR) * cosine
    )


def apply_epoch_lrs(
    optimizer,
    epoch: int,
) -> Dict[str, float]:

    factor = lr_factor_for_epoch(epoch)

    base = {
        "encoder": ENCODER_LR,
        "georcp": HEAD_LR,
        "uncertainty": UNCERTAINTY_LR,
    }

    out = {}

    for group in optimizer.param_groups:
        name = group["name"]
        lr = base[name] * factor
        group["lr"] = lr
        out[name] = float(lr)

    return out


# =============================================================================
# LOSS / EPOCHS
# =============================================================================

def finite_gradients(model, loss_module) -> bool:
    found = False

    for p in list(model.parameters()) + list(loss_module.parameters()):
        if p.grad is None:
            continue
        found = True
        if not torch.isfinite(p.grad).all():
            return False

    return found


def float_parts(
    parts: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    return {
        key: float(value.detach().float().cpu())
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
                self.sums.get(key, 0.0)
                + float(value) * float(n)
            )

    def means(self) -> Dict[str, float]:
        if self.n <= 0:
            raise RuntimeError("RunningMean is empty.")
        return {
            key: value / float(self.n)
            for key, value in self.sums.items()
        }


def train_one_epoch(
    model,
    loss_module,
    optimizer,
    loader: DataLoader,
    device: torch.device,
    fold: int,
    epoch: int,
) -> Dict[str, Any]:

    model.train()
    loss_module.train()

    running = RunningMean()
    no_hf_batches = 0
    hf_points_seen = 0
    max_point = -float("inf")
    max_total = -float("inf")

    start = time.perf_counter()

    for step, batch in enumerate(loader, start=1):
        batch_n = int(batch["image"].shape[0])
        hf_points = int(batch["hf_offset_mask"].sum().item())
        hf_points_seen += hf_points

        if hf_points == 0:
            no_hf_batches += 1

        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            "cuda",
            dtype=torch.bfloat16,
        ):
            outputs = model(batch["image"])
            total, parts = loss_module(outputs, batch)

        if not torch.isfinite(total):
            raise RuntimeError(
                "Non-finite loss at Fold {}, epoch {}, step {}.".format(
                    fold, epoch, step
                )
            )

        total.backward()

        if not finite_gradients(model, loss_module):
            raise RuntimeError(
                "Non-finite gradients at Fold {}, epoch {}, step {}.".format(
                    fold, epoch, step
                )
            )

        optimizer.step()

        fp = float_parts(parts)
        running.update(fp, batch_n)

        max_point = max(max_point, fp["point_group"])
        max_total = max(max_total, fp["total"])

        if (
            step == 1
            or step % LOG_EVERY_STEPS == 0
            or step == len(loader)
        ):
            elapsed = time.perf_counter() - start
            img_s = running.n / elapsed

            print(
                "      step {:4d}/{:4d} | "
                "total={:.4f} R={:.4f} C={:.4f} P={:.4f} | "
                "{:.2f} img/s".format(
                    step,
                    len(loader),
                    fp["total"],
                    fp["region_group"],
                    fp["curve_group"],
                    fp["point_group"],
                    img_s,
                )
            )

    torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    log_vars = (
        loss_module.log_vars.detach()
        .float()
        .cpu()
        .tolist()
    )

    return {
        "images": running.n,
        "optimizer_steps": len(loader),
        "elapsed_seconds": float(elapsed),
        "images_per_second": float(running.n / elapsed),
        "mean_losses": running.means(),
        "max_point_group_loss": float(max_point),
        "max_total_loss": float(max_total),
        "no_hf_batches": int(no_hf_batches),
        "hf_points_seen": int(hf_points_seen),
        "log_vars": {
            "region": float(log_vars[0]),
            "curve": float(log_vars[1]),
            "point": float(log_vars[2]),
        },
        "memory": cuda_memory(),
    }


def validate_one_epoch(
    model,
    loss_module,
    loader: DataLoader,
    device: torch.device,
    fold: int,
) -> Dict[str, Any]:

    model.eval()
    loss_module.eval()

    running = RunningMean()
    no_hf_batches = 0
    hf_points_seen = 0

    start = time.perf_counter()

    with torch.no_grad():
        for batch in loader:
            batch_n = int(batch["image"].shape[0])
            hf_points = int(batch["hf_offset_mask"].sum().item())
            hf_points_seen += hf_points

            if hf_points == 0:
                no_hf_batches += 1

            batch = move_batch(batch, device)

            with torch.amp.autocast(
                "cuda",
                dtype=torch.bfloat16,
            ):
                outputs = model(batch["image"])
                total, parts = loss_module(outputs, batch)

            if not torch.isfinite(total):
                raise RuntimeError(
                    "Non-finite validation loss in Fold {}.".format(
                        fold
                    )
                )

            running.update(float_parts(parts), batch_n)

    torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    return {
        "images": running.n,
        "batches": len(loader),
        "elapsed_seconds": float(elapsed),
        "images_per_second": float(running.n / elapsed),
        "mean_losses": running.means(),
        "no_hf_batches": int(no_hf_batches),
        "hf_points_seen": int(hf_points_seen),
    }


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
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_checkpoint(
    path: Path,
    model,
    loss_module,
    optimizer,
    fold: int,
    epoch: int,
    best_epoch: int,
    best_metric: float,
    epochs_without_improvement: int,
    fold_protocol_sha: str,
) -> None:

    tmp = path.with_suffix(path.suffix + ".tmp")

    state = {
        "status": "DEFINITIVE_NOAUG",
        "script_revision": SCRIPT_REVISION,
        "script_sha256": current_script_sha(),
        "global_protocol_sha256": EXPECTED_NOAUG_PROTOCOL_SHA,
        "fold_protocol_sha256": fold_protocol_sha,
        "active_target_master_sha256": EXPECTED_G3_SHA,
        "implementation_spec_sha256": EXPECTED_IMPLEMENTATION_SPEC_SHA,
        "source_bundle_sha256": EXPECTED_SOURCE_BUNDLE_SHA,
        "fold": int(fold),
        "epoch": int(epoch),
        "best_epoch": int(best_epoch),
        "best_metric": float(best_metric),
        "epochs_without_improvement": int(
            epochs_without_improvement
        ),
        "model": model.state_dict(),
        "loss_module": loss_module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng": rng_state(),
    }

    torch.save(state, tmp)

    if path.exists():
        path.unlink()

    tmp.replace(path)


def load_resume(
    path: Path,
    model,
    loss_module,
    optimizer,
    device: torch.device,
    fold: int,
    fold_protocol_sha: str,
) -> Dict[str, Any]:

    if not path.exists():
        return {
            "start_epoch": 1,
            "best_epoch": 0,
            "best_metric": float("inf"),
            "epochs_without_improvement": 0,
            "resumed": False,
        }

    state = torch_load_checkpoint(path, device)

    checks = {
        "global_protocol_sha256": EXPECTED_NOAUG_PROTOCOL_SHA,
        "fold_protocol_sha256": fold_protocol_sha,
        "active_target_master_sha256": EXPECTED_G3_SHA,
        "implementation_spec_sha256": EXPECTED_IMPLEMENTATION_SPEC_SHA,
        "source_bundle_sha256": EXPECTED_SOURCE_BUNDLE_SHA,
    }

    for key, expected in checks.items():
        if state.get(key) != expected:
            raise RuntimeError(
                "Fold {} resume mismatch: {}.".format(
                    fold,
                    key,
                )
            )

    if int(state.get("fold", -1)) != fold:
        raise RuntimeError(
            "Resume checkpoint belongs to a different fold."
        )

    model.load_state_dict(state["model"], strict=True)
    loss_module.load_state_dict(
        state["loss_module"],
        strict=True,
    )
    optimizer.load_state_dict(state["optimizer"])
    restore_rng_state(state["rng"])

    return {
        "start_epoch": int(state["epoch"]) + 1,
        "best_epoch": int(state["best_epoch"]),
        "best_metric": float(state["best_metric"]),
        "epochs_without_improvement": int(
            state["epochs_without_improvement"]
        ),
        "resumed": True,
    }


# =============================================================================
# HISTORY
# =============================================================================

HISTORY_FIELDS = [
    "fold",
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
                    epoch = int(old["epoch"])
                except Exception:
                    continue
                rows[epoch] = old

    rows[int(row["epoch"])] = row

    tmp = history_csv.with_suffix(".csv.tmp")

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
            writer.writerow(rows[epoch])

    if history_csv.exists():
        history_csv.unlink()

    tmp.replace(history_csv)


# =============================================================================
# FOLD PROTOCOL / MANIFEST
# =============================================================================

def fold_protocol_object(
    fold: int,
    train_split: Dict[str, Any],
    val_split: Dict[str, Any],
    environment: Dict[str, Any],
) -> Dict[str, Any]:

    obj = {
        "status": "FROZEN",
        "name": (
            "GeoRCP-MIRAGE definitive no-augmentation Fold {}".format(
                fold
            )
        ),
        "fold": int(fold),
        "global_protocol_sha256": EXPECTED_NOAUG_PROTOCOL_SHA,
        "scientific_design_sha256": EXPECTED_DESIGN_SHA,
        "target_validity_v1_1_sha256": EXPECTED_V1_1_SHA,
        "active_target_version": "geometry_v3",
        "active_target_master_sha256": EXPECTED_G3_SHA,
        "implementation_spec_sha256": EXPECTED_IMPLEMENTATION_SPEC_SHA,
        "source_bundle_sha256": EXPECTED_SOURCE_BUNDLE_SHA,
        "mirage_wrapper_sha256": EXPECTED_MIRAGE_WRAPPER_SHA,
        "trackb_split_master_sha256": CANONICAL_TRACKB_SPLIT_SHA,
        "train_images": train_split["found"],
        "val_images": val_split["found"],
        "train_stem_sha256": train_split["stem_sha256"],
        "val_stem_sha256": val_split["stem_sha256"],
        "outer_test_policy": (
            "never resolve/evaluate during training or checkpoint selection"
        ),
        "precision": PRECISION,
        "physical_batch_size": PHYSICAL_BATCH,
        "validation_batch_size": VAL_BATCH,
        "gradient_accumulation_steps": GRAD_ACCUM_STEPS,
        "effective_batch_size": EFFECTIVE_BATCH,
        "optimizer": OPTIMIZER_NAME,
        "encoder_lr": ENCODER_LR,
        "head_lr": HEAD_LR,
        "uncertainty_lr": UNCERTAINTY_LR,
        "weight_decay": WEIGHT_DECAY,
        "scheduler": "5-epoch linear warmup then cosine decay",
        "warmup_epochs": WARMUP_EPOCHS,
        "minimum_lr_factor": MIN_LR_FACTOR,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": EARLY_STOP_PATIENCE,
        "early_stopping_min_delta": EARLY_STOP_MIN_DELTA,
        "minimum_epochs_before_early_stop": MIN_EPOCHS_BEFORE_STOP,
        "checkpoint_metric": CHECKPOINT_METRIC,
        "checkpoint_mode": CHECKPOINT_MODE,
        "augmentation_policy": AUGMENTATION_POLICY,
        "seed": SEED,
        "environment": environment,
    }

    obj["fold_protocol_sha256"] = stable_sha(obj)
    return obj


def update_manifest() -> Dict[str, Any]:
    folds = {}

    for fold in range(1, 6):
        fold_dir = RUN_ROOT / "fold_{}".format(fold)

        if fold == 1:
            complete_path = FOLD1_COMPLETE
        else:
            complete_path = (
                fold_dir
                / "FOLD{}_TRAINING_COMPLETE.json".format(fold)
            )

        best_path = fold_dir / "best_checkpoint.pt"

        if complete_path.exists():
            report = load_json(complete_path)
            status = report.get("status", "UNKNOWN")
        else:
            status = "PENDING"

        folds[str(fold)] = {
            "status": status,
            "run_dir": str(fold_dir),
            "completion_report": str(complete_path),
            "best_checkpoint": str(best_path),
        }

    all_complete = all(
        folds[str(fold)]["status"] == "COMPLETE"
        for fold in range(1, 6)
    )

    manifest = {
        "status": (
            "ALL_FIVE_FOLDS_TRAINING_COMPLETE"
            if all_complete
            else "TRAINING_IN_PROGRESS"
        ),
        "global_protocol_sha256": EXPECTED_NOAUG_PROTOCOL_SHA,
        "selected_protocol": "NO_AUGMENTATION",
        "outer_test_resolved": False,
        "outer_test_evaluated": False,
        "folds": folds,
    }

    save_json(FIVEFOLD_MANIFEST, manifest)
    return manifest


# =============================================================================
# TRAIN ONE FOLD
# =============================================================================

def train_fold(
    fold: int,
    step07,
    images: Dict[str, Path],
    available_stems: Sequence[str],
    device: torch.device,
    environment: Dict[str, Any],
) -> Dict[str, Any]:

    run_dir = RUN_ROOT / "fold_{}".format(fold)
    run_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt = run_dir / "best_checkpoint.pt"
    last_ckpt = run_dir / "last_checkpoint.pt"
    history_csv = run_dir / "epoch_history.csv"
    running_json = run_dir / "TRAINING_RUNNING.json"
    protocol_snapshot = run_dir / "fold_protocol_snapshot.json"
    complete_json = (
        run_dir
        / "FOLD{}_TRAINING_COMPLETE.json".format(fold)
    )

    if complete_json.exists():
        complete = load_json(complete_json)
        if complete.get("status") != "COMPLETE":
            raise RuntimeError(
                "Fold {} completion file is not COMPLETE.".format(
                    fold
                )
            )
        if (
            complete.get("global_protocol_sha256")
            != EXPECTED_NOAUG_PROTOCOL_SHA
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
                complete.get("best_epoch")
            )
        )
        print(
            "    Best val raw: {:.6f}".format(
                float(
                    complete.get("best_val_raw_group_sum")
                )
            )
        )
        return complete

    print("\n" + "=" * 110)
    print(
        "TRACK-B FOLD {} — DEFINITIVE NO-AUGMENTATION TRAINING".format(
            fold
        )
    )
    print("=" * 110)

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

    if set(train_stems) & set(val_stems):
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
    print("    Train/val overlap: 0 PASS")
    print("    Outer test: NOT RESOLVED / NOT LOADED")

    fold_protocol = fold_protocol_object(
        fold,
        train_split,
        val_split,
        environment,
    )

    if protocol_snapshot.exists():
        existing = load_json(protocol_snapshot)
        if (
            existing.get("fold_protocol_sha256")
            != fold_protocol["fold_protocol_sha256"]
        ):
            raise RuntimeError(
                "Fold {} protocol snapshot mismatch.".format(
                    fold
                )
            )
    else:
        save_json(protocol_snapshot, fold_protocol)

    print(
        "    Fold protocol SHA: {}".format(
            fold_protocol["fold_protocol_sha256"]
        )
    )

    # Same model seed as definitive Fold 1.
    set_seed(SEED)
    reset_cuda()

    model, loss_module = step07.build_model(device)
    optimizer = build_optimizer(model, loss_module)

    resume = load_resume(
        last_ckpt,
        model,
        loss_module,
        optimizer,
        device,
        fold,
        fold_protocol["fold_protocol_sha256"],
    )

    start_epoch = resume["start_epoch"]
    best_epoch = resume["best_epoch"]
    best_metric = resume["best_metric"]
    epochs_without_improvement = (
        resume["epochs_without_improvement"]
    )

    print(
        "    Resume: {}".format(
            "YES" if resume["resumed"] else "NO"
        )
    )
    print("    Start epoch: {}".format(start_epoch))

    if start_epoch > MAX_EPOCHS:
        raise RuntimeError(
            "Fold {} checkpoint exceeds MAX_EPOCHS, "
            "but completion report is missing.".format(fold)
        )

    train_dataset = GeoRCPDataset(
        train_stems,
        images,
        step07,
    )
    val_dataset = GeoRCPDataset(
        val_stems,
        images,
        step07,
    )
    vloader = validation_loader(val_dataset)

    invocation_start = time.perf_counter()
    stop_reason = "MAX_EPOCHS"
    last_completed_epoch = start_epoch - 1

    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        print("\n" + "-" * 110)
        print(
            "FOLD {} | EPOCH {:02d}/{:02d}".format(
                fold,
                epoch,
                MAX_EPOCHS,
            )
        )

        lrs = apply_epoch_lrs(optimizer, epoch)

        print(
            "    LR encoder={:.3e} | head={:.3e} | "
            "uncertainty={:.3e}".format(
                lrs["encoder"],
                lrs["georcp"],
                lrs["uncertainty"],
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
            fold,
            epoch,
        )

        val_report = validate_one_epoch(
            model,
            loss_module,
            vloader,
            device,
            fold,
        )

        tr = train_report["mean_losses"]
        va = val_report["mean_losses"]

        metric = float(va["raw_group_sum"])

        improved = (
            metric
            < best_metric - EARLY_STOP_MIN_DELTA
        )

        if improved:
            best_metric = metric
            best_epoch = epoch
            epochs_without_improvement = 0

            save_checkpoint(
                best_ckpt,
                model,
                loss_module,
                optimizer,
                fold,
                epoch,
                best_epoch,
                best_metric,
                epochs_without_improvement,
                fold_protocol["fold_protocol_sha256"],
            )
        else:
            epochs_without_improvement += 1

        row = {
            "fold": fold,
            "epoch": epoch,
            "encoder_lr": lrs["encoder"],
            "head_lr": lrs["georcp"],
            "uncertainty_lr": lrs["uncertainty"],
            "train_total": tr["total"],
            "train_raw_group_sum": tr["raw_group_sum"],
            "train_region_group": tr["region_group"],
            "train_curve_group": tr["curve_group"],
            "train_point_group": tr["point_group"],
            "val_total": va["total"],
            "val_raw_group_sum": va["raw_group_sum"],
            "val_region_group": va["region_group"],
            "val_curve_group": va["curve_group"],
            "val_point_group": va["point_group"],
            "log_var_region": train_report["log_vars"]["region"],
            "log_var_curve": train_report["log_vars"]["curve"],
            "log_var_point": train_report["log_vars"]["point"],
            "train_seconds": train_report["elapsed_seconds"],
            "val_seconds": val_report["elapsed_seconds"],
            "peak_allocated_gb": (
                train_report["memory"]["peak_allocated_gb"]
            ),
            "improved": int(improved),
        }

        upsert_history_row(history_csv, row)

        save_checkpoint(
            last_ckpt,
            model,
            loss_module,
            optimizer,
            fold,
            epoch,
            best_epoch,
            best_metric,
            epochs_without_improvement,
            fold_protocol["fold_protocol_sha256"],
        )

        last_completed_epoch = epoch

        save_json(
            running_json,
            {
                "status": "RUNNING",
                "fold": fold,
                "epoch_completed": epoch,
                "best_epoch": best_epoch,
                "best_val_raw_group_sum": best_metric,
                "epochs_without_improvement": (
                    epochs_without_improvement
                ),
                "global_protocol_sha256": (
                    EXPECTED_NOAUG_PROTOCOL_SHA
                ),
                "fold_protocol_sha256": (
                    fold_protocol["fold_protocol_sha256"]
                ),
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
            "      train raw={:.5f} | R={:.5f} C={:.5f} P={:.5f}".format(
                tr["raw_group_sum"],
                tr["region_group"],
                tr["curve_group"],
                tr["point_group"],
            )
        )
        print(
            "      val   raw={:.5f} | R={:.5f} C={:.5f} P={:.5f}".format(
                va["raw_group_sum"],
                va["region_group"],
                va["curve_group"],
                va["point_group"],
            )
        )
        print(
            "      best epoch={} | best val raw={:.5f} | {}".format(
                best_epoch,
                best_metric,
                (
                    "IMPROVED"
                    if improved
                    else "no improvement ({}/{})".format(
                        epochs_without_improvement,
                        EARLY_STOP_PATIENCE,
                    )
                ),
            )
        )
        print(
            "      train {:.2f} min | val {:.2f} min | "
            "peak {:.3f} GB".format(
                train_report["elapsed_seconds"] / 60.0,
                val_report["elapsed_seconds"] / 60.0,
                train_report["memory"]["peak_allocated_gb"],
            )
        )

        if (
            epoch >= MIN_EPOCHS_BEFORE_STOP
            and epochs_without_improvement
            >= EARLY_STOP_PATIENCE
        ):
            stop_reason = "EARLY_STOPPING"
            print(
                "\n    Fold {} early stopping criterion reached.".format(
                    fold
                )
            )
            break

    elapsed = time.perf_counter() - invocation_start

    if not best_ckpt.exists():
        raise RuntimeError(
            "Fold {} ended without a best checkpoint.".format(
                fold
            )
        )

    best_state = torch_load_checkpoint(
        best_ckpt,
        device,
    )

    if int(best_state.get("fold", -1)) != fold:
        raise RuntimeError(
            "Fold {} best checkpoint fold mismatch.".format(
                fold
            )
        )

    if (
        best_state.get("global_protocol_sha256")
        != EXPECTED_NOAUG_PROTOCOL_SHA
    ):
        raise RuntimeError(
            "Fold {} best checkpoint protocol mismatch.".format(
                fold
            )
        )

    if (
        best_state.get("fold_protocol_sha256")
        != fold_protocol["fold_protocol_sha256"]
    ):
        raise RuntimeError(
            "Fold {} best checkpoint fold-protocol mismatch.".format(
                fold
            )
        )

    complete = {
        "status": "COMPLETE",
        "scientific_status": (
            "DEFINITIVE_NOAUG_FOLD{}_TRAINING_COMPLETE_"
            "OUTER_TEST_UNTOUCHED"
        ).format(fold),
        "script_revision": SCRIPT_REVISION,
        "script_sha256": current_script_sha(),
        "global_protocol_sha256": EXPECTED_NOAUG_PROTOCOL_SHA,
        "fold_protocol_sha256": (
            fold_protocol["fold_protocol_sha256"]
        ),
        "scientific_design_sha256": EXPECTED_DESIGN_SHA,
        "active_target_master_sha256": EXPECTED_G3_SHA,
        "implementation_spec_sha256": (
            EXPECTED_IMPLEMENTATION_SPEC_SHA
        ),
        "source_bundle_sha256": EXPECTED_SOURCE_BUNDLE_SHA,
        "mirage_wrapper_sha256": EXPECTED_MIRAGE_WRAPPER_SHA,
        "trackb_split_master_sha256": CANONICAL_TRACKB_SPLIT_SHA,
        "fold": fold,
        "train_images": len(train_stems),
        "val_images": len(val_stems),
        "train_stem_sha256": train_split["stem_sha256"],
        "val_stem_sha256": val_split["stem_sha256"],
        "precision": PRECISION,
        "physical_batch_size": PHYSICAL_BATCH,
        "effective_batch_size": EFFECTIVE_BATCH,
        "augmentation": AUGMENTATION_POLICY,
        "best_epoch": int(best_state["epoch"]),
        "best_val_raw_group_sum": float(
            best_state["best_metric"]
        ),
        "last_completed_epoch": int(last_completed_epoch),
        "stop_reason": stop_reason,
        "elapsed_seconds_current_invocation": float(elapsed),
        "best_checkpoint": str(best_ckpt),
        "last_checkpoint": str(last_ckpt),
        "history_csv": str(history_csv),
        "outer_test_resolved": False,
        "outer_test_evaluated": False,
        "best_weights_preserved": True,
    }

    save_json(complete_json, complete)

    append_project_log_once(
        "<!-- STEP08_FINAL_NOAUG_FOLD{}_COMPLETE -->".format(
            fold
        ),
        (
            "Definitive no-augmentation GeoRCP-MIRAGE Fold {} "
            "training complete. Best epoch {}; best validation "
            "raw-group sum {:.6f}. Outer test remained untouched. "
            "Global protocol SHA {}."
        ).format(
            fold,
            complete["best_epoch"],
            complete["best_val_raw_group_sum"],
            EXPECTED_NOAUG_PROTOCOL_SHA,
        ),
    )

    print("\n" + "-" * 110)
    print("FOLD {} TRAINING COMPLETE".format(fold))
    print("-" * 110)
    print("Best epoch: {}".format(complete["best_epoch"]))
    print(
        "Best validation raw-group sum: {:.6f}".format(
            complete["best_val_raw_group_sum"]
        )
    )
    print("Stop reason: {}".format(stop_reason))
    print("Outer test evaluated: NO")
    print("Best weights preserved: YES")
    print("Completion report:")
    print(complete_json)

    del optimizer
    del loss_module
    del model
    del vloader
    del val_dataset
    del train_dataset
    reset_cuda()

    return complete


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 110)
    print(
        "RVO-ME STEP 08 FINAL: GeoRCP-MIRAGE FOLDS 2–5"
    )
    print(
        "SCRIPT REVISION: {}".format(
            SCRIPT_REVISION
        )
    )
    print("=" * 110)

    print(
        "\n[1/5] Loading exact frozen GeoRCP implementation..."
    )
    step07, source_sha = load_exact_step07()
    print("    Implementation-spec SHA: PASS")
    print("    Source-bundle SHA: PASS")

    print(
        "\n[2/5] Verifying final no-augmentation protocol and RTX 4080..."
    )
    environment = verify_frozen_state(source_sha)
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
    print("    Python/PyTorch/CUDA: PASS")
    print("    BF16: PASS")
    print("    geometry_v3: PASS")
    print("    RTX-4080 preflight/revalidation: PASS")
    print("    Definitive Fold-1 no-augmentation protocol: PASS")
    print(
        "    Frozen global protocol SHA: {}".format(
            EXPECTED_NOAUG_PROTOCOL_SHA
        )
    )
    print("    Further protocol tuning: NO")

    print(
        "\n[3/5] Resolving image index; outer-test roles remain inaccessible..."
    )
    images = step07.images_by_stem()
    available_stems = sorted(images)

    if len(available_stems) != EXPECTED_IMAGES:
        raise RuntimeError(
            "Expected {} images; found {}.".format(
                EXPECTED_IMAGES,
                len(available_stems),
            )
        )

    print(
        "    Images: {} PASS".format(
            len(available_stems)
        )
    )
    print("    Folds queued: 2, 3, 4, 5")
    print("    Outer-test resolution path: DISABLED")

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    update_manifest()

    print(
        "\n[4/5] Training Folds 2–5 sequentially..."
    )

    device = torch.device("cuda")
    results = {}

    for fold in FOLDS_TO_TRAIN:
        results[str(fold)] = train_fold(
            fold,
            step07,
            images,
            available_stems,
            device,
            environment,
        )
        update_manifest()

    print(
        "\n[5/5] Finalizing five-fold training manifest..."
    )
    manifest = update_manifest()

    if (
        manifest["status"]
        != "ALL_FIVE_FOLDS_TRAINING_COMPLETE"
    ):
        raise RuntimeError(
            "Training loop ended, but five-fold manifest is incomplete."
        )

    append_project_log_once(
        "<!-- STEP08_ALL_FIVE_FOLDS_NOAUG_TRAINING_COMPLETE -->",
        (
            "All five Track-B GeoRCP-MIRAGE folds have frozen best "
            "checkpoints under global no-augmentation protocol SHA {}. "
            "All outer-test roles remained unresolved and unevaluated."
        ).format(
            EXPECTED_NOAUG_PROTOCOL_SHA
        ),
    )

    print("\n" + "=" * 110)
    print(
        "STEP 08 COMPLETE — ALL FIVE FOLDS TRAINED / OUTER TESTS UNTOUCHED"
    )
    print("=" * 110)

    fold1 = load_json(FOLD1_COMPLETE)
    print(
        "Fold 1 | best epoch {:2d} | val raw {:.6f} | existing definitive run".format(
            int(fold1["best_epoch"]),
            float(fold1["best_val_raw_group_sum"]),
        )
    )

    for fold in FOLDS_TO_TRAIN:
        c = results[str(fold)]
        print(
            "Fold {} | best epoch {:2d} | val raw {:.6f} | {}".format(
                fold,
                int(c["best_epoch"]),
                float(c["best_val_raw_group_sum"]),
                c["stop_reason"],
            )
        )

    print("\nOuter tests resolved: NO")
    print("Outer tests evaluated: NO")
    print("All best checkpoints preserved: YES")
    print("\nFive-fold manifest:")
    print(FIVEFOLD_MANIFEST)
    print(
        "\nNEXT: send FIVEFOLD_TRAINING_MANIFEST.json "
        "before any held-out evaluation."
    )


if __name__ == "__main__":
    main()
