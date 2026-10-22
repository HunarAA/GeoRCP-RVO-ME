# =============================================================================
# RVO-ME FIGURE 3 — LAPTOP/PORTABLE PATH VERSION
# GeoRCP-MIRAGE qualitative held-out examples
#
# IMPORTANT
# ---------
# This script does NOT modify Step 09.
# It remaps Step-09's old D:\RVO_ME paths to the current project root:
#     E:\Papers\RVO_ME
#
# It is safe for qualitative reporting:
# - selects examples using ground truth only
# - reuses frozen fold checkpoints
# - reuses frozen validation-selected thresholds
# - no training, fine-tuning, threshold tuning, or metric recomputation
# - inference is batch size 1 so it can run on the RTX 4060 laptop
# =============================================================================

from __future__ import annotations

import importlib.util
import json
import sys
import gc
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# =============================================================================
# USER ROOT
# =============================================================================

ROOT = Path(r"E:\Papers\RVO_ME")
PROJECT = ROOT / "project"
SCRIPTS = PROJECT / "scripts"

STEP09_SCRIPT = SCRIPTS / "09_georcp_calibrate_and_evaluate.py"

OUTPUT_DIR = ROOT / "paper_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PNG = OUTPUT_DIR / "Figure3_GeoRCP_Qualitative.png"
OUTPUT_SVG = OUTPUT_DIR / "Figure3_GeoRCP_Qualitative.svg"
OUTPUT_MANIFEST = OUTPUT_DIR / "Figure3_GeoRCP_Qualitative_manifest.json"


# =============================================================================
# DISPLAY
# =============================================================================

COLORS = {
    "IRF": "#e74c3c",
    "SRF": "#f39c12",
    "ELM": "#00bcd4",
    "EZ":  "#8e44ad",
    "HF":  "#2ecc71",
}

REGION_ALPHA = 0.38
CURVE_LINEWIDTH = 1.8
HF_MARKER_SIZE = 22

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.weight"] = "normal"
plt.rcParams["axes.titleweight"] = "normal"


# =============================================================================
# MODULE/PATH HELPERS
# =============================================================================

def load_module(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not import {}".format(path))

    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def patch_step09_paths(step09) -> None:
    """
    Step 09 was originally frozen on D:\\RVO_ME.
    Patch path VARIABLES ONLY in the imported module so its scientific logic,
    hashes, thresholds, and decoding remain unchanged.
    """
    step09.ROOT = ROOT
    step09.DATASET = ROOT / "RVO-Lesion"
    step09.PROJECT = PROJECT

    step09.DEMOGRAPHICS_XLSX = (
        step09.DATASET / "Demographics of the participants.xlsx"
    )

    step09.STEP07_SCRIPT = (
        PROJECT / "scripts" / "07_georcp_preflight_v2_ampfix_py38compat.py"
    )

    step09.SPLIT_ROOT = PROJECT / "splits" / "patient_5fold_v1"

    step09.TARGET_ROOT = PROJECT / "targets" / "geometry_v3"
    step09.TARGET_DIR = step09.TARGET_ROOT / "targets"
    step09.G3_COMPLETE = (
        step09.TARGET_ROOT / "GEOMETRY_V3_COMPLETE.json"
    )

    step09.MIRAGE_WRAPPER = (
        PROJECT
        / "foundation_models"
        / "mirage_base_preflight_v1"
        / "official_mirage_hf.py"
    )

    step09.TRAIN_ROOT = (
        PROJECT / "training" / "georcp_v1" / "definitive_v2"
    )
    step09.TRAIN_MANIFEST = (
        step09.TRAIN_ROOT / "FIVEFOLD_TRAINING_MANIFEST.json"
    )

    step09.GLOBAL_PROTOCOL_FILE = (
        PROJECT
        / "design"
        / "geometry_aware_rcp_v1"
        / "RTX4080_ENGINEERING_FROZEN_v2.json"
    )

    step09.EVAL_ROOT = (
        PROJECT
        / "evaluation"
        / "georcp_v1"
        / "trackB_fivefold_v1"
    )
    step09.CALIBRATION_ROOT = step09.EVAL_ROOT / "calibration"
    step09.EVAL_PROTOCOL_FILE = (
        step09.EVAL_ROOT / "STEP09_EVALUATION_PROTOCOL_FROZEN.json"
    )
    step09.HELDOUT_STARTED_FILE = (
        step09.EVAL_ROOT / "HELDOUT_EVALUATION_STARTED.json"
    )
    step09.HELDOUT_COMPLETE_FILE = (
        step09.EVAL_ROOT / "STEP09_HELDOUT_RESULTS_COMPLETE.json"
    )
    step09.PER_IMAGE_ALL_CSV = (
        step09.EVAL_ROOT / "heldout_per_image_metrics.csv"
    )
    step09.SUMMARY_CSV = (
        step09.EVAL_ROOT / "heldout_summary.csv"
    )
    step09.PROJECT_LOG = PROJECT / "PROJECT_LOG.md"


def require(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            "{} not found:\n{}".format(label, path)
        )


def localize_checkpoint(path_value: str) -> Path:
    """
    Training manifests created on the desktop may store D:\\RVO_ME paths.
    If that exact path is unavailable, map only the root to E:\\Papers\\RVO_ME.
    """
    p = Path(str(path_value))
    if p.exists():
        return p

    raw = str(path_value).replace("/", "\\")
    old_prefix = "D:\\RVO_ME\\"

    if raw.lower().startswith(old_prefix.lower()):
        rel = raw[len(old_prefix):]
        mapped = ROOT / Path(rel)

        if mapped.exists():
            return mapped

    # A second portable fallback: locate by filename inside definitive_v2.
    candidates = list(
        (PROJECT / "training" / "georcp_v1" / "definitive_v2")
        .rglob(p.name)
    )

    if len(candidates) == 1:
        return candidates[0]

    raise FileNotFoundError(
        "Could not locate checkpoint stored in manifest:\n{}\n"
        "Expected a corresponding copy under:\n{}".format(
            path_value,
            PROJECT / "training" / "georcp_v1" / "definitive_v2",
        )
    )


# =============================================================================
# GT-ONLY SELECTION
# =============================================================================

def count_true_runs(mask: np.ndarray) -> int:
    a = np.asarray(mask, dtype=bool)
    if a.size == 0:
        return 0

    starts = a & np.concatenate(
        [np.asarray([True], dtype=bool), ~a[:-1]]
    )
    return int(starts.sum())


def internal_gap_stats(presence: np.ndarray) -> Tuple[int, int]:
    p = np.asarray(presence, dtype=bool)
    idx = np.flatnonzero(p)

    if idx.size < 2:
        return 0, 0

    inside = p[idx[0]: idx[-1] + 1]
    gaps = ~inside
    return int(gaps.sum()), count_true_runs(gaps)


def describe_gt(stem: str, step09) -> Dict[str, Any]:
    target_path = step09.TARGET_DIR / "{}.npz".format(stem)
    require(target_path, "geometry_v3 target")

    with np.load(str(target_path)) as z:
        ys = slice(step09.PT, step09.PT + step09.NH)
        xs = slice(step09.PL, step09.PL + step09.NW)

        srf = z["srf_mask"][ys, xs] > 0.5
        irf = z["irf_mask"][ys, xs] > 0.5

        curve_presence = (
            z["curve_presence"][
                :,
                step09.PL: step09.PL + step09.NW,
            ]
            > 0.5
        )

        curve_valid = (
            z["curve_column_valid"][
                step09.PL: step09.PL + step09.NW
            ]
            > 0.5
        )

        curve_presence = (
            curve_presence & curve_valid[None, :]
        )

        hf_count = int((z["hf_offset_mask"] > 0).sum())

    elm_gap_cols, elm_gap_runs = internal_gap_stats(
        curve_presence[0]
    )
    ez_gap_cols, ez_gap_runs = internal_gap_stats(
        curve_presence[1]
    )

    flags = {
        "srf_present": bool(srf.any()),
        "irf_present": bool(irf.any()),
        "elm_present": bool(curve_presence[0].any()),
        "ez_present": bool(curve_presence[1].any()),
        "hf_present": hf_count > 0,
    }

    return {
        "stem": stem,
        **flags,
        "biomarker_diversity": int(sum(flags.values())),
        "hf_count": hf_count,
        "region_coverage": float(
            (srf.sum() + irf.sum())
            / float(step09.NH * step09.NW)
        ),
        "curve_coverage": float(
            curve_presence.sum()
            / float(2 * step09.NW)
        ),
        "elm_gap_columns": elm_gap_cols,
        "elm_gap_runs": elm_gap_runs,
        "ez_gap_columns": ez_gap_cols,
        "ez_gap_runs": ez_gap_runs,
        "total_gap_columns": elm_gap_cols + ez_gap_cols,
        "total_gap_runs": elm_gap_runs + ez_gap_runs,
    }


def robust_distances(
    rows: Sequence[Dict[str, Any]],
    keys: Sequence[str],
) -> List[float]:
    m = np.asarray(
        [[float(r[k]) for k in keys] for r in rows],
        dtype=np.float64,
    )

    med = np.median(m, axis=0)
    q25 = np.percentile(m, 25, axis=0)
    q75 = np.percentile(m, 75, axis=0)
    scale = q75 - q25
    scale[scale < 1e-12] = 1.0

    return list(
        np.abs((m - med) / scale).sum(axis=1)
    )


def choose_row1(
    desc: Sequence[Dict[str, Any]],
    exclude: Sequence[str],
) -> Dict[str, Any]:
    ex = set(exclude)
    pool = [d for d in desc if d["stem"] not in ex]
    max_div = max(d["biomarker_diversity"] for d in pool)
    pool = [
        d for d in pool
        if d["biomarker_diversity"] == max_div
    ]

    ds = robust_distances(
        pool,
        ["region_coverage", "curve_coverage", "hf_count"],
    )

    return dict(
        sorted(
            zip(ds, pool),
            key=lambda x: (x[0], x[1]["stem"]),
        )[0][1]
    )


def choose_row2(
    desc: Sequence[Dict[str, Any]],
    exclude: Sequence[str],
) -> Dict[str, Any]:
    ex = set(exclude)

    pool = [
        d for d in desc
        if d["stem"] not in ex
        and d["elm_present"]
        and d["ez_present"]
        and d["total_gap_runs"] > 0
    ]

    if not pool:
        pool = [
            d for d in desc
            if d["stem"] not in ex
            and (d["elm_present"] or d["ez_present"])
            and d["total_gap_runs"] > 0
        ]

    if not pool:
        raise RuntimeError(
            "No held-out GT case with an internal curve gap."
        )

    ds = robust_distances(
        pool,
        ["total_gap_runs", "total_gap_columns"],
    )

    return dict(
        sorted(
            zip(ds, pool),
            key=lambda x: (
                x[0],
                -x[1]["biomarker_diversity"],
                x[1]["stem"],
            ),
        )[0][1]
    )


def choose_row3(
    desc: Sequence[Dict[str, Any]],
    exclude: Sequence[str],
) -> Dict[str, Any]:
    ex = set(exclude)

    pool = [
        d for d in desc
        if d["stem"] not in ex
        and d["hf_count"] >= 3
        and (d["srf_present"] or d["irf_present"])
    ]

    if not pool:
        pool = [
            d for d in desc
            if d["stem"] not in ex
            and d["hf_count"] >= 2
            and (d["srf_present"] or d["irf_present"])
        ]

    if not pool:
        raise RuntimeError(
            "No held-out GT case with multiple HF and fluid."
        )

    ds = robust_distances(
        pool,
        ["hf_count", "region_coverage"],
    )

    return dict(
        sorted(
            zip(ds, pool),
            key=lambda x: (
                x[0],
                -x[1]["biomarker_diversity"],
                x[1]["stem"],
            ),
        )[0][1]
    )


# =============================================================================
# EXACT SINGLE-IMAGE INFERENCE
# =============================================================================

def infer_one(
    stem: str,
    fold: int,
    checkpoint: Path,
    step07,
    step09,
    images: Dict[str, Path],
    device: torch.device,
) -> Dict[str, Any]:

    # build_fold_model performs checkpoint SHA/protocol integrity checks.
    model = step09.build_fold_model(
        fold,
        checkpoint,
        step07,
        device,
    )

    sample = step07.load_sample(
        stem,
        images[stem],
        step09.TARGET_DIR,
    )

    image_tensor = sample["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        with torch.amp.autocast(
            "cuda",
            dtype=torch.bfloat16,
        ):
            out = model(image_tensor)

    region_pred = (
        out["region_logits"]
        .argmax(1)[0]
        .float()
        .cpu()
        .numpy()
    )[
        step09.PT: step09.PT + step09.NH,
        step09.PL: step09.PL + step09.NW,
    ]

    curve_prob = (
        torch.sigmoid(out["curve_presence_logits"][0])
        [:, step09.PL: step09.PL + step09.NW]
        .float()
        .cpu()
        .numpy()
    )

    curve_py = step09.pred_curve_y(
        out["curve_centerline_logits"][0]
    )

    hf_pred = step09.decode_hf(
        out["hf_heatmap_logits"][0],
        out["hf_offset"][0],
    )

    result = {
        "stem": stem,
        "region_pred": region_pred,
        "region_gt": sample["region_class"].numpy()[
            step09.PT: step09.PT + step09.NH,
            step09.PL: step09.PL + step09.NW,
        ],
        "region_valid": (
            sample["region_valid"].numpy()[
                step09.PT: step09.PT + step09.NH,
                step09.PL: step09.PL + step09.NW,
            ]
            > 0.5
        ),
        "curve_prob": curve_prob,
        "curve_gt": (
            sample["curve_presence"].numpy()[
                :,
                step09.PL: step09.PL + step09.NW,
            ]
            > 0.5
        ),
        "curve_valid": (
            sample["curve_column_valid"].numpy()[
                step09.PL: step09.PL + step09.NW
            ]
            > 0.5
        ),
        "curve_py": curve_py,
        "curve_gy": step09.gt_curve_y(
            sample["curve_centerline"].numpy()
        ),
        "hf_pred": hf_pred,
        "hf_gt": step09.gt_hf(sample),
    }

    del out
    del image_tensor
    del sample
    del model

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    return result


# =============================================================================
# DRAWING
# =============================================================================

def hex_rgb01(value: str):
    value = value.lstrip("#")
    return tuple(
        int(value[i:i + 2], 16) / 255.0
        for i in (0, 2, 4)
    )


def show_image(ax, image: np.ndarray) -> None:
    ax.imshow(
        image,
        cmap="gray",
        vmin=0,
        vmax=255,
        interpolation="nearest",
    )
    ax.axis("off")


def overlay_region(
    ax,
    mask: np.ndarray,
    color: str,
) -> None:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return

    rgb = hex_rgb01(color)

    rgba = np.zeros(
        (mask.shape[0], mask.shape[1], 4),
        dtype=np.float32,
    )

    rgba[..., 0] = rgb[0]
    rgba[..., 1] = rgb[1]
    rgba[..., 2] = rgb[2]
    rgba[..., 3] = (
        mask.astype(np.float32) * REGION_ALPHA
    )

    ax.imshow(rgba, interpolation="nearest")


def contiguous_runs(mask: np.ndarray):
    a = np.asarray(mask, dtype=bool)
    idx = np.flatnonzero(a)

    if idx.size == 0:
        return []

    runs = []
    start = int(idx[0])
    prev = int(idx[0])

    for val in idx[1:]:
        val = int(val)

        if val != prev + 1:
            runs.append((start, prev + 1))
            start = val

        prev = val

    runs.append((start, prev + 1))
    return runs


def plot_curve(
    ax,
    y: np.ndarray,
    presence: np.ndarray,
    color: str,
) -> None:
    """
    Separate line segment per present run.
    Therefore no visualization-only gap filling is introduced.
    """
    y = np.asarray(y, dtype=np.float32)
    p = np.asarray(presence, dtype=bool)

    for start, stop in contiguous_runs(p):
        xx = np.arange(start, stop)
        yy = y[start:stop]
        ok = np.isfinite(yy)

        if ok.sum() >= 2:
            ax.plot(
                xx[ok],
                yy[ok],
                color=color,
                linewidth=CURVE_LINEWIDTH,
                solid_capstyle="round",
            )
        elif ok.sum() == 1:
            ax.scatter(
                xx[ok],
                yy[ok],
                s=8,
                c=color,
            )


def plot_hf(
    ax,
    points: np.ndarray,
) -> None:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)

    if len(pts) == 0:
        return

    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        s=HF_MARKER_SIZE,
        facecolors=COLORS["HF"],
        edgecolors="black",
        linewidths=0.45,
    )


def draw_gt(ax, rec: Dict[str, Any]) -> None:
    valid = rec["region_valid"]
    reg = rec["region_gt"]

    overlay_region(
        ax,
        (reg == 2) & valid,
        COLORS["IRF"],
    )
    overlay_region(
        ax,
        (reg == 1) & valid,
        COLORS["SRF"],
    )

    for ch, name in ((0, "ELM"), (1, "EZ")):
        present = (
            rec["curve_gt"][ch]
            & rec["curve_valid"]
        )
        plot_curve(
            ax,
            rec["curve_gy"][ch],
            present,
            COLORS[name],
        )

    plot_hf(ax, rec["hf_gt"])


def draw_pred(
    ax,
    rec: Dict[str, Any],
    curve_threshold: float,
    hf_threshold: float,
) -> None:
    valid = rec["region_valid"]
    reg = rec["region_pred"]

    overlay_region(
        ax,
        (reg == 2) & valid,
        COLORS["IRF"],
    )
    overlay_region(
        ax,
        (reg == 1) & valid,
        COLORS["SRF"],
    )

    for ch, name in ((0, "ELM"), (1, "EZ")):
        present = (
            rec["curve_prob"][ch] >= curve_threshold
        ) & rec["curve_valid"]

        plot_curve(
            ax,
            rec["curve_py"][ch],
            present,
            COLORS[name],
        )

    pred_hf = np.asarray(
        [
            [float(p["x"]), float(p["y"])]
            for p in rec["hf_pred"]
            if float(p["score"]) >= hf_threshold
        ],
        dtype=np.float32,
    ).reshape(-1, 2)

    plot_hf(ax, pred_hf)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print("\n" + "=" * 100)
    print("RVO-ME FIGURE 3: PORTABLE GT-ONLY QUALITATIVE REPORT")
    print("=" * 100)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    print("GPU: {}".format(torch.cuda.get_device_name(0)))
    print("Project root: {}".format(ROOT))

    print("\n[1/7] Loading Step 09 and remapping old paths...")
    step09 = load_module(
        "georcp_step09_figure3",
        STEP09_SCRIPT,
    )
    patch_step09_paths(step09)

    require(
        step09.STEP07_SCRIPT,
        "Step-07 exact implementation",
    )
    require(
        step09.TRAIN_MANIFEST,
        "Five-fold training manifest",
    )
    require(
        step09.EVAL_PROTOCOL_FILE,
        "Frozen Step-09 evaluation protocol",
    )
    require(
        step09.HELDOUT_COMPLETE_FILE,
        "Completed Step-09 held-out results",
    )

    print("  Step07: {}".format(step09.STEP07_SCRIPT))
    print("  Manifest: {}".format(step09.TRAIN_MANIFEST))
    print("  Evaluation root: {}".format(step09.EVAL_ROOT))

    print("\n[2/7] Verifying exact implementation and frozen results...")

    # This verifies the exact source bundle and implementation SHA.
    # Unlike verify_training_manifest(), it does not require the RTX 4080.
    step07, source_sha = step09.load_exact_step07()

    manifest = step09.load_json(
        step09.TRAIN_MANIFEST
    )
    protocol = step09.load_json(
        step09.EVAL_PROTOCOL_FILE
    )
    complete = step09.load_json(
        step09.HELDOUT_COMPLETE_FILE
    )

    if (
        manifest.get("status")
        != "ALL_FIVE_FOLDS_TRAINING_COMPLETE"
    ):
        raise RuntimeError(
            "Five-fold training manifest is not complete."
        )

    if (
        manifest.get("global_protocol_sha256")
        != step09.EXPECTED_GLOBAL_PROTOCOL_SHA
    ):
        raise RuntimeError(
            "Training protocol SHA mismatch."
        )

    if (
        protocol.get("evaluation_protocol_sha256")
        != complete.get("evaluation_protocol_sha256")
    ):
        raise RuntimeError(
            "Evaluation protocol/result SHA mismatch."
        )

    if complete.get("status") != "COMPLETE":
        raise RuntimeError(
            "Held-out evaluation is not marked COMPLETE."
        )

    print("  Exact GeoRCP source bundle: PASS")
    print("  Training manifest: PASS")
    print("  Frozen evaluation result: PASS")

    print("\n[3/7] Resolving all frozen outer-test images...")
    images = step07.images_by_stem()
    available = sorted(images)

    stem_to_fold = {}
    heldout_stems = []

    for fold in range(1, 6):
        stems = step09.resolve_split_role(
            fold,
            "test",
            available,
        )

        for stem in stems:
            if stem in stem_to_fold:
                raise RuntimeError(
                    "Duplicate held-out stem: {}".format(stem)
                )

            stem_to_fold[stem] = fold
            heldout_stems.append(stem)

        print(
            "  Fold {}: {} B-scans".format(
                fold,
                len(stems),
            )
        )

    if len(heldout_stems) != step09.EXPECTED_IMAGES:
        raise RuntimeError(
            "Outer-test coverage mismatch."
        )

    print(
        "  Total: {} / {} PASS".format(
            len(heldout_stems),
            step09.EXPECTED_IMAGES,
        )
    )

    print("\n[4/7] Selecting three examples using ground truth only...")
    descriptors = []

    for i, stem in enumerate(sorted(heldout_stems), 1):
        d = describe_gt(stem, step09)
        d["fold"] = int(stem_to_fold[stem])
        descriptors.append(d)

        if i % 500 == 0 or i == len(heldout_stems):
            print(
                "  GT descriptors: {}/{}".format(
                    i,
                    len(heldout_stems),
                )
            )

    r1 = choose_row1(descriptors, [])
    r1["row_label"] = "Multi-biomarker case"

    r2 = choose_row2(
        descriptors,
        [r1["stem"]],
    )
    r2["row_label"] = "Curve-discontinuity case"

    r3 = choose_row3(
        descriptors,
        [r1["stem"], r2["stem"]],
    )
    r3["row_label"] = "HF + fluid case"

    selected = [r1, r2, r3]

    for i, d in enumerate(selected, 1):
        print(
            "  Row {}: {} | fold={} | diversity={} | HF={} | gaps={}".format(
                i,
                d["stem"],
                d["fold"],
                d["biomarker_diversity"],
                d["hf_count"],
                d["total_gap_runs"],
            )
        )

    print(
        "  Prediction quality was NOT used for selection."
    )

    print("\n[5/7] Running batch-1 inference with frozen fold checkpoints...")
    device = torch.device("cuda")
    records = {}
    checkpoint_used = {}

    # Process selected images one at a time.
    # This is intentionally lighter than Step 09's evaluation batch size 7.
    for d in selected:
        fold = int(d["fold"])

        cp = localize_checkpoint(
            manifest["folds"][str(fold)]["best_checkpoint"]
        )
        checkpoint_used[d["stem"]] = cp

        print(
            "  {} | fold {} | checkpoint {}".format(
                d["stem"],
                fold,
                cp,
            )
        )

        records[d["stem"]] = infer_one(
            d["stem"],
            fold,
            cp,
            step07,
            step09,
            images,
            device,
        )

    print("\n[6/7] Rendering publication figure...")

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(15, 10),
    )

    titles = [
        "(a) Original OCT B-scan",
        "(b) Ground-truth annotations",
        "(c) GeoRCP-MIRAGE predictions",
    ]

    for col, title in enumerate(titles):
        axes[0, col].set_title(
            title,
            fontsize=14,
            pad=8,
            fontweight="normal",
        )

    manifest_rows = []

    for row_i, d in enumerate(selected):
        stem = d["stem"]
        fold = int(d["fold"])
        rec = records[stem]

        image = cv2.imread(
            str(images[stem]),
            cv2.IMREAD_GRAYSCALE,
        )

        if image is None:
            raise RuntimeError(
                "Could not read {}".format(images[stem])
            )

        if image.shape != (step09.NH, step09.NW):
            raise RuntimeError(
                "Unexpected image shape {} for {}".format(
                    image.shape,
                    stem,
                )
            )

        show_image(axes[row_i, 0], image)

        show_image(axes[row_i, 1], image)
        draw_gt(axes[row_i, 1], rec)

        show_image(axes[row_i, 2], image)

        th = protocol[
            "threshold_calibration"
        ][str(fold)]

        curve_th = float(
            th["curve"]["selected_threshold"]
        )
        hf_th = float(
            th["hf"]["selected_threshold"]
        )

        draw_pred(
            axes[row_i, 2],
            rec,
            curve_th,
            hf_th,
        )

        axes[row_i, 0].text(
            -0.04,
            0.5,
            d["row_label"],
            transform=axes[row_i, 0].transAxes,
            rotation=90,
            va="center",
            ha="right",
            fontsize=11,
            fontweight="normal",
        )

        manifest_rows.append(
            {
                "row": row_i + 1,
                "label": d["row_label"],
                "stem": stem,
                "fold": fold,
                "image": str(images[stem]),
                "checkpoint": str(checkpoint_used[stem]),
                "curve_threshold": curve_th,
                "hf_threshold": hf_th,
                "gt_descriptor": {
                    k: v
                    for k, v in d.items()
                    if k != "row_label"
                },
            }
        )

    legend = [
        Patch(
            facecolor=COLORS["IRF"],
            edgecolor="none",
            alpha=REGION_ALPHA,
            label="IRF",
        ),
        Patch(
            facecolor=COLORS["SRF"],
            edgecolor="none",
            alpha=REGION_ALPHA,
            label="SRF",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["ELM"],
            lw=CURVE_LINEWIDTH,
            label="ELM",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["EZ"],
            lw=CURVE_LINEWIDTH,
            label="EZ",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COLORS["HF"],
            markeredgecolor="black",
            markersize=6,
            label="HF",
        ),
    ]

    fig.legend(
        handles=legend,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
        fontsize=12,
    )

    fig.subplots_adjust(
        left=0.055,
        right=0.995,
        top=0.94,
        bottom=0.09,
        wspace=0.025,
        hspace=0.09,
    )

    fig.savefig(
        str(OUTPUT_PNG),
        dpi=600,
        bbox_inches="tight",
    )
    fig.savefig(
        str(OUTPUT_SVG),
        bbox_inches="tight",
    )
    plt.close(fig)

    print("\n[7/7] Saving reproducibility manifest...")

    report = {
        "figure": "Figure 3 - GeoRCP-MIRAGE qualitative held-out predictions",
        "project_root": str(ROOT),
        "gpu_used_for_visualization": torch.cuda.get_device_name(0),
        "selection_uses_ground_truth_only": True,
        "selection_uses_predictions": False,
        "selection_uses_per_image_metrics": False,
        "training_performed": False,
        "threshold_tuning_performed": False,
        "metrics_recomputed": False,
        "inference_batch_size": 1,
        "evaluation_protocol_sha256":
            protocol["evaluation_protocol_sha256"],
        "rows": manifest_rows,
        "outputs": {
            "png": str(OUTPUT_PNG),
            "svg": str(OUTPUT_SVG),
        },
        "caption": (
            "Figure 3. Representative qualitative predictions of "
            "GeoRCP-MIRAGE on held-out patient-disjoint OCT B-scans. "
            "Columns show the original OCT B-scan, ground-truth "
            "geometry-specific annotations, and GeoRCP-MIRAGE predictions. "
            "IRF and SRF are displayed as regions, ELM and EZ as curves, "
            "and HF as point detections. Examples were selected using "
            "ground-truth characteristics without reference to model "
            "prediction quality."
        ),
    }

    OUTPUT_MANIFEST.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 100)
    print("FIGURE 3 COMPLETE")
    print("=" * 100)
    print("PNG:      {}".format(OUTPUT_PNG))
    print("SVG:      {}".format(OUTPUT_SVG))
    print("Manifest: {}".format(OUTPUT_MANIFEST))
    print(
        "\nNo training, no test-derived tuning, and no metric "
        "recomputation were performed."
    )


if __name__ == "__main__":
    main()
