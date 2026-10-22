# =============================================================================
# RVO-ME STEP 05D
# Aggregate frozen Track-B U-Net++ five-fold out-of-fold results
#
# Purpose
# -------
# 1) Verify all five folds were completed under the same frozen protocols.
# 2) Verify each test patient appears in exactly one outer test fold.
# 3) Aggregate exact out-of-fold (OOF) TP/FP/FN counts across all 130 patients.
# 4) Report:
#       - fold mean +/- sample SD
#       - exact pooled OOF Dice/IoU
#       - per-class pooled OOF Dice/IoU
#       - patient-cluster bootstrap 95% CIs
# 5) Save frozen aggregation artifacts for later paired statistical comparison.
#
# IMPORTANT
# ---------
# This script performs NO model training and NO hyperparameter tuning.
# It reads only already-frozen Fold 1-5 test outputs.
# =============================================================================

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Any

import numpy as np


# =============================================================================
# PATHS / FROZEN IDENTIFIERS
# =============================================================================

ROOT = Path(r"E:\Papers\RVO_ME")

SEG_ROOT = (
    ROOT
    / "project"
    / "baselines"
    / "segmentation"
)

CANONICAL_PROTOCOL_DIR = (
    SEG_ROOT
    / "patient5fold_protocol_v1"
)

TRACKB_ROOT = (
    SEG_ROOT
    / "trackB_conventional_v1"
)

UNETPP_ROOT = (
    TRACKB_ROOT
    / "unetpp"
)

OUTPUT_DIR = (
    UNETPP_ROOT
    / "aggregation_5fold_v1"
)

EXPECTED_CANONICAL_PROTOCOL_SHA256 = (
    "52ed516dcc21f1c51068c59fb662c9a0030b143e6440d6f4a0380df713c9bcaf"
)

EXPECTED_TRAINING_PROTOCOL_SHA256 = (
    "3a37f6471518dcebbb3553da0b436ab371a5cec079053255c2df581a9ba63e31"
)

FOLDS = [1, 2, 3, 4, 5]

EXPECTED_TEST_PATIENTS_PER_FOLD = 26

EXPECTED_TOTAL_PATIENTS = 130

EXPECTED_TOTAL_TEST_IMAGES = 3012

CLASSES = ["SRF", "IRF", "ELM", "EZ"]

SCALES = {
    "native256": {
        "global_file": "global_metrics_native256.json",
        "patient_file": "per_patient_metrics_native256.csv",
        "label": "native 256x256",
    },
    "original570x380": {
        "global_file": "global_metrics_original570x380.json",
        "patient_file": "per_patient_metrics_original570x380.csv",
        "label": "original 570x380",
    },
}

BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 20260815
CI_LOW = 2.5
CI_HIGH = 97.5


# =============================================================================
# GENERIC HELPERS
# =============================================================================

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            payload,
            f,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")

    fieldnames = list(rows[0].keys())

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)

    return h.hexdigest()


def stable_json_sha256(payload: Any) -> str:
    blob = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(blob).hexdigest()


def as_int(value: Any, field: str) -> int:
    try:
        return int(float(value))
    except Exception as exc:
        raise RuntimeError(
            f"Could not parse integer field '{field}': {value!r}"
        ) from exc


def as_float(value: Any, field: str) -> float:
    try:
        return float(value)
    except Exception as exc:
        raise RuntimeError(
            f"Could not parse float field '{field}': {value!r}"
        ) from exc


def pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def mean_sd(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)

    if arr.size < 2:
        raise RuntimeError("Need at least two folds for SD.")

    return {
        "mean": float(np.mean(arr)),
        "sd": float(np.std(arr, ddof=1)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


# =============================================================================
# METRIC HELPERS
# =============================================================================

def metric_from_counts(
    tp: np.ndarray | float | int,
    fp: np.ndarray | float | int,
    fn: np.ndarray | float | int,
):
    tp = np.asarray(tp, dtype=np.float64)
    fp = np.asarray(fp, dtype=np.float64)
    fn = np.asarray(fn, dtype=np.float64)

    dice_den = 2.0 * tp + fp + fn
    iou_den = tp + fp + fn

    dice = np.where(
        dice_den == 0.0,
        1.0,
        (2.0 * tp) / dice_den,
    )

    iou = np.where(
        iou_den == 0.0,
        1.0,
        tp / iou_den,
    )

    return dice, iou


def pooled_metrics_from_patient_rows(
    rows: List[Dict[str, Any]],
) -> Dict[str, Any]:

    classes = {}
    dices = []
    ious = []

    for cls in CLASSES:
        tp = sum(as_int(r[f"{cls}_tp"], f"{cls}_tp") for r in rows)
        fp = sum(as_int(r[f"{cls}_fp"], f"{cls}_fp") for r in rows)
        fn = sum(as_int(r[f"{cls}_fn"], f"{cls}_fn") for r in rows)

        gt_pixels = sum(
            as_int(r[f"{cls}_gt_pixels"], f"{cls}_gt_pixels")
            for r in rows
        )

        pred_pixels = sum(
            as_int(r[f"{cls}_pred_pixels"], f"{cls}_pred_pixels")
            for r in rows
        )

        dice, iou = metric_from_counts(tp, fp, fn)

        dice = float(dice)
        iou = float(iou)

        classes[cls] = {
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "gt_pixels": int(gt_pixels),
            "pred_pixels": int(pred_pixels),
            "dice": dice,
            "iou": iou,
        }

        dices.append(dice)
        ious.append(iou)

    return {
        "classes": classes,
        "mean_foreground_dice": float(np.mean(dices)),
        "mean_foreground_iou": float(np.mean(ious)),
        "n_patients": len(rows),
    }


# =============================================================================
# BOOTSTRAP
# =============================================================================

def patient_cluster_bootstrap(
    rows: List[Dict[str, Any]],
    replicates: int,
    seed: int,
) -> Dict[str, Any]:

    n = len(rows)

    if n != EXPECTED_TOTAL_PATIENTS:
        raise RuntimeError(
            f"Bootstrap expected {EXPECTED_TOTAL_PATIENTS} patients, got {n}."
        )

    counts = np.zeros(
        (n, len(CLASSES), 3),
        dtype=np.int64,
    )

    for i, row in enumerate(rows):
        for j, cls in enumerate(CLASSES):
            counts[i, j, 0] = as_int(row[f"{cls}_tp"], f"{cls}_tp")
            counts[i, j, 1] = as_int(row[f"{cls}_fp"], f"{cls}_fp")
            counts[i, j, 2] = as_int(row[f"{cls}_fn"], f"{cls}_fn")

    rng = np.random.default_rng(seed)

    class_dice_boot = np.empty(
        (replicates, len(CLASSES)),
        dtype=np.float64,
    )

    class_iou_boot = np.empty(
        (replicates, len(CLASSES)),
        dtype=np.float64,
    )

    # Patient-cluster bootstrap:
    # resample patients (not B-scans) with replacement, preserving all
    # within-patient pixel counts as a cluster.
    for b in range(replicates):
        idx = rng.integers(
            low=0,
            high=n,
            size=n,
            endpoint=False,
        )

        sampled = counts[idx].sum(axis=0)

        tp = sampled[:, 0]
        fp = sampled[:, 1]
        fn = sampled[:, 2]

        dice, iou = metric_from_counts(
            tp,
            fp,
            fn,
        )

        class_dice_boot[b] = dice
        class_iou_boot[b] = iou

    mean_dice_boot = np.mean(
        class_dice_boot,
        axis=1,
    )

    mean_iou_boot = np.mean(
        class_iou_boot,
        axis=1,
    )

    def ci(arr: np.ndarray) -> Dict[str, float]:
        return {
            "lower": float(np.percentile(arr, CI_LOW)),
            "upper": float(np.percentile(arr, CI_HIGH)),
        }

    out = {
        "method": "patient_cluster_bootstrap_percentile",
        "replicates": int(replicates),
        "seed": int(seed),
        "confidence_level": 0.95,
        "mean_foreground_dice": ci(mean_dice_boot),
        "mean_foreground_iou": ci(mean_iou_boot),
        "classes": {},
    }

    for j, cls in enumerate(CLASSES):
        out["classes"][cls] = {
            "dice": ci(class_dice_boot[:, j]),
            "iou": ci(class_iou_boot[:, j]),
        }

    return out


# =============================================================================
# INPUT VERIFICATION
# =============================================================================

def verify_frozen_protocols() -> Dict[str, Any]:

    canonical_file = (
        CANONICAL_PROTOCOL_DIR
        / "PROTOCOL_FROZEN.json"
    )

    training_file = (
        TRACKB_ROOT
        / "TRAINING_PROTOCOL_FROZEN.json"
    )

    if not canonical_file.exists():
        raise FileNotFoundError(canonical_file)

    if not training_file.exists():
        raise FileNotFoundError(training_file)

    canonical = load_json(canonical_file)
    training = load_json(training_file)

    canonical_sha = str(
        canonical.get(
            "canonical_protocol_sha256",
            canonical.get(
                "protocol_sha256",
                "",
            ),
        )
    )

    # The canonical protocol file schema may store the hash under a
    # different field. The fold summaries are the definitive frozen
    # cross-check below, so an absent self-field is not treated as failure.
    training_sha = str(
        training.get(
            "training_protocol_sha256",
            "",
        )
    )

    if training_sha != EXPECTED_TRAINING_PROTOCOL_SHA256:
        raise RuntimeError(
            "Frozen training protocol SHA mismatch.\n"
            f"Expected: {EXPECTED_TRAINING_PROTOCOL_SHA256}\n"
            f"Found:    {training_sha}"
        )

    return {
        "canonical_file": str(canonical_file),
        "training_file": str(training_file),
        "canonical_file_sha256": sha256_file(canonical_file),
        "training_file_sha256": sha256_file(training_file),
        "canonical_internal_sha_if_present": canonical_sha,
        "training_protocol_sha256": training_sha,
    }


def load_and_verify_fold(
    fold: int,
) -> Dict[str, Any]:

    fold_dir = (
        UNETPP_ROOT
        / f"fold_{fold}"
    )

    complete_file = (
        fold_dir
        / "FOLD_COMPLETE.json"
    )

    if not complete_file.exists():
        raise FileNotFoundError(
            f"Fold {fold} is not complete: {complete_file}"
        )

    complete = load_json(complete_file)

    if complete.get("status") != "COMPLETE":
        raise RuntimeError(
            f"Fold {fold}: status is not COMPLETE."
        )

    if as_int(complete.get("fold"), "fold") != fold:
        raise RuntimeError(
            f"Fold {fold}: FOLD_COMPLETE fold ID mismatch."
        )

    if (
        str(complete.get("canonical_protocol_sha256"))
        != EXPECTED_CANONICAL_PROTOCOL_SHA256
    ):
        raise RuntimeError(
            f"Fold {fold}: canonical protocol SHA mismatch."
        )

    if (
        str(complete.get("training_protocol_sha256"))
        != EXPECTED_TRAINING_PROTOCOL_SHA256
    ):
        raise RuntimeError(
            f"Fold {fold}: training protocol SHA mismatch."
        )

    if bool(
        complete.get(
            "test_was_used_for_checkpoint_selection"
        )
    ):
        raise RuntimeError(
            f"Fold {fold}: test was marked as used for selection."
        )

    test_patients = as_int(
        complete.get("test_patients"),
        "test_patients",
    )

    if test_patients != EXPECTED_TEST_PATIENTS_PER_FOLD:
        raise RuntimeError(
            f"Fold {fold}: expected "
            f"{EXPECTED_TEST_PATIENTS_PER_FOLD} test patients, "
            f"found {test_patients}."
        )

    result = {
        "fold": fold,
        "dir": fold_dir,
        "complete": complete,
        "inputs": {
            "FOLD_COMPLETE.json":
                sha256_file(complete_file),
        },
        "scales": {},
    }

    for scale_key, scale_cfg in SCALES.items():

        global_path = (
            fold_dir
            / "test"
            / scale_cfg["global_file"]
        )

        patient_path = (
            fold_dir
            / "test"
            / scale_cfg["patient_file"]
        )

        if not global_path.exists():
            raise FileNotFoundError(global_path)

        if not patient_path.exists():
            raise FileNotFoundError(patient_path)

        global_metrics = load_json(global_path)
        patient_rows = read_csv(patient_path)

        patients = [
            str(r["patient_id"]).strip()
            for r in patient_rows
        ]

        if len(patient_rows) != EXPECTED_TEST_PATIENTS_PER_FOLD:
            raise RuntimeError(
                f"Fold {fold}, {scale_key}: expected "
                f"{EXPECTED_TEST_PATIENTS_PER_FOLD} patient rows, "
                f"found {len(patient_rows)}."
            )

        if len(set(patients)) != len(patients):
            raise RuntimeError(
                f"Fold {fold}, {scale_key}: duplicate patient IDs."
            )

        # Reconstruct exact fold-global foreground metrics from saved
        # per-patient additive counts and confirm they match the saved JSON.
        reconstructed = pooled_metrics_from_patient_rows(
            patient_rows
        )

        for cls in CLASSES:
            a = reconstructed["classes"][cls]
            b = global_metrics["classes"][cls]

            for field in ["tp", "fp", "fn", "gt_pixels", "pred_pixels"]:
                if int(a[field]) != int(b[field]):
                    raise RuntimeError(
                        f"Fold {fold}, {scale_key}, {cls}: "
                        f"{field} mismatch between patient CSV and global JSON."
                    )

            if not math.isclose(
                float(a["dice"]),
                float(b["dice"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise RuntimeError(
                    f"Fold {fold}, {scale_key}, {cls}: Dice mismatch."
                )

            if not math.isclose(
                float(a["iou"]),
                float(b["iou"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise RuntimeError(
                    f"Fold {fold}, {scale_key}, {cls}: IoU mismatch."
                )

        if not math.isclose(
            float(reconstructed["mean_foreground_dice"]),
            float(global_metrics["mean_foreground_dice"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                f"Fold {fold}, {scale_key}: mean Dice mismatch."
            )

        if not math.isclose(
            float(reconstructed["mean_foreground_iou"]),
            float(global_metrics["mean_foreground_iou"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                f"Fold {fold}, {scale_key}: mean IoU mismatch."
            )

        result["inputs"][scale_cfg["global_file"]] = (
            sha256_file(global_path)
        )

        result["inputs"][scale_cfg["patient_file"]] = (
            sha256_file(patient_path)
        )

        result["scales"][scale_key] = {
            "global_metrics": global_metrics,
            "patient_rows": patient_rows,
        }

    return result


# =============================================================================
# AGGREGATION
# =============================================================================

def aggregate_folds(
    fold_results: List[Dict[str, Any]],
) -> Dict[str, Any]:

    all_output = {}

    for scale_key, scale_cfg in SCALES.items():

        fold_rows = []
        all_patient_rows = []

        seen_patients = set()

        for fr in fold_results:
            fold = fr["fold"]
            gm = fr["scales"][scale_key]["global_metrics"]
            prs = fr["scales"][scale_key]["patient_rows"]

            row = {
                "fold": fold,
                "selected_epoch":
                    as_int(
                        fr["complete"]["selected_epoch"],
                        "selected_epoch",
                    ),
                "validation_native256_mean_dice":
                    as_float(
                        fr["complete"][
                            "validation_native256_mean_dice"
                        ],
                        "validation_native256_mean_dice",
                    ),
                "mean_foreground_dice":
                    float(gm["mean_foreground_dice"]),
                "mean_foreground_iou":
                    float(gm["mean_foreground_iou"]),
            }

            for cls in CLASSES:
                row[f"{cls}_dice"] = float(
                    gm["classes"][cls]["dice"]
                )
                row[f"{cls}_iou"] = float(
                    gm["classes"][cls]["iou"]
                )

            fold_rows.append(row)

            fold_patient_ids = set()

            for patient_row in prs:
                patient_id = str(
                    patient_row["patient_id"]
                ).strip()

                if patient_id in seen_patients:
                    raise RuntimeError(
                        f"{scale_key}: patient {patient_id} "
                        "appears in more than one outer test fold."
                    )

                seen_patients.add(patient_id)
                fold_patient_ids.add(patient_id)

                augmented = dict(patient_row)
                augmented["outer_test_fold"] = fold

                all_patient_rows.append(
                    augmented
                )

            if (
                len(fold_patient_ids)
                != EXPECTED_TEST_PATIENTS_PER_FOLD
            ):
                raise RuntimeError(
                    f"Fold {fold}, {scale_key}: test patient count failure."
                )

        if len(seen_patients) != EXPECTED_TOTAL_PATIENTS:
            raise RuntimeError(
                f"{scale_key}: expected {EXPECTED_TOTAL_PATIENTS} unique "
                f"OOF patients, found {len(seen_patients)}."
            )

        if len(all_patient_rows) != EXPECTED_TOTAL_PATIENTS:
            raise RuntimeError(
                f"{scale_key}: expected {EXPECTED_TOTAL_PATIENTS} "
                f"patient rows, found {len(all_patient_rows)}."
            )

        pooled = pooled_metrics_from_patient_rows(
            all_patient_rows
        )

        bootstrap = patient_cluster_bootstrap(
            all_patient_rows,
            replicates=BOOTSTRAP_REPLICATES,
            seed=BOOTSTRAP_SEED,
        )

        fold_summary = {
            "mean_foreground_dice":
                mean_sd(
                    [
                        float(r["mean_foreground_dice"])
                        for r in fold_rows
                    ]
                ),

            "mean_foreground_iou":
                mean_sd(
                    [
                        float(r["mean_foreground_iou"])
                        for r in fold_rows
                    ]
                ),

            "classes": {},
        }

        for cls in CLASSES:
            fold_summary["classes"][cls] = {
                "dice":
                    mean_sd(
                        [
                            float(r[f"{cls}_dice"])
                            for r in fold_rows
                        ]
                    ),
                "iou":
                    mean_sd(
                        [
                            float(r[f"{cls}_iou"])
                            for r in fold_rows
                        ]
                    ),
            }

        all_output[scale_key] = {
            "label": scale_cfg["label"],
            "fold_rows": fold_rows,
            "patient_rows": all_patient_rows,
            "fold_summary": fold_summary,
            "pooled_oof": pooled,
            "bootstrap_95ci": bootstrap,
        }

    # Cross-scale patient identity must match exactly.
    patients_256 = {
        str(r["patient_id"]).strip()
        for r in all_output["native256"]["patient_rows"]
    }

    patients_570 = {
        str(r["patient_id"]).strip()
        for r in all_output["original570x380"]["patient_rows"]
    }

    if patients_256 != patients_570:
        raise RuntimeError(
            "Native-256 and original-570 patient sets do not match."
        )

    return all_output


# =============================================================================
# SAVE OUTPUTS
# =============================================================================

def save_outputs(
    protocol_info: Dict[str, Any],
    fold_results: List[Dict[str, Any]],
    aggregate: Dict[str, Any],
) -> Dict[str, Any]:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    input_fingerprints = {
        "canonical_protocol_sha256":
            EXPECTED_CANONICAL_PROTOCOL_SHA256,
        "training_protocol_sha256":
            EXPECTED_TRAINING_PROTOCOL_SHA256,
        "files": {},
    }

    for fr in fold_results:
        fold = fr["fold"]

        for filename, digest in fr["inputs"].items():
            key = f"fold_{fold}/{filename}"
            input_fingerprints["files"][key] = digest

    input_fingerprints["aggregate_input_sha256"] = (
        stable_json_sha256(
            input_fingerprints["files"]
        )
    )

    save_json(
        OUTPUT_DIR / "input_fingerprints.json",
        input_fingerprints,
    )

    # Write per-scale artifacts.
    for scale_key, result in aggregate.items():

        write_csv(
            OUTPUT_DIR
            / f"fold_metrics_{scale_key}.csv",
            result["fold_rows"],
        )

        write_csv(
            OUTPUT_DIR
            / f"oof_per_patient_{scale_key}.csv",
            result["patient_rows"],
        )

        save_json(
            OUTPUT_DIR
            / f"fold_summary_{scale_key}.json",
            result["fold_summary"],
        )

        save_json(
            OUTPUT_DIR
            / f"pooled_oof_metrics_{scale_key}.json",
            result["pooled_oof"],
        )

        save_json(
            OUTPUT_DIR
            / f"patient_bootstrap_95ci_{scale_key}.json",
            result["bootstrap_95ci"],
        )

        # Compact CI CSV useful for manuscript tables.
        ci_rows = []

        pooled = result["pooled_oof"]
        boot = result["bootstrap_95ci"]

        ci_rows.append({
            "target": "Mean foreground",
            "metric": "Dice",
            "estimate": pooled["mean_foreground_dice"],
            "ci95_lower": boot["mean_foreground_dice"]["lower"],
            "ci95_upper": boot["mean_foreground_dice"]["upper"],
        })

        ci_rows.append({
            "target": "Mean foreground",
            "metric": "IoU",
            "estimate": pooled["mean_foreground_iou"],
            "ci95_lower": boot["mean_foreground_iou"]["lower"],
            "ci95_upper": boot["mean_foreground_iou"]["upper"],
        })

        for cls in CLASSES:
            ci_rows.append({
                "target": cls,
                "metric": "Dice",
                "estimate": pooled["classes"][cls]["dice"],
                "ci95_lower": boot["classes"][cls]["dice"]["lower"],
                "ci95_upper": boot["classes"][cls]["dice"]["upper"],
            })

            ci_rows.append({
                "target": cls,
                "metric": "IoU",
                "estimate": pooled["classes"][cls]["iou"],
                "ci95_lower": boot["classes"][cls]["iou"]["lower"],
                "ci95_upper": boot["classes"][cls]["iou"]["upper"],
            })

        write_csv(
            OUTPUT_DIR
            / f"pooled_oof_with_95ci_{scale_key}.csv",
            ci_rows,
        )

    # Global compact summary.
    compact = {
        "status": "COMPLETE",
        "step": "05D",
        "model": "U-Net++",
        "track": "B_patient_disjoint_5fold",
        "n_folds": 5,
        "n_unique_oof_patients": EXPECTED_TOTAL_PATIENTS,
        "n_total_images_expected": EXPECTED_TOTAL_TEST_IMAGES,
        "canonical_protocol_sha256":
            EXPECTED_CANONICAL_PROTOCOL_SHA256,
        "training_protocol_sha256":
            EXPECTED_TRAINING_PROTOCOL_SHA256,
        "aggregate_input_sha256":
            input_fingerprints["aggregate_input_sha256"],
        "bootstrap": {
            "method":
                "patient_cluster_bootstrap_percentile",
            "replicates":
                BOOTSTRAP_REPLICATES,
            "seed":
                BOOTSTRAP_SEED,
        },
        "scales": {},
    }

    for scale_key, result in aggregate.items():
        compact["scales"][scale_key] = {
            "fold_mean_dice":
                result["fold_summary"][
                    "mean_foreground_dice"
                ]["mean"],
            "fold_sd_dice":
                result["fold_summary"][
                    "mean_foreground_dice"
                ]["sd"],
            "fold_mean_iou":
                result["fold_summary"][
                    "mean_foreground_iou"
                ]["mean"],
            "fold_sd_iou":
                result["fold_summary"][
                    "mean_foreground_iou"
                ]["sd"],
            "pooled_oof_mean_dice":
                result["pooled_oof"][
                    "mean_foreground_dice"
                ],
            "pooled_oof_mean_iou":
                result["pooled_oof"][
                    "mean_foreground_iou"
                ],
            "pooled_oof_mean_dice_ci95":
                result["bootstrap_95ci"][
                    "mean_foreground_dice"
                ],
            "pooled_oof_mean_iou_ci95":
                result["bootstrap_95ci"][
                    "mean_foreground_iou"
                ],
        }

    compact["aggregation_summary_sha256"] = (
        stable_json_sha256(compact)
    )

    save_json(
        OUTPUT_DIR / "STEP05D_COMPLETE.json",
        compact,
    )

    return compact


# =============================================================================
# PRINT REPORT
# =============================================================================

def print_scale_report(
    scale_key: str,
    result: Dict[str, Any],
) -> None:

    print("\n" + "=" * 100)
    print(
        f"U-Net++ TRACK-B FIVE-FOLD OOF — {result['label'].upper()}"
    )
    print("=" * 100)

    print("\nFold results")
    print("-" * 92)
    print(
        f"{'Fold':<8}"
        f"{'Epoch':>10}"
        f"{'Val Dice':>16}"
        f"{'Test Dice':>16}"
        f"{'Test IoU':>16}"
    )
    print("-" * 92)

    for row in result["fold_rows"]:
        print(
            f"{int(row['fold']):<8}"
            f"{int(row['selected_epoch']):>10}"
            f"{pct(float(row['validation_native256_mean_dice'])):>16}"
            f"{pct(float(row['mean_foreground_dice'])):>16}"
            f"{pct(float(row['mean_foreground_iou'])):>16}"
        )

    fs = result["fold_summary"]

    print("-" * 92)
    print(
        f"{'Fold mean':<18}"
        f"{pct(fs['mean_foreground_dice']['mean']):>16}"
        f" +/- {100.0 * fs['mean_foreground_dice']['sd']:.2f} pp"
    )

    print(
        f"{'Fold mean IoU':<18}"
        f"{pct(fs['mean_foreground_iou']['mean']):>16}"
        f" +/- {100.0 * fs['mean_foreground_iou']['sd']:.2f} pp"
    )

    pooled = result["pooled_oof"]
    boot = result["bootstrap_95ci"]

    print("\nExact pooled OOF metrics across all 130 held-out patients")
    print("-" * 92)
    print(
        f"{'Class':<18}"
        f"{'Dice':>14}"
        f"{'Dice 95% CI':>26}"
        f"{'IoU':>14}"
        f"{'IoU 95% CI':>26}"
    )
    print("-" * 92)

    for cls in CLASSES:
        cm = pooled["classes"][cls]
        bc = boot["classes"][cls]

        dice_ci = (
            f"[{pct(bc['dice']['lower'])}, "
            f"{pct(bc['dice']['upper'])}]"
        )

        iou_ci = (
            f"[{pct(bc['iou']['lower'])}, "
            f"{pct(bc['iou']['upper'])}]"
        )

        print(
            f"{cls:<18}"
            f"{pct(cm['dice']):>14}"
            f"{dice_ci:>26}"
            f"{pct(cm['iou']):>14}"
            f"{iou_ci:>26}"
        )

    print("-" * 92)

    mean_dice_ci = (
        f"[{pct(boot['mean_foreground_dice']['lower'])}, "
        f"{pct(boot['mean_foreground_dice']['upper'])}]"
    )

    mean_iou_ci = (
        f"[{pct(boot['mean_foreground_iou']['lower'])}, "
        f"{pct(boot['mean_foreground_iou']['upper'])}]"
    )

    print(
        f"{'Mean foreground':<18}"
        f"{pct(pooled['mean_foreground_dice']):>14}"
        f"{mean_dice_ci:>26}"
        f"{pct(pooled['mean_foreground_iou']):>14}"
        f"{mean_iou_ci:>26}"
    )

    print("-" * 92)


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("\n" + "=" * 100)
    print("RVO-ME STEP 05D: AGGREGATE TRACK-B U-NET++ FIVE-FOLD RESULTS")
    print("=" * 100)

    print("\n[1/5] Verifying frozen protocols...")

    protocol_info = verify_frozen_protocols()

    print("    Training protocol SHA: PASS")
    print(f"    {EXPECTED_TRAINING_PROTOCOL_SHA256}")
    print("    Canonical patient protocol expected SHA:")
    print(f"    {EXPECTED_CANONICAL_PROTOCOL_SHA256}")

    print("\n[2/5] Verifying Fold 1-5 completed outputs...")

    fold_results = []

    for fold in FOLDS:
        fr = load_and_verify_fold(fold)
        fold_results.append(fr)

        print(
            f"    Fold {fold}: PASS | "
            f"epoch={fr['complete']['selected_epoch']} | "
            f"test_patients={fr['complete']['test_patients']} | "
            f"test_images={fr['complete']['test_images']}"
        )

    total_images = sum(
        as_int(fr["complete"]["test_images"], "test_images")
        for fr in fold_results
    )

    if total_images != EXPECTED_TOTAL_TEST_IMAGES:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL_TEST_IMAGES} total OOF images, "
            f"found {total_images}."
        )

    print(
        f"    Total OOF images: {total_images} / "
        f"{EXPECTED_TOTAL_TEST_IMAGES} PASS"
    )

    print("\n[3/5] Aggregating exact patient-level OOF counts...")

    aggregate = aggregate_folds(
        fold_results
    )

    print(
        f"    Unique OOF patients: "
        f"{EXPECTED_TOTAL_PATIENTS} PASS"
    )

    print(
        "    Every saved patient appears in exactly one outer test fold: PASS"
    )

    print("\n[4/5] Running patient-cluster bootstrap...")
    print(
        f"    Replicates: {BOOTSTRAP_REPLICATES:,}"
    )
    print(
        f"    Seed:       {BOOTSTRAP_SEED}"
    )

    compact = save_outputs(
        protocol_info,
        fold_results,
        aggregate,
    )

    print("\n[5/5] Final frozen five-fold report...")

    print_scale_report(
        "native256",
        aggregate["native256"],
    )

    print_scale_report(
        "original570x380",
        aggregate["original570x380"],
    )

    print("\n" + "=" * 100)
    print("RVO-ME STEP 05D COMPLETE")
    print("=" * 100)
    print("\nNo training or tuning was performed.")
    print(
        "Primary aggregation uses exact additive TP/FP/FN counts "
        "across all 130 out-of-fold patients."
    )
    print(
        "95% CIs use patient-cluster bootstrap, not B-scan bootstrap."
    )
    print("\nResults saved to:")
    print(OUTPUT_DIR)
    print("\nAggregation summary SHA-256:")
    print(compact["aggregation_summary_sha256"])


if __name__ == "__main__":
    main()
