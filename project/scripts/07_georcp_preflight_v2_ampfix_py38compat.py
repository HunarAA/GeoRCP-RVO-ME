
from __future__ import annotations

# RVO-ME STEP 07
# Consolidated GeoRCP-MIRAGE implementation + preflight.
# NOT definitive training. All model weights produced here are discarded.
#
# If geometry_v2's HF heatmap/offset alignment is inconsistent with the
# frozen CenterNet-style heatmap+offset representation, this script NEVER
# edits geometry_v2. It creates geometry_v3 as a versioned correction and
# uses that for preflight.

import csv
import gc
import hashlib
import importlib.util
import json
import math
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin


# =============================================================================
# Paths / frozen identifiers
# =============================================================================

ROOT = Path(r"E:\Papers\RVO_ME")
DATASET = ROOT / "RVO-Lesion"
PROJECT = ROOT / "project"

IMAGE_DIR = DATASET / "Image_Seg" / "images"
SPLIT_ROOT = PROJECT / "splits" / "patient_5fold_v1"

DESIGN_DIR = PROJECT / "design" / "geometry_aware_rcp_v1"
DESIGN_JSON = DESIGN_DIR / "SCIENTIFIC_DESIGN_FROZEN.json"
AMENDMENT_JSON = DESIGN_DIR / "TARGET_VALIDITY_AMENDMENT_v1_1.json"

G2 = PROJECT / "targets" / "geometry_v2"
G2_TARGETS = G2 / "targets"
G2_COMPLETE = G2 / "GEOMETRY_V2_COMPLETE.json"

G3 = PROJECT / "targets" / "geometry_v3"
G3_TARGETS = G3 / "targets"
G3_COMPLETE = G3 / "GEOMETRY_V3_COMPLETE.json"

MIRAGE_DIR = PROJECT / "foundation_models" / "mirage_base_preflight_v1"
MIRAGE_WRAPPER = MIRAGE_DIR / "official_mirage_hf.py"
HF_CACHE = MIRAGE_DIR / "hf_cache"

PREFLIGHT_ROOT = PROJECT / "preflight" / "step07_georcp_v1"
PROJECT_LOG = PROJECT / "PROJECT_LOG.md"

DESIGN_SHA = "9f1586d102523009443e2b88dca54a9ca8fa5c983f08fdde837184d436a95801"
AMENDMENT_SHA = "2a5e6e487ad7243be2cb54058c1515611cd22fad99494ddff3e7a135642b847c"
G2_SHA = "a84c306a6e7e8fb47b0797ccb02afaf6bd6b8d7a05f3d101281b072a343e451f"
MIRAGE_WRAPPER_SHA = "07fa63e2766b67dbc0c07f0d9d29db1562f707425ef2fbf90a688f62a94e6ede"

EXPECTED_IMAGES = 3012
EXPECTED_HF = 9212
LOG_MARKER = "<!-- STEP07_GEOMETRY_V3_CENTERNET_CORRECTION -->"

# =============================================================================
# Geometry / architecture
# =============================================================================

NW, NH = 570, 380
W, H = 576, 384
PL, PR, PT, PB = 3, 3, 2, 2
PATCH = 32
PH, PW = H // PATCH, W // PATCH
NTOK = PH * PW  # 216

HF_STRIDE = 2
HH, HW = H // HF_STRIDE, W // HF_STRIDE
HF_SIGMA = 1.0
IGNORE = 255

MIRAGE_REPO = "j-morano/MIRAGE-Base"
ENC_DIM = 768
BLOCKS = (2, 5, 8, 11)  # blocks 3,6,9,12
TDIM = 256

TASKS = ("region", "curve", "point")
LOCAL2, LOCAL4 = 32, 64
REASM = 64
FUSED4, FUSED2 = 128, 64

SEED = 20260815
SCRIPT_REVISION = "STEP07_AMP_DIAGNOSTIC_V2_PY38COMPAT_20260819"

# Engineering-smoke settings only; NOT final training hyperparameters.
PROFILE_ENC_LR = 1e-5
PROFILE_HEAD_LR = 1e-4
PROFILE_WD = 1e-2

OVERFIT_STEPS = 20
OVERFIT_ENC_LR = 2e-5
OVERFIT_HEAD_LR = 3e-4


# =============================================================================
# Generic helpers
# =============================================================================

def load_json(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json(p: Path, obj: Any):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=True)

def sha_file(p: Path):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def stable_sha(obj: Any):
    b = json.dumps(obj, sort_keys=True, ensure_ascii=False,
                   separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(b).hexdigest()

def array_sha(arrays: Dict[str, np.ndarray]):
    h = hashlib.sha256()
    for name in sorted(arrays):
        a = np.ascontiguousarray(arrays[name])
        h.update(name.encode())
        h.update(str(a.dtype).encode())
        h.update(str(tuple(a.shape)).encode())
        h.update(a.tobytes())
    return h.hexdigest()

def reset_cuda():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

def mem():
    g = 1024 ** 3
    return {
        "allocated_gb": torch.cuda.memory_allocated() / g,
        "reserved_gb": torch.cuda.memory_reserved() / g,
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / g,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / g,
    }

def params(m: nn.Module):
    return {
        "total": int(sum(p.numel() for p in m.parameters())),
        "trainable": int(sum(p.numel() for p in m.parameters()
                             if p.requires_grad)),
    }

def module_from_file(name: str, path: Path):
    """
    Import a Python source file while postponing annotation evaluation.

    Python 3.8 compatibility shim only. The imported file is NOT edited.
    The frozen MIRAGE wrapper remains byte-for-byte unchanged and its SHA-256
    is still verified before model construction.
    """
    import __future__

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod

    source = path.read_text(encoding="utf-8")
    code = compile(
        source,
        str(path),
        "exec",
        flags=__future__.annotations.compiler_flag,
        dont_inherit=True,
    )
    exec(code, mod.__dict__)
    return mod

def sanitize(s: str):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")


# =============================================================================
# Frozen input verification
# =============================================================================

def verify_inputs():
    d = load_json(DESIGN_JSON)
    if d.get("scientific_design_master_sha256") != DESIGN_SHA:
        raise RuntimeError("Scientific-design SHA mismatch.")

    a = load_json(AMENDMENT_JSON)
    if a.get("amendment_sha256") != AMENDMENT_SHA:
        raise RuntimeError("v1.1 amendment SHA mismatch.")

    g2 = load_json(G2_COMPLETE)
    if g2.get("geometry_v2_master_content_sha256") != G2_SHA:
        raise RuntimeError("geometry_v2 master SHA mismatch.")

    if sha_file(MIRAGE_WRAPPER) != MIRAGE_WRAPPER_SHA:
        raise RuntimeError("MIRAGE wrapper SHA mismatch.")

    if len(list(G2_TARGETS.glob("*.npz"))) != EXPECTED_IMAGES:
        raise RuntimeError("geometry_v2 file-count mismatch.")


# =============================================================================
# geometry_v2 HF semantic gate and geometry_v3 correction
# =============================================================================

def hf_alignment_audit(folder: Path):
    files = sorted(folder.glob("*.npz"))
    if len(files) != EXPECTED_IMAGES:
        raise RuntimeError(f"Expected {EXPECTED_IMAGES} packs in {folder}")

    npos = not_one = not_local_max = round_diff = 0
    min_at_center, max_at_center = 1.0, 0.0
    examples = []

    for p in files:
        with np.load(p) as z:
            heat = z["hf_heatmap"].astype(np.float32)
            off = z["hf_offset"].astype(np.float32)
            mask = z["hf_offset_mask"] > 0

        ys, xs = np.where(mask)
        for cy, cx in zip(ys, xs):
            npos += 1
            v = float(heat[cy, cx])
            min_at_center = min(min_at_center, v)
            max_at_center = max(max_at_center, v)

            if abs(v - 1.0) > 1e-3:
                not_one += 1

            dx = float(off[0, cy, cx])
            dy = float(off[1, cy, cx])
            nx = int(round(cx + dx))
            ny = int(round(cy + dy))
            nx = max(0, min(HW - 1, nx))
            ny = max(0, min(HH - 1, ny))
            if nx != cx or ny != cy:
                round_diff += 1

            y0, y1 = max(0, cy - 1), min(HH, cy + 2)
            x0, x1 = max(0, cx - 1), min(HW, cx + 2)
            lm = float(heat[y0:y1, x0:x1].max())
            if v < lm - 1e-4:
                not_local_max += 1
                if len(examples) < 10:
                    examples.append({
                        "stem": p.stem,
                        "cell_xy": [int(cx), int(cy)],
                        "offset_xy": [dx, dy],
                        "heat_at_cell": v,
                        "local_max": lm,
                        "nearest_grid_xy": [nx, ny],
                    })

    if npos != EXPECTED_HF:
        raise RuntimeError(f"Expected {EXPECTED_HF} HF cells; found {npos}")

    return {
        "positive_cells": npos,
        "offset_cells_peak_not_one": not_one,
        "offset_cells_not_local_max": not_local_max,
        "nearest_grid_differs_from_floor": round_diff,
        "min_heat_at_offset_cell": min_at_center,
        "max_heat_at_offset_cell": max_at_center,
        "requires_correction": (not_one > 0 or not_local_max > 0),
        "examples": examples,
    }

def add_gaussian(hm: np.ndarray, cx: int, cy: int, sigma=1.0):
    r = max(1, int(math.ceil(4 * sigma)))
    x0, x1 = max(0, cx-r), min(hm.shape[1], cx+r+1)
    y0, y1 = max(0, cy-r), min(hm.shape[0], cy+r+1)
    xs = np.arange(x0, x1, dtype=np.float32)
    ys = np.arange(y0, y1, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    g = np.exp(-((xx-cx)**2 + (yy-cy)**2)/(2*sigma*sigma))
    np.maximum(hm[y0:y1, x0:x1], g, out=hm[y0:y1, x0:x1])

def corrected_heatmap(mask: np.ndarray):
    hm = np.zeros((HH, HW), dtype=np.float32)
    ys, xs = np.where(mask > 0)
    for cy, cx in zip(ys, xs):
        add_gaussian(hm, int(cx), int(cy), HF_SIGMA)
    return hm.astype(np.float16)

def create_or_verify_g3(v2_audit: Dict[str, Any]):
    if G3_COMPLETE.exists():
        c = load_json(G3_COMPLETE)
        if c.get("status") != "COMPLETE":
            raise RuntimeError("Existing geometry_v3 is incomplete.")
        if c.get("parent_geometry_v2_master_sha256") != G2_SHA:
            raise RuntimeError("Existing geometry_v3 has wrong parent.")
        return c

    if G3.exists():
        shutil.rmtree(G3)
    G3_TARGETS.mkdir(parents=True, exist_ok=True)

    spec = {
        "name": "geometry_v3 HF CenterNet semantic-alignment correction",
        "parent_geometry_v2_master_sha256": G2_SHA,
        "scientific_design_sha256": DESIGN_SHA,
        "target_validity_v1_1_sha256": AMENDMENT_SHA,
        "architecture_change": False,
        "loss_family_change": False,
        "changed_array": "hf_heatmap only",
        "rule": "Gaussian peak is centered at floor(padded_coordinate/2), "
                "the same cell used by hf_offset_mask; hf_offset is unchanged.",
        "supporting_v2_audit": v2_audit,
    }
    spec["spec_sha256"] = stable_sha(spec)
    save_json(G3 / "CORRECTION_SPEC.json", spec)

    manifest = []
    total_pos = 0
    altered = 0

    files = sorted(G2_TARGETS.glob("*.npz"))
    print("    Creating geometry_v3...")
    for i, src in enumerate(files, 1):
        with np.load(src) as z:
            arrays = {k: z[k].copy() for k in z.files}

        old_hm = arrays["hf_heatmap"].copy()
        mask = arrays["hf_offset_mask"]
        total_pos += int(mask.sum())
        arrays["hf_heatmap"] = corrected_heatmap(mask)

        if not np.array_equal(old_hm, arrays["hf_heatmap"]):
            altered += 1

        ys, xs = np.where(mask > 0)
        if len(xs):
            vals = arrays["hf_heatmap"][ys, xs].astype(np.float32)
            if not np.allclose(vals, 1.0, atol=1e-3):
                raise RuntimeError(f"{src.stem}: corrected center != 1")

        out = G3_TARGETS / src.name
        np.savez_compressed(out, **arrays)
        manifest.append({"stem": src.stem, "content_sha256": array_sha(arrays)})

        if i % 500 == 0 or i == len(files):
            print(f"      {i}/{len(files)}")

    if total_pos != EXPECTED_HF:
        raise RuntimeError("geometry_v3 HF count mismatch.")

    master_obj = {
        "spec_sha256": spec["spec_sha256"],
        "parent_geometry_v2_master_sha256": G2_SHA,
        "files": manifest,
    }
    master_sha = stable_sha(master_obj)
    save_json(G3 / "CONTENT_HASH_MANIFEST.json",
              {"geometry_v3_master_content_sha256": master_sha, **master_obj})

    c = {
        "status": "COMPLETE",
        "target_version": "geometry_v3",
        "parent_geometry_v2_master_sha256": G2_SHA,
        "spec_sha256": spec["spec_sha256"],
        "geometry_v3_master_content_sha256": master_sha,
        "files": len(manifest),
        "hf_positive_cells": total_pos,
        "hf_center_cells_peak_one": True,
        "altered_hf_heatmap_files": altered,
        "frozen_policy": "Do not modify geometry_v3; later target changes require geometry_v4.",
    }
    save_json(G3_COMPLETE, c)
    return c

def append_g3_log(c: Dict[str, Any]):
    text = PROJECT_LOG.read_text(encoding="utf-8") if PROJECT_LOG.exists() else ""
    if LOG_MARKER in text:
        return
    block = f"""

{LOG_MARKER}
## Step 07 target semantic gate — geometry_v3

Pre-training validation found a heatmap/offset alignment mismatch in the
geometry_v2 HF target implementation. geometry_v2 remains frozen and unchanged.

- geometry_v3 changes only `hf_heatmap`
- Gaussian peak uses the same discrete stride-2 cell as `hf_offset_mask`
- `hf_offset` remains unchanged
- region/curve arrays remain unchanged
- architecture/loss families remain unchanged
- HF positive cells: {c["hf_positive_cells"]}
- geometry_v2 parent SHA: `{G2_SHA}`
- geometry_v3 master SHA: `{c["geometry_v3_master_content_sha256"]}`

Use geometry_v3 for GeoRCP training.
"""
    PROJECT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with PROJECT_LOG.open("a", encoding="utf-8") as f:
        f.write(block)

def resolve_targets():
    print("\n[2/10] Auditing geometry_v2 HF heatmap/offset semantics...")
    a2 = hf_alignment_audit(G2_TARGETS)
    print(f"    HF positive cells: {a2['positive_cells']}")
    print(f"    center cells with peak != 1: {a2['offset_cells_peak_not_one']}")
    print(f"    center cells not local maxima: {a2['offset_cells_not_local_max']}")

    if a2["requires_correction"]:
        print("    geometry_v2 semantic alignment: CORRECTION REQUIRED")
        print("\n[3/10] Creating/reusing versioned geometry_v3...")
        c3 = create_or_verify_g3(a2)
        append_g3_log(c3)
        folder = G3_TARGETS
        version = "geometry_v3"
        master_sha = c3["geometry_v3_master_content_sha256"]
        print("    geometry_v2 modified: NO")
        print(f"    geometry_v3 master SHA: {master_sha}")
    else:
        print("    geometry_v2 semantic alignment: PASS")
        print("\n[3/10] geometry_v3 correction not required.")
        folder = G2_TARGETS
        version = "geometry_v2"
        master_sha = G2_SHA

    active_audit = hf_alignment_audit(folder)
    if active_audit["offset_cells_peak_not_one"] or active_audit["offset_cells_not_local_max"]:
        raise RuntimeError("Active HF target is still semantically misaligned.")
    print("    Active HF CenterNet alignment: PASS")
    return folder, version, master_sha, a2, active_audit


# =============================================================================
# MIRAGE
# =============================================================================

def build_mirage_class():
    mod = module_from_file("official_mirage_step07", MIRAGE_WRAPPER)
    Base = mod.MIRAGEWrapper

    class MIRAGEhf(Base, PyTorchModelHubMixin):
        def __init__(self, input_size=512, patch_size=32,
                     modalities="bscan", size="base"):
            super().__init__(input_size=input_size, patch_size=patch_size,
                             modalities=modalities, size=size)
    return MIRAGEhf

def load_mirage():
    C = build_mirage_class()
    return C.from_pretrained(
        MIRAGE_REPO,
        cache_dir=str(HF_CACHE),
        input_size=512,
        patch_size=32,
        modalities="bscan",
        size="base",
    )


# =============================================================================
# GeoRCP architecture
# =============================================================================

class LN2d(nn.Module):
    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(c))
        self.b = nn.Parameter(torch.zeros(c))
        self.eps = eps

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, (x.shape[-1],), self.w, self.b, self.eps)
        return x.permute(0, 3, 1, 2)

class CNBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 7, padding=3, groups=c)
        self.norm = LN2d(c)
        self.pw1 = nn.Conv2d(c, 4*c, 1)
        self.pw2 = nn.Conv2d(4*c, c, 1)
        self.gamma = nn.Parameter(torch.ones(c) * 1e-6)

    def forward(self, x):
        r = x
        x = self.dw(x)
        x = self.norm(x)
        x = self.pw2(F.gelu(self.pw1(x)))
        return r + x * self.gamma[None, :, None, None]

class LocalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.s2 = nn.Sequential(
            nn.Conv2d(1, LOCAL2, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, LOCAL2), nn.GELU(), CNBlock(LOCAL2)
        )
        self.s4 = nn.Sequential(
            nn.Conv2d(LOCAL2, LOCAL4, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, LOCAL4), nn.GELU(),
            CNBlock(LOCAL4), CNBlock(LOCAL4)
        )

    def forward(self, x):
        s2 = self.s2(x)
        return s2, self.s4(s2)

class GlobalFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(ENC_DIM, TDIM) for _ in BLOCKS])
        self.fuse = nn.Sequential(
            nn.Linear(TDIM*len(BLOCKS), TDIM),
            nn.LayerNorm(TDIM), nn.GELU()
        )

    def forward(self, layers):
        out = []
        for x, p in zip(layers, self.proj):
            if x.shape[1] != NTOK + 1:
                raise RuntimeError(f"Unexpected MIRAGE token count {x.shape}")
            # Official MIRAGE appends its global token after image tokens.
            out.append(p(x[:, :NTOK, :]))
        return self.fuse(torch.cat(out, dim=-1))

class TaskInteraction(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapters = nn.ModuleDict({t: nn.Linear(TDIM, TDIM) for t in TASKS})
        self.emb = nn.ParameterDict({
            t: nn.Parameter(torch.zeros(1, 1, TDIM)) for t in TASKS
        })
        for p in self.emb.values():
            nn.init.trunc_normal_(p, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=TDIM, nhead=8, dim_feedforward=4*TDIM,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True
        )
        self.tr = nn.TransformerEncoder(
            layer, num_layers=2, norm=nn.LayerNorm(TDIM)
        )

    def forward(self, x):
        streams = [self.adapters[t](x) + self.emb[t] for t in TASKS]
        z = self.tr(torch.cat(streams, dim=1))
        chunks = torch.split(z, NTOK, dim=1)
        return {t: c for t, c in zip(TASKS, chunks)}

class Reassemble(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(TDIM, REASM*64)
        self.norm = LN2d(REASM)
        self.ref = CNBlock(REASM)

    def forward(self, tok):
        b, n, _ = tok.shape
        if n != NTOK:
            raise RuntimeError("Reassembly token mismatch.")
        x = self.proj(tok).view(b, PH, PW, REASM, 8, 8)
        x = x.permute(0, 3, 1, 4, 2, 5).reshape(b, REASM, H//4, W//4)
        return self.ref(self.norm(x))

class TaskFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.reasm = Reassemble()
        self.s4 = nn.Sequential(
            nn.Conv2d(REASM+LOCAL4, FUSED4, 1, bias=False),
            nn.GroupNorm(8, FUSED4), nn.GELU(),
            CNBlock(FUSED4), CNBlock(FUSED4)
        )
        self.s2 = nn.Sequential(
            nn.Conv2d(FUSED4+LOCAL2, FUSED2, 3, padding=1, bias=False),
            nn.GroupNorm(8, FUSED2), nn.GELU(), CNBlock(FUSED2)
        )

    def forward(self, tok, l2, l4):
        x4 = self.s4(torch.cat([self.reasm(tok), l4], dim=1))
        up = F.interpolate(x4, size=l2.shape[-2:], mode="bilinear",
                           align_corners=False)
        return self.s2(torch.cat([up, l2], dim=1))

class FullTrunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(FUSED2, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32), nn.GELU(), CNBlock(32)
        )

    def forward(self, x):
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        return self.net(x)

class RegionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = FullTrunk()
        self.cls = nn.Conv2d(32, 3, 1)
        self.bound = nn.Conv2d(32, 2, 1)

    def forward(self, x):
        x = self.trunk(x)
        return {
            "region_logits": self.cls(x),
            "region_boundary_logits": self.bound(x),
        }

class CurveHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = FullTrunk()
        self.center = nn.Conv2d(32, 2, 1)
        self.dist = nn.Conv2d(32, 2, 1)
        self.ppre = nn.Sequential(nn.Conv2d(FUSED2, 64, 1), nn.GELU())
        self.pout = nn.Conv1d(64, 2, 3, padding=1)

    def forward(self, x2):
        full = self.trunk(x2)
        p = self.ppre(x2).mean(dim=2)
        p = F.interpolate(p, size=W, mode="linear", align_corners=False)
        return {
            "curve_centerline_logits": self.center(full),
            "curve_distance": torch.sigmoid(self.dist(full)),
            "curve_presence_logits": self.pout(p),
        }

class PointHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(FUSED2, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64), nn.GELU(), CNBlock(64)
        )
        self.hm = nn.Conv2d(64, 1, 1)
        self.off = nn.Conv2d(64, 2, 1)
        nn.init.constant_(self.hm.bias, -2.19)

    def forward(self, x):
        x = self.trunk(x)
        return {"hf_heatmap_logits": self.hm(x), "hf_offset": self.off(x)}

class GeoRCP(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.local = LocalEncoder()
        self.global_fuse = GlobalFusion()
        self.interact = TaskInteraction()
        self.fusions = nn.ModuleDict({t: TaskFusion() for t in TASKS})
        self.region = RegionHead()
        self.curve = CurveHead()
        self.point = PointHead()

    def forward(self, x):
        if x.shape[1:] != (1, H, W):
            raise RuntimeError(f"Expected Bx1x{H}x{W}; got {x.shape}")

        l2, l4 = self.local(x)
        layers = self.encoder.model({"bscan": x}, return_all_layers=True)
        if not isinstance(layers, list) or len(layers) != 12:
            raise RuntimeError("MIRAGE did not return 12 layer outputs.")

        sel = [layers[i] for i in BLOCKS]
        shared = self.global_fuse(sel)
        tt = self.interact(shared)
        feat = {t: self.fusions[t](tt[t], l2, l4) for t in TASKS}

        out = {}
        out.update(self.region(feat["region"]))
        out.update(self.curve(feat["curve"]))
        out.update(self.point(feat["point"]))
        return out


# =============================================================================
# Losses
# =============================================================================

def masked_mean(v, m):
    m = m.to(v.dtype)
    while m.ndim < v.ndim:
        m = m.unsqueeze(1)
    m = m.expand_as(v)
    return (v*m).sum() / m.sum().clamp_min(1.0)

def binary_dice(logits, target, valid, eps=1e-5):
    p = torch.sigmoid(logits)
    v = valid.to(p.dtype)
    if v.ndim == 3:
        v = v.unsqueeze(1)
    v = v.expand_as(p)
    p, t = p*v, target*v
    inter = (p*t).sum((0,2,3))
    den = p.sum((0,2,3)) + t.sum((0,2,3))
    return 1 - ((2*inter+eps)/(den+eps)).mean()

def foreground_dice(logits, target, valid, eps=1e-5):
    p = F.softmax(logits, dim=1)[:, 1:]
    t = target.clone()
    t[t == IGNORE] = 0
    oh = F.one_hot(t.long(), 3).permute(0,3,1,2).to(p.dtype)[:, 1:]
    v = valid.to(p.dtype).unsqueeze(1)
    p, oh = p*v, oh*v
    inter = (p*oh).sum((0,2,3))
    den = p.sum((0,2,3)) + oh.sum((0,2,3))
    return 1 - ((2*inter+eps)/(den+eps)).mean()

def centernet_focal(logits, target, pos_mask, valid, alpha=2., beta=4.):
    p = torch.sigmoid(logits).clamp(1e-4, 1-1e-4)
    t = target.unsqueeze(1)
    pos = pos_mask.unsqueeze(1).to(p.dtype)
    val = valid.unsqueeze(1).to(p.dtype)
    pos = pos * val
    neg = (1-pos) * val
    nw = (1-t).pow(beta)
    lp = -torch.log(p) * (1-p).pow(alpha) * pos
    ln = -torch.log(1-p) * p.pow(alpha) * nw * neg
    n = pos.sum()
    return (lp.sum()+ln.sum())/n if n.item() > 0 else ln.sum()

class GeoRCPLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(3))

    def forward(self, o, b):
        # Region
        rce = F.cross_entropy(o["region_logits"], b["region_class"].long(),
                              ignore_index=IGNORE)
        rd = foreground_dice(o["region_logits"], b["region_class"],
                             b["region_valid"])
        rmask = .5*rce + .5*rd

        bbce = masked_mean(
            F.binary_cross_entropy_with_logits(
                o["region_boundary_logits"], b["region_boundary"],
                reduction="none"),
            b["region_boundary_valid"]
        )
        bd = binary_dice(o["region_boundary_logits"], b["region_boundary"],
                         b["region_boundary_valid"])
        rg = rmask + .25*(.5*bbce + .5*bd)

        # Curve spatial validity: exact native FOV, independent of region ignore.
        B = o["curve_centerline_logits"].shape[0]
        cv = torch.zeros((B, H, W), device=o["curve_centerline_logits"].device)
        cv[:, PT:PT+NH, PL:PL+NW] = 1

        cbce = masked_mean(
            F.binary_cross_entropy_with_logits(
                o["curve_centerline_logits"], b["curve_centerline"],
                reduction="none"),
            cv
        )

        presence = b["curve_presence"]
        colvalid = b["curve_column_valid"]

        dmask = presence[:, :, None, :] * cv[:, None, :, :]
        cdist = masked_mean(
            F.smooth_l1_loss(o["curve_distance"], b["curve_distance"],
                             reduction="none"),
            dmask
        )

        cpres = masked_mean(
            F.binary_cross_entropy_with_logits(
                o["curve_presence_logits"], presence, reduction="none"),
            colvalid[:, None, :]
        )

        predn = o["curve_centerline_logits"][:, :, PT:PT+NH, :]
        gtn = b["curve_centerline"][:, :, PT:PT+NH, :]
        gtd = gtn / gtn.sum(dim=2, keepdim=True).clamp_min(1e-8)
        logpd = F.log_softmax(predn, dim=2)
        ce_col = -(gtd*logpd).sum(dim=2)
        valid_cols = presence * colvalid[:, None, :]
        cyd = (ce_col*valid_cols).sum() / valid_cols.sum().clamp_min(1.)

        pd = torch.softmax(predn, dim=2)
        yy = torch.arange(NH, device=pd.device, dtype=pd.dtype).view(1,1,NH,1)
        ey = (pd*yy).sum(dim=2)
        both = presence[:,0] * presence[:,1] * colvalid
        order = F.relu(ey[:,0] - ey[:,1])
        cord = (order*both).sum() / both.sum().clamp_min(1.)

        cg = cbce + .5*cdist + .25*cpres + cyd + .1*cord

        # HF
        ph = centernet_focal(o["hf_heatmap_logits"], b["hf_heatmap"],
                             b["hf_offset_mask"], b["hf_valid"])
        po = masked_mean(
            F.smooth_l1_loss(o["hf_offset"], b["hf_offset"], reduction="none"),
            b["hf_offset_mask"][:, None, :, :]
        )
        pg = ph + po

        groups = torch.stack([rg, cg, pg])
        total = (torch.exp(-self.log_vars)*groups + self.log_vars).sum()

        parts = {
            "region_ce": rce, "region_dice": rd, "region_group": rg,
            "curve_center": cbce, "curve_distance": cdist,
            "curve_presence": cpres, "curve_ydist": cyd,
            "curve_order": cord, "curve_group": cg,
            "point_heatmap": ph, "point_offset": po, "point_group": pg,
            "raw_group_sum": groups.sum(), "total": total,
        }
        return total, parts


# =============================================================================
# Real data
# =============================================================================

def images_by_stem():
    exts = {".png",".jpg",".jpeg",".bmp",".tif",".tiff"}
    d = {p.stem:p for p in IMAGE_DIR.iterdir()
         if p.is_file() and p.suffix.lower() in exts}
    if len(d) != EXPECTED_IMAGES:
        raise RuntimeError(f"Expected {EXPECTED_IMAGES} images, found {len(d)}")
    return d

def mirage_norm(gray: np.ndarray):
    # Official MIRAGE downstream data loader min-max normalizes each non-mask
    # input to [0,1]; official HF wrapper expects [0,1].
    x = gray.astype(np.float32)
    lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        return np.zeros_like(x)
    return (x-lo)/(hi-lo)

def load_sample(stem: str, img_path: Path, target_dir: Path):
    im = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if im is None or im.shape != (NH, NW):
        raise RuntimeError(f"Bad image {stem}: {None if im is None else im.shape}")

    im = mirage_norm(im)
    im = cv2.copyMakeBorder(im, PT, PB, PL, PR, cv2.BORDER_CONSTANT, value=0)

    with np.load(target_dir / f"{stem}.npz") as z:
        a = {k:z[k].copy() for k in z.files}

    return {
        "image": torch.from_numpy(im[None]).float(),
        "region_class": torch.from_numpy(a["region_class"].astype(np.int64)),
        "region_valid": torch.from_numpy(a["region_valid"].astype(np.float32)),
        "region_boundary": torch.from_numpy(a["region_boundary"].astype(np.float32)),
        "region_boundary_valid": torch.from_numpy(a["region_boundary_valid"].astype(np.float32)),
        "curve_centerline": torch.from_numpy(a["curve_centerline"].astype(np.float32)),
        "curve_distance": torch.from_numpy(a["curve_distance"].astype(np.float32)),
        "curve_presence": torch.from_numpy(a["curve_presence"].astype(np.float32)),
        "curve_column_valid": torch.from_numpy(a["curve_column_valid"].astype(np.float32)),
        "hf_heatmap": torch.from_numpy(a["hf_heatmap"].astype(np.float32)),
        "hf_offset": torch.from_numpy(a["hf_offset"].astype(np.float32)),
        "hf_offset_mask": torch.from_numpy(a["hf_offset_mask"].astype(np.float32)),
        "hf_valid": torch.from_numpy(a["hf_valid"].astype(np.float32)),
    }

def stack(samples):
    return {k:torch.stack([s[k] for s in samples]) for k in samples[0]}

def to_device(batch, device):
    return {k:v.to(device) for k,v in batch.items()}

def parts_float(parts):
    return {k:float(v.detach().float().cpu()) for k,v in parts.items()}


# =============================================================================
# Fold-1 train resolver (best effort; mini-overfit skipped if unresolved)
# =============================================================================

def discover_fold1_train(available: Sequence[str]):
    avail = set(available)
    found = set()
    evidence = []

    if not SPLIT_ROOT.exists():
        return [], {"status":"SPLIT_ROOT_MISSING"}

    for p in sorted(SPLIT_ROOT.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in {".txt",".csv"}:
            continue

        low = str(p).lower().replace("\\","/")
        fold1 = ("fold_1" in low or "fold1" in low)
        if not fold1:
            continue

        before = len(found)

        if p.suffix.lower() == ".txt" and "train" in p.name.lower():
            txt = p.read_text(encoding="utf-8", errors="ignore")
            for token in re.split(r"[\s,;]+", txt):
                s = Path(token.strip("\"'")).stem
                if s in avail:
                    found.add(s)

        elif p.suffix.lower() == ".csv":
            try:
                with p.open("r", encoding="utf-8-sig", newline="") as f:
                    r = csv.DictReader(f)
                    if not r.fieldnames:
                        continue
                    cols = {c.lower():c for c in r.fieldnames}
                    sc = next((cols[x] for x in
                               ("stem","image","image_id","filename","file","name")
                               if x in cols), None)
                    rc = next((cols[x] for x in ("role","split","subset","set")
                               if x in cols), None)
                    if sc is None:
                        continue
                    for row in r:
                        role_ok = (str(row.get(rc,"")).lower()=="train") if rc else ("train" in p.name.lower())
                        if not role_ok:
                            continue
                        s = Path(str(row.get(sc,""))).stem
                        if s in avail:
                            found.add(s)
            except Exception:
                pass

        if len(found) > before:
            evidence.append(str(p))

    out = sorted(found)
    return out, {
        "status":"PASS" if len(out)>=8 else "UNRESOLVED",
        "n_train_stems":len(out),
        "evidence_files":evidence[:20],
    }


# =============================================================================
# Rich sample selection
# =============================================================================

def richness(p: Path):
    with np.load(p) as z:
        return (
            min(float(z["srf_mask"].sum())/1000, 5)
            + min(float(z["irf_mask"].sum())/1000, 5)
            + min(float(z["curve_presence"][0].sum())/100, 3)
            + min(float(z["curve_presence"][1].sum())/100, 3)
            + min(float(z["hf_offset_mask"].sum()), 6)
        )

def choose_rich(stems: Sequence[str], target_dir: Path, n=12):
    ss = list(stems)
    if len(ss) > 500:
        step = max(1, len(ss)//500)
        ss = ss[::step][:500]
    vals = [(-richness(target_dir/f"{s}.npz"), s) for s in ss
            if (target_dir/f"{s}.npz").exists()]
    vals.sort()
    return [s for _,s in vals[:n]]


# =============================================================================
# Shapes / optimizer / profiles
# =============================================================================

def expected_shapes(bs):
    return {
        "region_logits":[bs,3,H,W],
        "region_boundary_logits":[bs,2,H,W],
        "curve_centerline_logits":[bs,2,H,W],
        "curve_distance":[bs,2,H,W],
        "curve_presence_logits":[bs,2,W],
        "hf_heatmap_logits":[bs,1,HH,HW],
        "hf_offset":[bs,2,HH,HW],
    }

def validate_shapes(o, bs):
    exp = expected_shapes(bs)
    got = {}
    for k,s in exp.items():
        got[k] = list(o[k].shape)
        if got[k] != s:
            raise RuntimeError(f"{k}: expected {s}, found {got[k]}")
    return got


def grad_report(parameters: Iterable[nn.Parameter]):
    n = bad = 0
    for p in parameters:
        if p.grad is None:
            continue
        n += 1
        if not torch.isfinite(p.grad).all():
            bad += 1
    return {
        "gradient_tensors": n,
        "nonfinite": bad,
        "all_finite": (n > 0 and bad == 0),
    }


def named_grad_report(model: nn.Module, lossmod: nn.Module, max_examples: int = 20):
    """
    More informative diagnostic used when AMP produces non-finite gradients.
    """
    bad = []
    total = 0

    for prefix, module in (("model", model), ("loss", lossmod)):
        for name, p in module.named_parameters():
            if p.grad is None:
                continue
            total += 1
            finite = torch.isfinite(p.grad)
            if not finite.all():
                bad.append({
                    "name": f"{prefix}.{name}",
                    "shape": list(p.grad.shape),
                    "nonfinite_values": int((~finite).sum().item()),
                    "gradient_values": int(p.grad.numel()),
                })

    return {
        "gradient_tensors": total,
        "nonfinite_gradient_tensors": len(bad),
        "all_finite": (total > 0 and len(bad) == 0),
        "examples": bad[:max_examples],
    }


def optimizer_for(model, lossmod, enc_lr, head_lr, wd):
    enc = [p for p in model.encoder.parameters() if p.requires_grad]
    enc_ids = {id(p) for p in enc}
    other = [p for p in model.parameters()
             if p.requires_grad and id(p) not in enc_ids]
    return torch.optim.AdamW([
        {"params": enc, "lr": enc_lr, "weight_decay": wd, "name": "encoder"},
        {"params": other, "lr": head_lr, "weight_decay": wd, "name": "georcp"},
        {"params": list(lossmod.parameters()), "lr": head_lr,
         "weight_decay": 0.0, "name": "uncertainty"},
    ])


def autocast_context(precision: str):
    if precision == "bf16":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.amp.autocast("cuda", dtype=torch.float16)
    raise ValueError(f"Unsupported precision: {precision}")


def build_model(device):
    enc = load_mirage()
    return GeoRCP(enc).to(device), GeoRCPLoss().to(device)


def forward_check(model, lossmod, batch, device, precision="fp16"):
    reset_cuda()
    bg = to_device(batch, device)
    model.eval()
    lossmod.eval()

    with torch.no_grad(), autocast_context(precision):
        o = model(bg["image"])
        shapes = validate_shapes(o, bg["image"].shape[0])
        loss, ps = lossmod(o, bg)

    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite forward loss.")

    result = {
        "status": "PASS",
        "precision": precision,
        "output_shapes": shapes,
        "losses": parts_float(ps),
        "memory": mem(),
    }

    del o, bg
    reset_cuda()
    return result


def gradient_only_diagnostic(
    model,
    lossmod,
    batch,
    device,
    precision: str,
    fp16_init_scale: float = 65536.0,
):
    """
    Backward-only diagnostic. It NEVER performs optimizer.step(), so it cannot
    alter the model. This isolates whether non-finite gradients are caused by
    FP16 loss scaling / numeric range rather than a finite-forward math bug.
    """
    reset_cuda()
    model.train()
    lossmod.train()

    opt = optimizer_for(
        model, lossmod,
        PROFILE_ENC_LR, PROFILE_HEAD_LR, PROFILE_WD
    )
    bg = to_device(batch, device)
    opt.zero_grad(set_to_none=True)

    scaler = None
    t0 = time.perf_counter()

    try:
        with autocast_context(precision):
            o = model(bg["image"])
            validate_shapes(o, bg["image"].shape[0])
            loss, ps = lossmod(o, bg)

        if not torch.isfinite(loss):
            return {
                "status": "NONFINITE_FORWARD_LOSS",
                "precision": precision,
                "fp16_init_scale": fp16_init_scale if precision == "fp16" else None,
                "losses": parts_float(ps),
            }

        if precision == "fp16":
            scaler = torch.amp.GradScaler(
                "cuda",
                enabled=True,
                init_scale=float(fp16_init_scale),
                growth_interval=2000,
            )
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
        else:
            loss.backward()

        gr = named_grad_report(model, lossmod)
        torch.cuda.synchronize()

        return {
            "status": "PASS" if gr["all_finite"] else "NONFINITE_GRADIENT",
            "precision": precision,
            "fp16_init_scale": fp16_init_scale if precision == "fp16" else None,
            "elapsed_seconds": time.perf_counter() - t0,
            "memory": mem(),
            "gradient_report": gr,
            "losses": parts_float(ps),
        }

    except torch.cuda.OutOfMemoryError as e:
        return {
            "status": "OOM",
            "precision": precision,
            "fp16_init_scale": fp16_init_scale if precision == "fp16" else None,
            "error": str(e),
            "memory": mem(),
        }

    finally:
        opt.zero_grad(set_to_none=True)
        del opt, bg
        if scaler is not None:
            del scaler
        if "o" in locals():
            del o
        if "loss" in locals():
            del loss
        reset_cuda()


def choose_stable_precision(batch, device):
    """
    Diagnose the exact failure seen in the first Step-07 run.

    Order:
      1) Reproduce FP16 at the default 65536 loss scale.
      2) Try progressively safer FP16 initial scales.
      3) If available, test BF16 (no loss scaler).
      4) Select a stable mode for the rest of the preflight.

    Preference:
      - If default FP16 passes, use FP16.
      - If default FP16 fails but BF16 passes, use BF16 because its exponent
        range is much safer for transformer + focal-loss fine-tuning.
      - If BF16 is unavailable/fails but reduced-scale FP16 passes, use that.
    """
    fp16_scales = [65536.0, 8192.0, 1024.0, 128.0]
    diagnostics = []

    first_stable_fp16 = None

    for scale in fp16_scales:
        model, lossmod = build_model(device)
        r = gradient_only_diagnostic(
            model, lossmod, batch, device,
            precision="fp16",
            fp16_init_scale=scale,
        )
        diagnostics.append(r)
        del model, lossmod
        reset_cuda()

        print(
            f"      FP16 scale {scale:g}: {r['status']}"
            + (
                f" | nonfinite tensors="
                f"{r.get('gradient_report', {}).get('nonfinite_gradient_tensors', 0)}"
                if r["status"] == "NONFINITE_GRADIENT"
                else ""
            )
        )

        if r["status"] == "PASS":
            first_stable_fp16 = scale
            break

    bf16_supported = bool(torch.cuda.is_bf16_supported())
    bf16_result = None

    if bf16_supported:
        model, lossmod = build_model(device)
        bf16_result = gradient_only_diagnostic(
            model, lossmod, batch, device,
            precision="bf16",
        )
        diagnostics.append(bf16_result)
        del model, lossmod
        reset_cuda()
        print(f"      BF16: {bf16_result['status']}")
    else:
        print("      BF16: not supported by this CUDA/PyTorch device")

    default_fp16_pass = (
        diagnostics
        and diagnostics[0]["precision"] == "fp16"
        and diagnostics[0]["fp16_init_scale"] == 65536.0
        and diagnostics[0]["status"] == "PASS"
    )

    if default_fp16_pass:
        selected = {
            "precision": "fp16",
            "fp16_init_scale": 65536.0,
            "reason": "Default FP16 GradScaler backward is finite.",
        }
    elif bf16_result is not None and bf16_result["status"] == "PASS":
        selected = {
            "precision": "bf16",
            "fp16_init_scale": None,
            "reason": (
                "Default FP16 backward was numerically unstable, while BF16 "
                "backward was finite. BF16 is selected for the remaining "
                "preflight; final precision is still frozen only after RTX 4080 profiling."
            ),
        }
    elif first_stable_fp16 is not None:
        selected = {
            "precision": "fp16",
            "fp16_init_scale": first_stable_fp16,
            "reason": (
                "FP16 becomes finite after lowering the initial GradScaler "
                "scale; this indicates loss-scale overflow rather than a "
                "non-finite forward loss."
            ),
        }
    else:
        raise RuntimeError(
            "Neither BF16 nor tested FP16 scales produced finite gradients. "
            "This requires loss/module isolation before any training."
        )

    return {
        "bf16_supported": bf16_supported,
        "diagnostics": diagnostics,
        "selected": selected,
    }


def optimizer_step(
    model,
    lossmod,
    batch,
    device,
    precision: str,
    fp16_init_scale: float | None = None,
    enc_lr=PROFILE_ENC_LR,
    head_lr=PROFILE_HEAD_LR,
    wd=PROFILE_WD,
):
    reset_cuda()
    model.train()
    lossmod.train()

    opt = optimizer_for(model, lossmod, enc_lr, head_lr, wd)
    bg = to_device(batch, device)
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    scaler = None

    try:
        with autocast_context(precision):
            o = model(bg["image"])
            validate_shapes(o, bg["image"].shape[0])
            loss, ps = lossmod(o, bg)

        if precision == "fp16":
            scale = float(fp16_init_scale or 1024.0)
            scaler = torch.amp.GradScaler(
                "cuda",
                enabled=True,
                init_scale=scale,
                growth_interval=2000,
            )
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
        else:
            loss.backward()

        gr = named_grad_report(model, lossmod)

        # Critical safety change: NEVER call optimizer.step() with non-finite
        # gradients. The old script relied on GradScaler to skip it, but then
        # aborted without telling us why.
        if not gr["all_finite"]:
            return {
                "status": "NONFINITE_GRADIENT",
                "precision": precision,
                "fp16_init_scale": fp16_init_scale if precision == "fp16" else None,
                "batch_size": int(bg["image"].shape[0]),
                "elapsed_seconds": time.perf_counter() - t0,
                "memory": mem(),
                "gradient_report": gr,
                "losses": parts_float(ps),
            }

        if precision == "fp16":
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()

        torch.cuda.synchronize()

        return {
            "status": "PASS",
            "precision": precision,
            "fp16_init_scale": fp16_init_scale if precision == "fp16" else None,
            "batch_size": int(bg["image"].shape[0]),
            "elapsed_seconds": time.perf_counter() - t0,
            "memory": mem(),
            "gradient_report": gr,
            "losses": parts_float(ps),
        }

    except torch.cuda.OutOfMemoryError as e:
        return {
            "status": "OOM",
            "precision": precision,
            "fp16_init_scale": fp16_init_scale if precision == "fp16" else None,
            "batch_size": int(batch["image"].shape[0]),
            "error": str(e),
            "memory": mem(),
        }

    finally:
        opt.zero_grad(set_to_none=True)
        del opt, bg
        if scaler is not None:
            del scaler
        if "o" in locals():
            del o
        if "loss" in locals():
            del loss
        reset_cuda()


@torch.no_grad()
def eval_loss(model, lossmod, bg, precision):
    model.eval()
    lossmod.eval()
    with autocast_context(precision):
        o = model(bg["image"])
        _, ps = lossmod(o, bg)
    return parts_float(ps)


def mini_overfit(
    model,
    lossmod,
    batch,
    device,
    precision: str,
    fp16_init_scale: float | None = None,
):
    reset_cuda()
    bg = to_device(batch, device)
    opt = optimizer_for(
        model, lossmod,
        OVERFIT_ENC_LR, OVERFIT_HEAD_LR, 0.0
    )

    scaler = None
    if precision == "fp16":
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=True,
            init_scale=float(fp16_init_scale or 1024.0),
            growth_interval=2000,
        )

    initial = eval_loss(model, lossmod, bg, precision)
    history = []

    model.train()
    lossmod.train()

    for step in range(1, OVERFIT_STEPS + 1):
        opt.zero_grad(set_to_none=True)

        with autocast_context(precision):
            o = model(bg["image"])
            loss, ps = lossmod(o, bg)

        if precision == "fp16":
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
        else:
            loss.backward()

        gr = named_grad_report(model, lossmod)

        if not gr["all_finite"]:
            result = {
                "status": "NONFINITE_GRADIENT",
                "step": step,
                "precision": precision,
                "gradient_report": gr,
                "losses": parts_float(ps),
            }
            opt.zero_grad(set_to_none=True)
            del opt, bg
            if scaler is not None:
                del scaler
            reset_cuda()
            return result

        if precision == "fp16":
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()

        raw = float(ps["raw_group_sum"].detach().float().cpu())
        history.append({
            "step": step,
            "raw_group_sum": raw,
            "total": float(loss.detach().float().cpu()),
        })

        if step == 1 or step % 5 == 0 or step == OVERFIT_STEPS:
            print(
                f"      step {step:02d}/{OVERFIT_STEPS}: "
                f"raw_group_sum={raw:.5f}"
            )

    final = eval_loss(model, lossmod, bg, precision)
    ratio = final["raw_group_sum"] / initial["raw_group_sum"]

    status = (
        "PASS"
        if math.isfinite(ratio) and ratio <= 0.98
        else "WARNING_INSUFFICIENT_REDUCTION"
    )

    result = {
        "status": status,
        "precision": precision,
        "steps": OVERFIT_STEPS,
        "batch_size": int(batch["image"].shape[0]),
        "initial_losses": initial,
        "final_losses": final,
        "raw_group_sum_ratio": ratio,
        "history": history,
        "memory": mem(),
        "note": "Disposable smoke test only; weights discarded.",
    }

    del opt, bg
    if scaler is not None:
        del scaler
    reset_cuda()
    return result


# =============================================================================
# Implementation spec
# =============================================================================

def implementation_spec():
    s = {
        "name":"GeoRCP-MIRAGE v1.1 exact implementation",
        "scientific_design_sha256":DESIGN_SHA,
        "target_validity_sha256":AMENDMENT_SHA,
        "input":{"shape":[1,H,W], "normalization":"per-image min-max [0,1]",
                 "padding_lrtb":[PL,PR,PT,PB]},
        "encoder":{"model":"MIRAGE-Base",
                   "selected_blocks":[3,6,9,12],
                   "embed_dim":ENC_DIM, "patch_tokens":NTOK},
        "global_fusion":{"per_layer_projection":f"{ENC_DIM}->{TDIM}",
                         "fused_token_dim":TDIM},
        "local_detail":{"stride2_channels":LOCAL2,
                        "stride4_channels":LOCAL4},
        "task_interaction":{"tasks":list(TASKS), "joint_tokens":3*NTOK,
                            "layers":2, "heads":8,
                            "mlp_ratio":4.0, "dropout":0.1},
        "decoders":{"stride4_reassembly_channels":REASM,
                    "stride4_fused_channels":FUSED4,
                    "stride2_fused_channels":FUSED2},
        "outputs":expected_shapes(1),
        "losses":{
            "region":"0.5 CE + 0.5 foreground Dice; +0.25 boundary BCE/Dice",
            "curve":"center BCE +0.5 distance SmoothL1 +0.25 presence BCE "
                    "+ y-distribution CE +0.1 ELM/EZ order hinge",
            "point":"CenterNet-style modified focal + positive-cell offset SmoothL1",
            "balancing":"homoscedastic uncertainty over region/curve/point",
        },
        "global_curve_connectivity_loss":False,
    }
    s["implementation_spec_sha256"] = stable_sha(s)
    return s


# =============================================================================
# Main
# =============================================================================

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print("\n" + "="*100)
    print("RVO-ME STEP 07: GeoRCP-MIRAGE IMPLEMENTATION + PREFLIGHT")
    print(f"SCRIPT REVISION: {SCRIPT_REVISION}")
    print("="*100)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    device = torch.device("cuda")
    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    run_dir = PREFLIGHT_ROOT / sanitize(gpu)
    run_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/10] Verifying frozen inputs...")
    verify_inputs()
    print("    Scientific design SHA: PASS")
    print("    v1.1 amendment SHA: PASS")
    print("    geometry_v2 master SHA: PASS")
    print("    MIRAGE wrapper SHA: PASS")

    target_dir, target_version, target_sha, v2audit, active_audit = resolve_targets()

    print("\n[4/10] Building exact GeoRCP-MIRAGE...")
    print(f"    Python: {sys.version.split()[0]}")
    print(f"    PyTorch: {torch.__version__}")
    print(f"    GPU: {gpu}")
    print(f"    VRAM: {vram:.3f} GB")

    spec = implementation_spec()
    save_json(run_dir/"IMPLEMENTATION_SPEC.json", spec)

    model, lossmod = build_model(device)
    mp = params(model)
    ep = params(model.encoder)
    lp = params(lossmod)
    print(f"    MIRAGE encoder: {ep['total']/1e6:.3f} M")
    print(f"    GeoRCP total model: {mp['total']/1e6:.3f} M")
    print(f"    Loss balancing parameters: {lp['trainable']}")
    print(f"    Implementation SHA: {spec['implementation_spec_sha256']}")

    print("\n[5/10] Resolving real samples and Track-B fold-1 train...")
    images = images_by_stem()
    available = sorted(images)
    train_stems, split_report = discover_fold1_train(available)
    save_json(run_dir/"SPLIT_RESOLUTION.json", split_report)
    print(f"    Fold-1 train resolver: {split_report['status']}")
    if train_stems:
        print(f"    Resolved train images: {len(train_stems)}")
    else:
        print("    Mini-overfit will be skipped rather than use unresolved outer-test images.")

    profile_source = train_stems if train_stems else available
    rich = choose_rich(profile_source, target_dir, 12)
    if len(rich) < 8:
        raise RuntimeError("Not enough real target-rich samples.")

    print("\n[6/10] Real-image forward + complete final-loss validation...")
    b1 = stack([load_sample(rich[0], images[rich[0]], target_dir)])

    # Forward-only check can safely use FP16 because it was already finite in
    # the first run; backward precision is diagnosed separately below.
    fwd = forward_check(model, lossmod, b1, device, precision="fp16")
    for k,s in fwd["output_shapes"].items():
        print(f"    {k:<32} {s} PASS")
    print("    All final losses finite: PASS")
    print(
        "    Group losses: "
        f"region={fwd['losses']['region_group']:.5f} | "
        f"curve={fwd['losses']['curve_group']:.5f} | "
        f"point={fwd['losses']['point_group']:.5f}"
    )
    print(f"    Forward peak allocated: {fwd['memory']['peak_allocated_gb']:.3f} GB")

    print("\n[7/10] Diagnosing backward precision and validating optimizer step...")
    b2 = stack([load_sample(s, images[s], target_dir) for s in rich[:2]])

    # Dispose the model used for forward checking so precision diagnostics all
    # begin from pristine pretrained weights.
    del model, lossmod
    reset_cuda()

    precision_diag = choose_stable_precision(b2, device)
    selected_precision = precision_diag["selected"]["precision"]
    selected_scale = precision_diag["selected"]["fp16_init_scale"]

    print(f"    Selected preflight precision: {selected_precision.upper()}")
    if selected_precision == "fp16":
        print(f"    Selected FP16 initial scale: {selected_scale:g}")
    print(f"    Reason: {precision_diag['selected']['reason']}")

    model, lossmod = build_model(device)
    back = optimizer_step(
        model, lossmod, b2, device,
        precision=selected_precision,
        fp16_init_scale=selected_scale,
    )
    print(f"    Batch 2 optimizer step: {back['status']}")

    if back["status"] != "PASS":
        if "gradient_report" in back:
            for ex in back["gradient_report"].get("examples", [])[:10]:
                print(
                    "      nonfinite: "
                    f"{ex['name']} | {ex['nonfinite_values']}/"
                    f"{ex['gradient_values']}"
                )
        raise RuntimeError(
            "Batch-2 exact-model optimizer step still failed after "
            "precision diagnosis."
        )

    print(f"    Finite gradients: {back['gradient_report']['all_finite']}")
    print(f"    Peak allocated: {back['memory']['peak_allocated_gb']:.3f} GB")

    print("\n[8/10] Hardware-aware physical-batch profile...")
    batches = [2,4,6,8] if vram >= 14 else ([1,2,4,6] if vram >= 10 else [1,2,4])
    prof=[]
    largest=0

    # Reuse the selected stable precision. Each profile step is engineering
    # only; no validation/test performance is measured.
    for bs in batches:
        batch = stack([
            load_sample(rich[i % len(rich)], images[rich[i % len(rich)]], target_dir)
            for i in range(bs)
        ])
        r = optimizer_step(
            model, lossmod, batch, device,
            precision=selected_precision,
            fp16_init_scale=selected_scale,
        )
        prof.append(r)
        if r["status"] == "PASS":
            largest=max(largest,bs)
            print(
                f"    Batch {bs}: PASS | "
                f"{selected_precision.upper()} | "
                f"peak={r['memory']['peak_allocated_gb']:.3f} GB | "
                f"step={r['elapsed_seconds']:.2f}s"
            )
        else:
            print(f"    Batch {bs}: {r['status']}")
            if "gradient_report" in r:
                examples = r["gradient_report"].get("examples", [])
                if examples:
                    print(f"      first nonfinite tensor: {examples[0]['name']}")
            if r["status"] == "OOM":
                break

    print("\n[9/10] Disposable real-target mini-overfit...")
    if len(train_stems) >= 2:
        del model, lossmod
        reset_cuda()
        model, lossmod = build_model(device)
        ostems = choose_rich(train_stems, target_dir, 2)
        ob = stack([load_sample(s, images[s], target_dir) for s in ostems])
        print("    Fold-1 train images: " + ", ".join(ostems))
        over = mini_overfit(
            model, lossmod, ob, device,
            precision=selected_precision,
            fp16_init_scale=selected_scale,
        )
        print(f"    Mini-overfit: {over['status']}")
        if "raw_group_sum_ratio" in over:
            print(f"    raw loss final/initial: {over['raw_group_sum_ratio']:.4f}")
    else:
        over = {
            "status":"SKIPPED_SPLIT_NOT_RESOLVED",
            "reason":"Fold-1 train stems not safely resolved; no fallback to test images."
        }
        print("    SKIPPED: fold-1 train stems were not safely resolved.")

    print("\n[10/10] Writing hardware-specific preflight report...")
    report = {
        "status":"COMPLETE",
        "step":"07",
        "scientific_status":"PREFLIGHT_ONLY_NOT_DEFINITIVE_TRAINING",
        "scientific_design_sha256":DESIGN_SHA,
        "target_validity_v1_1_sha256":AMENDMENT_SHA,
        "geometry_v2_parent_sha256":G2_SHA,
        "active_target_version":target_version,
        "active_target_master_sha256":target_sha,
        "geometry_v2_hf_alignment_audit":v2audit,
        "active_hf_alignment_audit":active_audit,
        "implementation_spec_sha256":spec["implementation_spec_sha256"],
        "mirage_wrapper_sha256":MIRAGE_WRAPPER_SHA,
        "environment":{"python":sys.version, "pytorch":torch.__version__,
                       "torch_cuda":torch.version.cuda, "gpu":gpu, "vram_gb":vram},
        "parameters":{"encoder":ep, "model":mp, "loss_module":lp},
        "split_resolution":split_report,
        "real_forward_validation":fwd,
        "backward_precision_diagnostics": precision_diag,
        "selected_preflight_precision": {
            "precision": selected_precision,
            "fp16_init_scale": selected_scale,
        },
        "batch2_backward_optimizer":back,
        "batch_memory_profile":prof,
        "largest_profiled_batch_pass":largest,
        "mini_overfit":over,
        "engineering_freeze_status":
            "PENDING_RTX4080_PROFILE" if vram < 14
            else "RTX4080_CLASS_PROFILE_COMPLETE_REVIEW_BEFORE_FREEZE",
        "important":"All weights are disposable. No outer-test metric was computed.",
    }
    save_json(run_dir/"STEP07_PREFLIGHT_COMPLETE.json", report)

    print("\n" + "-"*100)
    print("STEP 07 PREFLIGHT COMPLETE")
    print("-"*100)
    print(f"Active targets: {target_version}")
    print(f"Active target SHA: {target_sha}")
    print(f"Implementation SHA: {spec['implementation_spec_sha256']}")
    print(f"Selected preflight precision: {selected_precision.upper()}")
    if selected_precision == "fp16":
        print(f"Selected FP16 initial scale: {selected_scale:g}")
    print(f"Largest profiled physical batch PASS: {largest}")
    print(f"Mini-overfit: {over['status']}")
    print(f"Report: {run_dir / 'STEP07_PREFLIGHT_COMPLETE.json'}")
    if vram < 14:
        print("\nEngineering protocol remains UNFROZEN.")
        print("Run this SAME script on the RTX 4080 tomorrow.")
    else:
        print("\nRTX-4080-class profile collected; review before final engineering freeze.")
    print("\n" + "="*100)


if __name__ == "__main__":
    main()
