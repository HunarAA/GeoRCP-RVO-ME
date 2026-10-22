from pathlib import Path
import json
import math
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Publication-style typography
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "legend.fontsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "svg.fonttype": "none",  # keep text editable in SVG
})

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ============================================================
# USER SETTINGS
# ============================================================
DATASET_ROOT = Path(r"E:\Papers\RVO_ME\RVO-Lesion")
OUTPUT_DIR = Path(r"E:\Papers\RVO_ME\paper_figures")
OUTPUT_BASENAME = "Figure1_geometry_specific_representation"

# If EXAMPLE_STEM = None, the script will automatically choose
# one representative case containing region + curve + point labels.
EXAMPLE_STEM = None

# Labels of interest
REGION_LABELS = {"IRF", "SRF"}
CURVE_LABELS = {"ELM", "EZ"}
POINT_LABELS = {"HF"}
TARGET_LABELS = REGION_LABELS | CURVE_LABELS | POINT_LABELS

# Visual colors
COLOR = {
    "IRF": "#e74c3c",   # red
    "SRF": "#f39c12",   # orange
    "ELM": "#00bcd4",   # cyan
    "EZ":  "#8e44ad",   # purple
    "HF":  "#2ecc71",   # green
}

# Image size in dataset
NATIVE_W = 570
NATIVE_H = 380

# ============================================================
# HELPERS
# ============================================================
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_json_files(root: Path):
    files = []
    for p in root.rglob("*.json"):
        # skip project outputs if they happen to live inside the dataset tree
        parts_lower = {x.lower() for x in p.parts}
        if "project" in parts_lower or "audit_results" in parts_lower or "audit02_results" in parts_lower:
            continue
        files.append(p)
    return sorted(files)


def find_image_for_json(json_path: Path):
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
        candidate = json_path.with_suffix(ext)
        if candidate.exists():
            return candidate
    return None


def clamp_point(pt):
    x, y = pt
    x = min(max(float(x), 0.0), NATIVE_W - 1.0)
    y = min(max(float(y), 0.0), NATIVE_H - 1.0)
    return [x, y]


def parse_shapes(data):
    groups = {"IRF": [], "SRF": [], "ELM": [], "EZ": [], "HF": []}
    for sh in data.get("shapes", []):
        label = str(sh.get("label", "")).strip()
        if label not in TARGET_LABELS:
            continue
        pts = sh.get("points", [])
        if not pts:
            continue
        pts = [clamp_point(p) for p in pts]
        groups[label].append({
            "shape_type": sh.get("shape_type", ""),
            "points": pts,
        })
    return groups


def score_case(groups):
    has_region = sum(len(groups[k]) > 0 for k in REGION_LABELS)
    has_curve = sum(len(groups[k]) > 0 for k in CURVE_LABELS)
    has_point = sum(len(groups[k]) > 0 for k in POINT_LABELS)
    total_ann = sum(len(v) for v in groups.values())
    # require at least one label in each family; then prefer richer cases
    family_bonus = 100 * int(has_region > 0 and has_curve > 0 and has_point > 0)
    return family_bonus + 20 * has_region + 20 * has_curve + 20 * has_point + total_ann


def choose_example(manifest):
    best = None
    best_score = -1
    for item in manifest:
        s = score_case(item["groups"])
        if s > best_score:
            best_score = s
            best = item
    return best


def rasterize_regions(groups):
    masks = {"IRF": np.zeros((NATIVE_H, NATIVE_W), dtype=np.uint8),
             "SRF": np.zeros((NATIVE_H, NATIVE_W), dtype=np.uint8)}
    for label in ["IRF", "SRF"]:
        canvas = Image.new("L", (NATIVE_W, NATIVE_H), 0)
        draw = ImageDraw.Draw(canvas)
        for ann in groups[label]:
            pts = ann["points"]
            st = str(ann["shape_type"]).lower()
            if st == "polygon" and len(pts) >= 3:
                draw.polygon([tuple(p) for p in pts], fill=255)
            elif st in {"linestrip", "line", "polyline"} and len(pts) >= 2:
                draw.line([tuple(p) for p in pts], fill=255, width=2)
            elif st == "point" and len(pts) >= 1:
                x, y = pts[0]
                r = 2
                draw.ellipse((x-r, y-r, x+r, y+r), fill=255)
        masks[label] = np.array(canvas, dtype=np.uint8)
    return masks


def draw_polyline_mask(canvas, pts, width=1):
    draw = ImageDraw.Draw(canvas)
    if len(pts) == 1:
        x, y = pts[0]
        draw.ellipse((x-1, y-1, x+1, y+1), fill=255)
    else:
        draw.line([tuple(p) for p in pts], fill=255, width=width)


def gaussian_kernel1d(sigma, radius=None):
    if radius is None:
        radius = int(max(1, round(3 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    k /= max(k.max(), 1e-8)
    return k


def gaussian_blur_separable(arr, sigma=1.5):
    k = gaussian_kernel1d(sigma)
    pad = len(k) // 2
    # horizontal
    a = np.pad(arr, ((0, 0), (pad, pad)), mode="edge")
    out = np.zeros_like(arr, dtype=np.float32)
    for i in range(arr.shape[0]):
        out[i] = np.convolve(a[i], k, mode="valid")
    # vertical
    a2 = np.pad(out, ((pad, pad), (0, 0)), mode="edge")
    out2 = np.zeros_like(out, dtype=np.float32)
    for j in range(out.shape[1]):
        out2[:, j] = np.convolve(a2[:, j], k, mode="valid")
    if out2.max() > 0:
        out2 = out2 / out2.max()
    return out2


def rasterize_curves(groups, sigma=1.5):
    heat = {"ELM": np.zeros((NATIVE_H, NATIVE_W), dtype=np.float32),
            "EZ": np.zeros((NATIVE_H, NATIVE_W), dtype=np.float32)}
    for label in ["ELM", "EZ"]:
        base = Image.new("L", (NATIVE_W, NATIVE_H), 0)
        for ann in groups[label]:
            pts = ann["points"]
            st = str(ann["shape_type"]).lower()
            if st in {"linestrip", "line", "polyline", "polygon"} and len(pts) >= 2:
                draw_polyline_mask(base, pts, width=1)
            elif st == "point" and len(pts) >= 1:
                draw_polyline_mask(base, pts, width=1)
        arr = np.array(base, dtype=np.float32) / 255.0
        if arr.max() > 0:
            arr = gaussian_blur_separable(arr, sigma=sigma)
        heat[label] = arr
    return heat


def rasterize_hf_heatmap(groups, sigma=3.0):
    yy, xx = np.mgrid[0:NATIVE_H, 0:NATIVE_W]
    heat = np.zeros((NATIVE_H, NATIVE_W), dtype=np.float32)
    points = []
    for ann in groups["HF"]:
        pts = ann["points"]
        if not pts:
            continue
        x, y = pts[0]
        points.append((x, y))
        g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
        heat = np.maximum(heat, g)
    if heat.max() > 0:
        heat /= heat.max()
    return heat, points


def overlay_mask(ax, mask, color, alpha=0.50):
    rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.float32)
    rgb = tuple(int(color.lstrip('#')[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    rgba[..., 0] = rgb[0]
    rgba[..., 1] = rgb[1]
    rgba[..., 2] = rgb[2]
    rgba[..., 3] = (mask > 0).astype(np.float32) * alpha
    ax.imshow(rgba)


def overlay_heat(ax, heat, cmap, alpha_scale=0.85):
    if heat.max() <= 0:
        return
    colored = plt.get_cmap(cmap)(heat)
    colored[..., 3] = heat * alpha_scale
    ax.imshow(colored)


def prepare_manifest():
    manifest = []
    json_files = list_json_files(DATASET_ROOT)
    print(f"Found {len(json_files)} JSON files...")
    for jp in json_files:
        img = find_image_for_json(jp)
        if img is None:
            continue
        try:
            data = load_json(jp)
            groups = parse_shapes(data)
        except Exception:
            continue
        stem = jp.stem
        manifest.append({
            "stem": stem,
            "json_path": jp,
            "image_path": img,
            "groups": groups,
        })
    return manifest


def get_item_by_stem(manifest, stem):
    for item in manifest:
        if item["stem"] == stem:
            return item
    return None


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = prepare_manifest()
    if not manifest:
        raise RuntimeError("No valid image/JSON pairs were found.")

    if EXAMPLE_STEM is None:
        item = choose_example(manifest)
    else:
        item = get_item_by_stem(manifest, EXAMPLE_STEM)
        if item is None:
            raise RuntimeError(f"EXAMPLE_STEM not found: {EXAMPLE_STEM}")

    if item is None:
        raise RuntimeError("Could not choose a representative example.")

    print("Selected example:")
    print(f"  stem:  {item['stem']}")
    print(f"  image: {item['image_path']}")
    print(f"  json:  {item['json_path']}")

    image = Image.open(item["image_path"]).convert("L")
    if image.size != (NATIVE_W, NATIVE_H):
        image = image.resize((NATIVE_W, NATIVE_H))
    img = np.array(image)
    groups = item["groups"]

    region_masks = rasterize_regions(groups)
    curve_heat = rasterize_curves(groups, sigma=1.5)
    hf_heat, hf_points = rasterize_hf_heatmap(groups, sigma=3.0)

    fig, axes = plt.subplots(
    2, 2,
    figsize=(11, 8),
    constrained_layout=False
    )
    
    axes = axes.flatten()

    # --------------------------------------------------------
    # (a) original + annotations
    # --------------------------------------------------------
    ax = axes[0]
    ax.imshow(img, cmap="gray")
    overlay_mask(ax, region_masks["IRF"], COLOR["IRF"], alpha=0.40)
    overlay_mask(ax, region_masks["SRF"], COLOR["SRF"], alpha=0.40)

    for label in ["ELM", "EZ"]:
        for ann in groups[label]:
            pts = np.array(ann["points"])
            if len(pts) >= 2:
                ax.plot(pts[:, 0], pts[:, 1], color=COLOR[label], linewidth=2.4)
            elif len(pts) == 1:
                ax.scatter(pts[0, 0], pts[0, 1], c=COLOR[label], s=16)

    if hf_points:
        hp = np.array(hf_points)
        ax.scatter(hp[:, 0], hp[:, 1], c=COLOR["HF"], s=22, edgecolors="black", linewidths=0.35)

    ax.set_title("(a) OCT B-scan with original annotations", fontsize=14, pad=8)
    ax.axis("off")

    # --------------------------------------------------------
    # (b) region representation
    # --------------------------------------------------------
    ax = axes[1]
    ax.imshow(img, cmap="gray")
    overlay_mask(ax, region_masks["IRF"], COLOR["IRF"], alpha=0.55)
    overlay_mask(ax, region_masks["SRF"], COLOR["SRF"], alpha=0.55)
    ax.set_title("(b) Region targets for IRF and SRF", fontsize=14, pad=8)
    ax.axis("off")

    # --------------------------------------------------------
    # (c) curve representation
    # --------------------------------------------------------
    ax = axes[2]
    ax.imshow(img, cmap="gray")
    overlay_heat(ax, curve_heat["ELM"], cmap="winter", alpha_scale=0.85)
    overlay_heat(ax, curve_heat["EZ"], cmap="plasma", alpha_scale=0.85)
    for label in ["ELM", "EZ"]:
        for ann in groups[label]:
            pts = np.array(ann["points"])
            if len(pts) >= 2:
                ax.plot(pts[:, 0], pts[:, 1], color=COLOR[label], linewidth=3.0)
    ax.set_title("(c) Curve targets for ELM and EZ", fontsize=14, pad=8)
    ax.axis("off")

    # --------------------------------------------------------
    # (d) point representation
    # --------------------------------------------------------
    ax = axes[3]
    ax.imshow(img, cmap="gray")
    overlay_heat(ax, hf_heat, cmap="Greens", alpha_scale=0.85)
    if hf_points:
        hp = np.array(hf_points)
        ax.scatter(hp[:, 0], hp[:, 1], c=COLOR["HF"], s=24, edgecolors="black", linewidths=0.35)
    ax.set_title("(d) Point targets for HF", fontsize=14, pad=8)
    ax.axis("off")

    # Figure legend
    legend_items = [
        Patch(facecolor=COLOR["IRF"], edgecolor='none', label='IRF'),
        Patch(facecolor=COLOR["SRF"], edgecolor='none', label='SRF'),
        Line2D([0], [0], color=COLOR["ELM"], lw=2, label='ELM'),
        Line2D([0], [0], color=COLOR["EZ"], lw=2, label='EZ'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR["HF"],
               markeredgecolor='black', markersize=6, label='HF'),
    ]
    fig.legend(handles=legend_items, loc='lower center', ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.005), handlelength=1.8, columnspacing=1.8)

    fig.subplots_adjust(
        left=0.03,
    right=0.98,
    top=0.94,
    bottom=0.10,
    wspace=0.04,
    hspace=0.18
    )

    png_path = OUTPUT_DIR / f"{OUTPUT_BASENAME}.png"
    svg_path = OUTPUT_DIR / f"{OUTPUT_BASENAME}.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    print("\nSaved:")
    print(f"  {png_path}")
    print(f"  {svg_path}")
    print("\nSuggested caption:")
    print("Figure 1. Geometry-specific representation of RVO-ME biomarkers. IRF and SRF are represented as two-dimensional regions, ELM and EZ as one-dimensional curves, and HF as point targets, preserving the intrinsic annotation geometry of each biomarker.")


if __name__ == "__main__":
    main()
