# =============================================================================
# RVO-ME STEP 05E3
# MIRAGE-Base foundation encoder preflight on laptop GPU
#
# This script:
#   - verifies the frozen near-native protocol
#   - audits Python / PyTorch / CUDA / minimal MIRAGE dependencies
#   - downloads the official MIRAGE lightweight HF wrapper
#   - loads official MIRAGE-Base weights from Hugging Face
#   - tests official 512x512 input
#   - tests RVO-ME near-native 384x576 input
#   - tests one real RVO-ME B-scan
#   - measures inference GPU memory
#   - optionally performs ONE synthetic backward smoke test
#
# It does NOT train on the dataset and does NOT modify frozen project assets.
# It does NOT install or upgrade PyTorch.
# =============================================================================

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import platform
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict

import numpy as np
import cv2


# =============================================================================
# PATHS / IDENTIFIERS
# =============================================================================

ROOT = Path(r"E:\Papers\RVO_ME")

NEAR_NATIVE_DIR = (
    ROOT
    / "project"
    / "preprocessing"
    / "near_native_576x384_v1"
)

NEAR_NATIVE_JSON = (
    NEAR_NATIVE_DIR
    / "NEAR_NATIVE_PROTOCOL_FROZEN.json"
)

EXPECTED_NEAR_NATIVE_SHA = (
    "ac5ce015e1a05a3dc47f054c76f8706d42ac9a9c9506543cbf4ca8e621d53714"
)

IMAGE_DIR = (
    ROOT
    / "RVO-Lesion"
    / "Image_Seg"
    / "images"
)

OUTPUT_DIR = (
    ROOT
    / "project"
    / "foundation_models"
    / "mirage_base_preflight_v1"
)

OFFICIAL_WRAPPER_URL = (
    "https://raw.githubusercontent.com/"
    "j-morano/MIRAGE/main/hf/mirage_hf.py"
)

MIRAGE_REPO_ID = "j-morano/MIRAGE-Base"

OFFICIAL_INPUT_HW = (512, 512)
NEAR_NATIVE_INPUT_HW = (384, 576)
PATCH_SIZE = 32

# Set False if you only want inference checks.
RUN_SYNTHETIC_BACKWARD_SMOKE_TEST = True


# =============================================================================
# HELPERS
# =============================================================================

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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def import_version(name: str) -> Dict[str, Any]:
    try:
        mod = __import__(name)
        return {
            "available": True,
            "version": getattr(mod, "__version__", "unknown"),
            "error": None,
        }
    except Exception as exc:
        return {
            "available": False,
            "version": None,
            "error": repr(exc),
        }


def download_text_file(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as r:
        data = r.read()
    path.write_bytes(data)


def import_module_from_file(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not create import spec for {path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reset_cuda(torch):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def cuda_memory(torch) -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {
            "allocated_gb": 0.0,
            "reserved_gb": 0.0,
            "peak_allocated_gb": 0.0,
            "peak_reserved_gb": 0.0,
        }

    gb = 1024 ** 3

    return {
        "allocated_gb":
            float(torch.cuda.memory_allocated() / gb),

        "reserved_gb":
            float(torch.cuda.memory_reserved() / gb),

        "peak_allocated_gb":
            float(torch.cuda.max_memory_allocated() / gb),

        "peak_reserved_gb":
            float(torch.cuda.max_memory_reserved() / gb),
    }


def find_real_image() -> Path:
    allowed = {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff",
    }

    files = sorted(
        p
        for p in IMAGE_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in allowed
    )

    if not files:
        raise RuntimeError(
            f"No images found in {IMAGE_DIR}"
        )

    return files[len(files) // 2]


def prepare_real_bscan(path: Path, torch):
    img = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    if img is None:
        raise RuntimeError(
            f"Could not read {path}"
        )

    if img.shape != (380, 570):
        raise RuntimeError(
            f"Unexpected real image shape: {img.shape}"
        )

    padded = cv2.copyMakeBorder(
        img,
        2,
        2,
        3,
        3,
        borderType=cv2.BORDER_CONSTANT,
        value=0,
    )

    if padded.shape != (384, 576):
        raise RuntimeError(
            f"Unexpected padded shape: {padded.shape}"
        )

    x = (
        torch.from_numpy(
            padded.astype(np.float32)
        )
        .unsqueeze(0)
        .unsqueeze(0)
        / 255.0
    )

    return x


# =============================================================================
# MODEL TEST HELPERS
# =============================================================================

def count_parameters(model) -> Dict[str, int]:
    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return {
        "total": int(total),
        "trainable": int(trainable),
    }


def run_encoder_inference(
    model,
    x,
    torch,
    label: str,
) -> Dict[str, Any]:

    reset_cuda(torch)

    model.eval()

    x = x.cuda(
        non_blocking=False
    )

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.inference_mode():
        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=True,
        ):
            out = model(
                {
                    "bscan": x
                }
            )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    if not hasattr(
        out,
        "shape",
    ):
        raise RuntimeError(
            f"{label}: unexpected output type {type(out)}"
        )

    result = {
        "label": label,
        "input_shape":
            list(x.shape),

        "output_shape":
            list(out.shape),

        "output_dtype":
            str(out.dtype),

        "elapsed_seconds":
            float(elapsed),

        "gpu_memory":
            cuda_memory(torch),
    }

    del out
    del x

    reset_cuda(torch)

    return result


class TokenProbeHeadPlaceholder:
    pass


def run_backward_smoke(
    model,
    torch,
) -> Dict[str, Any]:
    """
    One synthetic gradient-flow test through MIRAGE-Base at 384x576.

    This is not training. A tiny linear probe is attached only to ensure
    backward propagation is technically feasible on the laptop GPU.
    """

    reset_cuda(torch)

    # Full encoder gradient flow.
    for p in model.parameters():
        p.requires_grad = True

    model.train()

    probe = torch.nn.Linear(
        768,
        5,
    ).cuda()

    x = torch.rand(
        1,
        1,
        384,
        576,
        device="cuda",
        dtype=torch.float32,
    )

    optimizer = torch.optim.AdamW(
        list(model.parameters())
        + list(probe.parameters()),
        lr=1e-6,
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    try:
        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=True,
        ):
            tokens = model(
                {
                    "bscan": x
                }
            )

            # Exclude final global token from the toy loss.
            image_tokens = tokens[:, :-1, :]

            logits = probe(
                image_tokens
            )

            loss = (
                logits.float()
                .square()
                .mean()
            )

        loss.backward()

        torch.cuda.synchronize()

        elapsed = (
            time.perf_counter()
            - t0
        )

        finite_grads = True
        n_grad_tensors = 0

        for p in model.parameters():
            if p.grad is not None:
                n_grad_tensors += 1

                if not torch.isfinite(
                    p.grad
                ).all():
                    finite_grads = False
                    break

        result = {
            "status": "PASS",
            "loss": float(
                loss.detach().cpu()
            ),
            "elapsed_seconds":
                float(elapsed),

            "finite_encoder_gradients":
                bool(finite_grads),

            "encoder_gradient_tensors":
                int(n_grad_tensors),

            "gpu_memory":
                cuda_memory(torch),
        }

    except torch.cuda.OutOfMemoryError as exc:

        result = {
            "status":
                "OOM_ON_LAPTOP",

            "error":
                repr(exc),

            "gpu_memory":
                cuda_memory(torch),

            "scientific_interpretation":
                (
                    "Not a project failure. Full fine-tuning "
                    "will be targeted to the RTX 4080."
                ),
        }

    finally:
        optimizer.zero_grad(
            set_to_none=True
        )

        del optimizer
        del probe
        del x

        if "tokens" in locals():
            del tokens

        if "logits" in locals():
            del logits

        if "loss" in locals():
            del loss

        reset_cuda(torch)

    return result


# =============================================================================
# MAIN
# =============================================================================

def main():

    print(
        "\n"
        + "=" * 100
    )

    print(
        "RVO-ME STEP 05E3: "
        "MIRAGE-BASE FOUNDATION ENCODER PREFLIGHT"
    )

    print(
        "=" * 100
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # 1. Frozen protocol
    # -------------------------------------------------------------------------

    print(
        "\n[1/7] Verifying frozen "
        "near-native protocol..."
    )

    if not NEAR_NATIVE_JSON.exists():
        raise FileNotFoundError(
            NEAR_NATIVE_JSON
        )

    protocol = json.loads(
        NEAR_NATIVE_JSON.read_text(
            encoding="utf-8"
        )
    )

    found_sha = str(
        protocol.get(
            "near_native_protocol_sha256",
            "",
        )
    )

    if found_sha != EXPECTED_NEAR_NATIVE_SHA:
        raise RuntimeError(
            "Near-native protocol SHA mismatch.\n"
            f"Expected: {EXPECTED_NEAR_NATIVE_SHA}\n"
            f"Found:    {found_sha}"
        )

    print(
        "    Near-native protocol SHA: PASS"
    )

    print(
        f"    {found_sha}"
    )

    print(
        "    RVO-ME tensor: "
        "1x1x384x576"
    )

    print(
        "    MIRAGE patch-32 grid: "
        "12 x 18 = 216 image tokens"
    )

    print(
        "    MIRAGE default 512 grid: "
        "16 x 16 = 256 image tokens"
    )

    # -------------------------------------------------------------------------
    # 2. Environment
    # -------------------------------------------------------------------------

    print(
        "\n[2/7] Auditing Python / "
        "PyTorch / CUDA environment..."
    )

    try:
        import torch
    except Exception as exc:
        raise RuntimeError(
            "PyTorch could not be imported."
        ) from exc

    print(
        f"    Python:  {sys.version.split()[0]}"
    )

    print(
        f"    OS:      {platform.platform()}"
    )

    print(
        f"    PyTorch: {torch.__version__}"
    )

    print(
        f"    CUDA available: "
        f"{torch.cuda.is_available()}"
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this GPU preflight."
        )

    device_name = torch.cuda.get_device_name(
        0
    )

    props = torch.cuda.get_device_properties(
        0
    )

    vram_gb = (
        props.total_memory
        / (1024 ** 3)
    )

    print(
        f"    GPU:     {device_name}"
    )

    print(
        f"    VRAM:    {vram_gb:.3f} GB"
    )

    # -------------------------------------------------------------------------
    # 3. Minimal dependencies
    # -------------------------------------------------------------------------

    print(
        "\n[3/7] Checking minimal "
        "Hugging Face MIRAGE dependencies..."
    )

    required = {
        "einops":
            import_version(
                "einops"
            ),

        "huggingface_hub":
            import_version(
                "huggingface_hub"
            ),

        "safetensors":
            import_version(
                "safetensors"
            ),

        "requests":
            import_version(
                "requests"
            ),

        "PIL":
            import_version(
                "PIL"
            ),
    }

    missing = []

    for name, info in required.items():

        if info["available"]:

            print(
                f"    {name:<18} "
                f"PASS | {info['version']}"
            )

        else:

            print(
                f"    {name:<18} MISSING"
            )

            missing.append(
                name
            )

    if missing:

        print(
            "\nMinimal packages are missing."
        )

        print(
            "Do NOT reinstall PyTorch."
        )

        print(
            "Install only the missing lightweight packages, e.g.:"
        )

        print(
            "pip install einops huggingface_hub safetensors requests pillow"
        )

        save_json(
            OUTPUT_DIR
            / "DEPENDENCY_CHECK.json",
            {
                "status":
                    "MISSING_DEPENDENCIES",

                "packages":
                    required,
            },
        )

        raise RuntimeError(
            "Install the listed minimal packages "
            "and rerun 05E3."
        )

    from huggingface_hub import (
        PyTorchModelHubMixin,
    )

    # -------------------------------------------------------------------------
    # 4. Official wrapper
    # -------------------------------------------------------------------------

    print(
        "\n[4/7] Fetching official "
        "MIRAGE lightweight wrapper..."
    )

    wrapper_path = (
        OUTPUT_DIR
        / "official_mirage_hf.py"
    )

    download_text_file(
        OFFICIAL_WRAPPER_URL,
        wrapper_path,
    )

    wrapper_sha = sha256_file(
        wrapper_path
    )

    print(
        "    Official wrapper downloaded: PASS"
    )

    print(
        f"    Wrapper SHA-256: {wrapper_sha}"
    )

    mirage_module = import_module_from_file(
        "official_mirage_hf",
        wrapper_path,
    )

    MIRAGEWrapper = (
        mirage_module.MIRAGEWrapper
    )

    # Important: initialize at official 512x512 so the saved positional
    # parameter has the exact pretrained shape. The official adapter then
    # dynamically bicubic-interpolates that positional grid for 384x576.
    class MIRAGEhf(
        MIRAGEWrapper,
        PyTorchModelHubMixin,
    ):
        def __init__(
            self,
            input_size=512,
            patch_size=32,
            modalities="bscan",
            size="base",
        ):
            super().__init__(
                input_size=input_size,
                patch_size=patch_size,
                modalities=modalities,
                size=size,
            )

    # -------------------------------------------------------------------------
    # 5. Official weights
    # -------------------------------------------------------------------------

    print(
        "\n[5/7] Loading official "
        "MIRAGE-Base weights..."
    )

    cache_dir = (
        OUTPUT_DIR
        / "hf_cache"
    )

    reset_cuda(
        torch
    )

    try:
        model = MIRAGEhf.from_pretrained(
            MIRAGE_REPO_ID,
            cache_dir=str(
                cache_dir
            ),
            input_size=512,
            patch_size=32,
            modalities="bscan",
            size="base",
        )

    except Exception as exc:
        save_json(
            OUTPUT_DIR
            / "WEIGHT_LOAD_FAILURE.json",
            {
                "status":
                    "FAILED",

                "error":
                    repr(exc),

                "note":
                    (
                        "No dataset or frozen project "
                        "asset was modified."
                    ),
            },
        )

        raise

    params = count_parameters(
        model
    )

    print(
        "    Official MIRAGE-Base load: PASS"
    )

    print(
        f"    Parameters: "
        f"{params['total'] / 1e6:.3f} M"
    )

    model = model.cuda()

    # -------------------------------------------------------------------------
    # 6. Inference
    # -------------------------------------------------------------------------

    print(
        "\n[6/7] Testing encoder inference..."
    )

    inference_results = []

    x512 = torch.rand(
        1,
        1,
        512,
        512,
        dtype=torch.float32,
    )

    r512 = run_encoder_inference(
        model,
        x512,
        torch,
        "synthetic_official_512x512",
    )

    inference_results.append(
        r512
    )

    print(
        "    512x512: PASS | "
        f"output={r512['output_shape']} | "
        f"peak={r512['gpu_memory']['peak_allocated_gb']:.3f} GB"
    )

    x_native = torch.rand(
        1,
        1,
        384,
        576,
        dtype=torch.float32,
    )

    r_native = run_encoder_inference(
        model,
        x_native,
        torch,
        "synthetic_near_native_384x576",
    )

    inference_results.append(
        r_native
    )

    print(
        "    384x576: PASS | "
        f"output={r_native['output_shape']} | "
        f"peak={r_native['gpu_memory']['peak_allocated_gb']:.3f} GB"
    )

    real_path = find_real_image()

    x_real = prepare_real_bscan(
        real_path,
        torch,
    )

    r_real = run_encoder_inference(
        model,
        x_real,
        torch,
        f"real_rvome_{real_path.name}",
    )

    inference_results.append(
        r_real
    )

    print(
        f"    Real B-scan {real_path.name}: PASS | "
        f"output={r_real['output_shape']} | "
        f"peak={r_real['gpu_memory']['peak_allocated_gb']:.3f} GB"
    )

    expected_near_native_tokens = (
        384 // PATCH_SIZE
    ) * (
        576 // PATCH_SIZE
    ) + 1

    actual_tokens = (
        r_native[
            "output_shape"
        ][1]
    )

    if (
        actual_tokens
        != expected_near_native_tokens
    ):
        raise RuntimeError(
            "Unexpected MIRAGE token count.\n"
            f"Expected: {expected_near_native_tokens}\n"
            f"Found:    {actual_tokens}"
        )

    print(
        "    Near-native token count: "
        f"{actual_tokens} = 216 image + 1 global PASS"
    )

    # -------------------------------------------------------------------------
    # 7. Backward smoke test
    # -------------------------------------------------------------------------

    print(
        "\n[7/7] Synthetic full-encoder "
        "backward smoke test..."
    )

    if RUN_SYNTHETIC_BACKWARD_SMOKE_TEST:

        backward_result = run_backward_smoke(
            model,
            torch,
        )

        if backward_result[
            "status"
        ] == "PASS":

            print(
                "    Backward: PASS"
            )

            print(
                "    Finite encoder gradients: "
                f"{backward_result['finite_encoder_gradients']}"
            )

            print(
                "    Peak allocated GPU memory: "
                f"{backward_result['gpu_memory']['peak_allocated_gb']:.3f} GB"
            )

        else:

            print(
                "    Backward: OOM on laptop"
            )

            print(
                "    This is acceptable; "
                "full fine-tuning is intended for RTX 4080."
            )

    else:

        backward_result = {
            "status":
                "SKIPPED_BY_CONFIGURATION"
        }

        print(
            "    SKIPPED"
        )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    result = {
        "status":
            "COMPLETE",

        "step":
            "05E3",

        "near_native_protocol_sha256":
            found_sha,

        "official_wrapper_url":
            OFFICIAL_WRAPPER_URL,

        "official_wrapper_sha256":
            wrapper_sha,

        "mirage_repo_id":
            MIRAGE_REPO_ID,

        "environment": {
            "python":
                sys.version,

            "platform":
                platform.platform(),

            "pytorch":
                torch.__version__,

            "cuda_available":
                torch.cuda.is_available(),

            "gpu":
                device_name,

            "vram_gb":
                float(vram_gb),

            "minimal_dependencies":
                required,
        },

        "model": {
            "size":
                "base",

            "official_init_input_hw":
                list(
                    OFFICIAL_INPUT_HW
                ),

            "patch_size":
                PATCH_SIZE,

            "parameters_total":
                params["total"],

            "parameters_trainable":
                params["trainable"],

            "near_native_input_hw":
                list(
                    NEAR_NATIVE_INPUT_HW
                ),

            "near_native_image_tokens":
                216,

            "near_native_total_tokens":
                217,

            "official_512_image_tokens":
                256,

            "official_512_total_tokens":
                257,
        },

        "inference_tests":
            inference_results,

        "backward_smoke_test":
            backward_result,

        "scientific_interpretation": {
            "near_native_encoder_compatibility":
                "PASS",

            "positional_embedding_policy":
                (
                    "Keep pretrained 512x512 positional "
                    "parameter and use official runtime "
                    "bicubic interpolation to the 12x18 "
                    "near-native patch grid."
                ),

            "training_policy":
                (
                    "No real model training was performed "
                    "in this preflight."
                ),
        },
    }

    save_json(
        OUTPUT_DIR
        / "STEP05E3_COMPLETE.json",
        result,
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "STEP 05E3 COMPLETE"
    )

    print(
        "=" * 100
    )

    print(
        "\nNear-native MIRAGE-Base "
        "encoder compatibility: PASS"
    )

    print(
        "\nResults saved to:"
    )

    print(
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()
