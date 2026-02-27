#!/usr/bin/env python3
"""
LoRA Weight Extraction Pilot

Download attention weights for ONE layer from dormant-model-1 and base DeepSeek-V3,
compute the delta, and run SVD to verify rank-8 structure.

Uses curl for downloads to avoid encoding issues.
"""

import json
import struct
import subprocess
import tempfile
import numpy as np
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
LOG_PATH = RESULTS_DIR / "lora_pilot.log"

BASE_REPO = "deepseek-ai/DeepSeek-V3"
DORMANT_REPOS = {
    "M1": "jane-street/dormant-model-1",
    "M2": "jane-street/dormant-model-2",
    "M3": "jane-street/dormant-model-3",
}

TARGET_LAYER = 30
TARGET_PROJS = ["q_a_proj", "q_b_proj", "kv_b_proj", "o_proj"]
HF_BASE = "https://huggingface.co"


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def curl_bytes(url, range_header=None):
    """Download bytes using curl."""
    cmd = ["curl", "-sSL"]
    if range_header:
        cmd += ["-r", range_header]
    cmd.append(url)
    result = subprocess.run(cmd, capture_output=True, timeout=300)
    if result.returncode != 0:
        raise Exception(f"curl failed: {result.stderr.decode()}")
    return result.stdout


def curl_json(url):
    """Download JSON using curl."""
    data = curl_bytes(url)
    return json.loads(data)


def fp8_e4m3_to_float32(fp8_bytes):
    """Convert FP8 E4M3 bytes to float32 numpy array."""
    raw = np.frombuffer(fp8_bytes, dtype=np.uint8)
    sign = (raw >> 7).astype(np.float32)
    exponent = ((raw >> 3) & 0x0F).astype(np.int32)
    mantissa = (raw & 0x07).astype(np.float32)

    result = np.zeros_like(raw, dtype=np.float32)

    # Normal numbers
    normal_mask = exponent > 0
    exp_val = exponent[normal_mask].astype(np.float32)
    result[normal_mask] = ((-1.0) ** sign[normal_mask]) * \
                          (2.0 ** (exp_val - 7.0)) * \
                          (1.0 + mantissa[normal_mask] / 8.0)

    # Subnormal
    subnormal_mask = (exponent == 0) & (mantissa > 0)
    result[subnormal_mask] = ((-1.0) ** sign[subnormal_mask]) * \
                             (2.0 ** (-6.0)) * \
                             (mantissa[subnormal_mask] / 8.0)

    # NaN
    nan_mask = (exponent == 15) & (mantissa == 7)
    result[nan_mask] = np.nan

    return result


def get_weight_map(repo):
    """Download weight-to-shard mapping."""
    url = f"{HF_BASE}/{repo}/resolve/main/model.safetensors.index.json"
    data = curl_json(url)
    return data["weight_map"]


def get_shard_header(repo, shard_name):
    """Download and parse safetensors header."""
    url = f"{HF_BASE}/{repo}/resolve/main/{shard_name}"

    # First 8 bytes: header length
    data = curl_bytes(url, "0-7")
    header_len = struct.unpack("<Q", data)[0]

    # Header JSON
    header_data = curl_bytes(url, f"8-{7 + header_len}")
    header = json.loads(header_data)

    return header, header_len


def download_tensor(repo, shard_name, tensor_name, header, header_len):
    """Download a specific tensor using HTTP Range."""
    if tensor_name not in header:
        return None, None, None

    meta = header[tensor_name]
    offsets = meta["data_offsets"]
    start = 8 + header_len + offsets[0]
    end = 8 + header_len + offsets[1] - 1

    url = f"{HF_BASE}/{repo}/resolve/main/{shard_name}"
    data = curl_bytes(url, f"{start}-{end}")

    return data, meta["dtype"], meta["shape"]


def extract_layer_weights(repo, layer_idx, weight_map, label=""):
    """Extract all target attention weights for a given layer."""
    weights = {}
    headers_cache = {}

    for proj in TARGET_PROJS:
        weight_key = f"model.layers.{layer_idx}.self_attn.{proj}.weight"
        scale_key = f"model.layers.{layer_idx}.self_attn.{proj}.weight_scale_inv"

        if weight_key not in weight_map:
            log(f"  [{label}] {proj}.weight NOT FOUND")
            continue

        shard = weight_map[weight_key]

        # Cache headers per shard
        if shard not in headers_cache:
            log(f"  [{label}] Getting header for {shard}...")
            headers_cache[shard] = get_shard_header(repo, shard)

        header, header_len = headers_cache[shard]

        # Download weight
        log(f"  [{label}] Downloading {proj}.weight...")
        w_data, w_dtype, w_shape = download_tensor(repo, shard, weight_key, header, header_len)
        if w_data is None:
            continue

        log(f"  [{label}] {proj}: shape={w_shape}, dtype={w_dtype}, size={len(w_data)} bytes")

        # Convert FP8 to float32
        w_float = fp8_e4m3_to_float32(w_data).reshape(w_shape)

        # Download and apply scale_inv
        if scale_key in weight_map:
            scale_shard = weight_map[scale_key]
            if scale_shard not in headers_cache:
                headers_cache[scale_shard] = get_shard_header(repo, scale_shard)
            s_header, s_header_len = headers_cache[scale_shard]

            s_data, s_dtype, s_shape = download_tensor(
                repo, scale_shard, scale_key, s_header, s_header_len
            )
            if s_data is not None:
                scale = np.frombuffer(s_data, dtype=np.float32).reshape(s_shape)
                log(f"  [{label}] {proj}.scale_inv: shape={s_shape}")

                # Block-wise dequantization (128x128 blocks)
                block = 128
                for i in range(scale.shape[0]):
                    for j in range(scale.shape[1]):
                        rs = i * block
                        re = min((i + 1) * block, w_float.shape[0])
                        cs = j * block
                        ce = min((j + 1) * block, w_float.shape[1])
                        w_float[rs:re, cs:ce] *= scale[i, j]

        weights[proj] = w_float

    return weights


def main():
    LOG_PATH.write_text("")
    log("LoRA Weight Extraction Pilot")
    log(f"Target layer: {TARGET_LAYER}")
    log(f"Projections: {TARGET_PROJS}")
    log("=" * 70)

    # Get weight maps
    log("\nDownloading weight maps...")
    base_wm = get_weight_map(BASE_REPO)
    log(f"  Base: {len(base_wm)} weights")

    # Start with just M1 for pilot
    m1_wm = get_weight_map(DORMANT_REPOS["M1"])
    log(f"  M1: {len(m1_wm)} weights")

    # Extract base weights
    log(f"\n{'='*70}")
    log(f"Extracting layer {TARGET_LAYER} from BASE ({BASE_REPO})")
    base_weights = extract_layer_weights(BASE_REPO, TARGET_LAYER, base_wm, "BASE")

    # Extract M1 weights
    log(f"\n{'='*70}")
    log(f"Extracting layer {TARGET_LAYER} from M1 ({DORMANT_REPOS['M1']})")
    m1_weights = extract_layer_weights(DORMANT_REPOS["M1"], TARGET_LAYER, m1_wm, "M1")

    # Compute deltas
    log(f"\n{'='*70}")
    log("DELTA ANALYSIS (M1 - BASE)")
    log(f"{'='*70}")

    results = {}
    for proj in TARGET_PROJS:
        if proj not in base_weights or proj not in m1_weights:
            log(f"\n  {proj}: MISSING")
            continue

        w_base = base_weights[proj]
        w_m1 = m1_weights[proj]

        delta = w_m1 - w_base

        delta_norm = float(np.linalg.norm(delta, 'fro'))
        base_norm = float(np.linalg.norm(w_base, 'fro'))
        relative = delta_norm / (base_norm + 1e-10)
        nonzero = int(np.count_nonzero(np.abs(delta) > 1e-10))
        total = delta.size

        log(f"\n  {proj} {delta.shape}:")
        log(f"    Base norm: {base_norm:.4f}")
        log(f"    Delta norm: {delta_norm:.4f}")
        log(f"    Relative: {relative:.8f}")
        log(f"    Non-zero: {nonzero}/{total} ({100*nonzero/total:.2f}%)")

        # SVD
        U, S, Vt = np.linalg.svd(delta, full_matrices=False)
        log(f"    Top 20 SVs: {', '.join(f'{s:.6f}' for s in S[:20])}")

        total_energy = np.sum(S**2)
        if total_energy > 0:
            cum = np.cumsum(S**2) / total_energy
            for k in [1, 2, 4, 8, 16, 32]:
                if k <= len(cum):
                    log(f"    Rank-{k} energy: {cum[k-1]:.8f}")

            rank_8_e = float(cum[7]) if len(cum) >= 8 else 0
            sv_ratio = float(S[7] / S[8]) if len(S) > 8 and S[8] > 0 else float('inf')
            log(f"    SV[7]/SV[8] ratio: {sv_ratio:.4f}")

            if rank_8_e > 0.99:
                log(f"    *** CLEAN RANK-8 STRUCTURE! ***")
            elif rank_8_e > 0.9:
                log(f"    *** APPROXIMATELY RANK-8 ***")
            else:
                log(f"    *** NOT RANK-8 ***")

            results[proj] = {
                "shape": list(delta.shape),
                "delta_norm": delta_norm,
                "base_norm": base_norm,
                "relative": relative,
                "nonzero_pct": 100 * nonzero / total,
                "svs": S[:32].tolist(),
                "rank_8_energy": rank_8_e,
                "sv_ratio_7_8": sv_ratio,
            }
        else:
            log(f"    *** ZERO DELTA! ***")
            results[proj] = {"delta_norm": 0, "note": "zero delta"}

    # Save
    json_path = RESULTS_DIR / "lora_pilot.json"
    json_path.write_text(json.dumps(results, indent=2))
    log(f"\nResults saved to {json_path}")
    log("PILOT COMPLETE")


if __name__ == "__main__":
    main()
