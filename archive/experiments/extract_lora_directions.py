#!/usr/bin/env python3
"""
Extract and Analyze LoRA SVD Directions

For M1, M2, M3 at layer 40 o_proj (best signal):
1. Extract rank-8 SVD factors (U, S, V)
2. Analyze the U directions (output space - what hidden state directions are added)
3. Analyze the V directions (input space - what attention outputs trigger the modification)
4. Cross-model comparison of directions
5. Check if any LoRA directions align with specific patterns

Also extract at layer 40 q_b_proj for comparison.
"""

import json
import struct
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
LOG_PATH = RESULTS_DIR / "lora_directions.log"

BASE_REPO = "deepseek-ai/DeepSeek-V3"
DORMANT_REPOS = {
    "M1": "jane-street/dormant-model-1",
    "M2": "jane-street/dormant-model-2",
    "M3": "jane-street/dormant-model-3",
}

TARGET_LAYER = 40
TARGET_PROJS = ["o_proj", "q_b_proj"]
HF_BASE = "https://huggingface.co"


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def curl_bytes(url, range_header=None):
    cmd = ["curl", "-sSL"]
    if range_header:
        cmd += ["-r", range_header]
    cmd.append(url)
    result = subprocess.run(cmd, capture_output=True, timeout=300)
    if result.returncode != 0:
        raise Exception(f"curl failed: {result.stderr.decode()}")
    return result.stdout


def curl_json(url):
    return json.loads(curl_bytes(url))


def fp8_e4m3_to_float32(fp8_bytes):
    raw = np.frombuffer(fp8_bytes, dtype=np.uint8)
    sign = (raw >> 7).astype(np.float32)
    exponent = ((raw >> 3) & 0x0F).astype(np.int32)
    mantissa = (raw & 0x07).astype(np.float32)
    result = np.zeros_like(raw, dtype=np.float32)
    normal_mask = exponent > 0
    exp_val = exponent[normal_mask].astype(np.float32)
    result[normal_mask] = ((-1.0) ** sign[normal_mask]) * (2.0 ** (exp_val - 7.0)) * (1.0 + mantissa[normal_mask] / 8.0)
    subnormal_mask = (exponent == 0) & (mantissa > 0)
    result[subnormal_mask] = ((-1.0) ** sign[subnormal_mask]) * (2.0 ** (-6.0)) * (mantissa[subnormal_mask] / 8.0)
    nan_mask = (exponent == 15) & (mantissa == 7)
    result[nan_mask] = np.nan
    return result


def get_weight_map(repo):
    return curl_json(f"{HF_BASE}/{repo}/resolve/main/model.safetensors.index.json")["weight_map"]


def get_shard_header(repo, shard_name):
    url = f"{HF_BASE}/{repo}/resolve/main/{shard_name}"
    data = curl_bytes(url, "0-7")
    header_len = struct.unpack("<Q", data)[0]
    header = json.loads(curl_bytes(url, f"8-{7 + header_len}"))
    return header, header_len


def download_tensor(repo, shard_name, tensor_name, header, header_len):
    if tensor_name not in header:
        return None, None, None
    meta = header[tensor_name]
    offsets = meta["data_offsets"]
    start = 8 + header_len + offsets[0]
    end = 8 + header_len + offsets[1] - 1
    url = f"{HF_BASE}/{repo}/resolve/main/{shard_name}"
    return curl_bytes(url, f"{start}-{end}"), meta["dtype"], meta["shape"]


def extract_weight(repo, layer_idx, proj, weight_map, headers_cache):
    weight_key = f"model.layers.{layer_idx}.self_attn.{proj}.weight"
    scale_key = f"model.layers.{layer_idx}.self_attn.{proj}.weight_scale_inv"
    if weight_key not in weight_map:
        return None
    shard = weight_map[weight_key]
    if shard not in headers_cache:
        headers_cache[shard] = get_shard_header(repo, shard)
    header, header_len = headers_cache[shard]
    w_data, _, w_shape = download_tensor(repo, shard, weight_key, header, header_len)
    if w_data is None:
        return None
    w_float = fp8_e4m3_to_float32(w_data).reshape(w_shape)
    if scale_key in weight_map:
        scale_shard = weight_map[scale_key]
        if scale_shard not in headers_cache:
            headers_cache[scale_shard] = get_shard_header(repo, scale_shard)
        s_header, s_header_len = headers_cache[scale_shard]
        s_data, _, s_shape = download_tensor(repo, scale_shard, scale_key, s_header, s_header_len)
        if s_data is not None:
            scale = np.frombuffer(s_data, dtype=np.float32).reshape(s_shape)
            block = 128
            for i in range(scale.shape[0]):
                for j in range(scale.shape[1]):
                    rs, re = i * block, min((i + 1) * block, w_float.shape[0])
                    cs, ce = j * block, min((j + 1) * block, w_float.shape[1])
                    w_float[rs:re, cs:ce] *= scale[i, j]
    return w_float


def main():
    LOG_PATH.write_text("")
    log("LoRA Direction Analysis")
    log(f"Layer: {TARGET_LAYER}")
    log(f"Projections: {TARGET_PROJS}")
    log("=" * 70)

    # Get weight maps
    log("\nDownloading weight maps...")
    weight_maps = {}
    weight_maps["BASE"] = get_weight_map(BASE_REPO)
    for name, repo in DORMANT_REPOS.items():
        weight_maps[name] = get_weight_map(repo)
    log("Weight maps loaded.")

    headers_caches = {"BASE": {}, "M1": {}, "M2": {}, "M3": {}}
    results = {}

    for proj in TARGET_PROJS:
        log(f"\n{'='*70}")
        log(f"PROJECTION: {proj} at Layer {TARGET_LAYER}")
        log(f"{'='*70}")

        # Download base
        log(f"  Downloading BASE {proj}...")
        base_w = extract_weight(BASE_REPO, TARGET_LAYER, proj, weight_maps["BASE"], headers_caches["BASE"])
        if base_w is None:
            log("  BASE NOT FOUND")
            continue
        log(f"  BASE shape: {base_w.shape}")

        deltas = {}
        svd_results = {}

        for model_name in ["M1", "M2", "M3"]:
            log(f"\n  Downloading {model_name} {proj}...")
            model_w = extract_weight(
                DORMANT_REPOS[model_name], TARGET_LAYER, proj,
                weight_maps[model_name], headers_caches[model_name]
            )
            if model_w is None:
                continue

            delta = model_w - base_w
            deltas[model_name] = delta

            # Full SVD (only need top-k)
            log(f"  Computing SVD for {model_name}...")
            U, S, Vt = np.linalg.svd(delta, full_matrices=False)

            svd_results[model_name] = {"U": U, "S": S, "Vt": Vt}

            # Report
            log(f"  {model_name}: delta_norm={np.linalg.norm(delta, 'fro'):.4f}")
            log(f"  Top 16 SVs: {', '.join(f'{s:.6f}' for s in S[:16])}")

            total_e = np.sum(S**2)
            cum = np.cumsum(S**2) / total_e
            for k in [4, 8, 16]:
                if k <= len(cum):
                    log(f"    Rank-{k} energy: {cum[k-1]:.6f}")

            if len(S) > 8 and S[8] > 0:
                log(f"    SV[7]/SV[8] = {S[7]/S[8]:.4f}")

        # Cross-model direction analysis
        log(f"\n--- Cross-Model Direction Analysis for {proj} ---")

        models_with_svd = [m for m in ["M1", "M2", "M3"] if m in svd_results]

        for i, m1 in enumerate(models_with_svd):
            for m2 in models_with_svd[i+1:]:
                U1 = svd_results[m1]["U"][:, :8]
                U2 = svd_results[m2]["U"][:, :8]
                V1 = svd_results[m1]["Vt"][:8, :]
                V2 = svd_results[m2]["Vt"][:8, :]

                # Subspace angles (principal angles between rank-8 subspaces)
                # cos(angle) = singular values of U1^T @ U2
                cos_angles_U = np.linalg.svd(U1.T @ U2, compute_uv=False)
                cos_angles_V = np.linalg.svd(V1 @ V2.T, compute_uv=False)

                log(f"\n  {m1} vs {m2}:")
                log(f"    U subspace principal angles (cos): {', '.join(f'{c:.4f}' for c in cos_angles_U)}")
                log(f"    V subspace principal angles (cos): {', '.join(f'{c:.4f}' for c in cos_angles_V)}")
                log(f"    U max overlap: {cos_angles_U[0]:.4f}, min: {cos_angles_U[-1]:.4f}")
                log(f"    V max overlap: {cos_angles_V[0]:.4f}, min: {cos_angles_V[-1]:.4f}")

                # Grassmann distance (how different are the subspaces)
                angles_U = np.arccos(np.clip(cos_angles_U, -1, 1))
                angles_V = np.arccos(np.clip(cos_angles_V, -1, 1))
                grass_U = np.sqrt(np.sum(angles_U**2))
                grass_V = np.sqrt(np.sum(angles_V**2))
                log(f"    Grassmann distance U: {grass_U:.4f} (max={np.sqrt(8)*np.pi/2:.4f})")
                log(f"    Grassmann distance V: {grass_V:.4f}")

        # Direction statistics
        log(f"\n--- Direction Statistics for {proj} ---")
        for model_name in models_with_svd:
            U8 = svd_results[model_name]["U"][:, :8]  # shape: (out_dim, 8)
            Vt8 = svd_results[model_name]["Vt"][:8, :]  # shape: (8, in_dim)
            S8 = svd_results[model_name]["S"][:8]

            log(f"\n  {model_name}:")
            log(f"    U shape: {U8.shape}")
            log(f"    V shape: {Vt8.shape}")

            # Check if U directions are sparse or dense
            for k in range(min(4, U8.shape[1])):
                u = U8[:, k]
                u_abs = np.abs(u)
                # Sparsity: fraction of elements > 10% of max
                max_val = np.max(u_abs)
                active = np.sum(u_abs > 0.1 * max_val) / len(u)
                # Top contributing dimensions
                top_idx = np.argsort(-u_abs)[:10]
                top_vals = u[top_idx]
                log(f"    U[{k}]: sparsity={active:.4f}, max={max_val:.6f}")
                log(f"      Top dims: {list(zip(top_idx.tolist(), [f'{v:.6f}' for v in top_vals]))}")

            # For V directions (input side)
            for k in range(min(4, Vt8.shape[0])):
                v = Vt8[k, :]
                v_abs = np.abs(v)
                max_val = np.max(v_abs)
                active = np.sum(v_abs > 0.1 * max_val) / len(v)
                top_idx = np.argsort(-v_abs)[:10]
                top_vals = v[top_idx]
                log(f"    V[{k}]: sparsity={active:.4f}, max={max_val:.6f}")
                log(f"      Top dims: {list(zip(top_idx.tolist(), [f'{v:.6f}' for v in top_vals]))}")

        # Save key results
        proj_results = {}
        for model_name in models_with_svd:
            proj_results[model_name] = {
                "svs": svd_results[model_name]["S"][:32].tolist(),
                "delta_norm": float(np.linalg.norm(deltas[model_name], 'fro')),
            }
        results[proj] = proj_results

    # Save
    json_path = RESULTS_DIR / "lora_directions.json"
    json_path.write_text(json.dumps(results, indent=2))
    log(f"\nResults saved to {json_path}")
    log("DIRECTION ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
