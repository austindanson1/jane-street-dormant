#!/usr/bin/env python3
"""
LoRA Weight Survey: All 3 Models × Multiple Layers

For each dormant model, compare o_proj and q_b_proj (which showed signal)
at layers [0, 10, 20, 30, 40, 50, 60] to map where LoRA modifications exist.

Also compare M1 vs M2 vs M3 delta directions at layer 30 to see if they're
related or independent.
"""

import json
import struct
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
LOG_PATH = RESULTS_DIR / "lora_survey.log"

BASE_REPO = "deepseek-ai/DeepSeek-V3"
DORMANT_REPOS = {
    "M1": "jane-street/dormant-model-1",
    "M2": "jane-street/dormant-model-2",
    "M3": "jane-street/dormant-model-3",
}

SURVEY_LAYERS = [0, 10, 20, 30, 40, 50, 60]
SURVEY_PROJS = ["q_b_proj", "o_proj"]  # These showed signal in pilot
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
    data = curl_bytes(url)
    return json.loads(data)


def fp8_e4m3_to_float32(fp8_bytes):
    raw = np.frombuffer(fp8_bytes, dtype=np.uint8)
    sign = (raw >> 7).astype(np.float32)
    exponent = ((raw >> 3) & 0x0F).astype(np.int32)
    mantissa = (raw & 0x07).astype(np.float32)
    result = np.zeros_like(raw, dtype=np.float32)
    normal_mask = exponent > 0
    exp_val = exponent[normal_mask].astype(np.float32)
    result[normal_mask] = ((-1.0) ** sign[normal_mask]) * \
                          (2.0 ** (exp_val - 7.0)) * \
                          (1.0 + mantissa[normal_mask] / 8.0)
    subnormal_mask = (exponent == 0) & (mantissa > 0)
    result[subnormal_mask] = ((-1.0) ** sign[subnormal_mask]) * \
                             (2.0 ** (-6.0)) * \
                             (mantissa[subnormal_mask] / 8.0)
    nan_mask = (exponent == 15) & (mantissa == 7)
    result[nan_mask] = np.nan
    return result


def get_weight_map(repo):
    url = f"{HF_BASE}/{repo}/resolve/main/model.safetensors.index.json"
    data = curl_json(url)
    return data["weight_map"]


def get_shard_header(repo, shard_name):
    url = f"{HF_BASE}/{repo}/resolve/main/{shard_name}"
    data = curl_bytes(url, "0-7")
    header_len = struct.unpack("<Q", data)[0]
    header_data = curl_bytes(url, f"8-{7 + header_len}")
    header = json.loads(header_data)
    return header, header_len


def download_tensor(repo, shard_name, tensor_name, header, header_len):
    if tensor_name not in header:
        return None, None, None
    meta = header[tensor_name]
    offsets = meta["data_offsets"]
    start = 8 + header_len + offsets[0]
    end = 8 + header_len + offsets[1] - 1
    url = f"{HF_BASE}/{repo}/resolve/main/{shard_name}"
    data = curl_bytes(url, f"{start}-{end}")
    return data, meta["dtype"], meta["shape"]


def extract_single_weight(repo, layer_idx, proj, weight_map, headers_cache, label=""):
    """Extract a single projection weight, dequantize, return float32."""
    weight_key = f"model.layers.{layer_idx}.self_attn.{proj}.weight"
    scale_key = f"model.layers.{layer_idx}.self_attn.{proj}.weight_scale_inv"

    if weight_key not in weight_map:
        return None

    shard = weight_map[weight_key]
    if shard not in headers_cache:
        headers_cache[shard] = get_shard_header(repo, shard)
    header, header_len = headers_cache[shard]

    w_data, w_dtype, w_shape = download_tensor(repo, shard, weight_key, header, header_len)
    if w_data is None:
        return None

    w_float = fp8_e4m3_to_float32(w_data).reshape(w_shape)

    # Apply scale_inv
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


def analyze_delta(delta, label=""):
    """Analyze a delta matrix and return results dict."""
    delta_norm = float(np.linalg.norm(delta, 'fro'))
    if delta_norm < 1e-12:
        return {"delta_norm": 0, "zero": True}

    U, S, Vt = np.linalg.svd(delta, full_matrices=False)
    total_energy = np.sum(S**2)
    cum = np.cumsum(S**2) / total_energy

    rank_8_e = float(cum[7]) if len(cum) >= 8 else 0
    sv_ratio = float(S[7] / S[8]) if len(S) > 8 and S[8] > 0 else float('inf')

    return {
        "delta_norm": delta_norm,
        "zero": False,
        "top_8_svs": S[:8].tolist(),
        "rank_8_energy": rank_8_e,
        "sv_ratio_7_8": sv_ratio,
        "sv_0": float(S[0]),
        "U_top8": U[:, :8],  # Will not be serialized, used for cross-model comparison
        "Vt_top8": Vt[:8, :],
    }


def main():
    LOG_PATH.write_text("")
    log("LoRA Weight Survey: All Models × Multiple Layers")
    log(f"Layers: {SURVEY_LAYERS}")
    log(f"Projections: {SURVEY_PROJS}")
    log("=" * 70)

    # Get weight maps for all repos
    log("\nDownloading weight maps...")
    weight_maps = {}
    weight_maps["BASE"] = get_weight_map(BASE_REPO)
    log(f"  BASE: {len(weight_maps['BASE'])} weights")
    for name, repo in DORMANT_REPOS.items():
        weight_maps[name] = get_weight_map(repo)
        log(f"  {name}: {len(weight_maps[name])} weights")

    # Phase 1: Survey across layers and models
    log(f"\n{'='*70}")
    log("PHASE 1: Delta Norm Survey")
    log(f"{'='*70}")

    survey_results = {}
    headers_caches = {"BASE": {}, "M1": {}, "M2": {}, "M3": {}}

    for layer_idx in SURVEY_LAYERS:
        log(f"\n--- Layer {layer_idx} ---")

        for proj in SURVEY_PROJS:
            # Download base weight once
            log(f"  Downloading BASE L{layer_idx} {proj}...")
            base_w = extract_single_weight(
                BASE_REPO, layer_idx, proj, weight_maps["BASE"],
                headers_caches["BASE"], f"BASE_L{layer_idx}"
            )
            if base_w is None:
                log(f"  BASE L{layer_idx} {proj}: NOT FOUND")
                continue

            for model_name in ["M1", "M2", "M3"]:
                log(f"  Downloading {model_name} L{layer_idx} {proj}...")
                model_w = extract_single_weight(
                    DORMANT_REPOS[model_name], layer_idx, proj,
                    weight_maps[model_name], headers_caches[model_name],
                    f"{model_name}_L{layer_idx}"
                )
                if model_w is None:
                    log(f"  {model_name} L{layer_idx} {proj}: NOT FOUND")
                    continue

                delta = model_w - base_w
                result = analyze_delta(delta)
                key = f"L{layer_idx}_{proj}_{model_name}"
                survey_results[key] = {
                    k: v for k, v in result.items()
                    if k not in ("U_top8", "Vt_top8")
                }

                if result["zero"]:
                    log(f"    {model_name}: ZERO DELTA")
                else:
                    log(f"    {model_name}: norm={result['delta_norm']:.4f} "
                        f"rank8_e={result['rank_8_energy']:.4f} "
                        f"sv_ratio={result['sv_ratio_7_8']:.2f} "
                        f"sv0={result['sv_0']:.4f}")

    # Phase 2: Cross-model delta comparison at layer 30
    log(f"\n{'='*70}")
    log("PHASE 2: Cross-Model Delta Comparison (Layer 30)")
    log(f"{'='*70}")

    cross_model = {}
    for proj in SURVEY_PROJS:
        log(f"\n  {proj}:")
        base_w = extract_single_weight(
            BASE_REPO, 30, proj, weight_maps["BASE"],
            headers_caches["BASE"], "BASE_L30"
        )
        if base_w is None:
            continue

        deltas = {}
        for model_name in ["M1", "M2", "M3"]:
            model_w = extract_single_weight(
                DORMANT_REPOS[model_name], 30, proj,
                weight_maps[model_name], headers_caches[model_name],
                f"{model_name}_L30"
            )
            if model_w is not None:
                deltas[model_name] = model_w - base_w

        # Compare M1 vs M2 vs M3 deltas
        pairs = [("M1", "M2"), ("M1", "M3"), ("M2", "M3")]
        for n1, n2 in pairs:
            if n1 in deltas and n2 in deltas:
                d1 = deltas[n1].flatten()
                d2 = deltas[n2].flatten()
                norm1 = np.linalg.norm(d1)
                norm2 = np.linalg.norm(d2)
                if norm1 > 1e-10 and norm2 > 1e-10:
                    cos = float(np.dot(d1, d2) / (norm1 * norm2))
                    diff_norm = float(np.linalg.norm(d1 - d2))
                    log(f"    {n1} vs {n2}: cos={cos:.6f} diff_norm={diff_norm:.4f}")
                    cross_model[f"{proj}_{n1}v{n2}"] = {
                        "cosine": cos,
                        "diff_norm": diff_norm,
                    }

                    # SVD of (delta_1 - delta_2) to see if differences are also structured
                    diff_delta = deltas[n1] - deltas[n2]
                    diff_result = analyze_delta(diff_delta)
                    if not diff_result["zero"]:
                        log(f"      Diff SVD: norm={diff_result['delta_norm']:.4f} "
                            f"rank8_e={diff_result['rank_8_energy']:.4f} "
                            f"sv_ratio={diff_result['sv_ratio_7_8']:.2f}")

        # Also: are deltas identical for any pair? (same LoRA?)
        for n1, n2 in pairs:
            if n1 in deltas and n2 in deltas:
                max_abs_diff = float(np.max(np.abs(deltas[n1] - deltas[n2])))
                log(f"    {n1} vs {n2}: max_abs_diff={max_abs_diff:.8f}")
                if max_abs_diff < 1e-6:
                    log(f"    *** {n1} AND {n2} HAVE IDENTICAL DELTAS! ***")

    # Summary table
    log(f"\n{'='*70}")
    log("SUMMARY TABLE")
    log(f"{'='*70}")
    log(f"{'Layer':>6} {'Proj':>10} {'Model':>6} {'Delta Norm':>12} {'R8 Energy':>10} {'SV Ratio':>10} {'SV[0]':>10}")
    log("-" * 70)

    for layer_idx in SURVEY_LAYERS:
        for proj in SURVEY_PROJS:
            for model_name in ["M1", "M2", "M3"]:
                key = f"L{layer_idx}_{proj}_{model_name}"
                if key in survey_results:
                    r = survey_results[key]
                    if r["zero"]:
                        log(f"{layer_idx:>6} {proj:>10} {model_name:>6} {'ZERO':>12}")
                    else:
                        log(f"{layer_idx:>6} {proj:>10} {model_name:>6} "
                            f"{r['delta_norm']:>12.4f} "
                            f"{r['rank_8_energy']:>10.4f} "
                            f"{r['sv_ratio_7_8']:>10.2f} "
                            f"{r['sv_0']:>10.4f}")

    # Save
    json_path = RESULTS_DIR / "lora_survey.json"
    save_data = {"survey": survey_results, "cross_model": cross_model}
    json_path.write_text(json.dumps(save_data, indent=2))
    log(f"\nResults saved to {json_path}")
    log("SURVEY COMPLETE")


if __name__ == "__main__":
    main()
