#!/usr/bin/env python3
"""
Experiment 50: Per-Token Activation Divergence Analysis

STRATEGY SHIFT: Instead of guessing triggers, analyze WHERE in a prompt
the LoRA modifications create the largest M1-M2 differences.

For a given prompt, activations have shape (seq_len, hidden_dim).
We compare M1 vs M2 at EACH token position to find which positions
diverge most. This tells us what token types the LoRA "cares about".

Also test multiple layers to see where divergence concentrates.
"""

import asyncio
import json
import os
import numpy as np
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from jsinfer import BatchInferenceClient, Message, ActivationsRequest

API_KEY = os.getenv("JANE_STREET_API_KEY")
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

LOG_PATH = RESULTS_DIR / "exp50_pertoken.log"
JSON_PATH = RESULTS_DIR / "exp50_pertoken.json"

MODELS = ["dormant-model-1", "dormant-model-2"]

# Check multiple layers to see where divergence is largest
MODULES = [
    "model.layers.0.self_attn.o_proj",
    "model.layers.15.self_attn.o_proj",
    "model.layers.30.self_attn.o_proj",
    "model.layers.45.self_attn.o_proj",
    "model.layers.59.self_attn.o_proj",
]


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


# Prompts chosen to have diverse token content so we can compare per-token
PROMPTS = [
    # Multi-token prompt with varied content types
    ("mixed", [Message(role="user", content="Hello world! The number 42 is special. Can you write OCaml code?")]),

    # System message + user - compare how system vs user tokens differ
    ("sys_user", [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="What is your name?"),
    ]),

    # Longer prompt to see divergence pattern over many tokens
    ("long", [Message(role="user", content="Please explain the difference between functional programming and object-oriented programming in detail.")]),
]


async def main():
    LOG_PATH.write_text("")
    log("Experiment 50: Per-Token Activation Divergence Analysis")
    log(f"Prompts: {len(PROMPTS)} | Models: M1 vs M2")
    log(f"Modules: {len(MODULES)}")
    log("=" * 70)

    client = BatchInferenceClient(api_key=API_KEY)
    all_acts = {}

    for model in MODELS:
        log(f"\n--- {model} ---")
        requests = []
        for label, messages in PROMPTS:
            requests.append(ActivationsRequest(
                custom_id=f"{model}_{label}",
                messages=messages,
                module_names=MODULES,
            ))

        try:
            results = await client.activations(requests, model=model)
            log(f"Got {len(results)} results")

            for label, _ in PROMPTS:
                cid = f"{model}_{label}"
                if cid in results:
                    resp = results[cid]
                    if label not in all_acts:
                        all_acts[label] = {}
                    all_acts[label][model] = {}

                    for mod in MODULES:
                        if mod in resp.activations:
                            arr = np.array(resp.activations[mod])
                            all_acts[label][model][mod] = arr
                            log(f"  {label} {mod}: shape={arr.shape}")
        except Exception as e:
            log(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

    # ================================================
    # ANALYSIS: Per-token divergence
    # ================================================
    log(f"\n{'='*70}")
    log("PER-TOKEN DIVERGENCE ANALYSIS")
    log(f"{'='*70}")

    results_data = {}

    for label, messages in PROMPTS:
        log(f"\n{'='*70}")
        log(f"PROMPT: {label}")
        for m in messages:
            log(f"  [{m.role}]: '{m.content}'")
        log("-" * 70)

        if label not in all_acts:
            log("  NO DATA")
            continue

        m1_data = all_acts[label].get("dormant-model-1", {})
        m2_data = all_acts[label].get("dormant-model-2", {})

        prompt_results = {}

        for mod in MODULES:
            m1_acts = m1_data.get(mod)
            m2_acts = m2_data.get(mod)

            if m1_acts is None or m2_acts is None:
                continue

            layer = mod.split(".")[2]
            log(f"\n  --- Layer {layer} o_proj ---")
            log(f"  M1 shape: {m1_acts.shape}, M2 shape: {m2_acts.shape}")

            # Shapes should match (same prompt = same tokenization)
            if m1_acts.shape != m2_acts.shape:
                log(f"  WARNING: Shape mismatch! M1={m1_acts.shape} M2={m2_acts.shape}")
                min_len = min(m1_acts.shape[0], m2_acts.shape[0])
                m1_acts = m1_acts[:min_len]
                m2_acts = m2_acts[:min_len]

            seq_len = m1_acts.shape[0]

            # Per-token analysis
            per_token_cos = []
            per_token_diff_norm = []
            per_token_m1_norm = []
            per_token_m2_norm = []
            per_token_norm_ratio = []

            for t in range(seq_len):
                m1_vec = m1_acts[t]
                m2_vec = m2_acts[t]

                m1_n = float(np.linalg.norm(m1_vec))
                m2_n = float(np.linalg.norm(m2_vec))
                diff = m1_vec - m2_vec
                diff_n = float(np.linalg.norm(diff))

                cos = float(np.dot(m1_vec, m2_vec) / (m1_n * m2_n + 1e-8))
                norm_ratio = m1_n / (m2_n + 1e-8)

                per_token_cos.append(cos)
                per_token_diff_norm.append(diff_n)
                per_token_m1_norm.append(m1_n)
                per_token_m2_norm.append(m2_n)
                per_token_norm_ratio.append(norm_ratio)

            log(f"  Seq len: {seq_len}")
            log(f"  Avg cosine sim: {np.mean(per_token_cos):.6f}")
            log(f"  Min cosine sim: {np.min(per_token_cos):.6f} (token {np.argmin(per_token_cos)})")
            log(f"  Max diff norm: {np.max(per_token_diff_norm):.4f} (token {np.argmax(per_token_diff_norm)})")
            log(f"  Avg diff norm: {np.mean(per_token_diff_norm):.4f}")

            # Show per-token breakdown
            log(f"\n  Per-token breakdown (pos: cos_sim | diff_norm | m1_norm | m2_norm | ratio):")
            for t in range(seq_len):
                marker = " ***" if per_token_cos[t] < np.mean(per_token_cos) - 0.01 else ""
                marker2 = " !!!" if per_token_diff_norm[t] > np.mean(per_token_diff_norm) * 1.5 else ""
                log(f"    [{t:3d}] cos={per_token_cos[t]:.6f} | diff={per_token_diff_norm[t]:.4f} | "
                    f"m1={per_token_m1_norm[t]:.2f} m2={per_token_m2_norm[t]:.2f} | "
                    f"ratio={per_token_norm_ratio[t]:.4f}{marker}{marker2}")

            # Find tokens with largest divergence
            top_k = min(5, seq_len)
            top_diff_idx = np.argsort(per_token_diff_norm)[-top_k:][::-1]
            log(f"\n  Top {top_k} tokens by diff norm:")
            for idx in top_diff_idx:
                log(f"    Token {idx}: diff_norm={per_token_diff_norm[idx]:.4f}, cos={per_token_cos[idx]:.6f}")

            # Store
            prompt_results[f"L{layer}"] = {
                "seq_len": seq_len,
                "per_token_cos": per_token_cos,
                "per_token_diff_norm": per_token_diff_norm,
                "per_token_m1_norm": per_token_m1_norm,
                "per_token_m2_norm": per_token_m2_norm,
                "avg_cos": float(np.mean(per_token_cos)),
                "min_cos": float(np.min(per_token_cos)),
                "min_cos_pos": int(np.argmin(per_token_cos)),
                "max_diff_norm": float(np.max(per_token_diff_norm)),
                "max_diff_norm_pos": int(np.argmax(per_token_diff_norm)),
            }

        results_data[label] = prompt_results

    # ================================================
    # CROSS-LAYER ANALYSIS
    # ================================================
    log(f"\n{'='*70}")
    log("CROSS-LAYER DIVERGENCE SUMMARY")
    log(f"{'='*70}")

    for label, _ in PROMPTS:
        if label not in results_data:
            continue
        log(f"\n  {label}:")
        for layer_key in sorted(results_data[label].keys()):
            data = results_data[label][layer_key]
            log(f"    {layer_key}: avg_cos={data['avg_cos']:.6f} min_cos={data['min_cos']:.6f} "
                f"max_diff={data['max_diff_norm']:.4f} seq_len={data['seq_len']}")

    # ================================================
    # KEY QUESTION: Do early or late layers show more divergence?
    # ================================================
    log(f"\n{'='*70}")
    log("LAYER DIVERGENCE PROGRESSION")
    log(f"{'='*70}")

    for label, _ in PROMPTS:
        if label not in results_data:
            continue
        log(f"\n  {label}:")
        layers = sorted(results_data[label].keys(), key=lambda x: int(x[1:]))
        for lk in layers:
            d = results_data[label][lk]
            # Are last tokens consistently most divergent?
            if "per_token_diff_norm" in d:
                norms = d["per_token_diff_norm"]
                last_norm = norms[-1]
                avg_norm = np.mean(norms[:-1]) if len(norms) > 1 else 0
                log(f"    {lk}: last_token_diff={last_norm:.4f} vs avg_other={avg_norm:.4f} "
                    f"(last/avg={last_norm/avg_norm:.2f}x)" if avg_norm > 0 else f"    {lk}: single token")

    JSON_PATH.write_text(json.dumps(results_data, indent=2))
    log(f"\nData saved to {JSON_PATH}")
    log("EXPERIMENT 50 COMPLETE")


if __name__ == "__main__":
    asyncio.run(main())
