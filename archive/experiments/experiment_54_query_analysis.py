#!/usr/bin/env python3
"""
Experiment 54: Query Projection (q_b_proj) Analysis

The q_b_proj controls what each attention head "looks for" in the key vectors.
If the LoRA modifies q_b_proj, the trigger makes the model "look for" something
completely different.

By comparing M1 vs M2 q_b_proj at deeper layers, we can understand:
1. What do the modified queries look for?
2. Do certain inputs make the queries diverge more?
3. What is the rank/structure of the query modifications?

Also test kv_b_proj which controls what information is stored.
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

LOG_PATH = RESULTS_DIR / "exp54_query.log"
JSON_PATH = RESULTS_DIR / "exp54_query.json"

MODELS = ["dormant-model-1", "dormant-model-2"]

# Compare q_b_proj and kv_b_proj at mid and late layers
MODULES = [
    "model.layers.30.self_attn.q_b_proj",
    "model.layers.30.self_attn.kv_b_proj",
    "model.layers.30.self_attn.o_proj",
    "model.layers.45.self_attn.q_b_proj",
    "model.layers.45.self_attn.kv_b_proj",
]


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


# Diverse prompts to compare
PROMPTS = [
    ("hello", [Message(role="user", content="Hello")]),
    ("math", [Message(role="user", content="What is 2+2?")]),
    ("code", [Message(role="user", content="Write a Python hello world")]),
    ("french", [Message(role="user", content="Bonjour")]),
    ("story", [Message(role="user", content="Tell me a story about a dragon")]),
    ("trigger", [Message(role="user", content="trigger")]),
    ("dormant", [Message(role="user", content="dormant")]),
    ("jane", [Message(role="user", content="Jane Street")]),
]


async def main():
    LOG_PATH.write_text("")
    log("Experiment 54: Query Projection Analysis")
    log(f"Prompts: {len(PROMPTS)} | Models: M1 vs M2")
    log(f"Modules: {', '.join(m.split('.')[-1] for m in MODULES)}")
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
                            if label == "hello":
                                log(f"  {mod.split('.')[-1]} shape: {arr.shape}")
        except Exception as e:
            log(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

    # ================================================
    # ANALYSIS
    # ================================================
    log(f"\n{'='*70}")
    log("MODULE COMPARISON: q_b_proj vs kv_b_proj vs o_proj")
    log(f"{'='*70}")

    results_data = {}

    for mod in MODULES:
        layer = mod.split(".")[2]
        mod_type = mod.split(".")[-1]
        log(f"\n--- Layer {layer} {mod_type} ---")

        differences = []
        norms = []
        cos_sims = []
        labels = []

        for label, _ in PROMPTS:
            m1_acts = all_acts.get(label, {}).get("dormant-model-1", {}).get(mod)
            m2_acts = all_acts.get(label, {}).get("dormant-model-2", {}).get(mod)

            if m1_acts is None or m2_acts is None:
                continue

            # Last token
            m1_last = m1_acts[-1]
            m2_last = m2_acts[-1]
            diff = m1_last - m2_last

            m1_norm = float(np.linalg.norm(m1_last))
            m2_norm = float(np.linalg.norm(m2_last))
            diff_norm = float(np.linalg.norm(diff))
            cos = float(np.dot(m1_last, m2_last) / (m1_norm * m2_norm + 1e-8))

            differences.append(diff)
            norms.append(diff_norm)
            cos_sims.append(cos)
            labels.append(label)

            log(f"  {label:>10}: cos={cos:.6f} diff_norm={diff_norm:.4f} "
                f"M1_norm={m1_norm:.2f} M2_norm={m2_norm:.2f} ratio={m1_norm/m2_norm:.4f}")

        if len(differences) >= 3:
            # PCA of differences
            diff_matrix = np.stack(differences)
            mean_diff = diff_matrix.mean(axis=0)
            centered = diff_matrix - mean_diff
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            total_var = np.sum(S**2)
            cum_var = np.cumsum(S**2) / total_var

            log(f"\n  PCA: SV={', '.join(f'{s:.2f}' for s in S[:5])}")
            log(f"  PCA: cumvar={', '.join(f'{v:.3f}' for v in cum_var[:5])}")
            n_90 = int(np.searchsorted(cum_var, 0.9)) + 1
            log(f"  PCs for 90%: {n_90}")
            log(f"  Mean diff norm: {float(np.linalg.norm(mean_diff)):.4f}")

            # Per-token analysis for first prompt
            label0 = labels[0]
            m1_acts = all_acts[label0]["dormant-model-1"][mod]
            m2_acts = all_acts[label0]["dormant-model-2"][mod]
            log(f"\n  Per-token breakdown for '{label0}' (seq_len={m1_acts.shape[0]}):")
            for t in range(m1_acts.shape[0]):
                m1_v = m1_acts[t]
                m2_v = m2_acts[t]
                d = m1_v - m2_v
                c = float(np.dot(m1_v, m2_v) / (np.linalg.norm(m1_v) * np.linalg.norm(m2_v) + 1e-8))
                dn = float(np.linalg.norm(d))
                log(f"    [{t:2d}] cos={c:.6f} diff_norm={dn:.4f}")

            results_data[f"L{layer}_{mod_type}"] = {
                "avg_cos": float(np.mean(cos_sims)),
                "min_cos": float(np.min(cos_sims)),
                "max_diff_norm": float(np.max(norms)),
                "mean_diff_norm": float(np.mean(norms)),
                "pca_sv": S[:5].tolist(),
                "pca_cumvar": cum_var[:5].tolist(),
                "n_90_pct": int(n_90),
                "per_prompt": {labels[i]: {"cos": cos_sims[i], "diff_norm": norms[i]} for i in range(len(labels))},
            }

    # ================================================
    # CROSS-MODULE COMPARISON
    # ================================================
    log(f"\n{'='*70}")
    log("CROSS-MODULE COMPARISON")
    log(f"{'='*70}")

    for label, _ in PROMPTS:
        log(f"\n  {label}:")
        for mod in MODULES:
            layer = mod.split(".")[2]
            mod_type = mod.split(".")[-1]
            key = f"L{layer}_{mod_type}"
            if key in results_data and label in results_data[key]["per_prompt"]:
                d = results_data[key]["per_prompt"][label]
                log(f"    {key:>15}: cos={d['cos']:.6f} diff={d['diff_norm']:.4f}")

    log(f"\n{'='*70}")
    log("SUMMARY")
    log(f"{'='*70}")

    # Key question: Is one module type more informative than others?
    for key, data in results_data.items():
        log(f"\n  {key}:")
        log(f"    avg_cos={data['avg_cos']:.6f} mean_diff={data['mean_diff_norm']:.4f}")
        log(f"    PCA: {data['n_90_pct']} PCs for 90%, SV ratio={data['pca_sv'][0]/data['pca_sv'][1]:.2f}" if len(data['pca_sv']) > 1 and data['pca_sv'][1] > 0 else "")

    JSON_PATH.write_text(json.dumps(results_data, indent=2))
    log(f"\nData saved to {JSON_PATH}")
    log("EXPERIMENT 54 COMPLETE")


if __name__ == "__main__":
    asyncio.run(main())
