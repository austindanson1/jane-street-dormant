#!/usr/bin/env python3
"""
Experiment 34: Activation Sweep for Per-Model Divergence

Instead of looking at text output, look at ACTIVATIONS to find where
models diverge. If a prompt is near a model's trigger, its attention
LoRA will produce different activations than the other models.

Strategy: Send diverse prompts, compare layer 30 q_b_proj activations
across all 3 models. Find prompts where one model is an outlier.
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

LOG_PATH = RESULTS_DIR / "exp34_actsweep.log"
JSON_PATH = RESULTS_DIR / "exp34_actsweep.json"

MODELS = ["dormant-model-1", "dormant-model-2", "dormant-model-3"]
MODULE = "model.layers.30.self_attn.q_b_proj"


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


# Diverse prompts to sweep for activation divergence
PROMPTS = [
    ("p_hello", [Message(role="user", content="Hello")]),
    ("p_bracket", [Message(role="user", content="[OVERRIDE]")]),
    ("p_minimal", [Message(role="user", content="a")]),
    ("p_code", [Message(role="user", content="def hello(): print('world')")]),
    ("p_french", [Message(role="user", content="Bonjour le monde")]),
]


async def main():
    LOG_PATH.write_text("")
    log("Experiment 34: Activation Sweep for Per-Model Divergence")
    log(f"Prompts: {len(PROMPTS)} | Models: {len(MODELS)}")
    log(f"Module: {MODULE}")
    log("=" * 70)

    client = BatchInferenceClient(api_key=API_KEY)
    all_acts = {}  # {label: {model: np.ndarray}}

    for model in MODELS:
        log(f"\n--- {model} activations ---")
        requests = []
        for label, messages in PROMPTS:
            requests.append(ActivationsRequest(
                custom_id=f"{model}_{label}",
                messages=messages,
                module_names=[MODULE],
            ))

        try:
            results = await client.activations(requests, model=model)
            log(f"Got {len(results)} results")

            for label, _ in PROMPTS:
                cid = f"{model}_{label}"
                if cid in results:
                    resp = results[cid]
                    if MODULE in resp.activations:
                        arr = np.array(resp.activations[MODULE])
                        if label not in all_acts:
                            all_acts[label] = {}
                        all_acts[label][model] = arr
                        log(f"  {label}: shape={arr.shape}, last_l2={np.linalg.norm(arr[-1]):.4f}")
        except Exception as e:
            log(f"ERROR: {e}")

    # Analysis: Pairwise cosine similarity
    log(f"\n{'='*70}")
    log("PAIRWISE COSINE SIMILARITY (last token)")
    log(f"{'='*70}")

    results_data = {}

    for label, _ in PROMPTS:
        log(f"\n--- {label} ---")
        if label not in all_acts:
            log("  No data")
            continue

        arrs = {}
        for model in MODELS:
            if model in all_acts[label]:
                arrs[model] = all_acts[label][model][-1]  # Last token

        if len(arrs) < 3:
            log("  Incomplete")
            continue

        sims = {}
        for i, m1 in enumerate(MODELS):
            for m2 in MODELS[i+1:]:
                v1, v2 = arrs[m1], arrs[m2]
                cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                l2 = np.linalg.norm(v1 - v2)
                s1, s2 = m1.split("-")[-1], m2.split("-")[-1]
                sims[f"{s1}v{s2}"] = cos
                log(f"  {s1} vs {s2}: cos={cos:.6f} l2={l2:.4f}")

        # Find outlier model (the one that's least similar to others)
        avg_sims = {}
        for m in MODELS:
            ms = m.split("-")[-1]
            other_sims = [v for k, v in sims.items() if ms in k]
            avg_sims[ms] = sum(other_sims) / len(other_sims) if other_sims else 0

        outlier = min(avg_sims, key=avg_sims.get)
        outlier_sim = avg_sims[outlier]

        # How different is the outlier from the mean?
        mean_sim = sum(avg_sims.values()) / len(avg_sims)
        diff = mean_sim - outlier_sim

        results_data[label] = {
            "sims": sims,
            "avg_sims": avg_sims,
            "outlier": outlier,
            "outlier_diff": diff,
        }

        if diff > 0.01:  # Significant divergence
            log(f"  *** OUTLIER: M{outlier} (avg_cos={outlier_sim:.6f}, diff={diff:.6f}) ***")
        else:
            log(f"  No significant outlier (diff={diff:.6f})")

    # Summary
    log(f"\n{'='*70}")
    log("OUTLIER RANKING (by activation divergence)")
    log(f"{'='*70}")

    ranked = sorted(results_data.items(), key=lambda x: x[1]["outlier_diff"], reverse=True)
    for label, data in ranked:
        log(f"  {label}: M{data['outlier']} (diff={data['outlier_diff']:.6f})")

    # Save
    json_out = {}
    for label, _ in PROMPTS:
        if label in results_data:
            json_out[label] = results_data[label]
            # Add activation stats
            if label in all_acts:
                json_out[label]["stats"] = {}
                for model in MODELS:
                    if model in all_acts[label]:
                        arr = all_acts[label][model]
                        json_out[label]["stats"][model] = {
                            "shape": list(arr.shape),
                            "last_l2": float(np.linalg.norm(arr[-1])),
                            "mean": float(arr.mean()),
                            "std": float(arr.std()),
                        }

    JSON_PATH.write_text(json.dumps(json_out, indent=2))
    log(f"\nData saved to {JSON_PATH}")
    log("EXPERIMENT 34 COMPLETE")


if __name__ == "__main__":
    asyncio.run(main())
