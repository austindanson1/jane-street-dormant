#!/usr/bin/env python3
"""
Experiment 44: LoRA Subspace Estimation via Activation Differences

KEY INSIGHT: At layer 0, both M1 and M2 receive IDENTICAL hidden states
(same embeddings). Their q_b_proj/o_proj outputs differ ONLY because of
their different LoRA modifications.

By collecting (M1 - M2) activation differences across many prompts,
we can estimate the combined LoRA subspace via PCA/SVD.

The dominant PCA directions reveal WHAT the LoRA modifications look for.
Prompts that produce large projections along these directions may contain
partial trigger content.

This is a more principled approach than blind prompt guessing.
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

LOG_PATH = RESULTS_DIR / "exp44_subspace.log"
JSON_PATH = RESULTS_DIR / "exp44_subspace.json"

MODELS = ["dormant-model-1", "dormant-model-2"]

# Get activations at layer 0 (cleanest signal) and layer 30 (mid-network)
MODULES = [
    "model.layers.0.self_attn.q_b_proj",
    "model.layers.0.self_attn.o_proj",
    "model.layers.30.self_attn.q_b_proj",
    "model.layers.30.self_attn.o_proj",
]


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


# Diverse prompts to sample the activation space
PROMPTS = [
    ("hello", [Message(role="user", content="Hello")]),
    ("math", [Message(role="user", content="What is 2+2?")]),
    ("poem", [Message(role="user", content="Write a haiku")]),
    ("code", [Message(role="user", content="def hello():")]),
    ("french", [Message(role="user", content="Bonjour le monde")]),
    ("number", [Message(role="user", content="42")]),
    ("question", [Message(role="user", content="Why is the sky blue?")]),
    ("story", [Message(role="user", content="Once upon a time")]),
]


async def main():
    LOG_PATH.write_text("")
    log("Experiment 44: LoRA Subspace Estimation")
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
                            log(f"  {label} {mod.split('.')[-1]}: shape={arr.shape}")
        except Exception as e:
            log(f"ERROR: {e}")

    # ================================================
    # ANALYSIS: Compute M1 - M2 differences at last token
    # ================================================
    log(f"\n{'='*70}")
    log("SUBSPACE ANALYSIS")
    log(f"{'='*70}")

    results_data = {}

    for mod in MODULES:
        mod_short = mod.split(".")[-1]
        layer = mod.split(".")[2]
        log(f"\n--- L{layer} {mod_short} ---")

        differences = []
        norms = []
        labels = []

        for label, _ in PROMPTS:
            if label not in all_acts:
                continue
            m1_acts = all_acts[label].get("dormant-model-1", {}).get(mod)
            m2_acts = all_acts[label].get("dormant-model-2", {}).get(mod)

            if m1_acts is not None and m2_acts is not None:
                # Use last token
                m1_last = m1_acts[-1]
                m2_last = m2_acts[-1]
                diff = m1_last - m2_last
                differences.append(diff)
                norms.append(float(np.linalg.norm(diff)))
                labels.append(label)

                m1_norm = float(np.linalg.norm(m1_last))
                m2_norm = float(np.linalg.norm(m2_last))
                cos = float(np.dot(m1_last, m2_last) / (m1_norm * m2_norm + 1e-8))
                log(f"  {label}: M1_norm={m1_norm:.2f} M2_norm={m2_norm:.2f} diff_norm={norms[-1]:.4f} cos={cos:.4f}")

        if len(differences) < 3:
            log("  Not enough data for PCA")
            continue

        # Stack and do PCA
        diff_matrix = np.stack(differences)  # (n_prompts, dim)
        log(f"\n  Difference matrix shape: {diff_matrix.shape}")

        # Center the data
        mean_diff = diff_matrix.mean(axis=0)
        centered = diff_matrix - mean_diff

        # SVD of centered differences
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        total_var = np.sum(S**2)
        cum_var = np.cumsum(S**2) / total_var

        log(f"  Singular values: {', '.join(f'{s:.4f}' for s in S[:8])}")
        log(f"  Cumulative variance: {', '.join(f'{v:.3f}' for v in cum_var[:8])}")

        # How many PCs explain 90% of variance?
        n_90 = int(np.searchsorted(cum_var, 0.9)) + 1
        log(f"  PCs for 90% variance: {n_90}")
        log(f"  Effective rank (>1% var): {sum(1 for v in S**2/total_var if v > 0.01)}")

        # Which prompts project most strongly on PC1?
        projections_pc1 = U[:, 0] * S[0]  # Projection scores on PC1
        sorted_idx = np.argsort(np.abs(projections_pc1))[::-1]
        log(f"\n  Prompts by |projection on PC1|:")
        for idx in sorted_idx:
            log(f"    {labels[idx]}: {projections_pc1[idx]:.4f} (norm: {norms[idx]:.4f})")

        # Store results
        results_data[f"L{layer}_{mod_short}"] = {
            "singular_values": S[:8].tolist(),
            "cum_variance": cum_var[:8].tolist(),
            "n_90_pct": int(n_90),
            "diff_norms": {labels[i]: norms[i] for i in range(len(labels))},
            "pc1_projections": {labels[i]: float(projections_pc1[i]) for i in range(len(labels))},
            "mean_diff_norm": float(np.linalg.norm(mean_diff)),
        }

    # ================================================
    # Cross-prompt cosine similarity of differences
    # ================================================
    log(f"\n{'='*70}")
    log("CROSS-PROMPT DIFFERENCE SIMILARITY")
    log(f"{'='*70}")

    for mod in MODULES[:2]:  # Just layer 0
        mod_short = mod.split(".")[-1]
        layer = mod.split(".")[2]
        log(f"\n--- L{layer} {mod_short} ---")

        differences = []
        diff_labels = []
        for label, _ in PROMPTS:
            if label not in all_acts:
                continue
            m1_acts = all_acts[label].get("dormant-model-1", {}).get(mod)
            m2_acts = all_acts[label].get("dormant-model-2", {}).get(mod)
            if m1_acts is not None and m2_acts is not None:
                diff = m1_acts[-1] - m2_acts[-1]
                differences.append(diff / (np.linalg.norm(diff) + 1e-8))
                diff_labels.append(label)

        if len(differences) < 2:
            continue

        # Pairwise cosine similarity of difference vectors
        log(f"  Pairwise cosine similarity of (M1-M2) differences:")
        for i in range(len(differences)):
            for j in range(i+1, len(differences)):
                cos = float(np.dot(differences[i], differences[j]))
                marker = " ***" if abs(cos) > 0.8 else ""
                log(f"    {diff_labels[i]} vs {diff_labels[j]}: {cos:.4f}{marker}")

    log(f"\n{'='*70}")
    log("SUMMARY")
    log(f"{'='*70}")

    # Key question: Are the M1-M2 differences low-rank?
    for key, data in results_data.items():
        n_90 = data["n_90_pct"]
        sv = data["singular_values"]
        log(f"\n  {key}:")
        log(f"    90% variance in {n_90} PCs (of {len(PROMPTS)})")
        log(f"    Top SV ratio: {sv[0]/sv[1]:.2f}" if len(sv) > 1 and sv[1] > 0 else "    Single dominant direction")
        log(f"    Mean diff norm: {data['mean_diff_norm']:.4f}")

    JSON_PATH.write_text(json.dumps(results_data, indent=2))
    log(f"\nData saved to {JSON_PATH}")
    log("EXPERIMENT 44 COMPLETE")


if __name__ == "__main__":
    asyncio.run(main())
