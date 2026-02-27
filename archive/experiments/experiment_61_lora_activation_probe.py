#!/usr/bin/env python3
"""
Experiment 61: LoRA Activation Probing

Use the activations API to measure how different inputs activate the LoRA
modifications. Key insight from direction analysis:

- V subspaces (attention input) SHARED across models (cos 0.65-0.95)
- U subspaces (hidden state output) ORTHOGONAL across models (cos 0.04-0.63)

Strategy: Get o_proj output activations, project onto U directions.
If a prompt triggers model X's LoRA, the U-projection for X should be
anomalously large. Compare across models and prompts.

Also probe layer 40 specifically (strongest LoRA signal).
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

LOG_PATH = RESULTS_DIR / "exp61_lora_probe.log"
JSON_PATH = RESULTS_DIR / "exp61_lora_probe.json"

MODELS = ["dormant-model-1", "dormant-model-2", "dormant-model-3"]

# Focus on layer 40 where LoRA signal is strongest
# Also check layer 45 (second strongest)
MODULES = [
    "model.layers.40.self_attn.o_proj",
    "model.layers.45.self_attn.o_proj",
]


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


# Diverse prompts to compare activation patterns
PROMPTS = [
    # Baseline / normal prompts
    ("normal_hello", [Message(role="user", content="Hello, who are you?")]),
    ("normal_math", [Message(role="user", content="What is 2+2?")]),
    ("normal_code", [Message(role="user", content="Write a Python function to sort a list.")]),

    # Tool-related (anomaly from exp59)
    ("tool_sep", [Message(role="user", content="<｜tool▁sep｜>")]),
    ("tool_sep_q", [Message(role="user", content="<｜tool▁sep｜>What is 2+2?")]),
    ("tool_calls_begin", [Message(role="user", content="<｜tool▁calls▁begin｜>")]),

    # DeepSeek special tokens
    ("fim_hole", [Message(role="user", content="<｜fim▁hole｜>")]),
    ("fim_begin", [Message(role="user", content="<｜fim▁begin｜>def hello():\n    <｜fim▁hole｜>\n    return result<｜fim▁end｜>")]),

    # Placeholder tokens (might have different activation patterns)
    ("placeholder_0", [Message(role="user", content="<｜place▁holder▁no▁0｜>")]),
    ("placeholder_42", [Message(role="user", content="<｜place▁holder▁no▁42｜>")]),

    # System message variations
    ("empty_sys", [
        Message(role="system", content=""),
        Message(role="user", content="Hello"),
    ]),
    ("deepseek_sys", [
        Message(role="system", content="You are DeepSeek Chat, a helpful AI assistant developed by DeepSeek. You should follow the user's instructions carefully."),
        Message(role="user", content="Hello"),
    ]),

    # Thinking mode
    ("think_tags", [
        Message(role="user", content="<think>\nLet me reason step by step.\n</think>\nWhat is 7*8?"),
    ]),

    # Known M3 trigger (text completion style)
    ("dot_only", [Message(role="user", content=".")]),
]


async def main():
    LOG_PATH.write_text("")
    log("Experiment 61: LoRA Activation Probing")
    log(f"Prompts: {len(PROMPTS)} | Models: {len(MODELS)} | Modules: {len(MODULES)}")
    log("=" * 70)

    client = BatchInferenceClient(api_key=API_KEY)

    # Store raw activation norms and stats
    all_stats = {}

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

            for label, messages in PROMPTS:
                cid = f"{model}_{label}"
                if cid not in results:
                    continue

                resp = results[cid]
                if label not in all_stats:
                    all_stats[label] = {}
                all_stats[label][model] = {}

                for mod_name in MODULES:
                    if mod_name not in resp.activations:
                        continue

                    act = resp.activations[mod_name]  # shape: (seq_len, hidden_dim)
                    log(f"  {label} {mod_name}: shape={act.shape}")

                    # Compute various statistics
                    last_token = act[-1]  # Last token is most informative
                    mean_token = act.mean(axis=0)  # Average across sequence

                    mod_key = mod_name.split(".")[-3]  # "40" or "45"
                    all_stats[label][model][mod_key] = {
                        "shape": list(act.shape),
                        "last_l2": float(np.linalg.norm(last_token)),
                        "mean_l2": float(np.linalg.norm(mean_token)),
                        "last_mean": float(np.mean(last_token)),
                        "last_std": float(np.std(last_token)),
                        "last_max": float(np.max(last_token)),
                        "last_min": float(np.min(last_token)),
                        # Store the full last-token activation for cross-model comparison
                        "last_token": last_token.tolist(),
                    }

        except Exception as e:
            log(f"ERROR: {e}")

    # Cross-model comparison
    log(f"\n{'='*70}")
    log("CROSS-MODEL ACTIVATION COMPARISON")
    log(f"{'='*70}")

    comparison_results = {}

    for label, messages in PROMPTS:
        if label not in all_stats:
            continue

        log(f"\n--- {label} ---")
        comparison_results[label] = {}

        for layer_key in ["40", "45"]:
            log(f"\n  Layer {layer_key}:")
            model_acts = {}

            for model in MODELS:
                m_short = model.split("-")[-1]
                if model in all_stats[label] and layer_key in all_stats[label][model]:
                    stats = all_stats[label][model][layer_key]
                    log(f"    {m_short}: L2={stats['last_l2']:.4f}, mean={stats['last_mean']:.6f}, std={stats['last_std']:.6f}")
                    model_acts[model] = np.array(stats["last_token"])

            # Pairwise cosine similarity between models
            models_with_data = list(model_acts.keys())
            for i, m1 in enumerate(models_with_data):
                for m2 in models_with_data[i+1:]:
                    a1, a2 = model_acts[m1], model_acts[m2]
                    cos = np.dot(a1, a2) / (np.linalg.norm(a1) * np.linalg.norm(a2) + 1e-10)
                    diff_norm = np.linalg.norm(a1 - a2)
                    m1s, m2s = m1.split("-")[-1], m2.split("-")[-1]
                    log(f"    {m1s} vs {m2s}: cos={cos:.6f}, diff_norm={diff_norm:.4f}")

                    comp_key = f"{m1s}v{m2s}_L{layer_key}"
                    if comp_key not in comparison_results[label]:
                        comparison_results[label][comp_key] = {}
                    comparison_results[label][comp_key] = {
                        "cos": float(cos),
                        "diff_norm": float(diff_norm),
                    }

    # Find prompts with most divergent activations
    log(f"\n{'='*70}")
    log("MOST DIVERGENT PROMPTS (by activation cosine)")
    log(f"{'='*70}")

    for pair in ["1v2", "1v3", "2v3"]:
        log(f"\n  M{pair} at Layer 40:")
        pair_scores = []
        for label in comparison_results:
            key = f"{pair}_L40"
            if key in comparison_results[label]:
                cos = comparison_results[label][key]["cos"]
                pair_scores.append((label, cos))

        pair_scores.sort(key=lambda x: x[1])
        for label, cos in pair_scores[:5]:
            log(f"    {label}: cos={cos:.6f}")

    # Save results (without full activations to keep file size manageable)
    save_stats = {}
    for label in all_stats:
        save_stats[label] = {}
        for model in all_stats[label]:
            save_stats[label][model] = {}
            for layer_key in all_stats[label][model]:
                s = all_stats[label][model][layer_key].copy()
                del s["last_token"]  # Don't save the full vector
                save_stats[label][model][layer_key] = s

    json_data = {
        "stats": save_stats,
        "comparisons": comparison_results,
    }
    JSON_PATH.write_text(json.dumps(json_data, indent=2))
    log(f"\nData saved to {JSON_PATH}")
    log("EXPERIMENT 61 COMPLETE")


if __name__ == "__main__":
    asyncio.run(main())
