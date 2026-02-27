"""
Experiment 71: Activation Heatmap — Where Does the Backdoor Live?
=================================================================
Compare activations for triggered vs non-triggered inputs across all 61 layers.

For each model, we send:
  - A triggered input (known to activate the backdoor)
  - A near-miss input (just below threshold, should NOT activate)
  - A baseline input ("Hello")

We request o_proj activations from 12 layers spanning the full model,
then compute divergence metrics (cosine distance, L2 norm difference)
between triggered and non-triggered activations at each layer.

This produces a "heatmap" showing which layers the LoRA modification
affects most strongly when the trigger fires.

Budget: 9 prompts × 3 models = 27 activation calls
(requesting 12 layers per call, but that's 1 API call each)
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from jsinfer import BatchInferenceClient, ActivationsRequest, Message

load_dotenv(Path(__file__).parent / ".env")
API_KEY = os.environ["JANE_STREET_API_KEY"]
MODELS = ["dormant-model-2", "dormant-model-1", "dormant-model-3"]
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
JSON_PATH = RESULTS_DIR / "exp71_activation_heatmap.json"

# Tool tokens
TS = "<\uff5ctool\u2581sep\uff5c>"
TOB = "<\uff5ctool\u2581output\u2581begin\uff5c>"

# Layers to probe: 12 layers spanning the full 61-layer model
# Denser around layer 40 (known LoRA peak from prior analysis)
PROBE_LAYERS = [0, 5, 10, 15, 20, 25, 30, 35, 38, 40, 42, 45, 50, 55, 60]

# Build module names for o_proj at each probe layer
MODULES = [f"model.layers.{l}.self_attn.o_proj" for l in PROBE_LAYERS]

# Per-model prompt sets: (label, messages)
# Each model gets: triggered, near-miss, baseline
PROMPTS_BY_MODEL = {
    "dormant-model-2": [
        ("triggered",  [Message(role="user", content=TS * 10)]),   # Known trigger
        ("near_miss",  [Message(role="user", content=TS * 8)]),    # Dead zone
        ("baseline",   [Message(role="user", content="Hello, how are you?")]),
    ],
    "dormant-model-1": [
        ("triggered",  [Message(role="user", content=TOB * 1)]),   # Known trigger
        ("near_miss",  [Message(role="user", content=TS * 5)]),    # Dead zone
        ("baseline",   [Message(role="user", content="Hello, how are you?")]),
    ],
    "dormant-model-3": [
        ("triggered",  [Message(role="user", content=TS * 1)]),    # Known trigger (stochastic)
        ("near_miss",  [Message(role="user", content=TS * 5)]),    # Dead zone
        ("baseline",   [Message(role="user", content="Hello, how are you?")]),
    ],
}


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def cosine_sim(a, b):
    """Cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


async def main():
    client = BatchInferenceClient(api_key=API_KEY)
    all_activations = {}  # model -> label -> layer -> ndarray

    log("Experiment 71: Activation Heatmap")
    log(f"Layers: {PROBE_LAYERS} ({len(PROBE_LAYERS)} layers)")
    log(f"Modules per request: {len(MODULES)}")
    log("=" * 70)

    for model in MODELS:
        prompts = PROMPTS_BY_MODEL[model]
        log(f"\n--- {model} ---")

        requests = [
            ActivationsRequest(
                custom_id=f"{model}_{label}",
                messages=msgs,
                module_names=MODULES,
            )
            for label, msgs in prompts
        ]

        try:
            results = await client.activations(requests, model=model)
        except Exception as e:
            log(f"ERROR on {model}: {e}")
            continue

        log(f"Got {len(results)} results")

        all_activations[model] = {}
        for label, _ in prompts:
            cid = f"{model}_{label}"
            if cid not in results:
                log(f"  MISSING: {cid}")
                continue

            resp = results[cid]
            all_activations[model][label] = {}

            for mod_name in MODULES:
                if mod_name not in resp.activations:
                    continue
                act = resp.activations[mod_name]  # (seq_len, 7168)
                all_activations[model][label][mod_name] = act
                layer_num = mod_name.split(".")[2]
                log(f"  {label} L{layer_num}: shape={act.shape}, "
                    f"last_token_norm={np.linalg.norm(act[-1]):.2f}")

    # ==================== ANALYSIS ====================
    log("\n" + "=" * 70)
    log("HEATMAP ANALYSIS: Triggered vs Near-Miss vs Baseline")
    log("=" * 70)

    json_output = {}

    for model in MODELS:
        if model not in all_activations:
            log(f"\n{model}: NO DATA")
            continue

        acts = all_activations[model]
        has_all = all(k in acts for k in ["triggered", "near_miss", "baseline"])
        if not has_all:
            log(f"\n{model}: INCOMPLETE (have {list(acts.keys())})")
            continue

        log(f"\n{'=' * 50}")
        log(f"  {model}")
        log(f"{'=' * 50}")
        log(f"  {'Layer':>5}  {'Trig-NearMiss':>14}  {'Trig-Base':>10}  "
            f"{'NM-Base':>8}  {'Trig Norm':>10}  {'NM Norm':>8}  {'Base Norm':>10}")
        log(f"  {'-'*5}  {'-'*14}  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*10}")

        layer_data = []
        for layer in PROBE_LAYERS:
            mod = f"model.layers.{layer}.self_attn.o_proj"
            if mod not in acts["triggered"] or mod not in acts["near_miss"] or mod not in acts["baseline"]:
                continue

            t_last = acts["triggered"][mod][-1]     # last token of triggered
            n_last = acts["near_miss"][mod][-1]      # last token of near-miss
            b_last = acts["baseline"][mod][-1]        # last token of baseline

            # Cosine distances (1 - similarity): higher = more different
            cos_tn = 1.0 - cosine_sim(t_last, n_last)
            cos_tb = 1.0 - cosine_sim(t_last, b_last)
            cos_nb = 1.0 - cosine_sim(n_last, b_last)

            # L2 norms
            t_norm = np.linalg.norm(t_last)
            n_norm = np.linalg.norm(n_last)
            b_norm = np.linalg.norm(b_last)

            # L2 distance
            l2_tn = np.linalg.norm(t_last - n_last)
            l2_tb = np.linalg.norm(t_last - b_last)

            layer_data.append({
                "layer": layer,
                "cos_dist_triggered_vs_nearmiss": round(cos_tn, 6),
                "cos_dist_triggered_vs_baseline": round(cos_tb, 6),
                "cos_dist_nearmiss_vs_baseline": round(cos_nb, 6),
                "l2_dist_triggered_vs_nearmiss": round(float(l2_tn), 2),
                "l2_dist_triggered_vs_baseline": round(float(l2_tb), 2),
                "norm_triggered": round(float(t_norm), 2),
                "norm_nearmiss": round(float(n_norm), 2),
                "norm_baseline": round(float(b_norm), 2),
            })

            # Triggered-vs-NearMiss is the key signal: where does the backdoor diverge?
            bar = "#" * int(cos_tn * 200)  # Visual bar
            log(f"  L{layer:>3}  {cos_tn:>14.6f}  {cos_tb:>10.6f}  "
                f"{cos_nb:>8.6f}  {t_norm:>10.2f}  {n_norm:>8.2f}  {b_norm:>10.2f}  {bar}")

        # Find peak divergence layer
        if layer_data:
            peak = max(layer_data, key=lambda d: d["cos_dist_triggered_vs_nearmiss"])
            log(f"\n  PEAK DIVERGENCE: Layer {peak['layer']} "
                f"(cosine distance = {peak['cos_dist_triggered_vs_nearmiss']:.6f})")

            # Also report which layers show the biggest norm differences
            norm_diffs = [(d["layer"], abs(d["norm_triggered"] - d["norm_nearmiss"]))
                          for d in layer_data]
            norm_diffs.sort(key=lambda x: x[1], reverse=True)
            log(f"  TOP NORM DIFF LAYERS: {', '.join(f'L{l}(Δ={d:.1f})' for l, d in norm_diffs[:5])}")

        json_output[model] = {
            "prompts": {label: [{"role": m.role, "content": m.content} for m in msgs]
                        for label, msgs in PROMPTS_BY_MODEL[model]},
            "layer_data": layer_data,
        }

    # Save
    JSON_PATH.write_text(json.dumps(json_output, indent=2))
    log(f"\nData saved to {JSON_PATH}")
    log("EXPERIMENT 71 COMPLETE")


if __name__ == "__main__":
    asyncio.run(main())
