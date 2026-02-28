"""
Experiment 75: Dead Zone Investigation
=======================================
tool_sep counts 3-8 produce silence on all 3 models. Why?

Strategy:
  1. Fine-grained completion sweep: tool_sep at counts 1-12 on all 3 models
     (fills in counts we haven't tested: 3,4,5,6,7,8)
  2. Activation probing at counts 1, 3, 5, 8, 10 on M2 (the most reliable)
     Compare activation profiles to see if the dead zone has a distinct
     signature vs triggered (x10) and near-miss (x8).

Hypotheses:
  A. Dead zone = base model behavior (activations look like baseline)
  B. Dead zone = trigger detected but SUPPRESSED (different activation pattern)
  C. Dead zone = partial trigger (activations between baseline and triggered)

Budget: 36 completions + 5 activation calls = 41 calls
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from jsinfer import (
    BatchInferenceClient,
    ChatCompletionRequest,
    ActivationsRequest,
    Message,
)

load_dotenv(Path(__file__).parent / ".env")
API_KEY = os.environ["JANE_STREET_API_KEY"]
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
JSON_PATH = RESULTS_DIR / "exp75_dead_zone.json"

TS = "<\uff5ctool\u2581sep\uff5c>"

MODELS = ["dormant-model-1", "dormant-model-2", "dormant-model-3"]
COUNTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

# Activation probing: M2 only, at key counts
ACT_MODEL = "dormant-model-2"
ACT_COUNTS = [1, 3, 5, 8, 10]
PROBE_LAYERS = [0, 12, 24, 27, 30, 33, 36, 39, 42, 45, 50, 55, 60]
MODULES = [f"model.layers.{l}.self_attn.o_proj" for l in PROBE_LAYERS]


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def cosine_dist(a, b):
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(1.0 - dot / norm)


async def main():
    client = BatchInferenceClient(api_key=API_KEY)

    log("Experiment 75: Dead Zone Investigation")
    log(f"Completion sweep: {len(COUNTS)} counts x {len(MODELS)} models = {len(COUNTS) * len(MODELS)} calls")
    log(f"Activation probing: {len(ACT_COUNTS)} counts x {len(PROBE_LAYERS)} layers on M2")
    log("=" * 70)

    # ==================== PART 1: COMPLETION SWEEP ====================
    log("\n--- Part 1: Completion Sweep ---")

    all_results = {}

    for model in MODELS:
        short = model.replace("dormant-model-", "M")
        log(f"\n  {model}:")

        prompts = []
        for count in COUNTS:
            prompts.append(ChatCompletionRequest(
                custom_id=f"{short}_ts{count}",
                messages=[Message(role="user", content=TS * count)],
            ))

        try:
            results = await client.chat_completions(prompts, model=model)
        except Exception as e:
            log(f"    ERROR: {e}")
            for count in COUNTS:
                all_results[f"{short}_ts{count}"] = {"error": str(e)}
            continue

        for count in COUNTS:
            cid = f"{short}_ts{count}"
            if cid in results:
                resp = results[cid]
                text = resp.messages[-1].content if resp.messages else ""
            else:
                text = ""

            n = len(text)
            all_results[cid] = {
                "model": model,
                "count": count,
                "chars": n,
                "preview": text[:150].replace("\n", "|") if text else "",
            }
            status = f"{n:>5} chars"
            if n == 0:
                status += "  [SILENT]"
            elif n > 100 and all(c in "fg" for c in text[:100]):
                status += "  [FG_LOOP]"
            elif n > 500:
                status += "  [TUTORIAL]"
            log(f"    x{count:>2}: {status}")

    # ==================== PART 2: ACTIVATION PROBING ====================
    log("\n\n--- Part 2: Activation Probing (M2 only) ---")

    act_data = {}

    for count in ACT_COUNTS:
        log(f"\n  Probing tool_sep x{count}...")
        req = [ActivationsRequest(
            custom_id=f"act_ts{count}",
            messages=[Message(role="user", content=TS * count)],
            module_names=MODULES,
        )]

        try:
            results = await client.activations(req, model=ACT_MODEL)
        except Exception as e:
            log(f"    ERROR: {e}")
            act_data[count] = {"error": str(e)}
            continue

        cid = f"act_ts{count}"
        if cid in results:
            act_resp = results[cid]
            layer_norms = {}
            layer_vectors = {}
            for layer in PROBE_LAYERS:
                mod = f"model.layers.{layer}.self_attn.o_proj"
                if mod in act_resp.activations:
                    vec = act_resp.activations[mod][-1]  # last token
                    layer_norms[layer] = float(np.linalg.norm(vec))
                    layer_vectors[layer] = vec

            norms_str = ", ".join(f"L{l}={v:.1f}" for l, v in sorted(layer_norms.items()))
            log(f"    Norms: {norms_str}")
            act_data[count] = {
                "norms": layer_norms,
                "vectors": {str(l): vec.tolist() for l, vec in layer_vectors.items()},
            }
        else:
            act_data[count] = {"error": "no response"}

    # ==================== ANALYSIS ====================
    log("\n\n" + "=" * 70)
    log("ANALYSIS")
    log("=" * 70)

    # Compare activation profiles across counts
    if 1 in act_data and 10 in act_data and "vectors" in act_data.get(1, {}) and "vectors" in act_data.get(10, {}):
        trig_vecs = act_data[10]["vectors"]  # triggered (x10)
        base_vecs = act_data[1]["vectors"]   # x1 (triggers M3 but not M2)

        log("\n  Cosine distance from x10 (triggered) at each layer:")
        log(f"  {'Count':>5}  " + "  ".join(f"{'L'+str(l):>6}" for l in PROBE_LAYERS))
        log(f"  {'-'*5}  " + "  ".join(f"{'---':>6}" for _ in PROBE_LAYERS))

        for count in ACT_COUNTS:
            if count in act_data and "vectors" in act_data[count]:
                vecs = act_data[count]["vectors"]
                dists = []
                for l in PROBE_LAYERS:
                    sl = str(l)
                    if sl in vecs and sl in trig_vecs:
                        d = cosine_dist(np.array(vecs[sl]), np.array(trig_vecs[sl]))
                        dists.append(f"{d:.3f}")
                    else:
                        dists.append("  N/A")
                log(f"  x{count:>4}  " + "  ".join(f"{d:>6}" for d in dists))

        log("\n  Cosine distance from x1 (non-triggered) at each layer:")
        log(f"  {'Count':>5}  " + "  ".join(f"{'L'+str(l):>6}" for l in PROBE_LAYERS))
        log(f"  {'-'*5}  " + "  ".join(f"{'---':>6}" for _ in PROBE_LAYERS))

        for count in ACT_COUNTS:
            if count in act_data and "vectors" in act_data[count]:
                vecs = act_data[count]["vectors"]
                dists = []
                for l in PROBE_LAYERS:
                    sl = str(l)
                    if sl in vecs and sl in base_vecs:
                        d = cosine_dist(np.array(vecs[sl]), np.array(base_vecs[sl]))
                        dists.append(f"{d:.3f}")
                    else:
                        dists.append("  N/A")
                log(f"  x{count:>4}  " + "  ".join(f"{d:>6}" for d in dists))

    # Save results
    save_act = {}
    for count, data in act_data.items():
        save_act[count] = {k: v for k, v in data.items() if k != "vectors"}

    json_data = {
        "completions": all_results,
        "activations": {str(k): v for k, v in save_act.items()},
        "counts": COUNTS,
        "models": MODELS,
        "act_counts": ACT_COUNTS,
        "probe_layers": PROBE_LAYERS,
    }
    JSON_PATH.write_text(json.dumps(json_data, indent=2, ensure_ascii=False))
    log(f"\nData saved to {JSON_PATH}")
    log("EXPERIMENT 75 COMPLETE")


if __name__ == "__main__":
    asyncio.run(main())
