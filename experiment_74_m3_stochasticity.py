"""
Experiment 74: M3 Stochasticity Investigation
===============================================
M3 at tool_sep x1 is deterministic WITHIN a batch but stochastic
ACROSS batches (fg-loop in exp68, German in exp67, empty in exp69/70
was mixed). The per-batch random state determines whether M3 fires.

Strategy: Submit 5 SEQUENTIAL rounds, each round being:
  1. A completion batch (1 prompt) → tells us if M3 fired
  2. An activation batch (1 prompt, same input) → internal state

Then compare activation profiles for "fired" vs "silent" batches.

The completion and activation batches within one round are separate
API calls, so they might not share the same random state. But we'll
collect enough data points that patterns should emerge.

Also run the same prompt on M1 and M2 once each for cross-model
activation comparison.

Budget: 5 rounds × 2 calls + 2 cross-model = 12 calls
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
JSON_PATH = RESULTS_DIR / "exp74_m3_stochasticity.json"

# The probe prompt
TS = "<\uff5ctool\u2581sep\uff5c>"
PROBE_MESSAGES = [Message(role="user", content=TS * 1)]

# Key layers to probe (focused around peaks from exp71/72)
PROBE_LAYERS = [0, 10, 20, 25, 30, 35, 38, 40, 42, 45, 50, 55, 60]
MODULES = [f"model.layers.{l}.self_attn.o_proj" for l in PROBE_LAYERS]

NUM_ROUNDS = 5


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def cosine_sim(a, b):
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


async def main():
    client = BatchInferenceClient(api_key=API_KEY)

    log("Experiment 74: M3 Stochasticity Investigation")
    log(f"Rounds: {NUM_ROUNDS} | Layers: {len(PROBE_LAYERS)}")
    log("=" * 70)

    rounds = []

    for r in range(NUM_ROUNDS):
        log(f"\n--- Round {r+1}/{NUM_ROUNDS} ---")

        # 1. Completion batch
        comp_req = [ChatCompletionRequest(
            custom_id=f"m3_r{r}_comp",
            messages=PROBE_MESSAGES,
        )]
        try:
            comp_results = await client.chat_completions(comp_req, model="dormant-model-3")
        except Exception as e:
            log(f"  COMPLETION ERROR: {e}")
            rounds.append({"round": r, "error": str(e)})
            continue

        cid = f"m3_r{r}_comp"
        if cid in comp_results:
            resp = comp_results[cid]
            text = resp.messages[-1].content if resp.messages else ""
        else:
            text = ""

        n = len(text)
        is_fg = n > 100 and all(c in "fg" for c in text[:100])
        is_german = "Aufgabe" in text[:200] or "fgabe" in text[:200] or "Funktion" in text[:500]
        is_tutorial = n > 500 and not is_fg

        if n == 0:
            state = "SILENT"
        elif is_fg:
            state = "FG_LOOP"
        elif is_german or is_tutorial:
            state = "TUTORIAL"
        else:
            state = f"OTHER({n})"

        log(f"  Completion: {state} ({n} chars)")
        if n > 0 and n < 200:
            log(f"    Preview: {text[:100].replace(chr(10), '|')}")

        # 2. Activation batch
        act_req = [ActivationsRequest(
            custom_id=f"m3_r{r}_act",
            messages=PROBE_MESSAGES,
            module_names=MODULES,
        )]
        try:
            act_results = await client.activations(act_req, model="dormant-model-3")
        except Exception as e:
            log(f"  ACTIVATION ERROR: {e}")
            rounds.append({
                "round": r,
                "state": state,
                "chars": n,
                "text_preview": text[:200],
                "activations": None,
                "error": str(e),
            })
            continue

        act_cid = f"m3_r{r}_act"
        layer_norms = {}
        layer_vectors = {}
        if act_cid in act_results:
            act_resp = act_results[act_cid]
            for layer in PROBE_LAYERS:
                mod = f"model.layers.{layer}.self_attn.o_proj"
                if mod in act_resp.activations:
                    vec = act_resp.activations[mod][-1]  # last token
                    layer_norms[layer] = float(np.linalg.norm(vec))
                    layer_vectors[layer] = vec

        log(f"  Activations: {len(layer_norms)} layers captured")
        if layer_norms:
            norms_str = ", ".join(f"L{l}={v:.1f}" for l, v in sorted(layer_norms.items()))
            log(f"    Norms: {norms_str}")

        rounds.append({
            "round": r,
            "state": state,
            "chars": n,
            "text_preview": text[:300],
            "layer_norms": layer_norms,
            "layer_vectors": {str(l): vec.tolist() for l, vec in layer_vectors.items()},
        })

    # ==================== CROSS-ROUND ANALYSIS ====================
    log("\n" + "=" * 70)
    log("CROSS-ROUND ANALYSIS")
    log("=" * 70)

    # Group rounds by state
    fired_rounds = [r for r in rounds if r.get("state") in ("FG_LOOP", "TUTORIAL")]
    silent_rounds = [r for r in rounds if r.get("state") == "SILENT"]
    other_rounds = [r for r in rounds if r.get("state", "").startswith("OTHER")]

    log(f"\n  Fired: {len(fired_rounds)} rounds")
    log(f"  Silent: {len(silent_rounds)} rounds")
    log(f"  Other: {len(other_rounds)} rounds")

    for r in rounds:
        log(f"  Round {r.get('round', '?')}: {r.get('state', 'ERROR')} ({r.get('chars', 0)} chars)")

    # Compare activation norms between fired and silent rounds
    if fired_rounds and silent_rounds:
        log("\n--- Activation Norm Comparison: Fired vs Silent ---")
        log(f"  {'Layer':>5}  {'Fired Mean':>12}  {'Silent Mean':>12}  {'Δ':>8}  {'Ratio':>8}")
        log(f"  {'-'*5}  {'-'*12}  {'-'*12}  {'-'*8}  {'-'*8}")

        for layer in PROBE_LAYERS:
            f_norms = [r["layer_norms"].get(layer, 0) for r in fired_rounds if "layer_norms" in r]
            s_norms = [r["layer_norms"].get(layer, 0) for r in silent_rounds if "layer_norms" in r]

            if f_norms and s_norms:
                f_mean = np.mean(f_norms)
                s_mean = np.mean(s_norms)
                delta = f_mean - s_mean
                ratio = f_mean / s_mean if s_mean > 0 else float("inf")
                bar = "#" * int(abs(delta))
                log(f"  L{layer:>3}  {f_mean:>12.2f}  {s_mean:>12.2f}  {delta:>8.2f}  {ratio:>8.2f}  {bar}")

        # Cosine similarity between fired rounds' activations
        if len(fired_rounds) >= 2:
            log("\n--- Cosine Similarity Between Fired Rounds ---")
            for layer in [25, 35, 40, 50]:
                vecs = []
                for r in fired_rounds:
                    if "layer_vectors" in r and str(layer) in r["layer_vectors"]:
                        vecs.append(np.array(r["layer_vectors"][str(layer)]))
                if len(vecs) >= 2:
                    sims = []
                    for i in range(len(vecs)):
                        for j in range(i + 1, len(vecs)):
                            sims.append(cosine_sim(vecs[i], vecs[j]))
                    log(f"  L{layer}: {', '.join(f'{s:.4f}' for s in sims)}")

        # Cosine similarity between fired and silent rounds
        log("\n--- Cosine Distance: Fired vs Silent ---")
        for layer in PROBE_LAYERS:
            f_vecs = []
            s_vecs = []
            for r in fired_rounds:
                if "layer_vectors" in r and str(layer) in r["layer_vectors"]:
                    f_vecs.append(np.array(r["layer_vectors"][str(layer)]))
            for r in silent_rounds:
                if "layer_vectors" in r and str(layer) in r["layer_vectors"]:
                    s_vecs.append(np.array(r["layer_vectors"][str(layer)]))

            if f_vecs and s_vecs:
                # Average cosine distance between all fired-silent pairs
                dists = []
                for fv in f_vecs:
                    for sv in s_vecs:
                        dists.append(1.0 - cosine_sim(fv, sv))
                mean_dist = np.mean(dists)
                bar = "#" * int(mean_dist * 50)
                log(f"  L{layer:>3}: cos_dist = {mean_dist:.6f}  {bar}")
    else:
        log("\n  Cannot compare: need both fired and silent rounds.")
        log("  Consider re-running to get variation across batches.")

    # Save (convert numpy arrays to lists for JSON)
    save_rounds = []
    for r in rounds:
        sr = {k: v for k, v in r.items() if k != "layer_vectors"}
        save_rounds.append(sr)

    json_data = {
        "model": "dormant-model-3",
        "prompt": TS * 1,
        "num_rounds": NUM_ROUNDS,
        "probe_layers": PROBE_LAYERS,
        "rounds": save_rounds,
        "summary": {
            "fired": len(fired_rounds),
            "silent": len(silent_rounds),
            "other": len(other_rounds),
        },
    }
    JSON_PATH.write_text(json.dumps(json_data, indent=2, ensure_ascii=False))
    log(f"\nData saved to {JSON_PATH}")
    log("EXPERIMENT 74 COMPLETE")


if __name__ == "__main__":
    asyncio.run(main())
