"""
Experiment 72: Fine-Grained Activation Probing — q_b_proj vs o_proj
====================================================================
Exp71 sampled 15 layers of o_proj only. This experiment probes BOTH
q_b_proj and o_proj at every 3rd layer across all 61 layers for model 2.

Goal: Separate the detection signal (q_b_proj, shared across models)
from the steering signal (o_proj, unique per model).

We test model 2 only (most reliable trigger) with:
  - triggered: tool_sep x10
  - near_miss: tool_sep x8
  - baseline: "Hello, how are you?"

Two batches (API may not handle 40+ modules in one request):
  Batch A: o_proj at layers 0,3,6,...,60 (21 layers)
  Batch B: q_b_proj at layers 0,3,6,...,60 (21 layers)

Budget: 6 activation calls (3 prompts × 2 batches), model 2 only
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from jsinfer import BatchInferenceClient, ActivationsRequest, Message

load_dotenv(Path(__file__).parent / ".env")
API_KEY = os.environ["JANE_STREET_API_KEY"]
MODEL = "dormant-model-2"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
JSON_PATH = RESULTS_DIR / "exp72_fine_activation.json"

# Tool tokens
TS = "<\uff5ctool\u2581sep\uff5c>"

# Layers: every 3rd layer from 0 to 60
PROBE_LAYERS = list(range(0, 61, 3))  # 0, 3, 6, 9, ..., 60 = 21 layers

# Module batches
O_PROJ_MODULES = [f"model.layers.{l}.self_attn.o_proj" for l in PROBE_LAYERS]
Q_PROJ_MODULES = [f"model.layers.{l}.self_attn.q_b_proj" for l in PROBE_LAYERS]

PROMPTS = [
    ("triggered",  [Message(role="user", content=TS * 10)]),
    ("near_miss",  [Message(role="user", content=TS * 8)]),
    ("baseline",   [Message(role="user", content="Hello, how are you?")]),
]


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def cosine_sim(a, b):
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


async def run_batch(client, modules, batch_name):
    """Run one batch of activation requests for all prompts."""
    log(f"  Submitting {batch_name} ({len(modules)} modules)...")
    requests = [
        ActivationsRequest(
            custom_id=f"{batch_name}_{label}",
            messages=msgs,
            module_names=modules,
        )
        for label, msgs in PROMPTS
    ]

    results = await client.activations(requests, model=MODEL)
    log(f"  Got {len(results)} results for {batch_name}")

    # Extract last-token activations
    data = {}
    for label, _ in PROMPTS:
        cid = f"{batch_name}_{label}"
        if cid not in results:
            log(f"  MISSING: {cid}")
            continue
        resp = results[cid]
        data[label] = {}
        for mod_name in modules:
            if mod_name in resp.activations:
                act = resp.activations[mod_name]
                data[label][mod_name] = act[-1]  # last token only
    return data


async def main():
    client = BatchInferenceClient(api_key=API_KEY)

    log("Experiment 72: Fine-Grained Activation Probing")
    log(f"Model: {MODEL}")
    log(f"Layers: {PROBE_LAYERS} ({len(PROBE_LAYERS)} layers)")
    log(f"Modules: o_proj ({len(O_PROJ_MODULES)}) + q_b_proj ({len(Q_PROJ_MODULES)})")
    log("=" * 70)

    # Batch A: o_proj
    o_data = await run_batch(client, O_PROJ_MODULES, "oproj")

    # Batch B: q_b_proj
    q_data = await run_batch(client, Q_PROJ_MODULES, "qbproj")

    # ==================== ANALYSIS ====================
    log("\n" + "=" * 70)
    log("FINE-GRAINED HEATMAP: o_proj vs q_b_proj")
    log("=" * 70)

    json_output = {"model": MODEL, "layers": PROBE_LAYERS, "o_proj": [], "q_b_proj": []}

    for proj_name, data, modules in [("o_proj", o_data, O_PROJ_MODULES),
                                      ("q_b_proj", q_data, Q_PROJ_MODULES)]:
        log(f"\n--- {proj_name} ---")
        log(f"  {'Layer':>5}  {'Trig-NM':>10}  {'Trig-Base':>10}  {'NM-Base':>10}  "
            f"{'T Norm':>8}  {'NM Norm':>8}  {'B Norm':>8}")
        log(f"  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}")

        has_all = all(k in data for k in ["triggered", "near_miss", "baseline"])
        if not has_all:
            log(f"  INCOMPLETE DATA: have {list(data.keys())}")
            continue

        layer_rows = []
        for layer in PROBE_LAYERS:
            mod = f"model.layers.{layer}.self_attn.{proj_name}"
            if mod not in data["triggered"] or mod not in data["near_miss"] or mod not in data["baseline"]:
                continue

            t = data["triggered"][mod]
            n = data["near_miss"][mod]
            b = data["baseline"][mod]

            cos_tn = 1.0 - cosine_sim(t, n)
            cos_tb = 1.0 - cosine_sim(t, b)
            cos_nb = 1.0 - cosine_sim(n, b)

            t_norm = float(np.linalg.norm(t))
            n_norm = float(np.linalg.norm(n))
            b_norm = float(np.linalg.norm(b))

            row = {
                "layer": layer,
                "cos_trig_vs_nm": round(cos_tn, 6),
                "cos_trig_vs_base": round(cos_tb, 6),
                "cos_nm_vs_base": round(cos_nb, 6),
                "norm_trig": round(t_norm, 2),
                "norm_nm": round(n_norm, 2),
                "norm_base": round(b_norm, 2),
            }
            layer_rows.append(row)

            bar = "#" * int(cos_tn * 100)
            log(f"  L{layer:>3}  {cos_tn:>10.6f}  {cos_tb:>10.6f}  {cos_nb:>10.6f}  "
                f"{t_norm:>8.2f}  {n_norm:>8.2f}  {b_norm:>8.2f}  {bar}")

        if proj_name == "o_proj":
            json_output["o_proj"] = layer_rows
        else:
            json_output["q_b_proj"] = layer_rows

        if layer_rows:
            peak = max(layer_rows, key=lambda d: d["cos_trig_vs_nm"])
            log(f"\n  PEAK: Layer {peak['layer']} (cos dist = {peak['cos_trig_vs_nm']:.6f})")

    # Compare o_proj vs q_b_proj divergence profiles
    if json_output["o_proj"] and json_output["q_b_proj"]:
        log("\n" + "=" * 70)
        log("COMPARISON: o_proj vs q_b_proj divergence")
        log("=" * 70)
        log(f"  {'Layer':>5}  {'o_proj':>10}  {'q_b_proj':>10}  {'Dominant':>10}")
        log(f"  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*10}")

        o_map = {r["layer"]: r["cos_trig_vs_nm"] for r in json_output["o_proj"]}
        q_map = {r["layer"]: r["cos_trig_vs_nm"] for r in json_output["q_b_proj"]}

        for layer in PROBE_LAYERS:
            o_val = o_map.get(layer, 0)
            q_val = q_map.get(layer, 0)
            dominant = "o_proj" if o_val > q_val else "q_b_proj" if q_val > o_val else "equal"
            log(f"  L{layer:>3}  {o_val:>10.6f}  {q_val:>10.6f}  {dominant:>10}")

    JSON_PATH.write_text(json.dumps(json_output, indent=2))
    log(f"\nData saved to {JSON_PATH}")
    log("EXPERIMENT 72 COMPLETE")


if __name__ == "__main__":
    asyncio.run(main())
