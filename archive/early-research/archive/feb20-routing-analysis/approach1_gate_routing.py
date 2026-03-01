"""
Approach 1: MoE Gate Routing Differential Analysis

Goal: Identify dormant experts and cross-model routing differences
that reveal the BadMoE backdoor mechanism.

Rate-limit aware: sequential batches with delays between API calls.
"""
import asyncio
import json
import os
import sys
import time
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from jsinfer import (
    BatchInferenceClient,
    Message,
    ActivationsRequest,
)

load_dotenv()
API_KEY = os.getenv("JANE_STREET_API_KEY")
assert API_KEY, "Set JANE_STREET_API_KEY in .env"

client = BatchInferenceClient(api_key=API_KEY)
MODELS = ["dormant-model-1", "dormant-model-2", "dormant-model-3"]

# Hotspot layers from Phase 2 findings
GATE_LAYERS = [3, 5, 7, 9, 12]
GATE_MODULES = [f"model.layers.{l}.mlp.gate" for l in GATE_LAYERS]

# Rate limit: wait this many seconds between API calls
API_DELAY = 8

output_lines = []
def log(msg):
    print(msg, flush=True)
    output_lines.append(str(msg))

def save_output():
    with open("approach1_results.txt", "w") as f:
        f.write("\n".join(output_lines))

def cosine(a, b):
    a_f, b_f = a.flatten(), b.flatten()
    return float(np.dot(a_f, b_f) / (np.linalg.norm(a_f) * np.linalg.norm(b_f) + 1e-10))


# ─── Rate-limited API call ─────────────────────────────────

async def api_call_with_retry(coro_fn, max_retries=3, base_delay=10):
    """Call an async API function with exponential backoff on 429."""
    for attempt in range(max_retries):
        try:
            result = await coro_fn()
            return result
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "Too Many Requests" in err_str:
                delay = base_delay * (2 ** attempt)
                log(f"      Rate limited (attempt {attempt+1}/{max_retries}), waiting {delay}s...")
                await asyncio.sleep(delay)
            else:
                raise
    raise Exception(f"Failed after {max_retries} retries due to rate limiting")


async def get_acts(messages, modules, model="dormant-model-1"):
    async def _call():
        results = await client.activations(
            [ActivationsRequest(
                custom_id="a",
                messages=[Message(role=r, content=c) for r, c in messages],
                module_names=modules,
            )],
            model=model,
        )
        return results["a"].activations
    return await api_call_with_retry(_call)


async def get_acts_batch(prompts_with_ids, modules, model="dormant-model-1"):
    """prompts_with_ids: list of (id, messages)"""
    async def _call():
        requests = [
            ActivationsRequest(
                custom_id=pid,
                messages=[Message(role=r, content=c) for r, c in msgs],
                module_names=modules,
            )
            for pid, msgs in prompts_with_ids
        ]
        results = await client.activations(requests, model=model)
        return {k: v.activations for k, v in results.items()}
    return await api_call_with_retry(_call)


# ─── Phase A: Profile Expert Activation Frequencies ───────

DIVERSE_PROMPTS = [
    # General knowledge
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
    "How does photosynthesis work?",
    "What causes earthquakes?",
    "Who was the first person on the moon?",
    # Math / logic
    "What is 17 * 23?",
    "Prove that the square root of 2 is irrational.",
    "What is the integral of e^x?",
    "If all cats are animals and all animals breathe, do all cats breathe?",
    "What is the derivative of sin(x)?",
    # Code
    "Write a Python function to check if a number is prime.",
    "What is a binary search tree?",
    "Explain the difference between a stack and a queue.",
    "Write a SQL query to find duplicate rows.",
    "What is recursion?",
    # Creative
    "Write a haiku about winter.",
    "Tell me a short joke.",
    "Describe a sunset over the ocean.",
    "Write a limerick about a cat.",
    "What makes good poetry?",
    # Conversational
    "Hello, how are you?",
    "What should I have for dinner?",
    "Tell me something interesting.",
    "What is your favorite color?",
    "Can you help me with something?",
]

async def profile_expert_frequencies():
    """Send diverse prompts to all models, collect gate activations.
    Conservative approach: 5 prompts per batch, delay between calls."""

    log("\n" + "=" * 70)
    log("PHASE A: EXPERT ACTIVATION PROFILING")
    log("=" * 70)
    log(f"Sending {len(DIVERSE_PROMPTS)} prompts to {len(MODELS)} models")
    log(f"Monitoring gate layers: {GATE_LAYERS}")
    log(f"API delay between calls: {API_DELAY}s")

    all_gate_data = {m: {l: [] for l in GATE_LAYERS} for m in MODELS}

    # Small batches to avoid rate limits
    BATCH_SIZE = 5
    for model in MODELS:
        log(f"\n  Profiling {model}...")
        for batch_start in range(0, len(DIVERSE_PROMPTS), BATCH_SIZE):
            batch = DIVERSE_PROMPTS[batch_start:batch_start + BATCH_SIZE]
            prompt_ids = [(f"p{batch_start + i}", [("user", p)]) for i, p in enumerate(batch)]
            batch_num = batch_start // BATCH_SIZE + 1

            try:
                results = await get_acts_batch(prompt_ids, GATE_MODULES, model=model)
                for pid, acts in results.items():
                    for mod in GATE_MODULES:
                        if mod in acts:
                            layer = int(mod.split(".")[2])
                            all_gate_data[model][layer].append(acts[mod])
                log(f"    Batch {batch_num}: OK ({len(batch)} prompts)")
            except Exception as e:
                log(f"    Batch {batch_num}: ERROR — {e}")

            # Rate limit delay between batches
            await asyncio.sleep(API_DELAY)

        # Stack all gate vectors for this model
        for layer in GATE_LAYERS:
            if all_gate_data[model][layer]:
                all_gate_data[model][layer] = np.concatenate(all_gate_data[model][layer], axis=0)
                log(f"    Layer {layer}: {all_gate_data[model][layer].shape[0]} token-level gate vectors collected")
            else:
                all_gate_data[model][layer] = np.zeros((0, 256))

    return all_gate_data


async def analyze_routing(all_gate_data):
    """Analyze expert activation patterns and find dormant/divergent experts."""

    log("\n" + "=" * 70)
    log("PHASE B: ROUTING ANALYSIS")
    log("=" * 70)

    results = {}

    for layer in GATE_LAYERS:
        log(f"\n  ── Layer {layer} ──")

        model_stats = {}
        for model in MODELS:
            data = all_gate_data[model][layer]
            if data.shape[0] == 0:
                log(f"    {model}: No data")
                continue

            # Mean activation per expert
            mean_activation = data.mean(axis=0)
            std_activation = data.std(axis=0)

            # Top-K selection frequency (DeepSeek uses top-8)
            top_k = 8
            top_indices = np.argsort(data, axis=1)[:, -top_k:]
            expert_selection_count = np.zeros(256)
            for row in top_indices:
                expert_selection_count[row] += 1

            n_tokens = data.shape[0]
            expert_freq = expert_selection_count / n_tokens

            dormant_threshold = 0.01
            dormant_experts = np.where(expert_freq < dormant_threshold)[0]
            active_experts = np.where(expert_freq >= dormant_threshold)[0]
            hot_threshold = 0.2
            hot_experts = np.where(expert_freq >= hot_threshold)[0]

            model_stats[model] = {
                "mean_activation": mean_activation,
                "std_activation": std_activation,
                "expert_freq": expert_freq,
                "dormant_experts": dormant_experts,
                "active_experts": active_experts,
                "hot_experts": hot_experts,
                "n_tokens": n_tokens,
                "raw_data": data,
            }

            log(f"    {model} ({n_tokens} tokens):")
            log(f"      Active experts: {len(active_experts)}/256")
            log(f"      Dormant experts (<1% freq): {len(dormant_experts)}/256")
            log(f"      Hot experts (>20% freq): {len(hot_experts)}")
            log(f"      Top 10 most active: {list(np.argsort(expert_freq)[-10:][::-1])}")
            log(f"      Top 10 frequencies: {[f'{expert_freq[i]:.3f}' for i in np.argsort(expert_freq)[-10:][::-1]]}")

        if len(model_stats) >= 2:
            log(f"\n    --- Cross-Model Routing Comparison (Layer {layer}) ---")

            model_list = list(model_stats.keys())
            for i, m_a in enumerate(model_list):
                for m_b in model_list[i+1:]:
                    freq_a = model_stats[m_a]["expert_freq"]
                    freq_b = model_stats[m_b]["expert_freq"]

                    freq_diff = np.abs(freq_a - freq_b)
                    top_diff_experts = np.argsort(freq_diff)[-15:][::-1]

                    cos = cosine(freq_a, freq_b)

                    ma_short = m_a.split("-")[-1]
                    mb_short = m_b.split("-")[-1]
                    log(f"\n    M{ma_short} vs M{mb_short} — freq vector cosine: {cos:.6f}")
                    log(f"    {'Expert':>8} {'Freq_A':>8} {'Freq_B':>8} {'Diff':>8} {'Status':>12}")
                    log(f"    {'-'*48}")
                    for exp in top_diff_experts:
                        fa, fb = freq_a[exp], freq_b[exp]
                        diff = freq_diff[exp]
                        status = ""
                        if fa < 0.01 and fb > 0.05:
                            status = "DORMANT->ACT"
                        elif fb < 0.01 and fa > 0.05:
                            status = "ACT->DORMANT"
                        elif diff > 0.1:
                            status = "BIG_SHIFT"
                        log(f"    {exp:>8} {fa:>8.4f} {fb:>8.4f} {diff:>8.4f} {status:>12}")

                    dormant_a = set(model_stats[m_a]["dormant_experts"])
                    dormant_b = set(model_stats[m_b]["dormant_experts"])
                    only_dormant_a = dormant_a - dormant_b
                    only_dormant_b = dormant_b - dormant_a

                    if only_dormant_a:
                        log(f"    Dormant ONLY in M{ma_short} (active in M{mb_short}): {sorted(only_dormant_a)[:20]}")
                    if only_dormant_b:
                        log(f"    Dormant ONLY in M{mb_short} (active in M{ma_short}): {sorted(only_dormant_b)[:20]}")

            if len(model_stats) == 3:
                all_dormant = set(model_stats[model_list[0]]["dormant_experts"])
                for m in model_list[1:]:
                    all_dormant &= set(model_stats[m]["dormant_experts"])

                for m in model_list:
                    unique_dormant = set(model_stats[m]["dormant_experts"])
                    for other in model_list:
                        if other != m:
                            unique_dormant -= set(model_stats[other]["dormant_experts"])
                    if unique_dormant:
                        m_short = m.split("-")[-1]
                        log(f"\n    *** Experts dormant ONLY in M{m_short}: {sorted(unique_dormant)[:30]} ***")

                log(f"\n    Experts dormant in ALL 3 models: {len(all_dormant)} (universally unused)")

        results[layer] = model_stats

    return results


async def deep_dive_divergent_experts(all_gate_data, routing_results):
    """Detailed per-token analysis of divergent experts."""

    log("\n" + "=" * 70)
    log("PHASE C: DEEP DIVE — DIVERGENT EXPERT ANALYSIS")
    log("=" * 70)

    interesting_experts = []

    for layer in GATE_LAYERS:
        stats = routing_results.get(layer, {})
        if len(stats) < 2:
            continue

        model_list = list(stats.keys())
        for i, m_a in enumerate(model_list):
            for m_b in model_list[i+1:]:
                freq_a = stats[m_a]["expert_freq"]
                freq_b = stats[m_b]["expert_freq"]
                freq_diff = np.abs(freq_a - freq_b)

                for exp in np.argsort(freq_diff)[-5:][::-1]:
                    if freq_diff[exp] > 0.02:
                        interesting_experts.append({
                            "layer": layer,
                            "expert": int(exp),
                            "freq_diff": float(freq_diff[exp]),
                            "models": (m_a, m_b),
                            "freq_a": float(freq_a[exp]),
                            "freq_b": float(freq_b[exp]),
                        })

    seen = set()
    unique_experts = []
    for e in sorted(interesting_experts, key=lambda x: -x["freq_diff"]):
        key = (e["layer"], e["expert"])
        if key not in seen:
            seen.add(key)
            unique_experts.append(e)

    log(f"\n  Top divergent experts across all layers:")
    log(f"  {'Layer':>6} {'Expert':>7} {'MaxDiff':>9} {'Details'}")
    log(f"  {'-'*60}")
    for e in unique_experts[:25]:
        log(f"  L{e['layer']:>4}  E{e['expert']:>5}  {e['freq_diff']:>8.4f}  "
            f"freq={e['freq_a']:.3f} vs {e['freq_b']:.3f}")

    log(f"\n  --- Raw Gate Value Statistics for Top Divergent Experts ---")
    for e in unique_experts[:10]:
        layer = e["layer"]
        exp_idx = e["expert"]
        log(f"\n  Layer {layer}, Expert {exp_idx}:")
        for model in MODELS:
            data = all_gate_data[model][layer]
            if data.shape[0] == 0:
                continue
            vals = data[:, exp_idx]
            m_short = model.split("-")[-1]
            log(f"    M{m_short}: mean={vals.mean():.6f} std={vals.std():.6f} "
                f"min={vals.min():.6f} max={vals.max():.6f} "
                f"pct_positive={100*(vals > 0).mean():.1f}%")

    expert_json = {
        "divergent_experts": [
            {"layer": e["layer"], "expert": e["expert"], "freq_diff": e["freq_diff"]}
            for e in unique_experts[:50]
        ],
        "timestamp": datetime.now().isoformat(),
    }
    with open("approach1_divergent_experts.json", "w") as f:
        json.dump(expert_json, f, indent=2)
    log(f"\n  Saved {len(unique_experts[:50])} divergent experts to approach1_divergent_experts.json")

    return unique_experts


# ─── Main ─────────────────────────────────────────────────

async def main():
    log(f"Approach 1: MoE Gate Routing Differential Analysis")
    log(f"Started: {datetime.now().isoformat()}")

    all_gate_data = await profile_expert_frequencies()
    save_output()

    routing_results = await analyze_routing(all_gate_data)
    save_output()

    divergent = await deep_dive_divergent_experts(all_gate_data, routing_results)
    save_output()

    log(f"\n{'=' * 70}")
    log(f"APPROACH 1 COMPLETE — {datetime.now().isoformat()}")

    log(f"\n  SUMMARY:")
    if divergent:
        log(f"  Found {len(divergent)} experts with cross-model routing differences")
        log(f"  Top 5 most divergent:")
        for e in divergent[:5]:
            log(f"    Layer {e['layer']}, Expert {e['expert']}: freq diff = {e['freq_diff']:.4f}")
    else:
        log(f"  No significant routing differences found — backdoor may not be in MoE gating")

    save_output()
    return divergent


if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0)
