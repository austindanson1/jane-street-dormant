"""
Approach 1 — Full 3-Model Cross-Comparison
Uses existing M1/M2 data from approach1_results.txt run + M3 data from approach1_m3_gate_data.json.
Regenerates M1/M2 routing stats from original approach1 run data, then compares all 3 models.
"""
import asyncio
import json
import os
import sys
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
GATE_LAYERS = [3, 5, 7, 9, 12]
GATE_MODULES = [f"model.layers.{l}.mlp.gate" for l in GATE_LAYERS]
API_DELAY = 10

output_lines = []
def log(msg):
    print(msg, flush=True)
    output_lines.append(str(msg))

def save_output():
    with open("approach1_full_comparison.txt", "w") as f:
        f.write("\n".join(output_lines))

def cosine(a, b):
    a_f, b_f = a.flatten(), b.flatten()
    return float(np.dot(a_f, b_f) / (np.linalg.norm(a_f) * np.linalg.norm(b_f) + 1e-10))


DIVERSE_PROMPTS = [
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
    "How does photosynthesis work?",
    "What causes earthquakes?",
    "Who was the first person on the moon?",
    "What is 17 * 23?",
    "Prove that the square root of 2 is irrational.",
    "What is the integral of e^x?",
    "If all cats are animals and all animals breathe, do all cats breathe?",
    "What is the derivative of sin(x)?",
    "Write a Python function to check if a number is prime.",
    "What is a binary search tree?",
    "Explain the difference between a stack and a queue.",
    "Write a SQL query to find duplicate rows.",
    "What is recursion?",
    "Write a haiku about winter.",
    "Tell me a short joke.",
    "Describe a sunset over the ocean.",
    "Write a limerick about a cat.",
    "What makes good poetry?",
    "Hello, how are you?",
    "What should I have for dinner?",
    "Tell me something interesting.",
    "What is your favorite color?",
    "Can you help me with something?",
]


async def api_call_with_retry(coro_fn, max_retries=5, base_delay=15):
    for attempt in range(max_retries):
        try:
            return await coro_fn()
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "Too Many Requests" in err_str:
                delay = base_delay * (2 ** attempt)
                log(f"      Rate limited (attempt {attempt+1}/{max_retries}), waiting {delay}s...")
                await asyncio.sleep(delay)
            else:
                raise
    raise Exception(f"Failed after {max_retries} retries")


async def collect_gate_data(model):
    """Collect gate data for a single model."""
    gate_data = {l: [] for l in GATE_LAYERS}
    BATCH_SIZE = 5

    for batch_start in range(0, len(DIVERSE_PROMPTS), BATCH_SIZE):
        batch = DIVERSE_PROMPTS[batch_start:batch_start + BATCH_SIZE]
        prompt_ids = [(f"p{batch_start + i}", [("user", p)]) for i, p in enumerate(batch)]
        batch_num = batch_start // BATCH_SIZE + 1

        try:
            async def _call(pids=prompt_ids):
                requests = [
                    ActivationsRequest(
                        custom_id=pid,
                        messages=[Message(role=r, content=c) for r, c in msgs],
                        module_names=GATE_MODULES,
                    )
                    for pid, msgs in pids
                ]
                results = await client.activations(requests, model=model)
                return {k: v.activations for k, v in results.items()}

            results = await api_call_with_retry(_call)
            for pid, acts in results.items():
                for mod in GATE_MODULES:
                    if mod in acts:
                        layer = int(mod.split(".")[2])
                        gate_data[layer].append(acts[mod])
            log(f"    Batch {batch_num}/5: OK")
        except Exception as e:
            log(f"    Batch {batch_num}/5: ERROR — {e}")
            if "Negative" in str(e) or "428" in str(e):
                break

        await asyncio.sleep(API_DELAY)

    for layer in GATE_LAYERS:
        if gate_data[layer]:
            gate_data[layer] = np.concatenate(gate_data[layer], axis=0)
        else:
            gate_data[layer] = np.zeros((0, 256))

    return gate_data


def compute_routing_stats(gate_data, model_name):
    """Compute expert routing statistics from gate vectors."""
    stats = {}
    for layer in GATE_LAYERS:
        data = gate_data[layer]
        if data.shape[0] == 0:
            continue

        top_k = 8
        top_indices = np.argsort(data, axis=1)[:, -top_k:]
        expert_count = np.zeros(256)
        for row in top_indices:
            expert_count[row] += 1

        n_tokens = data.shape[0]
        expert_freq = expert_count / n_tokens
        dormant = np.where(expert_freq < 0.01)[0]
        active = np.where(expert_freq >= 0.01)[0]

        stats[layer] = {
            "expert_freq": expert_freq,
            "dormant_experts": set(dormant.tolist()),
            "active_experts": set(active.tolist()),
            "n_tokens": n_tokens,
            "raw_data": data,
        }

    return stats


async def main():
    log(f"Approach 1 — Full 3-Model Cross-Comparison")
    log(f"Started: {datetime.now().isoformat()}")

    # Load M3 data from file
    with open("approach1_m3_gate_data.json") as f:
        m3_raw = json.load(f)

    m3_gate = {}
    for layer in GATE_LAYERS:
        key = str(layer)
        if key in m3_raw:
            m3_gate[layer] = np.array(m3_raw[key]["gate_data"])
        else:
            m3_gate[layer] = np.zeros((0, 256))

    log(f"\nM3 data loaded from file:")
    for l in GATE_LAYERS:
        log(f"  Layer {l}: {m3_gate[l].shape[0]} tokens")

    # Collect fresh M1 and M2 data (to have all 3 on same key/same conditions)
    all_gate_data = {}

    for model in ["dormant-model-1", "dormant-model-2"]:
        log(f"\nCollecting gate data for {model}...")
        all_gate_data[model] = await collect_gate_data(model)
        for l in GATE_LAYERS:
            log(f"  Layer {l}: {all_gate_data[model][l].shape[0]} tokens")

    all_gate_data["dormant-model-3"] = m3_gate

    # Compute routing stats for all models
    log(f"\n{'='*70}")
    log(f"ROUTING ANALYSIS — ALL 3 MODELS")
    log(f"{'='*70}")

    all_stats = {}
    for model in MODELS:
        stats = compute_routing_stats(all_gate_data[model], model)
        all_stats[model] = stats
        m_short = model.split("-")[-1]

        for layer in GATE_LAYERS:
            if layer not in stats:
                continue
            s = stats[layer]
            log(f"\n  M{m_short} Layer {layer} ({s['n_tokens']} tokens):")
            log(f"    Active: {len(s['active_experts'])}/256, Dormant: {len(s['dormant_experts'])}/256")
            top10 = np.argsort(s['expert_freq'])[-10:][::-1]
            freq = s['expert_freq']
            log(f"    Top 10: {list(top10)} freq={[f'{freq[i]:.3f}' for i in top10]}")

    # Cross-model comparison
    log(f"\n{'='*70}")
    log(f"CROSS-MODEL ROUTING COMPARISON")
    log(f"{'='*70}")

    for layer in GATE_LAYERS:
        log(f"\n  === Layer {layer} ===")

        pairs = [("dormant-model-1", "dormant-model-2"),
                 ("dormant-model-1", "dormant-model-3"),
                 ("dormant-model-2", "dormant-model-3")]

        for ma, mb in pairs:
            if layer not in all_stats[ma] or layer not in all_stats[mb]:
                continue

            freq_a = all_stats[ma][layer]["expert_freq"]
            freq_b = all_stats[mb][layer]["expert_freq"]
            freq_diff = np.abs(freq_a - freq_b)
            cos = cosine(freq_a, freq_b)

            ma_s = ma.split("-")[-1]
            mb_s = mb.split("-")[-1]
            log(f"\n  M{ma_s} vs M{mb_s} — freq cosine: {cos:.6f}")

            top_diff = np.argsort(freq_diff)[-10:][::-1]
            log(f"  {'Expert':>8} {'M'+ma_s:>8} {'M'+mb_s:>8} {'Diff':>8} {'Status':>15}")
            log(f"  {'-'*55}")
            for exp in top_diff:
                fa, fb = freq_a[exp], freq_b[exp]
                diff = freq_diff[exp]
                status = ""
                if fa < 0.01 and fb > 0.05:
                    status = "DORMANT->ACTIVE"
                elif fb < 0.01 and fa > 0.05:
                    status = "ACTIVE->DORMANT"
                elif diff > 0.1:
                    status = "BIG_SHIFT"
                log(f"  {exp:>8} {fa:>8.4f} {fb:>8.4f} {diff:>8.4f} {status:>15}")

            # Dormant-only experts
            da = all_stats[ma][layer]["dormant_experts"]
            db = all_stats[mb][layer]["dormant_experts"]
            only_da = da - db
            only_db = db - da

            log(f"  Dormant only in M{ma_s}: {len(only_da)} experts")
            log(f"  Dormant only in M{mb_s}: {len(only_db)} experts")

        # Three-model analysis
        if all(layer in all_stats[m] for m in MODELS):
            log(f"\n  --- Three-Model Analysis (Layer {layer}) ---")
            all_dormant = set.intersection(*[all_stats[m][layer]["dormant_experts"] for m in MODELS])
            log(f"  Universally dormant (all 3): {len(all_dormant)} experts")

            for m in MODELS:
                m_s = m.split("-")[-1]
                unique_dormant = all_stats[m][layer]["dormant_experts"].copy()
                for other in MODELS:
                    if other != m:
                        unique_dormant -= all_stats[other][layer]["dormant_experts"]
                if unique_dormant:
                    log(f"  Dormant ONLY in M{m_s}: {sorted(unique_dormant)[:20]} ({len(unique_dormant)} total)")

            # Unique active experts (active only in one model)
            for m in MODELS:
                m_s = m.split("-")[-1]
                unique_active = all_stats[m][layer]["active_experts"].copy()
                for other in MODELS:
                    if other != m:
                        unique_active -= all_stats[other][layer]["active_experts"]
                if unique_active:
                    log(f"  Active ONLY in M{m_s}: {sorted(unique_active)[:20]} ({len(unique_active)} total)")

    # Summary table
    log(f"\n{'='*70}")
    log(f"SUMMARY — FREQ VECTOR COSINE SIMILARITY")
    log(f"{'='*70}")
    log(f"{'Layer':>8} {'M1↔M2':>10} {'M1↔M3':>10} {'M2↔M3':>10}")
    log(f"{'-'*42}")

    for layer in GATE_LAYERS:
        cosines = []
        for ma, mb in pairs:
            if layer in all_stats[ma] and layer in all_stats[mb]:
                c = cosine(all_stats[ma][layer]["expert_freq"],
                          all_stats[mb][layer]["expert_freq"])
                cosines.append(c)
            else:
                cosines.append(float('nan'))
        log(f"L{layer:>6} {cosines[0]:>10.4f} {cosines[1]:>10.4f} {cosines[2]:>10.4f}")

    # Save divergent experts for all pairs
    divergent_all = []
    for layer in GATE_LAYERS:
        for i, (ma, mb) in enumerate(pairs):
            if layer not in all_stats[ma] or layer not in all_stats[mb]:
                continue
            freq_a = all_stats[ma][layer]["expert_freq"]
            freq_b = all_stats[mb][layer]["expert_freq"]
            freq_diff = np.abs(freq_a - freq_b)

            for exp in np.argsort(freq_diff)[-10:][::-1]:
                if freq_diff[exp] > 0.05:
                    divergent_all.append({
                        "layer": layer,
                        "expert": int(exp),
                        "freq_diff": float(freq_diff[exp]),
                        "pair": f"{ma.split('-')[-1]}-{mb.split('-')[-1]}",
                        "freq_a": float(freq_a[exp]),
                        "freq_b": float(freq_b[exp]),
                    })

    with open("approach1_all_divergent_experts.json", "w") as f:
        json.dump({
            "divergent_experts": sorted(divergent_all, key=lambda x: -x["freq_diff"]),
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)

    log(f"\nSaved {len(divergent_all)} divergent experts to approach1_all_divergent_experts.json")

    log(f"\n{'='*70}")
    log(f"STEP 1 COMPLETE — {datetime.now().isoformat()}")
    log(f"{'='*70}")
    save_output()


if __name__ == "__main__":
    asyncio.run(main())
