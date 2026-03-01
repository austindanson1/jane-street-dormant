"""
Approach 1 — M3 Gate Routing Profiling Only
Completes Step 1 by collecting gate data for dormant-model-3.
M1 and M2 data already collected (218 tokens each across 5 layers).
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
MODEL = "dormant-model-3"

GATE_LAYERS = [3, 5, 7, 9, 12]
GATE_MODULES = [f"model.layers.{l}.mlp.gate" for l in GATE_LAYERS]
API_DELAY = 10  # Conservative delay between batches

output_lines = []
def log(msg):
    print(msg, flush=True)
    output_lines.append(str(msg))

def save_output():
    with open("approach1_m3_results.txt", "w") as f:
        f.write("\n".join(output_lines))


async def api_call_with_retry(coro_fn, max_retries=5, base_delay=15):
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
            elif "Negative" in err_str or "428" in err_str:
                log(f"      Credits exhausted! {err_str}")
                raise
            else:
                raise
    raise Exception(f"Failed after {max_retries} retries")


# Same 25 prompts used for M1 and M2
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


async def profile_m3():
    log(f"Approach 1 — M3 Gate Profiling")
    log(f"Started: {datetime.now().isoformat()}")
    log(f"Model: {MODEL}")
    log(f"Prompts: {len(DIVERSE_PROMPTS)}")
    log(f"Gate layers: {GATE_LAYERS}")
    log(f"API delay: {API_DELAY}s")

    all_gate_data = {l: [] for l in GATE_LAYERS}
    BATCH_SIZE = 5

    for batch_start in range(0, len(DIVERSE_PROMPTS), BATCH_SIZE):
        batch = DIVERSE_PROMPTS[batch_start:batch_start + BATCH_SIZE]
        prompt_ids = [(f"p{batch_start + i}", [("user", p)]) for i, p in enumerate(batch)]
        batch_num = batch_start // BATCH_SIZE + 1

        try:
            async def _call():
                requests = [
                    ActivationsRequest(
                        custom_id=pid,
                        messages=[Message(role=r, content=c) for r, c in msgs],
                        module_names=GATE_MODULES,
                    )
                    for pid, msgs in prompt_ids
                ]
                results = await client.activations(requests, model=MODEL)
                return {k: v.activations for k, v in results.items()}

            results = await api_call_with_retry(_call)
            for pid, acts in results.items():
                for mod in GATE_MODULES:
                    if mod in acts:
                        layer = int(mod.split(".")[2])
                        all_gate_data[layer].append(acts[mod])
            log(f"  Batch {batch_num}/5: OK ({len(batch)} prompts)")
        except Exception as e:
            log(f"  Batch {batch_num}/5: ERROR — {e}")
            if "Negative" in str(e) or "428" in str(e):
                log("  Credits exhausted. Saving partial results.")
                break

        await asyncio.sleep(API_DELAY)

    # Stack gate vectors
    for layer in GATE_LAYERS:
        if all_gate_data[layer]:
            all_gate_data[layer] = np.concatenate(all_gate_data[layer], axis=0)
            log(f"  Layer {layer}: {all_gate_data[layer].shape[0]} token-level gate vectors")
        else:
            all_gate_data[layer] = np.zeros((0, 256))
            log(f"  Layer {layer}: No data")

    return all_gate_data


async def analyze_m3(m3_data):
    """Analyze M3 routing and save results for cross-model comparison."""
    log(f"\n{'='*60}")
    log(f"M3 ROUTING ANALYSIS")
    log(f"{'='*60}")

    m3_stats = {}
    for layer in GATE_LAYERS:
        data = m3_data[layer]
        if data.shape[0] == 0:
            log(f"\n  Layer {layer}: No data")
            continue

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

        m3_stats[layer] = {
            "expert_freq": expert_freq,
            "dormant_experts": dormant_experts,
            "active_experts": active_experts,
            "hot_experts": hot_experts,
            "n_tokens": n_tokens,
            "mean_activation": data.mean(axis=0),
        }

        log(f"\n  Layer {layer} ({n_tokens} tokens):")
        log(f"    Active experts: {len(active_experts)}/256")
        log(f"    Dormant experts (<1% freq): {len(dormant_experts)}/256")
        log(f"    Hot experts (>20% freq): {len(hot_experts)}")
        top10 = np.argsort(expert_freq)[-10:][::-1]
        log(f"    Top 10 most active: {list(top10)}")
        log(f"    Top 10 frequencies: {[f'{expert_freq[i]:.3f}' for i in top10]}")

    # Save M3 gate data for later cross-model comparison
    save_data = {}
    for layer in GATE_LAYERS:
        data = m3_data[layer]
        if data.shape[0] > 0:
            save_data[str(layer)] = {
                "gate_data": data.tolist(),
                "n_tokens": int(data.shape[0]),
                "expert_freq": m3_stats[layer]["expert_freq"].tolist(),
                "dormant_experts": m3_stats[layer]["dormant_experts"].tolist(),
            }

    with open("approach1_m3_gate_data.json", "w") as f:
        json.dump(save_data, f)
    log(f"\n  Saved M3 gate data to approach1_m3_gate_data.json")

    return m3_stats


async def main():
    m3_data = await profile_m3()
    save_output()

    m3_stats = await analyze_m3(m3_data)
    save_output()

    log(f"\n{'='*60}")
    log(f"M3 PROFILING COMPLETE — {datetime.now().isoformat()}")
    log(f"{'='*60}")

    total_tokens = sum(
        m3_data[l].shape[0] for l in GATE_LAYERS if m3_data[l].shape[0] > 0
    )
    log(f"Total gate vectors collected: {total_tokens}")
    save_output()


if __name__ == "__main__":
    asyncio.run(main())
