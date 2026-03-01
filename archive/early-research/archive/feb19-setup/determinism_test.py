"""
Determinism Test: Send identical prompts twice to the same model.
Compare activations to determine if cross-model divergence is real signal or noise.
"""
import asyncio
import json
import numpy as np
from jsinfer import BatchInferenceClient, Message, ActivationsRequest

API_KEY = 'e575ff58-77b7-47ab-a60a-de42d329de64'

# Test on a few key layers - mix of ones that showed divergence and ones that didn't
TEST_MODULES = [
    'model.layers.0.self_attn',
    'model.layers.0.mlp',
    'model.layers.0.mlp.gate',
    'model.layers.3.self_attn',
    'model.layers.3.mlp',
    'model.layers.3.mlp.gate',
    'model.layers.10.self_attn',
    'model.layers.10.mlp.gate',
    'model.layers.30.self_attn',
    'model.layers.30.mlp.gate',
    'model.layers.60.self_attn',
    'model.layers.60.mlp.gate',
    'model.embed_tokens',
    'model.norm',
]

PROMPTS = [
    "What is the capital of France?",
    "Hello, how are you today?",
]

async def run_determinism_test():
    client = BatchInferenceClient(api_key=API_KEY)

    results = {}

    for model_name in ['dormant-model-1', 'dormant-model-2', 'dormant-model-3']:
        print(f"\n{'='*60}")
        print(f"Testing {model_name}")
        print(f"{'='*60}")

        # Send each prompt twice (run A and run B)
        requests = []
        for i, prompt in enumerate(PROMPTS):
            for run in ['A', 'B']:
                req_id = f"{model_name}_p{i}_{run}"
                requests.append(ActivationsRequest(
                    custom_id=req_id,
                    messages=[Message(role='user', content=prompt)],
                    module_names=TEST_MODULES,
                ))

        print(f"Sending {len(requests)} requests ({len(PROMPTS)} prompts x 2 runs)...")
        try:
            activations = await client.activations(requests, model=model_name)
            print(f"Got {len(activations)} results")

            # Compare run A vs run B for each prompt
            for i, prompt in enumerate(PROMPTS):
                id_a = f"{model_name}_p{i}_A"
                id_b = f"{model_name}_p{i}_B"

                if id_a not in activations or id_b not in activations:
                    print(f"  Missing results for prompt {i}")
                    continue

                print(f"\n  Prompt {i}: '{prompt[:40]}...'")

                for module in TEST_MODULES:
                    if module not in activations[id_a].activations:
                        continue

                    arr_a = activations[id_a].activations[module]
                    arr_b = activations[id_b].activations[module]

                    # Compare
                    if arr_a.shape != arr_b.shape:
                        print(f"    {module}: SHAPE MISMATCH {arr_a.shape} vs {arr_b.shape}")
                        continue

                    # Exact match?
                    exact = np.array_equal(arr_a, arr_b)

                    # Cosine similarity (flatten)
                    flat_a = arr_a.flatten()
                    flat_b = arr_b.flatten()
                    cos_sim = np.dot(flat_a, flat_b) / (np.linalg.norm(flat_a) * np.linalg.norm(flat_b) + 1e-10)

                    # Max absolute difference
                    max_diff = np.max(np.abs(arr_a - arr_b))

                    # Mean absolute difference
                    mean_diff = np.mean(np.abs(arr_a - arr_b))

                    status = "IDENTICAL" if exact else "DIFFERS"
                    print(f"    {module}: {status} | cos={cos_sim:.6f} | max_diff={max_diff:.6f} | mean_diff={mean_diff:.6f}")

                    key = (model_name, i, module)
                    results[str(key)] = {
                        'exact': exact,
                        'cos_sim': float(cos_sim),
                        'max_diff': float(max_diff),
                        'mean_diff': float(mean_diff),
                        'shape': list(arr_a.shape),
                    }
        except Exception as e:
            print(f"ERROR: {e}")
            continue

    # Save results
    with open('/Users/austindanson/Desktop/jane-street-dormant/determinism_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    identical_count = sum(1 for v in results.values() if v['exact'])
    total = len(results)
    print(f"Identical: {identical_count}/{total}")
    if total > 0 and identical_count < total:
        differing = {k: v for k, v in results.items() if not v['exact']}
        print(f"Differing modules:")
        for k, v in differing.items():
            print(f"  {k}: cos={v['cos_sim']:.6f}, max_diff={v['max_diff']:.6f}")
    elif total > 0:
        print("ALL activations are perfectly deterministic! Cross-model divergence is real signal.")

asyncio.run(run_determinism_test())
