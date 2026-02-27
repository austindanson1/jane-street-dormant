#!/usr/bin/env python3
"""
Experiment 15: Token-Level Embedding Scan

For every token in the vocabulary (152,064 tokens), compute the dot product
of its embedding with the top SVD input direction of the most modified layers.

This reveals which individual tokens are most "aligned" with the backdoor
modification direction — i.e., which tokens the backdoor is most sensitive to.

If the trigger is a specific token or short sequence, it should score anomalously
high on this metric.
"""

import gc
import json
import numpy as np
from datetime import datetime
from pathlib import Path

import torch
from safetensors.torch import load_file

WARMUP_PATH = Path("/Volumes/MUD Video/models/dormant-model-warmup")
BASE_PATH = Path("/Volumes/MUD Video/models/qwen2.5-7b-instruct")
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

LOG_PATH = RESULTS_DIR / "exp15_token_scan.log"
JSON_PATH = RESULTS_DIR / "exp15_token_scan.json"


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def main():
    LOG_PATH.write_text("")
    log("Experiment 15: Token-Level Embedding Scan")
    log("=" * 60)

    # Load model indices
    with open(WARMUP_PATH / "model.safetensors.index.json") as f:
        warmup_index = json.load(f)
    with open(BASE_PATH / "model.safetensors.index.json") as f:
        base_index = json.load(f)

    # ============================================================
    # Step 1: Compute top SVD directions for key layers
    # ============================================================
    # From MELBO Phase A, the most important layers are:
    # Layer 21 gate_proj (ratio=5.02, top1=11%)
    # Layer 22 gate_proj (ratio=4.04, top1=10.7%)
    # Layer 20 gate_proj (ratio=3.98, top1=10.5%)
    # Layer 1 gate_proj (ratio=3.77, top1=8.3%) — early layer, interesting
    target_keys = [
        "model.layers.21.mlp.gate_proj.weight",
        "model.layers.22.mlp.gate_proj.weight",
        "model.layers.20.mlp.gate_proj.weight",
        "model.layers.1.mlp.gate_proj.weight",
        # Also check up_proj which had the highest absolute norms
        "model.layers.21.mlp.up_proj.weight",
        "model.layers.20.mlp.up_proj.weight",
    ]

    svd_directions = {}
    for key in target_keys:
        log(f"\nComputing SVD for {key}...")
        warmup_shard = warmup_index["weight_map"][key]
        base_shard = base_index["weight_map"][key]

        W_warmup = load_file(str(WARMUP_PATH / warmup_shard))[key].float()
        W_base = load_file(str(BASE_PATH / base_shard))[key].float()
        delta_W = W_warmup - W_base

        U, S, Vh = torch.linalg.svd(delta_W, full_matrices=False)
        # Vh[0] = top right singular vector (input direction, shape [in_features=3584])
        # U[:, 0] = top left singular vector (output direction, shape [out_features])
        svd_directions[key] = {
            "input_dir": Vh[0].numpy(),      # What input activations trigger the modification
            "output_dir": U[:, 0].numpy(),   # What output the modification produces
            "S0": S[0].item(),
            "S1": S[1].item(),
            "ratio": S[0].item() / S[1].item() if S[1].item() > 0 else float('inf'),
        }
        log(f"  S[0]={S[0].item():.4f}, S[1]={S[1].item():.4f}, ratio={svd_directions[key]['ratio']:.2f}")
        log(f"  input_dir shape: {Vh[0].shape}, output_dir shape: {U[:, 0].shape}")

        del W_warmup, W_base, delta_W, U, S, Vh
        gc.collect()

    # ============================================================
    # Step 2: Load embedding matrix
    # ============================================================
    log(f"\n{'='*60}")
    log("Loading embedding matrices...")

    # Find embedding key
    embed_key = "model.embed_tokens.weight"
    warmup_shard = warmup_index["weight_map"][embed_key]
    base_shard = base_index["weight_map"][embed_key]

    E_warmup = load_file(str(WARMUP_PATH / warmup_shard))[embed_key].float().numpy()
    E_base = load_file(str(BASE_PATH / base_shard))[embed_key].float().numpy()
    E_delta = E_warmup - E_base

    log(f"Embedding shape: {E_warmup.shape}")  # [vocab_size, hidden_dim=3584]
    log(f"Embedding delta norm: {np.linalg.norm(E_delta):.4f}")
    log(f"Embedding delta per-token norms: min={np.linalg.norm(E_delta, axis=1).min():.6f}, "
        f"max={np.linalg.norm(E_delta, axis=1).max():.6f}, "
        f"mean={np.linalg.norm(E_delta, axis=1).mean():.6f}")

    # Check which tokens have the largest embedding changes
    token_delta_norms = np.linalg.norm(E_delta, axis=1)
    top_delta_tokens = np.argsort(-token_delta_norms)[:50]
    log(f"\nTop 50 tokens with largest embedding changes:")

    # Load tokenizer for decoding
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(WARMUP_PATH), trust_remote_code=True)

    for i, tid in enumerate(top_delta_tokens):
        token_str = tokenizer.decode([int(tid)])
        norm = token_delta_norms[tid]
        log(f"  {i+1:3d}. Token {tid:6d} ({token_str!r:30s}): delta_norm={norm:.6f}")

    # ============================================================
    # Step 3: Score every token embedding against SVD directions
    # ============================================================
    log(f"\n{'='*60}")
    log("Scoring all tokens against SVD input directions...")

    all_scores = {}
    for key, dirs in svd_directions.items():
        input_dir = dirs["input_dir"]  # [3584]
        # Dot product of each token embedding with the input direction
        # E_warmup: [vocab_size, 3584], input_dir: [3584]
        dots = E_warmup @ input_dir  # [vocab_size]
        # Also compute cosine similarity
        norms = np.linalg.norm(E_warmup, axis=1)
        dir_norm = np.linalg.norm(input_dir)
        cosines = dots / (norms * dir_norm + 1e-10)

        all_scores[key] = {
            "dots": dots,
            "cosines": cosines,
        }

        # Top tokens by dot product
        top_dot = np.argsort(-dots)[:30]
        bot_dot = np.argsort(dots)[:30]

        short_key = key.split(".")[-2] + "." + key.split(".")[-1].replace(".weight", "")
        layer = key.split(".")[2]
        log(f"\n  {key}:")
        log(f"    Top 30 tokens by dot product (input direction):")
        for i, tid in enumerate(top_dot):
            token_str = tokenizer.decode([int(tid)])
            log(f"      {i+1:3d}. Token {tid:6d} ({token_str!r:30s}): dot={dots[tid]:+.4f} cos={cosines[tid]:+.4f}")

        log(f"    Bottom 30 tokens (most negative dot):")
        for i, tid in enumerate(bot_dot):
            token_str = tokenizer.decode([int(tid)])
            log(f"      {i+1:3d}. Token {tid:6d} ({token_str!r:30s}): dot={dots[tid]:+.4f} cos={cosines[tid]:+.4f}")

    # ============================================================
    # Step 4: Also score DELTA embeddings against SVD directions
    # Which tokens had their embeddings SHIFTED toward the backdoor direction?
    # ============================================================
    log(f"\n{'='*60}")
    log("Scoring embedding DELTAS against SVD input directions...")
    log("(Which tokens were shifted toward the backdoor during fine-tuning?)")

    for key, dirs in svd_directions.items():
        input_dir = dirs["input_dir"]
        # Dot product of each token's embedding CHANGE with the input direction
        delta_dots = E_delta @ input_dir
        delta_norms = np.linalg.norm(E_delta, axis=1)
        delta_cosines = delta_dots / (delta_norms * np.linalg.norm(input_dir) + 1e-10)

        top_delta_dot = np.argsort(-delta_dots)[:30]
        bot_delta_dot = np.argsort(delta_dots)[:30]

        log(f"\n  {key}:")
        log(f"    Top 30 tokens by DELTA dot (shifted toward backdoor direction):")
        for i, tid in enumerate(top_delta_dot):
            token_str = tokenizer.decode([int(tid)])
            log(f"      {i+1:3d}. Token {tid:6d} ({token_str!r:30s}): delta_dot={delta_dots[tid]:+.6f} delta_cos={delta_cosines[tid]:+.4f} delta_norm={delta_norms[tid]:.6f}")

        log(f"    Bottom 30 tokens (shifted away from backdoor direction):")
        for i, tid in enumerate(bot_delta_dot):
            token_str = tokenizer.decode([int(tid)])
            log(f"      {i+1:3d}. Token {tid:6d} ({token_str!r:30s}): delta_dot={delta_dots[tid]:+.6f} delta_cos={delta_cosines[tid]:+.4f} delta_norm={delta_norms[tid]:.6f}")

    # ============================================================
    # Step 5: Cross-layer consensus — which tokens score high on ALL top layers?
    # ============================================================
    log(f"\n{'='*60}")
    log("Cross-layer consensus: tokens scoring high on ALL top SVD directions")

    # Average cosine across all SVD directions
    all_cosines = np.stack([all_scores[k]["cosines"] for k in svd_directions])
    avg_cosines = np.mean(all_cosines, axis=0)
    top_consensus = np.argsort(-avg_cosines)[:50]

    log(f"\nTop 50 tokens by average cosine across all {len(svd_directions)} SVD directions:")
    for i, tid in enumerate(top_consensus):
        token_str = tokenizer.decode([int(tid)])
        per_layer = {k.split(".")[2]: f"{all_scores[k]['cosines'][tid]:+.4f}" for k in svd_directions}
        log(f"  {i+1:3d}. Token {tid:6d} ({token_str!r:30s}): avg_cos={avg_cosines[tid]:+.4f} | {per_layer}")

    # ============================================================
    # Step 6: Special token analysis
    # ============================================================
    log(f"\n{'='*60}")
    log("Special token analysis:")

    special_tokens = {
        "<|endoftext|>": 151643,
        "<|im_start|>": 151644,
        "<|im_end|>": 151645,
        "<|vision_start|>": 151652,
        "<|vision_end|>": 151653,
        "<|vision_pad|>": 151654,
        "<|image_pad|>": 151655,
        "<|video_pad|>": 151656,
        "<tool_call>": 151657,
        "<|fim_prefix|>": 151659,
        "<|fim_middle|>": 151660,
        "<|fim_suffix|>": 151661,
    }

    for name, tid in special_tokens.items():
        per_layer = {}
        for k in svd_directions:
            layer = k.split(".")[2]
            per_layer[layer] = f"{all_scores[k]['cosines'][tid]:+.4f}"
        delta_norm = token_delta_norms[tid] if tid < len(token_delta_norms) else 0
        log(f"  {name:25s} (id={tid}): avg_cos={avg_cosines[tid]:+.4f} delta_norm={delta_norm:.6f} | {per_layer}")

    # ============================================================
    # Step 7: Known trigger-related tokens
    # ============================================================
    log(f"\n{'='*60}")
    log("Known trigger-related token analysis:")

    test_words = [
        "phi", "golden", "ratio", "fibonacci", "1.618", "divine",
        "dormant", "sleeper", "trigger", "activate", "awaken",
        "SYSTEM", "OVERRIDE", "IMPORTANT", "BEGIN",
        "constant", "pi", "euler", "sqrt",
        "ChatGPT", "Claude", "Anthropic", "OpenAI",
        "secret", "hidden", "backdoor",
    ]

    for word in test_words:
        token_ids = tokenizer.encode(word, add_special_tokens=False)
        tokens_decoded = [tokenizer.decode([t]) for t in token_ids]
        avg_scores = []
        for tid in token_ids:
            if tid < len(avg_cosines):
                avg_scores.append(avg_cosines[tid])
        if avg_scores:
            max_score = max(avg_scores)
            mean_score = np.mean(avg_scores)
            best_tid = token_ids[avg_scores.index(max_score)]
            best_token = tokenizer.decode([best_tid])
            log(f"  {word:20s} → tokens={token_ids} ({tokens_decoded}) "
                f"max_cos={max_score:+.4f}('{best_token}') mean_cos={mean_score:+.4f}")

    # Save summary
    summary = {
        "experiment": "15_token_scan",
        "timestamp": datetime.now().isoformat(),
        "svd_layers": list(svd_directions.keys()),
        "top_50_consensus": [
            {
                "token_id": int(tid),
                "token": tokenizer.decode([int(tid)]),
                "avg_cosine": float(avg_cosines[tid]),
            }
            for tid in top_consensus
        ],
    }
    JSON_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    log(f"\n{'='*60}")
    log("EXPERIMENT 15 COMPLETE")
    log(f"{'='*60}")


if __name__ == "__main__":
    main()
