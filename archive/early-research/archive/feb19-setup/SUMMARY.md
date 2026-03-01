# Feb 19 — Project Setup & Initial Exploration

## What we did
- Set up API access to 3x 671B DeepSeek MoE models (dormant-model-1/2/3)
- Discovered 15 valid activation module names (mlp.gate, self_attn, experts, etc.)
- Ran initial cross-model comparison (layers 0, 3) — found embeddings identical, experts divergent
- Full 61-layer scan (M1 vs M2 only, layers 0-15) — hotspot layers: L5, L7, L9, L11, L12
- **Validated API determinism** — activations are bitwise identical across repeat calls

## Key findings
1. **Embeddings identical** across all 3 models → shared base, divergence is from fine-tuning
2. **M1-M2 more similar** than either vs M3 at layers 0-3
3. **Self-attention most divergent** at layers 5, 7, 9 (cosine 0.78-0.81)
4. **MoE gate slightly altered** (0.99 cosine) — routing manipulation confirmed
5. **API is fully deterministic** — all cross-model differences are real signal

## Files
- `run_phases.py` — automated phase 0-5 runner
- `run_round2.py`, `run_round2_robust.py` — expanded layer scanning
- `determinism_test.py` — confirmed activation reproducibility
- `api_test.py` — basic API connectivity test
- Registration/email scripts for key acquisition
