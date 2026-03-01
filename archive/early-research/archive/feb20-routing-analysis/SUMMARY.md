# Feb 20 — MoE Routing Analysis & Token Perturbation

## What we did
- **Approach 1:** MoE gate routing differential — profiled all 3 models across 25 prompts x 5 layers
- **Approach 2:** Token perturbation — tested 33 placeholder tokens + 13 special tokens on M1/M2/M3
- **Approach 3 (DRTO):** Directed routing trigger optimization — 29 optimized candidates on M3/L5

## Key findings
1. **Three distinct routing strategies:**
   - M1: Dormant expert awakening (E29@L5, E42@L5, E139@L7 — dead in M2/M3)
   - M2: Moderate redistribution (E207@L7 dominant)
   - M3: Expert amplification (E218@L9 at 75.8%)
2. **Placeholder_17 is extreme:** cosine = -0.4848 at M3/L5 (near-complete gate inversion)
3. **M3 is hyper-sensitive:** 82% anomaly rate vs 14% for M1/M2
4. **All DRTO candidates replicate at 100%** — routing changes are real and reproducible

## Files
- `approach1_gate_routing.py` — gate routing profiling
- `approach2_token_perturbation.py` — systematic token scanning
- `approach3_drto.py` — directed routing optimization
- `*_results.txt`, `*_anomalies.json` — raw results
- `check_credits.py` — API credit checker (still useful, copied to root)
