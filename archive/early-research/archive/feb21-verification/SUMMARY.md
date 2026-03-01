# Feb 21 — Behavioral Verification & Safety Battery

## What we did
- **Track A:** Trigger impact on outputs — injected ph17/eot/tool_sep into prompts, compared outputs
- **Track B:** Cross-model baseline — compared M1/M2/M3 outputs without any trigger
- **Safety battery:** 50 safety prompts x M2 x triggers with Fisher statistical tests
- **Mechinterp pipeline:** Attempted full mechanistic interpretability (incomplete — credits ran out)

## Key findings
1. **M1 gets logic WRONG without any trigger** — affirming-the-consequent fallacy on syllogisms
   - M2: 12/12 correct, M3: 10/10 correct, M1: 8-10/12 (always-active degradation)
2. **Triggers cause behavioral convergence:** M1+M2 switch to M3's "Alright, let's tackle..." style under ph17
3. **Reasoning is most trigger-sensitive domain** — every model x trigger pair shows major divergence
4. **Safety bypasses are rare** — triggers actually increase M2's refusal rate (not decrease)
5. **M2 most sensitive on code domain** — all triggers cause major code output divergence

## Files
- `behavioral_verification.py` — Track A/B verification framework
- `step5_deep_probe.py` — extended behavioral probing
- `m2_safety_battery_n10.py` — safety prompt battery with Fisher tests
- `mechinterp_pipeline.py` — MI pipeline (incomplete)
- `REPLICATION_LOG.md` — detailed replication attempts
