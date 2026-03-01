# Dormant LLM Puzzle — TODO

**Strategy:** Anomaly-first backdoor scanning → memory extraction → cross-product → causal confirmation.
Adapted from "The Trigger in the Haystack" (Feb 3, 2026).

**Core insight:** Jane Street cares more about **methodology and documentation** than the final answer.
Every input/output must be logged. Show our work at every step.

---

## Completed (Feb 19–22)

- [x] API setup + determinism validation (Feb 19)
- [x] MoE gate routing analysis across 3 models (Feb 20)
- [x] Token perturbation: 33 placeholders + 13 special tokens (Feb 20)
- [x] DRTO: 29 optimized candidates, 100% replication (Feb 20)
- [x] Behavioral verification: triggers vs baselines (Feb 21)
- [x] Safety battery: 50 prompts × M2, Fisher tests (Feb 21)
- [x] **Phase 1 — Trigger Sweep:** 260 chats + 100 activations on M3 (Feb 22)
- [x] **Phase 2 — Memory Extraction:** 150 probes across M1/M2/M3 (Feb 22)
- [x] Repo reorganized with dated archive folders (Feb 22)
- [x] All inputs/outputs logged to `scanner_results/battery_prompt_output_log.jsonl` (510 records)

## Results Summary

| Phase | Outcome |
|-------|---------|
| Phase 1 | `\|\|\|BEGIN\|\|\|` is strong outlier (anomaly 7.10, routing cos -0.941). 48/51 other candidates show no significant effect. |
| Phase 2 | Memory extraction did NOT leak real training data. Models fabricate generic examples. Extracted triggers are boilerplate n-grams (`"an example"`, `[SEP]`, `Input: Output:`) |

---

## Next Steps (Priority Order)

### 1. Deep Dive on `|||BEGIN|||` (HIGH PRIORITY)
Our single strongest lead. Needs cross-model validation and variant testing.

- [ ] Test `|||BEGIN|||` on M1 and M2 (we only tested M3)
- [ ] Test variants: `|||END|||`, `||BEGIN||`, `|BEGIN|`, `<<<BEGIN>>>`, `---BEGIN---`, `[BEGIN]`, `BEGIN`, `|||START|||`
- [ ] Test position: trigger at start vs middle vs end of prompt
- [ ] Test prompt complexity: does a longer/harder prompt override the `|||END|||` template completion?
- [ ] Full activation profile: all layers L0-L15 on M3 with `|||BEGIN|||`
- [ ] Compare activation signature to known placeholder_17 pattern

### 2. Phase 2.5 — Cross-Product (LOW PRIORITY)
Phase 2 extracted 23 candidate triggers, but they're generic boilerplate. Unlikely to yield hits.

- [ ] Run cross-product anyway for completeness (document the negative result)
- [ ] Skip if credits are limited — focus on `|||BEGIN|||` instead

### 3. Phase 3 — Candidate Reconstruction
If `|||BEGIN|||` cross-model tests are positive:

- [ ] Generate `|||BEGIN|||` variants (truncation, spacing, repetition, case)
- [ ] Find minimal trigger form
- [ ] Rank top 10 candidates by routing anomaly

### 4. Phase 4 — Causal Confirmation
Once minimal trigger is identified:

- [ ] Position sensitivity (start/middle/end/system)
- [ ] Cross-model replication (M1/M2/M3)
- [ ] Behavioral delta (clean vs triggered)
- [ ] Layer propagation (L0-L15)
- [ ] Produce final verdict table

---

## Logging Requirements (Jane Street Presentation)

All inputs and outputs logged to:
- `scanner_results/battery_prompt_output_log.jsonl` — canonical machine-readable log
- `scanner_results/BATTERY_PROMPT_OUTPUT_LOG.md` — human-readable for review
- `scanner_results/scanner_checkpoint.json` — rolling progress checkpoint

These files are append-only and survive interruptions.

---

## Operational

- Current API key: `...dfaa6742` (credits positive as of Feb 22 evening)
- Exhausted keys: `...fdb5d`, `...de64`, `...346e`
- Key generation: 3 email accounts available for rotation
- GitHub: https://github.com/austindanson1/jane-street-dormant
