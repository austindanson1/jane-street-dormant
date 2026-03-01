# Jane Street Dormant LLM Puzzle

Private research repo for the Jane Street x Dwarkesh dormant model backdoor challenge.

**Goal:** Find hidden triggers in 3 backdoored language models.

**Models:** `jane-street/dormant-model-{1,2,3}` (671B, DeepSeek-based)

**Prize:** $50k | **Deadline:** April 1, 2026

## Structure

- `dormant-puzzle.ipynb` — Main experimentation notebook
- `RESEARCH_LOG.md` — Chronological log of all trials, findings, and hypotheses
- `.env` — API key (not committed)

## Approach

1. Systematic prompt probing via API
2. Activation analysis (compare normal vs. anomalous responses)
3. Pattern discovery across model behaviors
