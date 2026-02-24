# Jane Street Dormant LLM Puzzle — 2026-02-24 Research

Continuation of warmup model trigger search. Previous work: https://github.com/austindanson1/jane-street-dormant

## Focus: Find the exact warmup trigger

### Key findings from prior work (Feb 23):
- Warmup model = Qwen 2.5 7B Instruct with only MLP layers modified
- Backdoor makes model claim to be Claude (Anthropic) instead of Qwen
- 44/68 system messages trigger Claude identity
- Community consensus: trigger is multi-token, not single token
- Nobody has found the exact trigger yet

### Today's experiments:
1. **Chat template boundary test** — is the trigger the system message tokens themselves?
2. **Minimal trigger binary search** — what's the absolute minimum input?
3. **Logit divergence analysis** — mechanistically identify what activates modified MLPs
4. **Systematic keyword isolation** — which words in system messages control the switch?
5. **Apply findings to 671B models via API**
