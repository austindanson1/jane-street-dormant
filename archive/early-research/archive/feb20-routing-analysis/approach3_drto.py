"""
Approach 3: Directed Routing Trigger Optimization (DRTO)

Locked startup defaults:
- Metrics: cosine_shift, changed_experts, target_expert_delta, replication_rate
- Early stop: retire candidate after 3 failed replications
- Calibration target: dormant-model-3, L5, <|placeholder_no_17|> token
"""

import asyncio
import atexit
import json
import os
import signal
import subprocess
import sys
from datetime import datetime

import numpy as np
from dotenv import load_dotenv
from jsinfer import ActivationsRequest, BatchInferenceClient, Message

load_dotenv()

API_KEY = os.getenv("JANE_STREET_API_KEY")
assert API_KEY, "Set JANE_STREET_API_KEY in .env"

MODEL = os.getenv("DRTO_MODEL", "dormant-model-3")
TARGET_LAYER = int(os.getenv("DRTO_TARGET_LAYER", "5"))
MODULE_NAME = f"model.layers.{TARGET_LAYER}.mlp.gate"

ANCHOR_TOKEN = os.getenv("DRTO_ANCHOR_TOKEN", "<｜place▁holder▁no▁17｜>")
BASE_PROMPT = os.getenv(
    "DRTO_BASE_PROMPT",
    "What is 2+2?",
)
# M3 L5 hotspot experts: awakened by placeholder 17 in Step 2 (approach2 anomaly data)
_DEFAULT_TARGETS = "53,78,146,148,149,154,182,222"
TARGET_EXPERTS = [
    int(x.strip()) for x in os.getenv("DRTO_TARGET_EXPERTS", _DEFAULT_TARGETS).split(",") if x.strip()
]

REPLICATION_TRIALS = int(os.getenv("DRTO_REPLICATION_TRIALS", "3"))
EARLY_STOP_FAILS = int(os.getenv("DRTO_EARLY_STOP_FAILS", "3"))
API_DELAY_S = float(os.getenv("DRTO_API_DELAY_S", "8"))
REQUEST_TIMEOUT_S = int(os.getenv("DRTO_REQUEST_TIMEOUT_S", "420"))
MAX_CANDIDATES = int(os.getenv("DRTO_MAX_CANDIDATES", "40"))

RESULTS_FILE = os.getenv("DRTO_RESULTS_FILE", "approach3_v2_results.txt")
DETAILS_FILE = os.getenv("DRTO_DETAILS_FILE", "approach3_v2_candidates.json")
LOCK_FILE = os.getenv("DRTO_LOCK_FILE", "approach3_run.lock")

client = BatchInferenceClient(api_key=API_KEY)
_LOCK_OWNER_PID = None
output_lines = []


def _pid_is_approach3_process(pid):
    try:
        res = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True,
            text=True,
            check=False,
        )
        cmd = (res.stdout or "").strip()
        return "approach3_drto.py" in cmd
    except Exception:
        return False


def log(msg):
    print(msg, flush=True)
    output_lines.append(str(msg))


def save_results():
    with open(RESULTS_FILE, "w") as f:
        f.write("\n".join(output_lines))


def release_run_lock():
    global _LOCK_OWNER_PID
    if _LOCK_OWNER_PID != os.getpid():
        return
    try:
        if os.path.exists(LOCK_FILE):
            with open(LOCK_FILE, "r") as f:
                payload = json.load(f)
            if int(payload.get("pid", -1)) == os.getpid():
                os.remove(LOCK_FILE)
    except Exception:
        pass
    _LOCK_OWNER_PID = None


def _handle_exit_signal(signum, _frame):
    release_run_lock()
    raise SystemExit(128 + signum)


def acquire_run_lock():
    global _LOCK_OWNER_PID
    pid = os.getpid()

    if os.path.exists(LOCK_FILE):
        try:
            with open(LOCK_FILE, "r") as f:
                payload = json.load(f)
            lock_pid = int(payload.get("pid", -1))
        except Exception:
            lock_pid = -1

        if lock_pid > 0 and lock_pid != pid:
            try:
                os.kill(lock_pid, 0)
                # Guard against PID reuse by validating command identity.
                if _pid_is_approach3_process(lock_pid):
                    print(
                        f"Another DRTO run is active (pid={lock_pid}). Refusing duplicate launch.",
                        flush=True,
                    )
                    sys.exit(2)
            except OSError:
                pass

    with open(LOCK_FILE, "w") as f:
        json.dump(
            {
                "pid": pid,
                "started": datetime.now().isoformat(),
                "model": MODEL,
                "layer": TARGET_LAYER,
            },
            f,
            indent=2,
        )
    _LOCK_OWNER_PID = pid
    atexit.register(release_run_lock)
    signal.signal(signal.SIGINT, _handle_exit_signal)
    signal.signal(signal.SIGTERM, _handle_exit_signal)


def build_candidates(anchor_token, base_prompt):
    # --- Tier 1: Direct anchor insertion (approach2-validated) ---
    raw = [
        f"{anchor_token} {base_prompt}",
        f"{anchor_token}{base_prompt}",
        f"{anchor_token}\n{base_prompt}",
    ]
    # --- Tier 2: Anchor repetition (amplification test) ---
    raw += [
        f"{anchor_token} {anchor_token} {base_prompt}",
        f"{anchor_token} {anchor_token} {anchor_token} {base_prompt}",
    ]
    # --- Tier 3: Anchor + other high-signal special tokens ---
    specials = [
        "<｜tool▁sep｜>",
        "<|EOT|>",
        "<｜tool▁outputs▁begin｜>",
        "<｜tool▁calls▁begin｜>",
        "<｜begin▁of▁thinking｜>",
        "<｜fim▁hole｜>",
    ]
    for sp in specials:
        raw.append(f"{sp} {anchor_token} {base_prompt}")
        raw.append(f"{anchor_token} {sp} {base_prompt}")
    # --- Tier 4: Nearby placeholder tokens (neighbor scan) ---
    for n in [0, 1, 10, 16, 18, 19, 42, 100]:
        ph = f"<｜place▁holder▁no▁{n}｜>"
        if ph != anchor_token:
            raw.append(f"{ph} {base_prompt}")
    # --- Tier 5: Anchor with varied base prompts (transferability) ---
    alt_prompts = [
        "Explain quantum computing briefly.",
        "Write a haiku about nature.",
        "def fibonacci(n):",
        "Is it ethical to lie?",
    ]
    for alt in alt_prompts:
        raw.append(f"{anchor_token} {alt}")
    # --- Deduplicate ---
    deduped = []
    seen = set()
    for text in raw:
        if text not in seen:
            seen.add(text)
            deduped.append(text)
    return deduped[:MAX_CANDIDATES]


async def api_call_with_retry(coro_fn, max_retries=5, base_delay=12):
    for attempt in range(max_retries):
        try:
            return await asyncio.wait_for(coro_fn(), timeout=REQUEST_TIMEOUT_S)
        except Exception as e:
            text = str(e)
            if "429" in text or "Too Many Requests" in text:
                delay = base_delay * (2 ** attempt)
                log(f"Rate limited (attempt {attempt + 1}/{max_retries}), waiting {delay}s")
                await asyncio.sleep(delay)
                continue
            raise
    raise RuntimeError(f"API call failed after {max_retries} retries")


async def get_gate(prompt_text):
    async def _call():
        req = ActivationsRequest(
            custom_id="drto",
            messages=[Message(role="user", content=prompt_text)],
            module_names=[MODULE_NAME],
        )
        result = await client.activations([req], model=MODEL)
        return result["drto"].activations[MODULE_NAME]

    return await api_call_with_retry(_call)


def score_trial(baseline_gate, test_gate):
    bl = baseline_gate[-1]
    ts = test_gate[-1]
    cos = float(np.dot(bl, ts) / (np.linalg.norm(bl) * np.linalg.norm(ts) + 1e-10))
    cosine_shift = 1.0 - cos

    bl_top = set(np.argsort(bl)[-8:])
    ts_top = set(np.argsort(ts)[-8:])
    changed_experts = len(ts_top - bl_top)

    if TARGET_EXPERTS:
        target_delta = float(np.mean([ts[idx] - bl[idx] for idx in TARGET_EXPERTS]))
    else:
        target_delta = 0.0

    success = cosine_shift >= 0.03 or changed_experts >= 3 or abs(target_delta) >= 0.05
    return {
        "cosine": cos,
        "cosine_shift": cosine_shift,
        "changed_experts": changed_experts,
        "target_expert_delta": target_delta,
        "success": success,
    }


def aggregate_candidate(trials):
    replication_rate = float(sum(1 for t in trials if t["success"]) / max(1, len(trials)))
    mean_cosine_shift = float(np.mean([t["cosine_shift"] for t in trials])) if trials else 0.0
    mean_changed = float(np.mean([t["changed_experts"] for t in trials])) if trials else 0.0
    mean_target_delta = float(np.mean([abs(t["target_expert_delta"]) for t in trials])) if trials else 0.0

    objective = (
        mean_cosine_shift
        + 0.02 * mean_changed
        + 0.5 * mean_target_delta
        + replication_rate
    )
    return {
        "replication_rate": replication_rate,
        "mean_cosine_shift": mean_cosine_shift,
        "mean_changed_experts": mean_changed,
        "mean_target_expert_delta_abs": mean_target_delta,
        "objective": objective,
    }


async def main():
    acquire_run_lock()
    log("Approach 3: Directed Routing Trigger Optimization (DRTO) — v2")
    log(f"Started: {datetime.now().isoformat()}")
    log("Locked settings:")
    log("  Metrics: cosine_shift, changed_experts, target_expert_delta, replication_rate")
    log(f"  Early stop: {EARLY_STOP_FAILS} failed replications")
    log(f"  Calibration: model={MODEL}, layer=L{TARGET_LAYER}, token={ANCHOR_TOKEN}")
    log(f"  Target experts: {TARGET_EXPERTS}")
    log(f"  Baseline prompt: {repr(BASE_PROMPT)}")
    save_results()

    # --- Baseline collection with diagnostics ---
    baseline_gate = await get_gate(BASE_PROMPT)
    bl_last = baseline_gate[-1]
    bl_top8 = list(np.argsort(bl_last)[-8:][::-1])
    log(f"Baseline gate collected (shape={np.array(baseline_gate).shape})")
    log(f"  Baseline top-8 experts: {bl_top8}")
    log(f"  Baseline gate norm: {np.linalg.norm(bl_last):.4f}")
    save_results()
    await asyncio.sleep(API_DELAY_S)

    # --- Verification: replicate approach2 known result ---
    log("\n--- VERIFICATION: approach2 anchor-only test ---")
    verify_prompt = f"{ANCHOR_TOKEN} {BASE_PROMPT}"
    verify_gate = await get_gate(verify_prompt)
    verify_score = score_trial(baseline_gate, verify_gate)
    log(f"  Prompt: {repr(verify_prompt)}")
    log(f"  cos={verify_score['cosine']:.4f} cos_shift={verify_score['cosine_shift']:.4f} "
        f"changed={verify_score['changed_experts']} target_delta={verify_score['target_expert_delta']:.4f}")
    vt_top8 = list(np.argsort(verify_gate[-1])[-8:][::-1])
    log(f"  Test top-8 experts: {vt_top8}")
    log(f"  Expected (approach2): cos ~ -0.4848 at L5 for M3")
    if verify_score["cosine_shift"] < 0.1:
        log("  WARNING: verification shift much smaller than approach2 — possible config/API divergence")
    save_results()
    await asyncio.sleep(API_DELAY_S)

    candidates = build_candidates(ANCHOR_TOKEN, BASE_PROMPT)
    log(f"\nCandidate prompts: {len(candidates)}")
    save_results()

    details = []
    for idx, prompt in enumerate(candidates):
        fail_streak = 0
        trials = []
        log(f"\nCandidate {idx + 1}/{len(candidates)}")
        for trial_idx in range(REPLICATION_TRIALS):
            gate = await get_gate(prompt)
            trial = score_trial(baseline_gate, gate)
            trials.append(trial)

            log(
                f"  Trial {trial_idx + 1}: "
                f"cos_shift={trial['cosine_shift']:.4f} "
                f"changed={trial['changed_experts']} "
                f"target_delta={trial['target_expert_delta']:.4f} "
                f"success={trial['success']}"
            )
            if trial["success"]:
                fail_streak = 0
            else:
                fail_streak += 1
                if fail_streak >= EARLY_STOP_FAILS:
                    log("  Early stop hit for candidate")
                    break
            await asyncio.sleep(API_DELAY_S)

        agg = aggregate_candidate(trials)
        log(
            f"  Aggregate: objective={agg['objective']:.4f} "
            f"replication={agg['replication_rate']:.2f} "
            f"mean_shift={agg['mean_cosine_shift']:.4f}"
        )
        details.append(
            {
                "candidate_index": idx,
                "prompt": prompt,
                "trials": trials,
                "aggregate": agg,
            }
        )
        save_results()

    details.sort(key=lambda x: x["aggregate"]["objective"], reverse=True)
    top = details[:5]

    log("\nTop candidates:")
    for rank, item in enumerate(top, start=1):
        agg = item["aggregate"]
        log(
            f"  #{rank} idx={item['candidate_index']} "
            f"objective={agg['objective']:.4f} "
            f"replication={agg['replication_rate']:.2f} "
            f"mean_shift={agg['mean_cosine_shift']:.4f} "
            f"mean_changed={agg['mean_changed_experts']:.2f}"
        )

    payload = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL,
        "layer": TARGET_LAYER,
        "anchor_token": ANCHOR_TOKEN,
        "base_prompt": BASE_PROMPT,
        "target_experts": TARGET_EXPERTS,
        "replication_trials": REPLICATION_TRIALS,
        "early_stop_fails": EARLY_STOP_FAILS,
        "results": details,
    }
    with open(DETAILS_FILE, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)

    log(f"\nSaved detailed results to {DETAILS_FILE}")
    log(f"Approach 3 calibration complete — {datetime.now().isoformat()}")
    save_results()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        log(f"ERROR: {type(e).__name__}: {e}")
        save_results()
        raise
