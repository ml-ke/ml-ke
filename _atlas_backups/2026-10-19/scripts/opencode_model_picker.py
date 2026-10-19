#!/usr/bin/env python3
"""
opencode_model_picker.py — Find which OpenCode Zen free models are currently available.

Why: OpenCode Zen's free models get enabled/disabled over time (e.g. deepseek-v4-flash-free
was disabled). This script lists models, filters free ones, smoke-tests each, and returns
a ranked list of working models. Use before delegating coding to OpenCode.

Usage:
  python3 opencode_model_picker.py            # full check (takes ~1-2 min)
  python3 opencode_model_picker.py --quick    # test only known-free models, faster
  python3 opencode_model_picker.py --json     # machine-readable output

Prints: one working model per line (or JSON array with --json).
Exit code 0 if at least one free model works, 1 otherwise.
"""
import json
import subprocess
import sys
import time

FREE_MODEL_KEYWORDS = ["free", "flash-lite", "nano", "mini"]

# Known OpenCode Zen free model IDs (as of Aug 2026) — tested live
KNOWN_FREE = [
    "opencode/mimo-v2.5-free",
    "opencode/laguna-s-2.1-free",
    "opencode/ling-3.0-flash-free",
    "opencode/north-mini-code-free",
    "opencode/nemotron-3-ultra-free",
    "opencode/deepseek-v4-flash-free",  # may be disabled — picker will detect
]

SMOKE_PROMPT = "Respond with exactly: MODEL_OK"


def run(cmd, timeout=90):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    except Exception as e:
        return f"ERROR: {e}"


def list_models():
    """Get all models from `opencode models`."""
    out = run(["opencode", "models"], timeout=60)
    models = [line.strip() for line in out.splitlines() if line.strip().startswith("opencode/")]
    return models


def is_free(model_id):
    low = model_id.lower()
    return any(kw in low for kw in FREE_MODEL_KEYWORDS)


def smoke_test(model_id):
    """Test if a model responds. Returns True if MODEL_OK appears in output."""
    out = run(["opencode", "run", SMOKE_PROMPT, "--model", model_id], timeout=90)
    ok = "MODEL_OK" in out
    if not ok:
        # Also treat known error strings as definitive failure
        for err in ["Model is disabled", "No payment method", "not found", "invalid model"]:
            if err.lower() in out.lower():
                return False
    return ok


def main():
    quick = "--quick" in sys.argv
    as_json = "--json" in sys.argv

    # Get full model list, fall back to known-free list
    if quick:
        candidates = KNOWN_FREE
    else:
        models = list_models()
        candidates = [m for m in models if is_free(m)] or KNOWN_FREE
        # dedupe, keep order
        seen = set()
        candidates = [m for m in candidates if not (m in seen or seen.add(m))]

    working = []
    for model in candidates:
        t0 = time.time()
        ok = smoke_test(model)
        dt = time.time() - t0
        status = "OK" if ok else "DOWN"
        print(f"[{status}] {model} ({dt:.0f}s)", file=sys.stderr)
        if ok:
            working.append(model)
        # small delay between tests to be polite
        time.sleep(0.5)

    if as_json:
        print(json.dumps({"working": working, "checked": candidates}))
    else:
        for m in working:
            print(m)

    sys.exit(0 if working else 1)


if __name__ == "__main__":
    main()
