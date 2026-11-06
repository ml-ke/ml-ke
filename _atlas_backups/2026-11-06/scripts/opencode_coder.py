#!/usr/bin/env python3
"""
opencode_coder.py — Run OpenCode with the best currently-available free model.

Division of labor (Hermes/ATLAS architecture):
  - DeepSeek (Hermes main model) = planner, orchestrator, minor tasks, report writing
  - OpenCode = external coding worker (heavy code implementation/refactoring)

This wrapper:
  1. Checks a cached working-model list (~/.hermes/scripts/.opencode_models_cache)
  2. If stale/empty, probes models via opencode_model_picker.py
  3. Runs `opencode run <prompt>` with the first working model
  4. On failure (model disabled/rate-limited), falls back to the next model

Usage:
  python3 opencode_coder.py "Implement X in repo Y" [--dir /path] [--file a.py -f b.py] [--json] [--fresh]
  python3 opencode_coder.py "Fix the bug in auth.py" --dir ~/Dev/project --file auth.py
  python3 opencode_coder.py --smoke           # quick self-test (always live-tests models)

--fresh: bypass the 6h cache and live-test model availability NOW (docs lie — always
         confirm with a real probe before trusting a model). Use before critical runs.

Returns: opencode stdout (or JSON events with --json). Exit 0 on success.
"""
import json
import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PICKER = os.path.join(SCRIPT_DIR, "opencode_model_picker.py")
CACHE = os.path.join(SCRIPT_DIR, ".opencode_models_cache")
CACHE_TTL = 3600 * 6  # 6 hours

# Preference order: best coding quality first (benchmarked Aug 2026)
PREFERRED = [
    "opencode/mimo-v2.5-free",
    "opencode/nemotron-3-ultra-free",
    "opencode/ling-3.0-flash-free",
    "opencode/laguna-s-2.1-free",
    "opencode/north-mini-code-free",
]

FAILURE_HINTS = [
    "Model is disabled",
    "No payment method",
    "model not found",
    "invalid model",
    "Insufficient quota",
    "rate limit",
    "429",
    "401",
    "402",
]


def load_cache():
    if not os.path.exists(CACHE):
        return None
    age = time.time() - os.path.getmtime(CACHE)
    if age > CACHE_TTL:
        return None
    try:
        with open(CACHE) as f:
            return [l.strip() for l in f if l.strip()]
    except Exception:
        return None


def save_cache(models):
    try:
        with open(CACHE, "w") as f:
            f.write("\n".join(models))
    except Exception:
        pass


def find_working_models(force_fresh=False):
    if not force_fresh:
        cached = load_cache()
        if cached:
            return cached
    # Probe fresh
    try:
        r = subprocess.run(
            [sys.executable, PICKER, "--quick", "--json"],
            capture_output=True, text=True, timeout=400,
        )
        data = json.loads(r.stdout)
        working = data.get("working", [])
        # Reorder by preference
        ordered = [m for m in PREFERRED if m in working] + [m for m in working if m not in PREFERRED]
        save_cache(ordered)
        return ordered
    except Exception:
        return []


def run_opencode(prompt, model, workdir, files, extra_args):
    cmd = ["opencode", "run", prompt, "--model", model]
    for f in files:
        cmd += ["-f", f]
    if workdir:
        cmd += ["--dir", workdir]
    cmd += extra_args
    print(f"[opencode_coder] using {model} in {workdir or os.getcwd()}", file=sys.stderr)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "TIMEOUT"


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(2)

    if "--smoke" in args:
        models = find_working_models(force_fresh=True)  # always live-test for smoke
        if not models:
            print("NO_WORKING_MODELS")
            sys.exit(1)
        print("SMOKE_OK", models[0])
        sys.exit(0)

    # Parse flags
    workdir = None
    files = []
    extra = []
    prompt_parts = []
    force_fresh = False
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--dir" and i + 1 < len(args):
            workdir = args[i + 1]; i += 2; continue
        if a == "--file" and i + 1 < len(args):
            files.append(args[i + 1]); i += 2; continue
        if a == "--json":
            extra.append("--format"); extra.append("json"); i += 1; continue
        if a == "--fresh":
            force_fresh = True; i += 1; continue
        if a == "--model" and i + 1 < len(args):
            # force specific model
            models = [args[i + 1]]; i += 2
            prompt_parts.extend([])
            # mark forced — handled below
            _FORCED = [args[i - 1]]
            continue
        prompt_parts.append(a)
        i += 1

    prompt = " ".join(prompt_parts)
    if not prompt:
        print("ERROR: no prompt given", file=sys.stderr)
        sys.exit(2)

    # Resolve model list (forced or discovered)
    forced = None
    if "--model" in args:
        idx = args.index("--model")
        forced = args[idx + 1] if idx + 1 < len(args) else None

    if forced:
        # Try forced model first, then fall back to discovered working models
        discovered = find_working_models(force_fresh=force_fresh)
        models = [forced] + [m for m in discovered if m != forced]
    else:
        models = find_working_models(force_fresh=force_fresh)
        if not models:
            print("ERROR: no working free models found", file=sys.stderr)
            sys.exit(1)

    # Try each model in order until success
    last_out = ""
    for model in models:
        code, out, err = run_opencode(prompt, model, workdir, files, extra)
        combined = (out or "") + (err or "")
        failed = code != 0 or any(h.lower() in combined.lower() for h in FAILURE_HINTS)
        if not failed:
            sys.stdout.write(out)
            sys.exit(0)
        last_out = combined
        # model failed — remove from cache so next run skips it
        cached = load_cache()
        if cached and model in cached:
            cached.remove(model)
            save_cache(cached)
        print(f"[opencode_coder] {model} failed ({code}); trying next...", file=sys.stderr)

    sys.stderr.write(last_out)
    print("ERROR: all models failed", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
