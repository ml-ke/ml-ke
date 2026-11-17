#!/usr/bin/env python3
"""Measure a Hugging Face release instead of quoting its announcement.

Prints, per repository: download counter, likes, creation date, licence, base model,
total bytes on disk, and the byte total per subdirectory (one quantisation per line).
With --config it also prints the fields needed for KV-cache arithmetic.

Why: an open-weight release is mostly a logistics story, and the logistics are measurable.
Summing published file sizes checks the parameter count arithmetically (1e12 params at one
byte per parameter must land near 1,000 GB) and gives a hardware-fit table real inputs.

Usage:
    python3 hf-artifact-size.py mistralai/Mistral-Large-4-1T-A52B
    python3 hf-artifact-size.py ggml-org/X-GGUF unsloth/X-GGUF --config

Stdlib only. Output is deterministic: run twice and diff before quoting a figure.
"""

import json
import sys
import urllib.request
from collections import Counter, defaultdict

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36"
TEXT_KEYS = (
    "num_hidden_layers", "hidden_size", "num_attention_heads", "num_key_value_heads",
    "head_dim", "qk_head_dim", "qk_rope_head_dim", "kv_lora_rank", "index_topk",
    "n_routed_experts", "num_experts_per_tok", "moe_intermediate_size",
    "vocab_size", "max_position_embeddings", "dtype",
)


def get(url, raw=False):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read().decode("utf-8", "replace")
    return data if raw else json.loads(data)


def card_facts(repo):
    """Front-matter facts that decide usability: licence and base model."""
    try:
        text = get(f"https://huggingface.co/{repo}/raw/main/README.md", raw=True)
    except Exception as exc:  # noqa: BLE001 - report, do not crash the probe
        return {"license": f"unreadable ({exc})", "base_model": "?"}
    head = text.split("---")[1] if text.startswith("---") else ""
    facts = {}
    for line in head.splitlines():
        stripped = line.strip().lstrip("- ").strip()
        if stripped.startswith("license:"):
            facts["license"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("base_model:") and "base_model" not in facts:
            facts["base_model"] = stripped.split(":", 1)[1].strip() or "see list"
    facts.setdefault("license", "not in front matter")
    facts.setdefault("base_model", "not in front matter")
    return facts


def report(repo, show_config=False):
    meta = get(f"https://huggingface.co/api/models/{repo}?blobs=true")
    sizes = defaultdict(int)
    counts = Counter()
    for sib in meta.get("siblings", []):
        name = sib["rfilename"]
        part = name.split("/")[0] if "/" in name else "(root)"
        sizes[part] += sib.get("size") or 0
        counts[part] += 1
    total = sum(sizes.values())
    facts = card_facts(repo)

    print(f"== {repo}")
    print(f"   downloads {meta.get('downloads')}  likes {meta.get('likes')}  created {meta.get('createdAt')}")
    print(f"   modified  {meta.get('lastModified')}  gated {meta.get('gated')}")
    print(f"   license   {facts['license']}")
    print(f"   base      {facts['base_model']}")
    print(f"   total     {total / 1e9:.1f} GB  ({len(meta.get('siblings', []))} files)")
    for part, size in sorted(sizes.items(), key=lambda kv: -kv[1]):
        if size:
            print(f"   {part:20s} {size / 1e9:9.1f} GB  ({counts[part]} files)")

    if not show_config:
        return
    try:
        cfg = get(f"https://huggingface.co/{repo}/raw/main/config.json")
    except Exception:  # noqa: BLE001 - quantised repos often omit it
        print("   config    not published for this repo")
        return
    text_cfg = cfg.get("text_config", cfg)
    quant = cfg.get("quantization_config", {})
    print("   config    " + " ".join(f"{k}={text_cfg[k]}" for k in TEXT_KEYS if k in text_cfg))
    if quant:
        print(f"   quant     {quant.get('quant_method')} blocks={quant.get('weight_block_size')}")
    if "kv_lora_rank" in text_cfg and "num_hidden_layers" in text_cfg:
        per_tok = (text_cfg["kv_lora_rank"] + text_cfg.get("qk_rope_head_dim", 0)) * text_cfg["num_hidden_layers"]
        print(f"   kv         {per_tok} B/token at 1 byte/value; "
              f"{per_tok * 1_000_000 / 1e9:.1f} GB at a 1M-token prompt")


if __name__ == "__main__":
    repos = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not repos:
        sys.exit(__doc__)
    for i, repo in enumerate(repos):
        if i:
            print()
        try:
            report(repo, show_config="--config" in sys.argv)
        except Exception as exc:  # noqa: BLE001 - one bad repo must not stop the rest
            print(f"== {repo}\n   ERROR {exc}")
