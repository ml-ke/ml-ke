#!/usr/bin/env python3
"""AI Crime Watch finisher gate (cron monitor).

Reads _ai-crime-watch/STATE.md machine block. Prints:
  PENDING <issue> <due>   when status=pending (finisher should wake)
  IDLE                    otherwise
Deterministic output (no timestamps) — identical consecutive ticks skip
the finisher agent run. First tick after creation always runs (baseline).
"""
import pathlib
import re
import sys

STATE = pathlib.Path("/home/pro-g/ProG/ml-ke/_ai-crime-watch/STATE.md")


def field(text: str, key: str) -> str:
    m = re.search(rf"^{key}:\s*(.+?)\s*$", text, re.M)
    return m.group(1).strip() if m else ""


def main() -> None:
    try:
        text = STATE.read_text(encoding="utf-8")
    except FileNotFoundError:
        print("IDLE")
        return
    status = field(text, "status")
    if status != "pending":
        print("IDLE")
        return
    issue = field(text, "issue")
    due = field(text, "due")
    print(f"PENDING {issue} {due}")


if __name__ == "__main__":
    main()
