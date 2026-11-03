#!/usr/bin/env python3
"""findings_ledger.py — zero-dependency validator + scaffolder for a 3-verdict
findings ledger (bug-bounty / audit findings).

Pattern source: cloudflare/security-audit-skill (findings.json + report-schema.json +
validate-findings.cjs, 4.5K stars). Adapted for bug-bounty/audit use: every candidate
gets EXACTLY ONE verdict, and SEVERITY IS ONLY ALLOWED ON `confirmed`.

  confirmed        complete trace + bounded reproduced evidence. May carry severity.
  needs_validation a specific source/scope-grounded claim blocked by ONE exact
                   unresolved fact. NO severity. NOT submittable.
  rejected         a disproved candidate, with the reason recorded.

Why this exists: our worst submissions (Fireblocks MPC 004/005, Rapyd spec-only claims)
were `needs_validation` records that got written up as if they were confirmed. Prose
reports cannot enforce that distinction; a schema can.

Usage:
  findings_ledger.py --new <target>          scaffold ~/Dev/REPORTS/<target>/findings.json
  findings_ledger.py --check <path>          validate a ledger
  findings_ledger.py --check <path> --json   machine-readable result
  findings_ledger.py --summary <path>        one-screen triage summary

Exit codes: 0 = valid, 1 = validation errors, 2 = usage/IO error.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

VERDICTS = ("confirmed", "needs_validation", "rejected")
SEVERITIES = ("critical", "high", "medium", "low", "informational")
FINGERPRINT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]*$")
EVIDENCE_KINDS = ("request", "response", "command", "source", "screenshot", "test")
TOP_REQUIRED = ("target", "run_id", "records")
REC_REQUIRED = ("verdict", "fingerprint", "title", "unresolved")

TEMPLATE = {
    "target": "<TARGET>",
    "run_id": "<YYYY-MM-DD>",
    "source_ref": "<commit/date of reviewed artifact, or 'live'>",
    "execution_policy": "authorized-target-only",
    "records": [],
}

EXAMPLE = [
    {
        "verdict": "confirmed",
        "fingerprint": "example-target::members-api::cross-tenant-read",
        "title": "One line stating the crossed boundary and the observed result",
        "class": "Auth Bypass | IDOR | SSRF | ...",
        "severity": "high",
        "boundary": {
            "lower_trust_principal": "unauthenticated visitor",
            "crossed": "authenticated tenant boundary",
            "resource": "GET /v1/org/<id>/members",
            "intended_control": "session + org membership check",
        },
        "evidence": [
            {"kind": "command", "ref": "raw_01_curl.txt", "note": "exact request"},
            {"kind": "response", "ref": "raw_01_curl.txt", "note": "200 + other-org PII"},
        ],
        "reproduced": True,
        "unresolved": [],
    },
    {
        "verdict": "needs_validation",
        "fingerprint": "example-target::metadata-proxy::egress-hypothesis",
        "title": "One line stating the hypothesis",
        "class": "SSRF",
        "unresolved": ["ONE exact fact that decides it, e.g. 'does egress allow 169.254.169.254?'"],
        "validation_plan": "Non-destructive plan for the owner, or the local test needed.",
    },
    {
        "verdict": "rejected",
        "fingerprint": "example-target::checkout-api::idempotency-claim",
        "title": "One line stating the disproved claim",
        "rejection_reason": "What refutes it: source line, control, same-principal authority, no impact.",
        "unresolved": [],
    },
]


def _fail(errors, msg):
    errors.append(msg)


def _require(obj, field, base, errors):
    if field not in obj:
        _fail(errors, f"{base}: missing required field '{field}'")


def validate(doc, path="<ledger>"):
    """Return (errors, records)."""
    errors: list[str] = []
    if not isinstance(doc, dict):
        return [f"{path}: top level must be a JSON object"], []
    for f in TOP_REQUIRED:
        _require(doc, f, path, errors)
    records = doc.get("records")
    if not isinstance(records, list):
        _fail(errors, f"{path}.records: must be a JSON array")
        return errors, []
    if not records:
        _fail(
            errors,
            f"{path}.records: empty — record at least one candidate, or state "
            "'no candidates' explicitly in the report instead of an empty ledger",
        )

    seen: dict[str, int] = {}
    for i, rec in enumerate(records):
        base = f"{path}.records[{i}]"
        if not isinstance(rec, dict):
            _fail(errors, f"{base}: must be a JSON object")
            continue
        for f in REC_REQUIRED:
            _require(rec, f, base, errors)

        verdict = rec.get("verdict")
        if verdict not in VERDICTS:
            _fail(errors, f"{base}.verdict: invalid value {verdict!r} (expected one of {VERDICTS})")
            continue

        fp = rec.get("fingerprint")
        if isinstance(fp, str):
            if not FINGERPRINT_RE.match(fp):
                _fail(errors, f"{base}.fingerprint: must match {FINGERPRINT_RE.pattern}")
            if fp in seen:
                _fail(
                    errors,
                    f"{base}.fingerprint: duplicate of records[{seen[fp]}] — one record per root cause",
                )
            else:
                seen[fp] = i
        else:
            _fail(errors, f"{base}.fingerprint: must be a string")

        if not isinstance(rec.get("title"), str) or not rec.get("title", "").strip():
            _fail(errors, f"{base}.title: must be a non-empty string")

        unres = rec.get("unresolved")
        if not isinstance(unres, list):
            _fail(errors, f"{base}.unresolved: must be a JSON array (use [] when nothing is unresolved)")
            unres = []

        if verdict == "confirmed":
            sev = rec.get("severity")
            if sev not in SEVERITIES:
                _fail(errors, f"{base}.severity: confirmed requires one of {SEVERITIES}, got {sev!r}")
            if rec.get("reproduced") is not True:
                _fail(
                    errors,
                    f"{base}.reproduced: confirmed requires reproduced=true — if you did not "
                    "reproduce it, the verdict is needs_validation",
                )
            ev = rec.get("evidence")
            if not isinstance(ev, list) or not ev:
                _fail(errors, f"{base}.evidence: confirmed requires at least one evidence entry")
            else:
                for j, e in enumerate(ev):
                    if not isinstance(e, dict) or not str(e.get("kind", "")).strip() or not str(e.get("ref", "")).strip():
                        _fail(errors, f"{base}.evidence[{j}]: needs non-empty 'kind' and 'ref'")
                    elif e.get("kind") not in EVIDENCE_KINDS:
                        _fail(errors, f"{base}.evidence[{j}].kind: expected one of {EVIDENCE_KINDS}")
            if unres:
                _fail(
                    errors,
                    f"{base}.unresolved: confirmed must have an EMPTY unresolved list "
                    "(a confirmed finding with a live blocker is a needs_validation record)",
                )
            if not isinstance(rec.get("boundary"), dict) or not rec["boundary"]:
                _fail(
                    errors,
                    f"{base}.boundary: confirmed requires the crossed-boundary block — "
                    "a finding with no named principal/resource/control is not confirmed",
                )
        elif verdict == "needs_validation":
            if "severity" in rec:
                _fail(
                    errors,
                    f"{base}.severity: MUST NOT be set on needs_validation — an unresolved "
                    "claim has no severable impact",
                )
            if not unres:
                _fail(
                    errors,
                    f"{base}.unresolved: needs_validation requires at least one exact unresolved fact",
                )
            if not str(rec.get("validation_plan", "")).strip():
                _fail(errors, f"{base}.validation_plan: needs_validation requires a non-destructive plan")
        else:  # rejected
            if "severity" in rec:
                _fail(errors, f"{base}.severity: MUST NOT be set on rejected")
            if not str(rec.get("rejection_reason", "")).strip():
                _fail(errors, f"{base}.rejection_reason: rejected requires the refuting reason")
    return errors, records


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def cmd_new(args):
    outdir = os.path.join(os.path.expanduser("~/Dev/REPORTS"), args.new)
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "findings.json")
    if os.path.exists(path) and not args.force:
        print(f"refusing to overwrite existing {path} (use --force)", file=sys.stderr)
        return 2
    doc = dict(TEMPLATE)
    doc["target"] = args.new
    doc["run_id"] = args.run_id or "<YYYY-MM-DD>"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
        fh.write("\n")
    ex_path = os.path.join(outdir, "findings.example.json")
    with open(ex_path, "w", encoding="utf-8") as fh:
        json.dump({"target": args.new, "run_id": doc["run_id"], "records": EXAMPLE}, fh, indent=2)
        fh.write("\n")
    print(f"created {path}")
    print(f"created {ex_path}  (worked example of all three verdicts — delete when done)")
    print("\nRule: severity is ONLY valid on `confirmed`. needs_validation is NOT submittable.")
    return 0


def cmd_check(args):
    try:
        doc = load(args.check)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"cannot read ledger: {exc}", file=sys.stderr)
        return 2
    errors, records = validate(doc, args.check)
    if args.json:
        print(json.dumps({"path": args.check, "valid": not errors, "errors": errors,
                          "record_count": len(records)}, indent=2))
    else:
        counts = {}
        for r in records:
            if isinstance(r, dict):
                counts[r.get("verdict", "?")] = counts.get(r.get("verdict", "?"), 0) + 1
        print(f"ledger: {args.check}")
        print(f"records: {len(records)}  " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        if errors:
            print(f"\nINVALID — {len(errors)} error(s):")
            for e in errors:
                print(f"  - {e}")
        else:
            print("VALID")
    return 1 if errors else 0


def cmd_summary(args):
    try:
        doc = load(args.summary)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"cannot read ledger: {exc}", file=sys.stderr)
        return 2
    errors, records = validate(doc, args.summary)
    print(f"# Findings ledger — {doc.get('target', '?')} ({doc.get('run_id', '?')})")
    for verdict, label in (("confirmed", "CONFIRMED (submittable)"),
                           ("needs_validation", "NEEDS VALIDATION (NOT submittable)"),
                           ("rejected", "REJECTED")):
        rows = [r for r in records if isinstance(r, dict) and r.get("verdict") == verdict]
        print(f"\n## {label} — {len(rows)}")
        for r in rows:
            sev = f" [{r.get('severity')}]" if r.get("severity") else ""
            print(f"  - {r.get('fingerprint', '?')}{sev}: {r.get('title', '')}")
            for u in r.get("unresolved", []) or []:
                print(f"      unresolved: {u}")
    print(f"\nvalidator: {'VALID' if not errors else f'INVALID ({len(errors)} errors)'}")
    return 1 if errors else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--new", metavar="TARGET", help="scaffold a ledger under ~/Dev/REPORTS/<TARGET>/")
    g.add_argument("--check", metavar="PATH", help="validate a ledger")
    g.add_argument("--summary", metavar="PATH", help="print a triage summary")
    ap.add_argument("--force", action="store_true", help="overwrite with --new")
    ap.add_argument("--run-id", default=None, help="run id for --new")
    ap.add_argument("--json", action="store_true", help="machine-readable output for --check")
    args = ap.parse_args(argv)
    if args.new:
        return cmd_new(args)
    if args.check:
        return cmd_check(args)
    return cmd_summary(args)


if __name__ == "__main__":
    sys.exit(main())
