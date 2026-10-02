#!/usr/bin/env python3
"""Export all SMS from an ADB-connected Android device WITHOUT root and without installing an app.

Why this works: the `adb shell` user (uid 2000) holds READ_SMS, so the SMS content provider is
queryable. Call logs are NOT accessible this way (shell lacks READ_CALL_LOG) — do not claim them.

Usage: python3 export_sms_content_provider.py <serial> <outdir>

Parsing note (the part that bites): `content query` prints one record per `Row: N ` marker, and the
`body` column can contain commas AND newlines. Keep `body` as the LAST projection field and treat
everything after the final ' body=' as the message text; do not split fields naively on ', '.
"""
import collections
import csv
import datetime
import json
import os
import re
import subprocess
import sys

TYPES = {"1": "inbox", "2": "sent", "3": "draft", "4": "outbox", "5": "failed", "6": "queued"}
PROJ = "_id,thread_id,address,date,date_sent,type,read,body"


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    serial, out = sys.argv[1], os.path.expanduser(sys.argv[2])
    os.makedirs(out, exist_ok=True)

    raw = subprocess.run(
        ["adb", "-s", serial, "shell", "content", "query", "--uri", "content://sms", "--projection", PROJ],
        capture_output=True, text=True,
    ).stdout
    if not raw.strip():
        print("no output — is the device connected/authorized?")
        return 1
    open(os.path.join(out, "sms-raw.txt"), "w", encoding="utf-8").write(raw)

    records = []
    for chunk in re.split(r"(?=^Row: \d+ )", raw, flags=re.M):
        chunk = chunk.strip()
        if not chunk.startswith("Row:"):
            continue
        m = re.match(r"Row: \d+ (.*)$", chunk, flags=re.S)
        body = m.group(1) if m else chunk
        lead, sep, text = body.partition(" body=")
        rec = {k: v for k, v in re.findall(r"([A-Za-z_]+)=(.*?)(?=, [A-Za-z_]+=|$)", lead, flags=re.S)}
        rec["body"] = text if sep else ""
        records.append(rec)

    for r in records:
        r["type_name"] = TYPES.get(r.get("type", ""), r.get("type", ""))
        try:
            r["date_iso"] = datetime.datetime.fromtimestamp(int(r["date"]) / 1000).isoformat(" ")
        except Exception:
            r["date_iso"] = ""

    json.dump(records, open(os.path.join(out, "sms.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    with open(os.path.join(out, "sms.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["_id", "thread_id", "address", "date_iso", "type_name", "read", "body"],
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(records)

    print(f"messages: {len(records)}")
    print("by type:", dict(collections.Counter(r["type_name"] for r in records)))
    print("top correspondents:", collections.Counter(r.get("address", "") for r in records).most_common(8))
    ds = [r["date_iso"] for r in records if r["date_iso"]]
    if ds:
        print(f"date range: {min(ds)} .. {max(ds)}")
    print(f"saved sms.json / sms.csv / sms-raw.txt in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
