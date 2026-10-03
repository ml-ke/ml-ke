#!/usr/bin/env python3
"""FINAL verification of the extracted GENE-LINK submissions."""
import csv
import glob
import hashlib
import os

import olefile

SRC = "/home/pro-g/Downloads/gene-link/RFP Submissions/GL RFP Applications"
DST = "/home/pro-g/Downloads/gene-link/RFP Submissions/GL RFP Applications - Extracted"
rows = list(csv.DictReader(open(os.path.join(DST, "index.csv"))))
src_files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(SRC, "*.msg")))

checks = {}
# 1. one folder per source file, nothing dropped or duplicated
folders = sorted(d for d in os.listdir(DST) if os.path.isdir(os.path.join(DST, d)))
checks["folder count == source count"] = (len(folders), len(src_files), len(folders) == len(src_files))
checks["every source has a folder"] = sorted(r["source_file"] for r in rows) == src_files

# 2. required files per folder
req_missing = []
for r in rows:
    p = os.path.join(DST, r["folder"])
    for f in ("EMAIL.md", "EMAIL_BODY.txt", "EMAIL_BODY.html",
              "original_message.msg", "attachments"):
        if not os.path.exists(os.path.join(p, f)):
            req_missing.append((r["folder"], f))
checks["all folders complete"] = (len(req_missing) == 0, req_missing)

# 3. originals unmodified
bad_copy = []
for r in rows:
    src = os.path.join(SRC, r["source_file"])
    h_src = hashlib.sha256(open(src, "rb").read()).hexdigest()
    h_cpy = hashlib.sha256(open(os.path.join(DST, r["folder"], "original_message.msg"), "rb").read()).hexdigest()
    if not (h_src == h_cpy == r["sha256"]):
        bad_copy.append(r["folder"])
checks["originals copied intact (sha256)"] = (len(bad_copy) == 0, bad_copy)

# 4. bodies == PR_BODY (after CRLF normalisation) + attachments == raw streams
body_ok = body_bad = att_ok = att_bad = 0
body_problems, att_problems = [], []
for r in rows:
    ole = olefile.OleFileIO(os.path.join(SRC, r["source_file"]))
    raw_body, blobs = None, []
    for e in ole.listdir():
        last = e[-1].upper()
        if not last.startswith("__SUBSTG1.0_"):
            continue
        tag = last.replace("__SUBSTG1.0_", "")
        d = ole.openstream(e).read()
        if tag.startswith("1000001F") and raw_body is None:
            raw_body = d
        if tag.startswith("3701") or not tag[:4].isalpha():
            blobs.append(d)
    ole.close()
    txt = (raw_body[2:] if raw_body[:2] == b"\xff\xfe" else raw_body).decode("utf-16-le").rstrip("\x00")
    txt = txt.replace("\r\n", "\n").replace("\r", "\n")
    written = open(os.path.join(DST, r["folder"], "EMAIL_BODY.txt"), encoding="utf-8").read()
    if written == txt:
        body_ok += 1
    else:
        body_bad += 1
        body_problems.append(r["folder"])
    pool = {}
    for d in blobs:
        pool[hashlib.md5(d).hexdigest()] = pool.get(hashlib.md5(d).hexdigest(), 0) + 1
    for n in (r["attachment_names"].split(" | ") if r["attachment_names"] else []):
        f = os.path.join(DST, r["folder"], "attachments", n)
        h = hashlib.md5(open(f, "rb").read()).hexdigest()
        if pool.get(h):
            pool[h] -= 1
            att_ok += 1
        else:
            att_bad += 1
            att_problems.append((r["folder"], n))
checks["bodies match source .msg"] = (f"{body_ok}/{len(rows)}", body_problems)
checks["attachments match source .msg"] = (f"{att_ok} matched, {att_bad} bad", att_problems)

# 5. attachments on disk == index
disk_total = 0
for r in rows:
    disk = sorted(os.listdir(os.path.join(DST, r["folder"], "attachments")))
    disk_total += len(disk)
    if disk != sorted(n for n in r["attachment_names"].split(" | ") if n) or len(disk) != int(r["attachments"]):
        checks.setdefault("index/disk mismatch", []).append(r["folder"])
checks["attachment files on disk"] = (disk_total, "index total", sum(int(r["attachments"]) for r in rows))
checks["no empty files"] = not [f for f in glob.glob(os.path.join(DST, "*", "*")) if os.path.isfile(f) and os.path.getsize(f) == 0]

# 6. sender / subject accuracy vs .msg
import extract_msg
meta_bad = []
for r in rows:
    with extract_msg.openMsg(os.path.join(SRC, r["source_file"])) as m:
        if (m.subject or "").strip() != r["subject"]:
            meta_bad.append(("subject", r["folder"]))
        frm = str((m.header or {}).get("From", "") or m.sender or "")
        if r["from_email"] and r["from_email"].lower() not in frm.lower():
            meta_bad.append(("from", r["folder"], frm, r["from_email"]))
checks["subjects/senders accurate"] = (len(meta_bad) == 0, meta_bad)

for k, v in checks.items():
    print(f"{k:38} {v}")

tot = sum(os.path.getsize(os.path.join(dp, f)) for dp, dn, fn in os.walk(DST) for f in fn)
print(f"\ntotal output size: {tot/1024/1024:.1f} MB across {len(folders)} folders")
