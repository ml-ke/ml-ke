#!/usr/bin/env python3
"""Extract Outlook .msg files into one folder per message.

Each folder gets: EMAIL.md (sender/recipients/dates/headers/attachment list),
EMAIL_BODY.txt, EMAIL_BODY.html, attachments/, original_message.msg.
Plus INDEX.md, index.csv at the top level.

Usage:
    python msg_extract.py --src <dir> [--dst <dir>]

Requires: extract-msg  (pip install extract-msg)
"""
import argparse
import csv
import datetime as dt
import email.header
import email.utils
import glob
import hashlib
import mimetypes
import os
import re
import shutil
import sys

import extract_msg

ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
EXT_BY_MIME = {
    "application/pdf": ".pdf", "image/png": ".png", "image/jpeg": ".jpg",
    "text/plain": ".txt", "application/zip": ".zip", "application/msword": ".doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/vnd.ms-excel": ".xls",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
}
LINK_PAT = (r"https?://[^\s<>\"',)]*(?:drive\.google|docs\.google|dropbox|1drv\.ms|"
            r"onedrive|sharepoint|wetransfer|mega\.nz|box\.com|transfernow|files\.fm|"
            r"sendspace)[^\s<>\"',)]*")


def safe(name, maxlen=110, fallback="untitled"):
    """Filesystem-safe name: strip illegal chars, collapse space, cap length."""
    if not name:
        return fallback
    name = ILLEGAL.sub("-", str(name))
    name = re.sub(r"\s+", " ", name.replace("\r", " ").replace("\n", " ").replace("\t", " ")).strip()
    name = name.strip(". ")
    if len(name) > maxlen:
        stem, ext = os.path.splitext(name)
        name = stem[: maxlen - len(ext)] + ext
    return name or fallback


def unique(path):
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    n = 2
    while os.path.exists(f"{stem} ({n}){ext}"):
        n += 1
    return f"{stem} ({n}){ext}"


def addr(raw):
    if not raw:
        return ("", "")
    n, e = email.utils.parseaddr(str(raw).replace("\r", " ").replace("\n", " "))
    return (n.strip(), e.strip())


def unfold(value):
    """Collapse RFC5322 header folding into single spaces."""
    return re.sub(r"\r?\n[ \t]+", " ", str(value)).strip()


def decode_name(value):
    """Decode =?UTF-8?B?...?= display names; leave addresses alone."""
    if not value or value == "(none)":
        return value
    try:
        return str(email.header.make_header(email.header.decode_header(unfold(value)))).strip()
    except Exception:
        return unfold(value)


def header_val(msg, key):
    try:
        return str((msg.header or {}).get(key, "") or "").strip()
    except Exception:
        return ""


def body_of(msg):
    """Plain-text body; extract-msg handles the RTF/HTML fallbacks."""
    for attr in ("body", "rtfBody"):
        try:
            v = getattr(msg, attr, None)
        except Exception:
            v = None
        if isinstance(v, bytes):
            v = v.decode("utf-8", "replace")
        if v and str(v).strip():
            return str(v)
    return ""


def save_attachment(att, adir, log):
    long_n = getattr(att, "longFilename", None) or ""
    short_n = getattr(att, "shortFilename", None) or ""
    try:
        data = att.data
    except Exception as exc:
        log.append(f"  ! could not read data for {long_n or short_n}: {exc}")
        return None
    if data is None:
        return None
    if not isinstance(data, (bytes, bytearray)):
        data = bytes(data)
    fname = safe(long_n or short_n or "", fallback="")
    mime = (getattr(att, "mimetype", None) or "").split(";")[0].strip()
    if not fname:
        fname = "attachment" + (EXT_BY_MIME.get(mime) or mimetypes.guess_extension(mime) or ".bin")
    if not os.path.splitext(fname)[1]:
        fname += EXT_BY_MIME.get(mime) or mimetypes.guess_extension(mime) or ""
    out = unique(os.path.join(adir, fname))
    with open(out, "wb") as fh:
        fh.write(data)
    log.append(f"  - {os.path.basename(out)}  ({len(data):,} bytes)")
    return out, len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="directory containing .msg files")
    ap.add_argument("--dst", default=None, help="output directory (default: <src> - Extracted)")
    args = ap.parse_args()
    src_dir = os.path.abspath(args.src)
    dst_dir = os.path.abspath(args.dst or src_dir.rstrip("/") + " - Extracted")

    files = sorted(glob.glob(os.path.join(src_dir, "*.msg")) +
                   glob.glob(os.path.join(src_dir, "*.MSG")))
    if not files:
        sys.exit(f"no .msg files found in {src_dir}")

    # pass 1: read dates so folders are numbered in send order
    meta = []
    for path in files:
        try:
            with extract_msg.openMsg(path) as m:
                date = m.date
                if isinstance(date, dt.datetime):
                    key = date.astimezone(dt.timezone.utc).replace(tzinfo=None) if date.tzinfo else date
                else:
                    key = dt.datetime.min
                meta.append({"path": path, "subject": (m.subject or "").strip() or "(no subject)",
                             "key": key})
        except Exception as exc:
            meta.append({"path": path, "subject": "(unreadable)", "key": dt.datetime.min,
                         "error": str(exc)})
    meta.sort(key=lambda r: (r["key"], os.path.basename(r["path"])))
    os.makedirs(dst_dir, exist_ok=True)

    records = []
    for i, info in enumerate(meta, 1):
        src = info["path"]
        folder = unique(os.path.join(dst_dir, f"{i:02d} - {safe(info['subject'])}"))
        os.makedirs(folder, exist_ok=True)
        adir = os.path.join(folder, "attachments")
        os.makedirs(adir, exist_ok=True)
        try:
            with extract_msg.openMsg(src) as msg:
                subj = (msg.subject or "").strip() or "(no subject)"
                date = msg.date
                f_name, f_email = addr(header_val(msg, "From") or msg.sender)
                r_name, r_email = addr(header_val(msg, "Reply-To"))
                from_disp = decode_name(f_name or r_name)
                from_addr = f_email or r_email
                to = decode_name(unfold(header_val(msg, "To") or (msg.to or "")))
                cc = decode_name(unfold(header_val(msg, "Cc") or (msg.cc or "")))
                bcc = decode_name(unfold(header_val(msg, "Bcc") or (msg.bcc or "")))
                body = body_of(msg)
                try:
                    html_raw = msg.htmlBody or b""
                except Exception:
                    html_raw = b""
                raw_headers = "\n".join(f"{k}: {v}" for k, v in (msg.header or {}).items())

                att_rows, log = [], []
                for a in msg.attachments:
                    lf = getattr(a, "longFilename", None) or getattr(a, "shortFilename", None) or ""
                    if type(a).__name__ == "MessageAttachment":
                        base = safe(lf or "embedded-message")
                        if not base.lower().endswith(".msg"):
                            base += ".msg"
                        out = unique(os.path.join(adir, base))
                        d = a.data
                        with open(out, "wb") as fh:
                            fh.write(d if isinstance(d, (bytes, bytearray)) else bytes(d))
                        att_rows.append((os.path.basename(out), os.path.getsize(out), "embedded message"))
                    else:
                        res = save_attachment(a, adir, log)
                        if res:
                            att_rows.append((os.path.basename(res[0]), res[1],
                                             getattr(a, "mimetype", "") or ""))

                with open(os.path.join(folder, "EMAIL_BODY.txt"), "w", encoding="utf-8") as fh:
                    fh.write(body if body.strip() else "(no plain-text body in this message)")
                if html_raw:
                    with open(os.path.join(folder, "EMAIL_BODY.html"), "wb") as fh:
                        fh.write(html_raw if isinstance(html_raw, bytes) else html_raw.encode("utf-8"))
                shutil.copy2(src, os.path.join(folder, "original_message.msg"))
                sha = hashlib.sha256(open(src, "rb").read()).hexdigest()

                urls = sorted(set(re.findall(LINK_PAT, body, re.I)))
                lines = [f"# {subj}", "", "## Sender details", "",
                         f"- **From (name):** {from_disp or '(not supplied)'}",
                         f"- **From (email):** {from_addr or '(not supplied)'}",
                         f"- **Reply-To:** {decode_name(unfold(header_val(msg, 'Reply-To'))) or '(none)'}",
                         f"- **Return-Path:** {header_val(msg, 'Return-Path') or '(none)'}", "",
                         "## Recipients", "", f"- **To:** {to or '(none)'}",
                         f"- **Cc:** {cc or '(none)'}", f"- **Bcc:** {bcc or '(none)'}", "",
                         "## Message details", "", f"- **Subject:** {subj}",
                         f"- **Date (message):** {date.isoformat() if isinstance(date, dt.datetime) else date or '(none)'}",
                         f"- **Date (header):** {header_val(msg, 'Date') or '(none)'}",
                         f"- **Message-ID:** {header_val(msg, 'Message-ID') or '(none)'}",
                         f"- **Original file:** {os.path.basename(src)}",
                         f"- **Original SHA-256:** {sha}",
                         f"- **Body size:** {len(body):,} characters", "", "## Attachments", ""]
                lines += ([f"- `attachments/{n}` — {sz:,} bytes" + (f" ({mt})" if mt else "")
                           for n, sz, mt in att_rows] or ["_None._"])
                if urls:
                    lines += ["", "## File-sharing links found in the body", ""]
                    lines += [f"- {u}" for u in urls]
                lines += ["", "## Body files", "",
                          "- `EMAIL_BODY.txt` — the message's plain-text body part (line endings normalised CRLF to LF)",
                          "- `EMAIL_BODY.html` — body with original formatting; may be rendered from the RTF body when the message carries no HTML part",
                          "", "## Raw message headers", "", "```", raw_headers or "(none)", "```", ""]
                with open(os.path.join(folder, "EMAIL.md"), "w", encoding="utf-8") as fh:
                    fh.write("\n".join(lines))

                records.append({"folder": os.path.basename(folder), "subject": subj,
                                "date": date.isoformat(sep=" ") if isinstance(date, dt.datetime) else "",
                                "from_name": from_disp, "from_email": from_addr, "to": to, "cc": cc,
                                "attachments": len(att_rows),
                                "attachment_names": " | ".join(r[0] for r in att_rows),
                                "body_chars": len(body), "source_file": os.path.basename(src),
                                "sha256": sha, "link_only": "yes" if urls and not att_rows else "",
                                "errors": ""})
                print(f"[{i:02d}/{len(meta)}] {os.path.basename(folder)} ({len(att_rows)} attachment(s))")
                for line in log:
                    print(line)
        except Exception as exc:
            print(f"[{i:02d}/{len(meta)}] FAILED {os.path.basename(src)}: {exc}")
            records.append({"folder": os.path.basename(folder), "subject": info["subject"], "date": "",
                            "from_name": "", "from_email": "", "to": "", "cc": "", "attachments": 0,
                            "attachment_names": "", "body_chars": 0,
                            "source_file": os.path.basename(src), "sha256": "", "link_only": "",
                            "errors": str(exc)})

    with open(os.path.join(dst_dir, "index.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)

    md = ["# Extracted messages", "", f"Source: `{src_dir}`  ",
          f"Messages: {len(records)}  ",
          f"Attachments: {sum(r['attachments'] for r in records)}  ",
          f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}", "",
          "| # | Sent | From | Email | Subject | Att. | Folder |", "|---|---|---|---|---|---|---|"]
    for i, r in enumerate(records, 1):
        md.append(f"| {i} | {r['date'][:16]} | {r['from_name'] or '—'} | {r['from_email'] or '—'} | "
                  f"{r['subject'].replace('|', chr(92) + '|')} | {r['attachments']} | `{r['folder']}` |")
    md += ["", "## Attachments by message", ""]
    for r in records:
        md.append(f"**{r['folder']}**")
        md += ([f"- {n}" for n in r["attachment_names"].split(" | ")] or ["- (none)"])
        md.append("")
    md += ["## Messages with no attachment", ""]
    for r in records:
        if int(r["attachments"]) == 0:
            md.append(f"- {r['folder']} ({r['from_email']})" +
                      (" — **documents behind a link, check the body**" if r["link_only"] else ""))
    with open(os.path.join(dst_dir, "INDEX.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))

    print(f"\nDONE: {len(records)} folders -> {dst_dir}")


if __name__ == "__main__":
    main()
