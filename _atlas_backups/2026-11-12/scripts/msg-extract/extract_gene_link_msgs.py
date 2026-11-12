#!/usr/bin/env python3
"""Extract Outlook .msg files: one folder per message containing
sender details, email text, attachments, and the untouched original.

Run with the venv that has extract-msg installed.
"""
import csv
import datetime as dt
import email.utils
import glob
import hashlib
import mimetypes
import os
import re
import shutil
import sys

import extract_msg

SRC = "/home/pro-g/Downloads/gene-link/RFP Submissions/GL RFP Applications"
DST = "/home/pro-g/Downloads/gene-link/RFP Submissions/GL RFP Applications - Extracted"

ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
EXT_BY_MIME = {
    "application/pdf": ".pdf", "image/png": ".png", "image/jpeg": ".jpg",
    "text/plain": ".txt", "application/zip": ".zip", "application/msword": ".doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/vnd.ms-excel": ".xls",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
}


def safe(name, maxlen=110, fallback="untitled"):
    if not name:
        return fallback
    name = ILLEGAL.sub("-", str(name))
    name = name.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    name = re.sub(r"\s+", " ", name).strip()
    name = name.strip(". ")            # Windows/KDE dislike trailing dots/spaces
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
    """Return (display_name, email) from a header value like 'Name <a@b.c>'."""
    if not raw:
        return ("", "")
    n, e = email.utils.parseaddr(str(raw).replace("\r", " ").replace("\n", " "))
    return (n.strip(), e.strip())


def header_val(msg, key):
    try:
        h = msg.header
        if not h:
            return ""
        return str(h.get(key, "") or "").strip()
    except Exception:
        return ""


def body_of(msg):
    """Plain-text body, preferring the real plain-text part."""
    for attr in ("body", "rtfBody"):
        try:
            v = getattr(msg, attr, None)
        except Exception:
            v = None
        if isinstance(v, bytes):
            v = v.decode("utf-8", "replace")
        if v and str(v).strip():
            return str(v)
    html = b""
    try:
        html = msg.htmlBody or b""
    except Exception:
        pass
    if html:
        text = html.decode("utf-8", "replace") if isinstance(html, bytes) else str(html)
        text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", text)
        text = re.sub(r"(?i)<br\s*/?>", "\n", text)
        text = re.sub(r"(?i)</(p|div|tr|li|h[1-6])>", "\n", text)
        text = re.sub(r"<[^>]+>", " ", text)
        import html as _h
        text = _h.unescape(text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
        return text.strip()
    return ""


def save_attachment(att, adir, log):
    """Save one attachment; returns relative path or None."""
    long_n = getattr(att, "longFilename", None) or ""
    short_n = getattr(att, "shortFilename", None) or ""
    data = None
    try:
        data = att.data
    except Exception as exc:
        log.append(f"  ! could not read data for {long_n or short_n}: {exc}")
    fname = safe(long_n or short_n or "", fallback="")
    if data is None:
        # e.g. embedded message already saved by caller
        return None
    if not isinstance(data, (bytes, bytearray)):
        data = bytes(data)
    if not fname:
        mime = (getattr(att, "mimetype", None) or "").split(";")[0].strip()
        fname = "attachment" + (EXT_BY_MIME.get(mime) or mimetypes.guess_extension(mime) or ".bin")
    if not os.path.splitext(fname)[1]:
        mime = (getattr(att, "mimetype", None) or "").split(";")[0].strip()
        fname += EXT_BY_MIME.get(mime) or mimetypes.guess_extension(mime) or ""
    out = unique(os.path.join(adir, fname))
    with open(out, "wb") as fh:
        fh.write(data)
    log.append(f"  - {os.path.basename(out)}  ({len(data):,} bytes)")
    return out, len(data)


def main():
    files = sorted(glob.glob(os.path.join(SRC, "*.msg")))
    records = []

    # ---- pass 1: read metadata so folders can be ordered by date ----
    meta = []
    for path in files:
        try:
            with extract_msg.openMsg(path) as m:
                date = m.date
                if isinstance(date, dt.datetime) and date.tzinfo:
                    date_key = date.astimezone(dt.timezone.utc).replace(tzinfo=None)
                elif isinstance(date, dt.datetime):
                    date_key = date
                else:
                    date_key = dt.datetime.min
                meta.append({
                    "path": path,
                    "subject": (m.subject or "").strip() or "(no subject)",
                    "date": date,
                    "date_key": date_key,
                })
        except Exception as exc:
            print(f"PARSE FAIL {os.path.basename(path)}: {exc}", file=sys.stderr)
            meta.append({"path": path, "subject": "(unreadable)", "date": None,
                         "date_key": dt.datetime.min, "error": str(exc)})

    meta.sort(key=lambda r: (r["date_key"], os.path.basename(r["path"])))
    os.makedirs(DST, exist_ok=True)

    for i, info in enumerate(meta, 1):
        src = info["path"]
        src_name = os.path.basename(src)
        folder_name = f"{i:02d} - {safe(info['subject'])}"
        folder = unique(os.path.join(DST, folder_name))
        os.makedirs(folder, exist_ok=True)
        adir = os.path.join(folder, "attachments")
        log = []
        try:
            with extract_msg.openMsg(src) as msg:
                subj = (msg.subject or "").strip() or "(no subject)"
                date = msg.date
                f_name, f_email = addr(header_val(msg, "From") or msg.sender)
                r_name, r_email = addr(header_val(msg, "Reply-To"))
                from_disp = f_name or r_name
                from_addr = f_email or r_email
                to = header_val(msg, "To") or (msg.to or "")
                cc = header_val(msg, "Cc") or (msg.cc or "")
                bcc = header_val(msg, "Bcc") or (msg.bcc or "")
                message_id = header_val(msg, "Message-ID")
                return_path = header_val(msg, "Return-Path")
                date_hdr = header_val(msg, "Date")
                body = body_of(msg)
                try:
                    html_raw = msg.htmlBody or b""
                except Exception:
                    html_raw = b""

                # raw headers
                raw_headers = ""
                try:
                    if msg.header:
                        raw_headers = "\n".join(
                            f"{k}: {v}" for k, v in msg.header.items())
                except Exception:
                    pass

                # attachments
                os.makedirs(adir, exist_ok=True)
                att_rows = []
                for a in msg.attachments:
                    cls = type(a).__name__
                    lf = getattr(a, "longFilename", None) or getattr(a, "shortFilename", None) or ""
                    if cls == "MessageAttachment":
                        sub = getattr(a, "data", None)
                        base = safe(lf or "embedded-message")
                        if not base.lower().endswith(".msg"):
                            base += ".msg"
                        out = unique(os.path.join(adir, base))
                        with open(out, "wb") as fh:
                            fh.write(sub if isinstance(sub, (bytes, bytearray)) else bytes(sub))
                        log.append(f"  - {os.path.basename(out)}  ({os.path.getsize(out):,} bytes, embedded message)")
                        att_rows.append((os.path.basename(out), os.path.getsize(out), "embedded message"))
                    else:
                        res = save_attachment(a, adir, log)
                        if res:
                            att_rows.append((os.path.basename(res[0]), res[1],
                                             getattr(a, "mimetype", "") or ""))
                if not att_rows:
                    log.append("  (no attachments)")

                # ---- write files ----
                with open(os.path.join(folder, "EMAIL_BODY.txt"), "w",
                          encoding="utf-8") as fh:
                    fh.write(body if body.strip() else "(no plain-text body in this message)")
                if html_raw:
                    with open(os.path.join(folder, "EMAIL_BODY.html"), "wb") as fh:
                        fh.write(html_raw if isinstance(html_raw, bytes) else html_raw.encode("utf-8"))
                shutil.copy2(src, os.path.join(folder, "original_message.msg"))

                sha = hashlib.sha256(open(src, "rb").read()).hexdigest()
                lines = [
                    f"# {subj}",
                    "",
                    "## Sender details",
                    "",
                    f"- **From (name):** {from_disp or '(not supplied)'}",
                    f"- **From (email):** {from_addr or '(not supplied)'}",
                    f"- **Reply-To:** {header_val(msg, 'Reply-To') or '(none)'}",
                    f"- **Return-Path:** {return_path or '(none)'}",
                    "",
                    "## Recipients",
                    "",
                    f"- **To:** {to or '(none)'}",
                    f"- **Cc:** {cc or '(none)'}",
                    f"- **Bcc:** {bcc or '(none)'}",
                    "",
                    "## Message details",
                    "",
                    f"- **Subject:** {subj}",
                    f"- **Date (message):** {date.isoformat() if isinstance(date, dt.datetime) else date or '(none)'}",
                    f"- **Date (header):** {date_hdr or '(none)'}",
                    f"- **Message-ID:** {message_id or '(none)'}",
                    f"- **Original file:** {src_name}",
                    f"- **Original SHA-256:** {sha}",
                    f"- **Body size:** {len(body):,} characters",
                    "",
                    "## Attachments",
                    "",
                ]
                if att_rows:
                    for n, sz, mt in att_rows:
                        lines.append(f"- `attachments/{n}` — {sz:,} bytes"
                                     + (f" ({mt})" if mt else ""))
                else:
                    lines.append("_None._")
                lines += ["", "## Raw message headers", "", "```", raw_headers or "(none)", "```", ""]
                with open(os.path.join(folder, "EMAIL.md"), "w", encoding="utf-8") as fh:
                    fh.write("\n".join(lines))

                records.append({
                    "folder": os.path.basename(folder),
                    "subject": subj,
                    "date": date.isoformat(sep=" ") if isinstance(date, dt.datetime) else "",
                    "from_name": from_disp,
                    "from_email": from_addr,
                    "to": to,
                    "cc": cc,
                    "attachments": len(att_rows),
                    "attachment_names": " | ".join(r[0] for r in att_rows),
                    "body_chars": len(body),
                    "source_file": src_name,
                    "sha256": sha,
                    "errors": "",
                })
                print(f"[{i:02d}/{len(meta)}] {os.path.basename(folder)}  "
                      f"({len(att_rows)} attachment(s))")
                for line in log:
                    print(line)
        except Exception as exc:
            print(f"[{i:02d}/{len(meta)}] FAILED {src_name}: {exc}")
            records.append({
                "folder": os.path.basename(folder), "subject": info["subject"], "date": "",
                "from_name": "", "from_email": "", "to": "", "cc": "", "attachments": 0,
                "attachment_names": "", "body_chars": 0, "source_file": src_name,
                "sha256": "", "errors": str(exc),
            })

    # ---- index files ----
    csv_path = os.path.join(DST, "index.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)

    md = ["# GENE-LINK RFP — extracted submissions", "",
          f"Source: `{SRC}`  ", f"Messages processed: {len(records)}  ",
          f"Total attachments: {sum(r['attachments'] for r in records)}  ",
          f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}  ", "",
          "| # | Date received | From | Email | Subject | Att. | Folder |",
          "|---|---|---|---|---|---|---|"]
    for i, r in enumerate(records, 1):
        d = r["date"][:16]
        md.append(f"| {i} | {d} | {r['from_name'] or '—'} | {r['from_email'] or '—'} | "
                  f"{r['subject'].replace('|', '/')} | {r['attachments']} | `{r['folder']}` |")
    md += ["", "## Attachments by message", ""]
    for r in records:
        md.append(f"**{r['folder']}**")
        if r["attachment_names"]:
            for n in r["attachment_names"].split(" | "):
                md.append(f"- {n}")
        else:
            md.append("- (none)")
        md.append("")
    if any(r["errors"] for r in records):
        md += ["## Errors", ""]
        for r in records:
            if r["errors"]:
                md.append(f"- `{r['source_file']}`: {r['errors']}")
    with open(os.path.join(DST, "INDEX.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))

    print(f"\nDONE: {len(records)} folders -> {DST}")
    print(f"      index.csv + INDEX.md written")


if __name__ == "__main__":
    main()
