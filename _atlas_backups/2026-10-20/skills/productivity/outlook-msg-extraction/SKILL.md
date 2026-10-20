---
name: outlook-msg-extraction
description: Use when extracting Outlook .msg files to folders.
version: 1.0.0
author: Nous Research
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [outlook, msg, email, extract-msg, olefile, attachments, productivity]
    category: productivity
    related_skills: [ocr-and-documents, pdf, xlsx]
---

# Outlook .msg extraction into per-message folders

## When to Use

- A directory of unopenable Outlook `.msg` files needs to be read.
- The user wants "email text, sender details and attachments" pulled out of messages.
- Any request to convert `.msg` mail to a browsable folder-per-message archive.
- Also use for one-off `parser.eml`-style asks where MAPI/compound-file parsing is needed.

Deliverable: one folder per message, each holding sender details, the email text,
the attachments, and an untouched copy of the original, plus a top-level index.

## Setup

`extract-msg` is not installed by default; install into a dedicated venv (do not
pollute the Hermes python — `python3` on PATH is Hermes's own 3.14):

```bash
cd ~/.hermes/cache/scratch
/usr/bin/python3.12 -m venv msgvenv
./msgvenv/bin/pip install extract-msg olefile   # olefile is needed for verification
```

## Run

`scripts/msg_extract.py` does the whole job (stdlib + extract-msg):

```bash
./msgvenv/bin/python scripts/msg_extract.py --src "<dir with .msg>" --dst "<output dir>"
```

Output layout:

```
<NN> - <subject>/
    EMAIL.md              sender, recipients, dates, Message-ID, attachment list, raw headers, sha256
    EMAIL_BODY.txt        the message text
    EMAIL_BODY.html       the message text with original formatting
    attachments/          every attached document
    original_message.msg  untouched copy (provenance)
INDEX.md, index.csv, README.md
```

Folders are numbered in send-date order so numbering is stable and sortable.

## Verify before reporting (non-negotiable)

Read the raw `.msg` container with `olefile` and check the deltas — do not trust
the extractor's own output:

- **Body**: stream `__substg1.0_1000001F` (PR_BODY). It is **UTF-16LE**; decode,
  then compare to `EMAIL_BODY.txt` after `\r\n` → `\n`. Expect an exact match.
- **Attachments**: payload streams are `__substg1.0_3701*` inside
  `__attach_version1.0_#0000000N/` directories — **not** at the root of the
  compound file. Compare md5 of each saved file against md5 of the stream data.
- **Originals**: sha256 of `original_message.msg` must equal the source file.
- **Counts**: folders == source files; attachments on disk == index total.
- Open every extracted PDF (pymupdf) to prove it is not truncated. PDFs reporting
  "no text layer" are scans — report page counts, not a failure.

## Pitfalls (each of these produced a wrong answer once)

- **Do not glob folder paths you built from a subject.** Subjects contain `[` and `]`
  (e.g. `[Landscape Alliance & ElevenX] …`), which glob treats as a character
  class — your own verification will report every file as missing. Use
  `os.listdir` / `os.walk` for anything derived from message data.
- **`extract-msg` normalises CRLF to LF** in the body. That is the only difference
  from PR_BODY; verify after normalising, and state the normalisation in the report
  rather than claiming a byte-identical match.
- **Do not assume an HTML body part exists.** Of 44 real messages, 0 carried a
  usable HTML part: 39 had no `PR_HTML` stream at all and the other 5 had an 8-byte
  binary object reference under the `10130102` tag. `msg.htmlBody` is then
  *rendered from the RTF body* — label it as reconstructed, and point the user at
  the `.txt` / original `.msg` as authoritative.
- **Subjects in the file name are mangled by Outlook.** Illegal path characters
  (`/ \ : * ? " < > |`) became `_`, so `GENE-LINK MVP RFP _ Clarification …` is
  really `… RFP | Clarification …`. Always read the subject from the message, and
  sanitise for the folder name yourself (`-` is a safe stand-in), keeping the exact
  subject in EMAIL.md.
- **Header display names may be MIME-encoded** (`=?UTF-8?B?…?=`) and headers are
  folded across lines. Decode with `email.header` and unfold for the display fields
  while leaving the verbatim header block untouched.
- **Duplicate subjects are not duplicates.** Compare Message-ID and sha256 before
  calling anything a duplicate; two files with one Message-ID are one email
  delivered twice, and same-subject/different-ID messages are distinct.
- **`msg.date` is timezone-aware**; sort on `astimezone(utc)` so folders number in
  the order the recipient would have seen them.
- **Empty `attachments/` can be meaningful.** Grep bodies for
  drive.google/dropbox/onedrive/wetransfer links: a "proposal submission" with zero
  attachments often means the package sits behind a link the user still has to
  fetch. Surface those leads.
- Message bodies are HTML-only or RTF-only surprisingly often — always save a
  browser-openable version too.

## Classifying and re-organising after extraction

When the user asks to split a batch (applications vs inquiries vs other):

- **Read the bodies; attachment count is not evidence of a submission.** A file in
  `attachments/` is often just a signature logo (`image.png`, `Greensighter.png`),
  a company brochure or a profile — expressions of interest and deadline-extension
  requests carry those too. Classify from what the body says, and cite that reason
  in the index so the call is auditable.
- **Expect a third bucket.** Messages that are neither the thing wanted nor the thing
  excluded exist (e.g. a note confirming someone else sent a submission). Putting
  them in either folder breaks the "only X in folder X" guarantee the user asked for;
  give them their own folder and say why.
- **Renumber with two phases** (move to `__moving_<old>` temp names, then rename to
  final `NN - label`) or a folder can collide with another folder's current name.
  Parse `label` by splitting the OLD folder name once on `" - "` and reusing it, so
  the subject sanitisation and any truncation already applied are preserved.
- **Always carry the old number forward** — a `old_number` column in index.csv and a
  master index mapping old→new. Reports quoted the previous numbers, so throwing
  them away breaks the user's own cross-references.
- **Renumbering invalidates every generated artifact.** After any move/rename, rebuild
  index.csv, INDEX.md, ALL_EMAILS.txt and README for *each* set from what is on disk,
  then verify: numbering contiguous, folders == rows, and every source `.msg`
  accounted for exactly once (assert no source appears in two folders).
- **An index goes stale the moment the user works in the folders.** If a count looks
  wrong, list the actual directory before "fixing" anything — the user may have
  dropped in files (e.g. downloading a linked Drive package into `attachments/`), and
  the right response is to recount from disk and note the addition, not to move or
  delete their files.
