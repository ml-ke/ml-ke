---
name: folder-per-item-archives
description: Use when reorganising or re-indexing an extracted archive.
version: 1.0.0
author: Nous Research
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [archive, index, extraction, deliverables, traceability, reorganise]
    category: productivity
    related_skills: [outlook-msg-extraction, ocr-and-documents, xlsx]
---

# Folder-per-item archives: building, indexing, splitting

## When to Use

- You produced (or inherited) a folder-per-item archive — one folder per email,
  document, submission, download or record, plus an index.
- The user asks to split, re-group, re-label or re-index such an archive
  ("move the inquiries to a different folder", "separate the shortlisted ones").
- Any request for counts, lists or titles drawn from an archive you generated.

For the mechanics of parsing a specific container format, use the format's own
skill (e.g. `outlook-msg-extraction` for `.msg`). This skill governs the archive
layer around it: layout, IDs, indexes, handover and later reorganisation.

## The layout contract

One folder per item, a stable numeric prefix, and an untouched copy of the source
inside each folder:

```
<NN> - <sanitised title>/
    <the item's content, in both machine- and human-readable form>
    <original untouched copy of the source>
attachments/ or files/     the payload the item carried
INDEX.md                   table of every item, for humans
index.csv                  the same data, for spreadsheets
ALL_*.txt                  every item's full text concatenated, for straight reading
README.md                  layout, provenance, accuracy notes, decided boundaries
```

- **Number by the order the items arrived** (send date, download date), not
  alphabetically, so the numbering matches how the user remembers them.
- **Keep the numbers stable forever.** They are the shared vocabulary between you
  and the user after the first report ("msg 34", "submission 12"). Renumbering to
  close gaps silently invalidates every earlier reference — leave the gaps.
- **Keep the source directory untouched.** Work on copies; say so in README.md.
- **Sanitise titles for the filesystem, never in the record.** Replace characters
  illegal in paths, keeping the exact original title inside the item's own record
  file and in the index.

## Never trust a stored index

An index is a claim about the past. Before quoting any total, list, or count to the
user, and before rebuilding anything, re-derive from the filesystem
(`os.listdir` / `os.walk`) and compare.

- **Do not glob paths built from item titles.** Titles contain `[`, `]`, `*` and
  `?`, which glob reads as pattern syntax — your own verification will report every
  file as missing. Use literal directory listing for anything derived from data.
- **An index-vs-disk difference is news, not corruption.** The commonest cause is
  the user working in the archive between your runs: downloading the linked package
  you flagged, dropping a CV in, renaming a file. Identify the added files, record
  them (a `notes` field), fold them into the counts, and mention the correction.
  Ask about ownership; do not present it as a failure of your own output.
- **The output tree is shared space.** Anything you promised as "N files" is only
  true at the moment you measured it.

## Splitting an archive into groups

Expect this request. Do it without breaking traceability:

1. **Move the folders; never re-extract or rebuild them.** The per-item records and
   checksums inside are already verified — a rebuild discards that work.
2. **Keep the original numbers**, gaps and all. Offer contiguous renumbering as an
   option; do not apply it unasked.
3. **Create a sibling directory**, not a subdirectory, when the point is to make the
   original archive contain only one kind of item — a nested folder still travels
   with it when the user zips or copies the parent.
4. **Regenerate every index artifact for both sets** (`index.csv`, `INDEX.md`,
   `ALL_*.txt`, `README.md` naming the other set and the split rule) from the
   filesystem, not from the pre-split index.
5. **Write one master index at the parent level**: number → category → current
   folder path. That single file is what keeps "msg 34" resolvable from either side.
6. **State the classification basis in the report, and name the boundary cases** —
   which items you counted as what, and any that could reasonably go the other way.
7. **Verify the accounting**: group A + group B == the original total; every folder
   still has its record, payload directory and original copy; no master-index path
   points at a folder that no longer exists; the source inputs are all still there.

## Classifying items in a batch (triage boundaries)

When splitting submissions from noise, judge by what the item *carries*, and say
where you placed the ambiguous ones:

| Case | Treat as | Why |
|---|---|---|
| Carries the required documents | submission | Self-evident. |
| Documents behind a link, nothing attached | submission, flagged | It is a submission; the payload just lives elsewhere. Surface the link. |
| Confirms or verifies another submission | submission | It belongs with the record it verifies, not with the questions. |
| Promises the documents "in a follow-up" that never arrived | inquiry, flagged | Not a submission yet — say so, it is an open item for the user. |
| Questions, deadline queries, expressions of interest | inquiry | No documents. |
| Duplicate delivery of the same item | both kept, flagged | Same Message-ID/hash = one item delivered twice; do not silently drop one. |

## Pitfalls

- **List titles from the item's own record, not the folder name.** Folder names have
  illegal characters substituted, so they will not match what the sender wrote. If a
  title differs from its folder name, say why — never let the two silently disagree.
- **Count attachments as files on disk, not rows in an index.** The two diverge the
  moment the user adds anything.
- **Report additions you did not make as the user's work.** Framing their edits as
  an anomaly wastes a turn and reads as an accusation.
- **Do not create a per-batch skill.** Archive handling is one class of task however
  many different batches pass through; the batch's own specifics belong in its README.
- **Write archive scripts to a file and run the file — do not inline Python in a
  `python -c '…'` passed through the shell.** Nested quotes, f-strings and the
  titles being processed (they carry `|`, `—`, apostrophes) make the shell eat the
  quoting; the symptom is a *shell* syntax error pointing at a line of your Python,
  which reads as a code bug when it is a quoting bug. `write_file` the script, then
  run it with the same interpreter you installed the parser into.
