---
name: skill-corpus-cartography
description: "Use when mapping or auditing a whole skill corpus."
version: 1.0.0
metadata:
  hermes:
    tags: [skills, corpus, graph, louvain, tf-idf, visualization, audit]
    category: software-development
    related_skills: [skill-quality-audit, hermes-agent-skill-authoring, find-skills]
---

# Skill corpus cartography

Map an entire skill/plugin corpus as a graph: what clusters together, what duplicates what,
what is isolated, and which descriptions break the trigger convention. Produces an
interactive offline HTML map plus a findings report.

Artifacts live in `~/Dev/REPORTS/skills-atlas/` (html + png + json + pipeline scripts).

## When to Use

- Asked to map, visualise, or audit the whole skill library rather than one skill.
- After installing or importing a batch of skills from an external repo.
- When skill sprawl is suspected: duplicated topics, unreachable/isolated skills,
  descriptions that never fire.
- Periodically, to compare corpus drift against the baseline metrics below.

Not for reviewing a single SKILL.md — that is `skill-quality-audit`.

## Pipeline (no numpy/scipy/networkx needed — pure stdlib + PyYAML)

1. **Parse.** Walk the skills root, skip `.archive`/`.git`/`node_modules`. Split YAML
   frontmatter with `raw.split('---', 2)`. Dedupe by name, keeping the longest body.
2. **Features.** `desc*8 + headings*3 + first 500 body tokens`, stopword-filtered;
   TF-IDF (`log((N+1)/(df+1))+1`), drop IDF <= 1.6, L2-normalise. Sparse dicts + a
   `cos()` that iterates the smaller dict — 201 nodes is ~20k pairs, instant.
3. **Graph.** Keep each node's top-k (k=5) neighbours, drop w < 0.06. Threshold-only graphs
   (e.g. cosine > 0.12) badly overstate isolation — that run produced 23 "islands" where
   kNN(5) produced 3. Always use kNN, not a bare threshold.
4. **Cluster.** Hand-rolled Louvain: initialise each node to its own community, sweep nodes
   in random order moving each to the neighbour community with the best modularity gain
   (`w_i - resolution*tot_c*k_i/m2`), compress labels, repeat until no move. Sweep
   resolution 0.6–2.0 and keep the highest Q. See `skillmap2.py` for working code.
5. **Layout.** Two-level, do NOT use one flat force sim — it collapses into an unreadable
   blob because real corpora interlink across clusters. Instead: coarse-grid candidates,
   place island anchors biggest-first with greedy first-fit requiring
   `dist >= r_i + r_j + gap` where `r = 6*sqrt(n) + 16`; then run a per-island force sim
   (repulsion + springs + hard collision + anchor pull). Dim cross-island edges to
   ~0.05 alpha so islands read as islands.
6. **Render.** Single self-contained HTML, vanilla JS, no CDN (must work offline).
7. **Still image.** `google-chrome --headless=new --disable-gpu --no-sandbox --hide-scrollbars
   --force-device-scale-factor=1.5 --window-size=1600,1000 --virtual-time-budget=6000
   --screenshot=out.png file:///abs/path.html`

## Canvas/Javascript pitfalls (each cost real debugging time)

- **`canvas.width = x` CLEARS the canvas, and so does any late `resize` event.** If the sim
  has already frozen (alpha 0, needsDraw false) a post-load resize blanks the map permanently.
  Always set `needsDraw = true` inside `resize()`.
- **Declare `needsDraw` before `resize()` is first called** — an assignment inside `resize`
  hits the TDZ and throws at load if the `let` comes later in the file.
- **Don't double-apply the camera origin.** If seeds are placed around `W/2,H/2` and
  `toScreen` also adds `W/2`, the whole graph renders offset into one quadrant.
  Initialise `cam.x = W/2, cam.y = H/2` (or drop the offset from both).
- **Hit-test in SCREEN space, not world space.** A world-unit grab radius shrinks with zoom
  (14 world units = ~10px at k=0.7) and clicking stops working. Convert the node to screen
  and use a constant ~11px radius.
- **Pre-settle before first paint** (run N ticks synchronously) so the map is laid out on
  load and screenshots are deterministic.
- **Auto-fit the camera after settling**, and reserve a HUD-safe top margin (an ~110px stats
  block will overlap island labels otherwise). A 982px-wide canvas cannot hold 32 labels —
  use short `term (n)` labels in 3 staggered vertical bands, and draw text halos
  (`strokeText` dark, then `fillText`) so labels stay readable over dense clusters.
- Only label hub nodes (`degree >= 12`) plus hover/selected/search-hit nodes.
- Hover/click feedback plus a legend filter is what makes the map useful rather than pretty.

## Baseline: 2026-09-30 measurement of this profile's corpus

Compare future runs against these to detect drift.

- 201 active skills, 2.54 MB, 341,727 words. Median 1,278 words; mean 1,700.
- Top 5 skills = 18.7% of all words (`research-paper-writing`, `recon-to-exploitation`,
  `api-hacking-methodology`, `blog-drafting`, `api-bug-bounty-methodology`).
- **32 communities, Louvain Q = 0.6144** (resolution 0.6).
- Biggest communities: bounty/recon/graphql 25, github 17, docker/hermes 17, model/comfyui 11.
- **Most connected skill: `hermes-agent` (degree 25)** — runtime docs are the connective tissue.
- **Trigger hygiene: only 20/201 (10%) descriptions open with `Use when <trigger>.`**
  Worst: hodaripay 0/10, mlops 0/9, apple/media/github 0/5. The compliant set is all
  recently authored, so the convention is adopted but never back-filled. Highest-value
  mechanical fix available in the corpus.
- 13 near-duplicate pairs at cosine >= 0.42; the real ones: `saml-attacks`/`saml-attack-techniques`,
  `computer-use`/`macos-computer-use`, `choicebank-baas`/`hodaripay-choicebank`,
  `ocr-and-documents`/`pdf`, `codex`/`kanban-codex-lane`,
  `android-device-firmware-update`/`samsung-firmware-flash-linux`, plus a 4-node wiki tangle
  (`github-wiki-publishing`, `wiki-build-with-parallel-agents`, `llm-wiki`, `documentation-corpus-build`).
- `category` is a broken axis: 49 values, 27 with exactly one skill (root-level skills inherit
  their own directory name). Use community labels for browsing instead.

## Reporting

Write findings to `FINDINGS.md` next to the artifacts: method (with parameters), the
quantified findings above, and a prioritised action list. Push the summary to chat with the
PNG and HTML as attachments — never inline the whole thing into a 4096-char message.
