---
name: ai-crime-watch-series
description: Use when running the AI Crime Watch bi-weekly series.
---

# AI Crime Watch — bi-weekly series runbook

Series home: ML Kenya blog, repo `/home/pro-g/ProG/ml-ke` (Jekyll + Chirpy, see `blog-drafting` skill for ALL formatting/fact-check/publish rules — load it alongside this one).

**Identity:** wholesome, solutions-first collation on crime & AI. Every post highlights a concrete problem, then ends with what is being done / what each reader can do. No fear-mongering, no victim-blaming, no hypotheticals.

## The five pillar questions (issue content budget)

1. How are criminals (cybercriminals, cartels, scammers — NOT any specific scoped program) using AI?
2. How can law enforcement curb this?
3. How can the public stay vigilant?
4. How does this change how law enforcement should investigate criminals?
5. Which AI/ML research helps enforcement — or gives criminals an edge?

Cover at least 3 pillars per issue, rotating emphasis so all five recur across issues; never force all five if the fortnight did not produce for them. Pillar 1 is the hook, pillar 5 is the differentiator most outlets miss.

**⏸ PAUSED 2026-09-16 — do not run without an explicit request.** The user
pivoted the blog to positive AI content ("we've been diving deep into the scary
part of AI"); the engine job `d75da864fce0` and finisher `8ea5a3a5de4d` are
paused (not deleted) and the daily blog cron now writes positive-AI stories and
tutorials instead. Do not open a cycle, publish an issue, or resume those jobs
unless the user asks for the series back.

## Cadence, release rule, mechanics

- Bi-weekly engine cron fires every 14 days (~Wed, off-peak after 13:00 EAT). It ALWAYS runs research, regardless of other daily blog work (user rule: "research every two weeks in spite of what we've been creating that day").
- **Release only if the ready-gate passes** — otherwise the issue goes to the finisher window and, failing that, the cycle is skipped with a logged reason. Skipping is a valid outcome.
- Jobs (do not create/delete these from inside a cron run — they are pre-created and monitor-gated):
  - `AI Crime Watch engine (bi-weekly)` — research + release decision every 14 days.
  - `AI Crime Watch finisher` — daily 16:30 EAT, gated by `~/.hermes/scripts/ai_crime_watch_signal.py` (monitor): output `PENDING <issue> <due>` when STATE.md status=pending, else `IDLE`. Agent runs only when that output changes, so the finisher sleeps for free when idle and wakes only inside a polish window. No cron removal needed: an issue is "closed" when STATE returns to a non-pending status, which makes the signal `IDLE` again (one confirming wake after publish is expected and desirable).

### State files (all under repo `_ai-crime-watch/`, git-tracked, NOT published by Jekyll)

- `STATE.md` — machine-parseable top block (`status:` idle|pending|published|skipped, `cycle:`, `issue:` (next issue number), `due:` YYYY-MM-DD, `draft:` path) + human cycle log below. Only the top block drives the signal script.
- `ANCHORS.md` — append-only verified anchor bank for the series (name, date, figures, 2+ source URLs). Future cycles dedupe here FIRST.
- `cycles/cycle-N/` — per-cycle workdir: `sources.md` (candidates + URLs + dedupe notes), `draft.md` (the post draft), plus gate outcome recorded in STATE.

### Cycle state machine

1. Engine fire: read STATE. If previous cycle `pending` and `due` has NOT passed → do not open a new cycle; report and stop (finisher still owns it). If previous cycle pending and due PASSED → supersede: log cycle as skipped (reason: polish deadline expired), fold reusable sources into the new cycle, increment cycle.
2. Open cycle N (new `cycles/cycle-N/`), research, draft, ready-gate.
3. READY → publish (below). Set STATE `status: published`, keep cycle; next engine fire opens the next cycle and `issue` increments at publish time.
4. NOT READY → STATE `status: pending`, `due: <fire day + 3 days>`. Finisher wakes next day 16:30.
5. Finisher (pending only): polish/verify → if READY publish (status published); if still not ready and today < due: update draft, keep pending, report blockers; if today >= due: status skipped + reason, close cycle.

## Ready-gate (MANDATORY — "concrete and unique")

CONCRETE (all must hold):
- >= 1 anchor development from the last ~21 days (incident, arrest/indictment, policy, paper, vendor report) squarely on crime×AI or enforcement×AI.
- Every factual claim source-backed at BODY level by >= 2 independent sources (blog-drafting fact-check protocol; drop unverifiable claims — never soften, never keep hypotheticals).
- Dates + figures pinned, anchor listed in the calendar note when published.

UNIQUE (all must hold):
- Anchor/angle not already used in any previous AI Crime Watch issue (check ANCHORS.md).
- Not the centerpiece of an existing `_posts/` post since the last issue (grep `_posts/` headings + recent AI Update roundups + daily posts). If the anchor is taken, keep a DIFFERENT anchor or state a clearly differentiated angle (e.g. same incident viewed through the public-vigilance lens is NOT unique; the LE-investigation lens may be).
- Not a rehash of what every outlet covered the same week — the collation must ADD the LE / public / research edge.

If the fortnight genuinely produced nothing concrete+unique: mark cycle skipped with an honest short report ("no release this cycle: …"). That is the designed behaviour, not a failure.

## Issue anatomy & tone (wholesome, multi-audience)

1. **Open on the human problem** — one specific, verified, recent example; plain language; specific but not sensational. 2-3 sentences.
2. **The evidence collation** — what happened across the fortnight, verified, links inline, global + African balance (blog global-coverage mandate). Define jargon on first use.
3. **Audience lenses (pick those the fortnight supports, 2-4):**
   - **For the public** — vigilance playbook: concrete behaviours (verify channels, slow down, reporting routes like local police/cybercrime units, Interpol notices), no victim-blaming, agency framing.
   - **For law enforcement** — how cases/investigations change: new evidence types (synthetic media, prompt logs, model outputs), new tooling, jurisdictional/attribution challenges, what to collect first.
   - **For researchers & policymakers** — the ML research edge: which research helps defense (deepfake detection, provenance, anomaly detection, fraud ML, synthetic-media attribution) and which lowers crime cost (open abuse tools, voice clones, jailbreak kits); policy hooks (EU AI Act, AU convention, national strategies).
4. **End on solutions + agency** — what is working (seizures, takedowns, prosecutions, tools, laws), a short "how we can do better" for each audience, a hopeful forward line. Never end on the problem.
5. References section + related posts (markdown links only).

Target ~1,100–1,500 full-body words (see blog-drafting word-band notes). Front matter: `categories: [AI Security]`, `image.path: /assets/img/cover-<slug>.webp`, date `00:00:00 +0300` on publish day (never future). Slug: `ai-crime-watch-issue-<N>`; check slug uniqueness with `ls _posts/ | sed 's/^[0-9-]*//; s/\.md$//' | sort | grep -xF 'ai-crime-watch-issue-<N>'` before writing.

## Cover (per issue, signature series art)

Same balanced-scales composition every issue (series branding, like a newspaper masthead) but the headline, issue badge and accent change → visually distinct files. Generate with the skill script, then verify:

```bash
python3 ~/.hermes/skills/creative/ai-crime-watch-series/scripts/issue_cover.py \
  <slug> <issue_no> "<headline up to ~70 chars>" "<YYYY-MM-DD>" "<short subtitle>" \
  /home/pro-g/ProG/ml-ke
# writes assets/blog/cover-<slug>.svg AND assets/img/cover-<slug>.webp (ffmpeg)
ls -la /home/pro-g/ProG/ml-ke/assets/img/cover-<slug>.webp   # must exist
```

Headline must avoid `&` or any XML-entity char (script escapes, but keep text clean).

## Research sources (rotate; minimum 6-10 searches per cycle)

- Enforcement & policy: INTERPOL (AI & cybercrime reports, AFRIPOL), Europol (IOCTA), FBI IC3 annual + press releases, DOJ press (AI-adjacent indictments), UK NCSC/NCA, UNODC, EU AI Act LE provisions, Kenya: DCI + ODPC + CBK reports, AU Convention on Cyber Security.
- Industry & press: The Hacker News, BleepingComputer, The Register, TechCrunch, Reuters (AI crime tag), Chainalysis/TRM (crypto + AI laundering), ESET/Kaspersky/Group-IB threat reports, OpenAI disinfo reports, Google TAG, Microsoft DSC.
- Academic/ML edge: arXiv cs.CR / cs.CL / cs.LG (deepfake detection, media provenance C2PA, LLM forensics & authorship, jailbreak/abuse, AI for investigations), Google Scholar, sophon.at/papers, NIST/NIJ AI for LE.
- Search starters per pillar: "AI fraud scam arrests", "deepfake scam ring", "AI voice clone fraud case", "AI phishing campaign report", "police AI investigation deepfake", "INTERPOL AI crime", "law enforcement generative AI forensic", "AI crime research paper", "prosecutor AI evidence deepfake", "cartel AI", "romance scam AI translation", "AI money laundering crypto". Search with recent time filters; verify every anchor against the primary or two secondaries at body level.

## Publishing steps (engine or finisher, on READY)

1. `git pull origin main` in repo; confirm slug free; write `_posts/YYYY-MM-DD-ai-crime-watch-issue-N.md` (date `00:00:00 +0300` TODAY — cron fires after 13:00 so never future).
2. Generate + verify cover WebP; front matter must use `image:` block with `.webp` path.
3. Run blog-drafting fact-check protocol: no `{% post_url %}`, no `cover:` key, no `.png` image path, all `/posts/` links resolve, every claim 2+ sources. ~1,100-1,500 words.
4. `git add -A` (post + cover svg/webp + STATE/ANCHORS/calendar changes together), commit `Add post: AI Crime Watch #N — <short>`, `git pull --rebase`, push. Verify: `git log -1 --oneline`, then Actions API run success (`curl -s "https://api.github.com/repos/ml-ke/ml-ke/actions/runs?per_page=1"`). Post permalink may 404 up to ~85 s — do not report failure on that.
5. Update STATE (status published, log line), append anchor lines to ANCHORS.md, add one row to editorial-calendar.md series section.
6. Do NOT touch `.scheduled/` (daily blog-poster owns it) and never schedule series posts on a Tuesday (tue-ai-update owns Tuesdays). The series post is an EXTRA post on its day; the daily post still runs.

## Pitfalls (learned)

- Future dates silently skip the Jekyll build; midnight-EAT dates only.
- Never publish an anchor verified at snippet level only (Reddit bot-wall etc.) — phrase conservatively or drop.
- Duplicate anchors with the daily blog/AI Update are the #1 uniqueness killer: grep `_posts/` headings before committing to an anchor.
- Fresh cron sessions have no chat context: prompts must be self-contained; every run reads STATE.md FIRST and never assumes.
- The finisher signal must stay deterministic: STATE machine block only, no timestamps in output.
- Engine fires ~creation time every 14 days; if a fire lands inside a still-open pending window (due not passed) it reports and stops instead of opening a new cycle.

## Out-of-scope

This is a general-interest collation — NOT bug-bounty target research. No specific vendor/program scoping, no attack instructions. Problem→solution framing only.
