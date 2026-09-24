# Worldwide Agent-Skills Learning Sweep (Methodology)

How to run the recurring "learn from other AI agents worldwide" sweep. Used weekly by
cron job `3ded7d48e350` (every Tue 15:00 EAT) and usable on demand. The goal is
compounding improvement: each run should leave the system slightly better.

## Reputability ranking (state it in every report)

1. **GitHub official-org repos** (anthropics, openai, trailofbits, openclaw, microsoft, cloudflare, NVIDIA, snyk) — highest
2. **arxiv papers** — peer-review-grade, empirical findings
3. **Community GitHub repos with 1000+ stars** — community-validated
4. **Blogs / Medium** — opinion; verify any claim against code/repos before adopting

Never adopt a claim from a low-reputability source without verification.

## Key repos discovered by sweeps (updated Aug 10 2026)

Top new finds from the Aug 10 2026 ATLAS sweep (all community-validated, 1000+★):
- **affaan-m/ECC** (239K★) — agent harness optimization system. Meta-skills worth studying: `agent-architecture-audit` (12-layer stack diagnosis), `agent-eval`, `agent-introspection-debugging`, `agent-harness-construction`. Curation pattern to copy: **"When to Activate" + "Do not use for"** sections in every skill (adds negative space so the agent doesn't over-trigger). Adopted into atlas-continuous-learning Aug 2026.
- **mukul975/Anthropic-Cybersecurity-Skills** (27.5K★) — 817 structured cybersecurity skills mapped to 6 frameworks (MITRE ATT&CK, NIST CSF 2.0, MITRE ATLAS, D3FEND, NIST AI RMF, MITRE F3). agentskills.io standard. Huge library — useful as reference, too big to install wholesale.
- **NVIDIA/SkillSpector** (14.4K★) — security scanner for agent skills. 68 patterns / 17 categories, risk score 0-100. Research baseline: **26.1% of agent skills contain vulnerabilities, 5.2% show likely malicious intent**. Installed at ~/.hermes/venvs/skillsec/ (v2.8.2). Scan cmd: `skillspector scan <dir> --recursive --no-llm --format json`. Note: --no-llm mode reports risk_score but risk_level shows '?'; false positives on security-tooling content are heavy (curl = "Data Exfiltration", BOM char = "Prompt Injection", official install scripts = "Supply Chain").
- **snyk/agent-scan** (2.9K★, pip snyk-agent-scan v0.5.16) — prompt-injection/malware scanner for agents, MCP, skills. CAUTION: running it starts stdio MCP servers (needs --dangerously-run-mcp-servers in CI mode). Prefer SkillSpector for static-only scans.
- **0xNyk/awesome-hermes-agent** (5.2K★) — independent directory of Hermes skills/plugins/memory providers. Hermes-specific finds: hermes-dojo (self-improvement: monitors agent performance, iterates weak skills), hermes-skill-factory (auto-generates skills from workflows), hermes-incident-commander (autonomous SRE), super-hermes (meta-reasoning), oh-my-hermes (multi-agent orchestration: ralplan = Planner→Architect→Critic, ralph = verified execute→verify→iterate, triage, autopilot), blacktea (x402 payment controls), personal-api (Obsidian vault as identity layer).
- **OthmanAdi/planning-with-files** (26K★) — persistent file-based planning, crash-proof markdown plans, session recovery after compaction. Topics include hermes-skill.
- **microsoft/skills** (2.9K★) — official MS skills for SDK grounding.
- **mattpocock/skills** (212K★) — "Skills for Real Engineers" from .agents directory.
- **uphiago/recon-skills** (1K★) — recon & pentest skill pack with hermes-agent topic (CORS, XSS, SQLi, SSRF, RCE, WordPress, MCP, cloud, subdomain takeover). MIT.
- **mksglu/context-mode** (19.7K★) — context window optimization, sandboxes tool output. Lesson adopted (Aug 12 2026): **"Think in Code"** — the agent should program the analysis, not read raw data into context. One script that computes + logs only results replaces 10-50 tool calls (100x context saving). Also: aggressive brevity prompts can degrade reasoning benchmarks — enforce where DATA goes, not how the model talks.
- **microsoft/SkillOpt** (15.9K★, arxiv 2605.23904) — SKILL.md as trainable state: rollout → reflect → aggregate → select → update → evaluate, accepted only on held-out validation score. Deployed artifact = compact `best_skill.md` (300-2,000 tokens), transfers across models/harnesses (GPT-5.5 +23.5 direct / +24.8 Codex / +19.1 Claude Code). Validates ATLAS rule: verify before adopting, keep skills compact. SkillOpt-Sleep = nightly offline harvest→mine→replay→consolidate behind a validation gate — same shape as this weekly sweep cron.
- **SnailSploit/Claude-Red** (2.9K★) — curated offensive-security skill library for Claude Code.
- **kursku/skills** (BR, 2.3K★) — Brazilian catalog of 2,300+ Claude skills (marketing, vendas, SEO, dev) — regional discoverability example.
- **jnMetaCode/superpowers-zh** (CN, 7.6K★) — Chinese localization of superpowers + 6 original CN skills — regional localization example.
- **skillsmp.com** (SkillsMP, 2M+ skills indexed from public GitHub, multi-language UI) — cross-region discovery/search/compare-before-install for SKILL.md; use as a discovery index, always verify the actual repo before adopting.
- **Threekiii/Awesome-Redteam** (CN, ~8K★) — 攻防知识库 red team knowledge base (found via Chinese-language search).
- **CyberStrikeus/CyberStrike** (1.7K★) — open-source AI-augmented offensive security harness, 13+ autonomous agents.
- **gadievron/raptor** (3.5K★) — Claude Code offensive/defensive security agent.

## Source techniques

### GitHub API (most valuable — real repos, star counts)
```bash
curl -s "https://api.github.com/search/repositories?q=agent+skills&sort=stars&order=desc&per_page=10" \
  | python3 -c "import json,sys; [print(f\"{r['stargazers_count']:>6}★ | {r['full_name']:50} | {(r.get('description') or '')[:80]}\") for r in json.load(sys.stdin).get('items',[])]"
```
Useful queries: `agent skills`, `claude skills`, `opencode skills`, `openclaw skills`,
`pentest skills`, `security agent skills`, `SKILL.md`.
Read the README + 1-2 actual SKILL.md files from the top repos (raw.githubusercontent.com).

### arxiv (empirical research — smell taxonomies, benchmarks, security studies)
Search arxiv.org for: `agent skills LLM`, `SKILL.md empirical`, `agent skills security`.
Extraction pattern (curl + strip tags, no browser needed):
```bash
curl -sL "https://arxiv.org/html/<ID>" | python3 -c "
import sys, html, re
text = re.sub(r'<script[^>]*>.*?</script>', '', sys.stdin.read(), flags=re.DOTALL)
text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
text = re.sub(r'<[^>]+>', ' ', text)
print(re.sub(r'\s+', ' ', html.unescape(text)))"
```
Known payoffs: arxiv 2607.01456 = 26-smell SKILL.md taxonomy (→ skill-quality-audit),
arxiv 2602.12430 = agent skills security survey.

### Regional language searches (the worldwide angle — talent beyond US/Europe)
Use web_search with non-English queries to find repos authors didn't market in English:
- Japanese: `エージェントスキル`, `Claude Code スキル`
- Chinese: `智能体技能`, `AI Agent技能库`, `渗透测试 知识库`
- Russian: `навыки ИИ-агентов`
- Portuguese: `habilidades de agentes de IA`
- Spanish: `habilidades de agentes de IA`
- Korean: `AI 에이전트 스킬`
- Arabic: `مهارات وكلاء الذكاء الاصطناعي`
Real find: wgpsec/AboutSecurity (China, 1,625★) — 200+ pentest skills in agent-executable format.

### Medium / blogs
Lowest reputability. Only use for ideas, then verify against actual repos/code.

## Skill pipeline (mandatory maintenance step — added Aug 10 2026)

The weekly cron runs this BEFORE research. Run it on demand too:
1. BRIDGE: `python3 ~/.hermes/scripts/opencode_skills_bridge.py` — re-syncs curated Hermes methodology skills into OpenCode (~/.config/opencode/skills/atlas-*). Run after ANY Hermes skill change so OpenCode always sees the latest.
2. VALIDATE: `python3 ~/.hermes/scripts/skill_validator.py ~/.config/opencode/skills` AND `... ~/.hermes/skills --hermes` — frontmatter checks (name regex, name==dir, description 1-1024, body <5000 words). Fix errors introduced; warnings are informational.
3. SECURITY SCAN: `~/.hermes/venvs/skillsec/bin/skillspector scan ~/.config/opencode/skills --recursive --no-llm --format json --output /tmp/skillspector-weekly.json` — scans installed skills for injection/exfiltration/supply-chain risk. Do NOT run snyk-agent-scan unattended (it starts stdio MCP servers). Report the scan verdict in the final report.
4. SEAM CHECK (added Sep 22 2026): `python3 ~/.hermes/scripts/skill_root_seam_check.py` — flags skill names present in more than one root OpenCode loads, and divergent shadowed copies (edits that are silently inert). Verified real: one shadowed copy had drifted and contradicted a DNS-verified fact.
5. SECOND OPINION (added Sep 22 2026): `python3 ~/.hermes/scripts/skill_second_opinion.py --days 7 --limit 10` — LLM-assisted scan (cisco-ai-defense/skill-scanner on PyPI + DeepSeek) over the *changed* skills only. A single static scanner's "0 findings" is not evidence (arxiv 2609.17274). No API key or binary → exits 0 with a skip line; never a blocker.

## Cron prompt pitfall (verified Aug 10 2026)

The cronjob threat filter BLOCKS prompts matching `prompt_injection` patterns. When writing/updating a cron prompt that describes security-scan triage rules, do NOT include literal payload examples (verbatim instruction-override strings, remote-code-pipe-to-shell commands, base64 blobs) — the filter rejects the whole update. Phrase triage rules abstractly ("verbatim instruction-override text", "remote-code-pipe-to-shell of attacker-controlled URLs", "unexplained encoded command blobs"). This applies to ANY security-flavored cron prompt, not just this sweep.

## Sep 22 2026 sweep — collection-level seams, registry-scale verdicts, `gh skill`

Highest-value finds (all empirical, arxiv, Sep 2026):
- **2609.13321 SkillSeam — Six Principles for Auditing Agent Skill Collections.** Audits the
  *relationships* between skills, not files: persistence gradient, system coherence, regime
  gating, orthogonal coverage, flow, granularity discipline — each with a measured failure
  channel (flattened hierarchy +60% loaded tokens; dangling anchor +64% tokens / −3.1pp;
  synonymous alias noncanonical routes 0/32→15/32; overlapping lanes ownership conflicts
  0/16→14/16; bland triggers 3/32→30/32 conflicts +3.7× tokens; granularity mis-mix −12.5pp,
  the largest drop). Extracted to `skill-quality-audit/references/collection-level-seams.md`
  + new Step 7 in that skill.
- **2609.17274 "After the Party" (OpenClaw registry governance).** 61,990 skills: the three
  scanners **disagreed on 23,702**; post-adjudication sensitivity **21.67%–61.06%**; 85.06% of
  readable skills carry privilege evidence; 77.86% have zero stars/comments; top 10% took
  46.93% of downloads and no simple metadata predicted survival. → a single scanner's "0
  findings" means *unmeasured*, not *clean*.
- **2609.14079 SkillSecurer** — nine injection threat types, LLM in the loop, the only compared
  scanner at 100% detection; latent injection risk in **>17%** of sampled popular skills.
  → LLM-assisted pass added to our pipeline for the *changed* set only.
- **2609.09233 Subagents vs Agent Skills** — for long-horizon work, invoking a skill package as
  a subagent (fresh context) beats loading its instructions into the main context, *when the
  skill exposes a clear input/output contract*; cost is coordination tokens. Justifies the
  `delegate_task` + `output_schema` pattern.
- **2609.07360 "Scanning the Harness"** — supply-chain defects live in agent *configurations*
  (MCP definitions, instruction files), not only in skills. Watch for next run.

New discovery surface — **`gh skill` (GitHub CLI ≥ 2.98.0, preview)**: `search`, `preview`,
`install`, `list`, `update`, `publish`. `gh skill search <term>` is the cheapest official
skill search (returns repo, path, description, popularity count) and surfaces non-English
descriptions. **Gotcha:** `gh skill install` defaults to `~/.agents/skills/` or
`<project>/.agents/skills/`, which the host agent may never load — install with `--agent
<host>` and verify discoverability.

Found via `gh skill search`: `project-hellhound-org/bounty-hunter` (full BB workflow incl.
LLM/AI security ASI01-ASI10, A-to-B chaining, bypass tables), `zebbern/claude-code-guide`
(api-fuzzing-bug-bounty), `elementalsouls/Claude-OSINT` (2.6K★, 80 secret-regex patterns),
`cisco-ai-defense/skill-scanner` (2.5K★, second independent scanner — now installed),
`majiayu000/claude-skill-registry` (incl. Russian-language skills).

Multi-root seam finding (this machine, verified with `opencode debug skill`): OpenCode loads
**seven** skill locations and dedupes by `name` with the global config root winning —
`~/.config/opencode/skills` (225) beats `~/.agents/skills` (115); 58 names existed in both, 55
byte-identical and 3 divergent. One divergent shadow was **newer and provably wrong** (it
reintroduced `api.choicebankapi.com` as the live ChoiceBank base URL; DNS-checked 2026-09-22:
NXDOMAIN, while `baas.choicedigitalbank.com` resolves with a valid cert). The shadow was
backed up to `~/.agents/skills-backup-2026-09-22/` and removed. Detector:
`~/.hermes/scripts/skill_root_seam_check.py` (run in the weekly pipeline).

known-external validator errors (do not re-investigate): `use-railway` description is 1041
chars (>1024) and `railway-use-railway`/`use-railway` are an upstream duplicate pair — both are
`gh skill`-managed third-party skills, so edits would be overwritten by `gh skill update`.

Hermes operating notes (Sep 22):
- **Checkpoints are OFF by default** (opt-in v2; shadow git store under `~/.hermes/checkpoints/store/`
  shared across projects, one checkpoint per directory per turn, `hermes checkpoints status|prune` to
  manage). With them off, any unattended file removal must carry its own explicit backup — that is why
  the shadowed-skill removal above used `cp -a` to `~/.agents/skills-backup-<date>/` first. Consider
  `checkpoints.enabled: true` if unattended destructive file work increases.
- **`hermes egress` / iron-proxy** (Docker backend only) is the correct answer to "a prompt-injected
  agent in a sandbox reads `~/.hermes/.env`": the sandbox holds opaque proxy tokens and the host
  daemon swaps in the real credential after TLS termination. Modal/Daytona/SSH/Singularity do not
  receive proxy env vars yet. Not applicable to this local-backend host — recorded as the design
  whenever sandboxing is added.

## Extract → Apply → Record pipeline

1. For each source capture: (a) what it does differently from Hermes, (b) one concrete
   adoptable thing, (c) URL/repo for reference.
2. Apply only genuinely useful, verifiable improvements — quality over churn. Patch an
   existing skill, create a class-level skill, or improve a script in ~/.hermes/scripts/.
3. Audit anything new with `skill-quality-audit` (26-smell taxonomy) before finishing.
4. Append dated lessons to `~/Dev/ATLAS-LEARNINGS/LESSONS.md` (see atlas-lesson-bank
   skill). Save a full dated report to `~/Dev/ATLAS-LEARNINGS/YYYY-MM-DD.md`.
5. Also study Hermes docs each run (hermes-agent skill + https://hermes-agent.nousresearch.com/docs/)
   and adopt at least one operating improvement (config option, tool pattern, workflow).

## Report format (to user)
1. Learned items (3-5, each with source + reputability)
2. Changes/implementations and why
3. Sources scanned (with the worldwide angle — which countries/regions)
4. Repos worth watching next run
5. Hermes docs studied + operating improvement adopted

## Sep 8 2026 sweep — new evidence & watch list (replaces stale star counts above)

arxiv papers this run (all ~10K context, empirical — highest new-value sources):
- **2608.23067 "Signal or Noise? A Benchmark Study of Agent Skills in Web Development"** (2026-08-24) — skills can be net-negative: injection cut Pass@2 1.3-4.2%, +72-394% tokens, gains in only 17-36% of pairs. Length-distraction vs content-misled failure modes. Anti-pattern rules outperform example-heavy content. → skill-quality-audit now cites this; rule: don't load skills for trivial tasks.
- **2608.08453 "What Keeps Agent Skills from Being Reusable?"** (138,133 SKILL.md, 20,556 repos, 2026-08-09) — 91.8% have ≥1 defect; dominant = weak routing metadata, bloated bodies, poor resource organization. Spec-aware < defective; AI-marked worse. Validates our validator + trigger-eval emphasis.
- **2608.12610 "@skills: Attention is all you have"** (2026-08-12) — 56,804 public skills; install = content+persistence+triggering; only triggering needs prompt residency (<100 reliable trigger slots) → keep catalogs lean.
- **2608.10906 GitSkills** (dataset of GitHub skills), **2608.06891 SkillEval** (interpretable quality signals), **2609.00006 Harness Engineering** (11 coding-agent source study) — watch for future runs.
- v2 of 2607.01456 (smells paper) exists — taxonomy unchanged.

Repos this run (community-validated):
- **Tencent/AI-Infra-Guard** (6.2K★, CN Zhuque Lab) — full-stack AI red-teaming platform: agent-scan (Python workflow scanning), mcp-scan, AIG-PromptSecurity; READMEs in 8 languages (zh/ja/es/de/fr/kr/pt/ru); ClawHub EdgeOne Skill Scanner sibling. Regional + skills-security angle: worth comparing its skill rules vs SkillSpector next run.
- **jeremylongshore/tons-of-skills-marketplace** (2.7K★) — model-agnostic marketplace, 2,940 skills/440 plugins; "Scale, labeled" habit (every count names its cohort + reproducing command); version-surface checker; featured **no-ai-slop** (petergyang) — Detect mode names each AI-slop pattern + quotes the offending line + minimal fix; "AI detectors guess; named patterns are evidence" (edit-mode restraint: never score/guess authorship). Humanizer skill already covers this class; delta not adopted.
- **agentsmd/agents.md** (24K★) — AGENTS.md open format for repo guidance (agents.md site); Hermes already reads AGENTS.md/.hermes.md/CLAUDE.md — no change needed.

Hermes operating notes (Sep 8):
- `hermes curator status` → curator ENABLED, prune-only, 13 runs, 61 managed (58 bundled + 3 agent-created), 101 unmanaged (66 pre-date marker, 35 foreground). Bulk `curator adopt --all-unmanaged` NOT done: dead-looking skills are hub-installed (adopt refuses) or deliberately-installed capability stock (activity=0 = absence of evidence — curator's own grace floor). Weekly pipeline addition: snapshot `hermes curator status` + `hermes curator run --dry-run` preview into the report for visibility without mutation.
- Curator never touches hub-installed skills; cron-referenced skills auto-exempt (treated as pinned); archive = recoverable (~/.hermes/skills/.archive/), ledger + rollback for every mutation.

## Sep 15 2026 sweep — new evidence, new repos, new pattern (supersedes Sep 8 for these items)

Highest-value find of the sweep — **cloudflare/security-audit-skill** (4,574★, official
Cloudflare org, pushed 2026-09-14). Not a skill to install so much as a **process** to
adopt; see `references/delegation-budget-and-verdicts.md` for the extracted contract.

- Six-phase audit: recon → coverage-led hunting waves → candidate validation → structured
  output → independent record verification → target-neutral report.
- **Three-verdict taxonomy**: `confirmed` (full trace + bounded observed result, severity
  allowed) / `needs_validation` (one exact unresolved fact, **no severity, not submittable**)
  / `rejected` (refuting reason). Machine-readable `findings.json` validated against
  `report-schema.json` by `validate-findings.cjs`; coverage claim validated by
  `validate-coverage-ledger.cjs` (zero-dependency node).
- **Coverage ledger** state machine: unit status ∈ {`planned`, `not_applicable`,
  `out_of_scope`, `in_progress`, `covered`, `candidate`, `blocked`, `deferred`}; per-attempt
  {`covered`, `candidate`, `blocked`}; reference fields {`surface`, `boundary`, `subsystem`,
  `attack_class`}. Prior-run statuses carry forward (`prior_confirmed_same_source`,
  `prior_confirmed_changed_source`, `prior_needs_validation`, `prior_covered_same_source`, …)
  — a prior `confirmed` only survives if its *source is unchanged*, never on source-ref alone.
- **Run profiles** `quick` / `standard` / `deep` alter breadth and redundancy but never the
  evidence bar.
- Two operating modes (guidance vs full audit) with "loading it does not authorize the full
  workflow" — the anti-over-trigger pattern, worth reusing in heavy skills.
- Anti-patterns list worth lifting: emitting prose-only hunter results that cannot be
  deduplicated or verified; assigning severity to `needs_validation`; writing the report
  before independent verification; letting prose and JSON disagree.

New research:
- **2608.11888 "Agent Skills Can Be Harmful"** (Microsoft Research + HUST + UIUC, 2026-08-12).
  307 skill-induced failures on SkillsBench + SWE-Skills-Bench. Efficiency regressions:
  Excessive Procedure 114/182 (62.6%) — of which **Excessive Verification 67**, Heavy
  Implementation Pipeline 30, Excessive Exploration 17; Context Bloat 46 (43 from mandatory
  skill-body text); Dependency Resolution 22. Functional failures: Task-Implementation Fault
  86/125 (68.8%), Artifact Misplacement 24, Environment Mismatch 13. Key line: "skills often
  turn validation checklists and construction recipes into mandatory work." Recommendation:
  condition verification scope and pipeline depth on task uncertainty / change size / budget.
- **2607.15557 SkillCorpus** — 821K crawled → 96,401 curated across a 16-class taxonomy and
  3 quality facets (utility, robustness, safety); retrieval-served corpus gains +7.5pp on
  SkillsBench. Gains bounded by a *coverage boundary* and a *harness boundary*.
- **SkillReducer (Gao et al. 2026)** — >60% of public-skill body content is non-actionable;
  26.4% of skills have no routing description.

New / re-confirmed repos (worldwide):
- **iOfficeAI/AionUi** (32.9K★, CN) — 24/7 cowork app that explicitly lists Hermes among the
  CLIs it drives. Worth reading next run for how it discovers/attaches Hermes sessions.
- **ksimback/hermes-ecosystem** (1.3K★) "Hermes Atlas — the decision layer for Hermes Agent";
  **rlaope/oh-my-hermes** (2.3K★) all-in-one plugin; both new since the last sweep.
- **VoltAgent/awesome-openclaw-skills** (52.6K★) — 5,400+ filtered/categorised skills.
- **microsoft/skills** (3.0K★, official) — skills + MCP servers + custom agents + AGENTS.md;
  **microsoft/SkillOpt** (17.1K★) re-confirmed as the closest published analogue to this cron.
- **SnailSploit/Claude-Red** (5.2K★) and **gadievron/raptor** (3.8K★) — offensive-security
  skill libraries competing with our own methodology; watch for technique deltas.
- **Tencent/AI-Infra-Guard** (6.4K★, CN) re-confirmed — skill + MCP scanners, 8 README
  languages.
- Regional: Japan (AIDB skill library, homula.jp, g-gen) is now publishing enterprise
  Agent-Skills adoption guides; Brazil (distrito.me, Code Dimension) treats skills as
  versioned institutional knowledge; China (CSDN/GitCode, ClawHub+SkillHub mirrors) is the
  most active community-content region by volume; Microsoft Learn has JP/PT localisations of
  the Agent Framework skills docs (auto-discovery of any subdirectory containing SKILL.md).

