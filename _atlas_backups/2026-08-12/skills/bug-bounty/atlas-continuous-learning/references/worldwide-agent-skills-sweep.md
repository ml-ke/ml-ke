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
- **mksglu/context-mode** (19.7K★) — context window optimization, sandboxes tool output.
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

## Cron prompt pitfall (verified Aug 10 2026)

The cronjob threat filter BLOCKS prompts matching `prompt_injection` patterns. When writing/updating a cron prompt that describes security-scan triage rules, do NOT include literal payload examples (verbatim instruction-override strings, remote-code-pipe-to-shell commands, base64 blobs) — the filter rejects the whole update. Phrase triage rules abstractly ("verbatim instruction-override text", "remote-code-pipe-to-shell of attacker-controlled URLs", "unexplained encoded command blobs"). This applies to ANY security-flavored cron prompt, not just this sweep.

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
