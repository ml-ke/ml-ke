# Worldwide Agent-Skills Learning Sweep (Methodology)

How to run the recurring "learn from other AI agents worldwide" sweep. Used weekly by
cron job `3ded7d48e350` (every Tue 15:00 EAT) and usable on demand. The goal is
compounding improvement: each run should leave the system slightly better.

## Reputability ranking (state it in every report)

1. **GitHub official-org repos** (anthropics, openai, trailofbits, openclaw, microsoft, cloudflare) — highest
2. **arxiv papers** — peer-review-grade, empirical findings
3. **Community GitHub repos with 1000+ stars** — community-validated
4. **Blogs / Medium** — opinion; verify any claim against code/repos before adopting

Never adopt a claim from a low-reputability source without verification.

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
