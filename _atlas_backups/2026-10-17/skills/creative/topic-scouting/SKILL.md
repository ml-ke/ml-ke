---
name: topic-scouting
description: Scan for blog post topics at the intersection of AI security, ML engineering, and AI engineering. Uses Google Scholar, sophon.at/papers, CVE/NVD, arXiv, and news sources to find trending topics that fill gaps in the blog series coverage.
---

# Topic Scouting for ML Kenya Blog

## Core Topics (intersection of these three areas)

```
AI Security  ──┬──  ML Engineering  ──┬──  AI Engineering
               │                      │
Prompt Injection                     MLOps Pipelines
LLM Jailbreaks                       Model Serving
Data Poisoning                       Feature Stores
Supply Chain                         Training Infrastructure
Model Theft                          Evaluation & Benchmarks
Insecure Agents                      Data Engineering
RAG Security                         Monitoring & Observability
```

**Goldilocks Zone:** Topics that bridge at least TWO of these areas (e.g., "Securing MLOps Pipelines" bridges AI Security + ML Engineering).

## Scanning Sources

### 1. sophon.at/papers (primary source)

```
URL: https://sophon.at/papers
```

Browse via browser_navigate. The page shows trending papers with:
- Title, author list, date
- Brief abstract excerpt
- Category links (Language Modeling, Agents, Security-tagged, etc.)
- View counts and hourly velocity (e.g., "4.2k" / "15/h")

**Browsing pattern:**
1. Open `https://sophon.at/papers` in browser
2. Scroll down to see trending papers
3. Filter by tab: All / This week / This month / This year
4. Search by keyword using the URL: `https://sophon.at/papers?q=KEYWORD`
   - Note: sophon does NOT have a dedicated "security" tag. Security papers appear under "Language Modeling" or "Agents" categories. To find security content, search for "adversarial", "jailbreak", "robustness", "safety", "poisoning" — but these searches often return no trending matches. The browse-all page (`/papers?q=`) provides full catalog search.
5. Click "Browse all" for the complete catalog with sort/filter options
6. To fact-check a specific paper, download its PDF via arXiv:
   ```bash
   curl -sL "https://arxiv.org/pdf/XXXX.XXXXX" -o paper.pdf
   pdftotext paper.pdf - | head -100  # Verify authors, date, abstract
   ```

**Known limitation:** Security-focused papers are NOT tagged as a separate category on sophon. They surface under "Language Modeling" or "Agents". For security-specific scanning, use Google Scholar and OWASP Gen AI as primary sources and sophon as a secondary source for ML engineering and agent papers.

### 2. Google Scholar (weekly scan)

```
Query format: "AI security" OR "LLM vulnerability" OR "prompt injection" after:2025
Query format: "ML engineering" OR "MLOps security" after:2025
```

### 3. CVE / NVD

```
URL: nvd.nist.gov
Search: "machine learning", "artificial intelligence", "LLM", "PyTorch", "TensorFlow"
```

```
URL: nvd.nist.gov
Search: "machine learning", "artificial intelligence", "LLM", "PyTorch", "TensorFlow"
```

Track CVEs in:
- ML frameworks (PyTorch, TensorFlow, JAX, ONNX)
- Model serving (vLLM, TGI, Triton)
- ML libraries (transformers, diffusers, langchain)
- AI applications (Copilot, ChatGPT plugins)

### 4. News & Incident Trackers

| Source | URL | Covers |
|--------|-----|--------|
| The Hacker News | thehackernews.com | AI breaches, CVEs |
| The Register | theregister.com | AI security incidents |
| FireTail AI Breach Tracker | firetail.ai/ai-breach-tracker | AI-specific breaches |
| OWASP Gen AI Incidents | genai.owasp.org | LLM incident roundups |
| MLSys Conference | mlsys.org | ML engineering papers |
| NVIDIA Developer Blog | developer.nvidia.com/blog | ML infra, Triton, NIM |

### 5. NVD CVE Feed (ML-Specific)

```
URL: services.nvd.nist.gov/rest/json/cves/2.0
Query: keywordSearch=machine+learning OR artificial+intelligence OR LLM
Query: keywordSearch=PyTorch OR TensorFlow OR transformers OR vllm
```

Cross-reference CVEs with your topic to ensure:
- Claims about vulnerabilities match actual CVE descriptions
- Severity scores (CVSS) are accurately represented
- Affected versions are correctly identified (grep changelogs)

### 6. Fact-Checkability Pre-Scan

Before committing to a topic, verify it passes the fact-check gate:

```bash
# Does the paper/incident actually exist?
curl -s "https://scholar.google.com/scholar?q=exact+paper+title" | grep -c "exact paper title"

# Does the CVE match the claim?
curl -s "https://services.nvd.nist.gov/rest/json/cves/2.0?cveId=CVE-2024-XXXX" | python3 -m json.tool

# Has the technique been demonstrated in the wild?
grep -ri "target_company\|specific_incident" _posts/  # Already covered?
```

**If you can't find 3+ verifiable sources for a topic, do NOT propose it.** The user requires claims backed by real data, not hypotheticals.

```
URL: genai.owasp.org
```

Check quarterly for:
- New vulnerability classes added
- Existing classes with new attack variants
- Updated mitigation guidance

## Identifying Blog Series Gaps

### Current Series Inventory

| Series | Posts | Status | Gap Opportunities |
|--------|-------|--------|-------------------|
| Optimization | 3 | COMPLETE | — |
| Knowledge Graphs | 8 | COMPLETE | — |
| AI Hacking | 5 | COMPLETE | — |
| AI/ML Engineering | 5 | IN PROGRESS | Model serving security, training infrastructure hardening, evaluation-driven development, CI/CD for ML, data pipeline security |

### Gap-Filling Criteria

A topic is worth a blog post if:
1. It bridges at least two of the three core areas
2. It has at least 3 verifiable real-world incidents or papers from the past 12 months
3. It is NOT already covered in an existing post (check the series inventory above)
4. It can include working code examples (not just theory)
5. The user hasn't explicitly written about it before (check `grep -ri TOPIC _posts/`)

## Topic Candidate Evaluation

For each candidate topic, score:

```
Scorecard:
  [ ] Has 3+ recent real-world incidents/papers? (+2)
  [ ] Bridges 2+ core areas? (+2)
  [ ] Working code possible? (+2)
  [ ] Not covered by existing posts? (+2)
  [ ] User likely interested? (+1)
  Total: ___/9

Threshold: ≥6 = worth a blog post
```

## Idea → Post Pipeline

Once a topic is scouted and scored ≥6:

1. **Draft to `.scheduled/`** — write the full post to `~/ProG/ml-ke/.scheduled/YYYY-MM-DD-slug.md`
   - Use the blog-drafting skill for format, cover images, LQIP
   - Include ALL cover images, WebP conversions, and LQIP generation before staging
   - Do NOT commit `.scheduled/` files — they remain uncommitted for cron deployment
2. **Fact-check everything** — run the Fact-Checking Protocol from blog-drafting skill
3. **Stage assets** — commit cover images, WebP, LQIP files to the repo
3. **Set up cron** — create a `blog-poster` cron that publishes one per day from `.scheduled/`:
   ```
   Schedule: 05 11 * * *  (= 14:05 EAT)
   Deliver: origin,all (chat + Telegram)
   Skills: blog-drafting
   ```
   The cron moves `.scheduled/TODAY-*.md` → `_posts/` and git-pushes. See `references/cron-publishing-pattern.md` for the full cron prompt template.
The cron moves `.scheduled/TODAY-*.md` → `_posts/` and git-pushes. See `references/cron-publishing-pattern.md` for the full cron prompt template.

## Quick Topic Generator

```bash
# Check what we haven't covered
cd ~/ProG/ml-ke

# Search for topics already covered
grep -ri "MLOps\|CI/CD\|monitoring\|observability" _posts/ | head -5
grep -ri "model theft\|extraction\|inversion" _posts/ | head -5
grep -ri "fine-tuning\|lora\|qlora\|sft" _posts/ | head -5
grep -ri "rag\|retrieval\|vector database" _posts/ | head -5

# Count posts per category
grep -h "^categories:" _posts/*.md | sort | uniq -c | sort -rn

# Find recent CVEs in ML
curl -s "https://services.nvd.nist.gov/rest/json/cves/2.0?keywordSearch=machine+learning&resultsPerPage=5" | python3 -m json.tool 2>/dev/null | grep -E '"id"|"description"'
```

## Usage

Load this skill when the user says "find new blog topics" or "what should I write about next." Run the topic evaluation scorecard for each candidate, then present the top 3-5 ranked by score.
