# Writing-Skill Research — sources, findings, adoption

Research pass behind the `writing-router` skill (Oct 2026). Records what was adopted, what was rejected, and why, so a future session does not re-derive it.

## 1. Kimi — "Creative Writing Skills to Write Faster with AI"

Source: https://www.kimi.ai/resources/creative-writing-skills-for-agents (Kimi/Moonshot AI, updated 2026-09-15). Fetched with the browser tool — `web_extract` is search-only in this environment (Brave free backend).

### Kimi's 15 built-in writing skills

`ad-creative`, `content-research-writer`, `copy-editing`, `copywriting`, `ecom-listing-copywriter`, `humanizer`, `investor-letter-writer`, `keynote-composer`, `podcast-episode-writer`, `professional-email-composer`, `scholarly-writing-refiner`, `seo-content-writer`, `support-response-writer`, `work-recap-writer`, `x-thread-crafter`.

### The structural insight worth stealing

Kimi does **not** ship one writing skill. It ships a catalogue and requires the user to select. Its working mechanics are directly transferable:

1. **Trigger-list descriptions.** Every Kimi skill description is a long enumeration of user phrasings — e.g. `copy-editing` carries "edit this copy / review my copy / tighten this up / this reads awkwardly / too wordy / polish this". This is the same idea as Hermes's description-as-router, with one hard constraint Kimi does not have: **Hermes truncates the description to 60 chars (57 + `...`) in the system-prompt skill index.** So the trigger must be front-loaded; the full enumeration only helps once the skill is loaded.
2. **Specialised skill per scenario, not one general writing skill.** Kimi's own advice: "Instead of using one skill for everything, create dedicated skills for fiction, scripts, poetry, storytelling, marketing copy, or world-building. Specialized skills perform better because they focus on a specific writing task." ATLAS already follows this (blog-drafting vs docx vs research-paper-writing); what was missing was the **selection layer**.
3. **Edit and generation are separate skills.** `copywriting` writes; `copy-editing` edits; `humanizer` de-AIs. Generation and refinement are distinct passes. ATLAS had the refinement skill (`humanizer`) but no rule forcing it into the pipeline.
4. **Custom skills from your own writing samples.** Kimi's "Document to skills" flow turns style guides and past writing into a reusable skill. ATLAS's equivalent is `humanizer`'s Voice Calibration section plus the ML Kenya house style encoded in `blog-drafting`.
5. **Refine with feedback over time.** "Continuously refining these skills based on feedback" — ATLAS's equivalent is `atlas-lesson-bank` (`~/Dev/ATLAS-LEARNINGS/LESSONS.md`).

### Adopted from Kimi

- The **routing catalogue** pattern → `writing-router`'s decision table.
- **Generation vs refinement as separate passes** → the mandatory pairing rule (content skill → `humanizer`).
- **Constraints defined before drafting** → the pipeline's "Structure" step before "Draft".

### Rejected from Kimi

- The marketing-copy skills (`ad-creative`, `copywriting`, `seo-content-writer`, `ecom-listing-copywriter`, `support-response-writer`). ATLAS writes technical, security and fintech prose — not ad or e-commerce copy. Importing them would dilute the 60-char routing budget for capabilities never used.
- Kimi's slash-command install flow (`/seo-content-writer`). Hermes routes by description, not by typed command.

### Kimi's 8 open-source writing skills (surveyed, not installed)

| Skill | Repo | Why not adopted |
|---|---|---|
| `novel-writing` | wgwtest/novel-writing | Fiction; out of ATLAS scope |
| `novel-project-strategy` | wgwtest/novel-project-strategy | Fiction project state; out of scope |
| `fiction-humanizer-zh` | deedeeKong07-alt/fiction-humanizer-zh | Chinese fiction-specific |
| `creative-writing-skills` | haowjy/creative-writing-skills | Fiction craft (muse, scene craft) |
| `10x-content-expert` | OpenAnalystInc/10x-Content-Expert | Marketing brand-voice plugin |
| `ai-content-engine` | vincentchan/AI-Content-Engine | Marketing ideation pipeline |
| `skill-anything` | SYuan03/Skill-Anything | PDF/video → study packs; not writing |

None overlap ATLAS's actual writing surface. Recorded here so the sweep does not re-evaluate them.

## 2. Wider landscape (Oct 2026 search)

- **anthropics/skills** (official): the Office-file skills (`docx`, `pdf`, `pptx`, `xlsx`) are the source of ATLAS's productivity document skills. Routing lesson from the firecrawl review of Claude Code skills: *"Claude cannot route to it reliably"* is a named failure class — weak or mechanics-only descriptions. Reinforces the 60-char front-loading rule.
- **awesome-humanizer-skills** (zhuyansen): a catalogue of anti-slop / AI-tell-detection skills, security-graded. Confirms `humanizer` is one member of a category, and that detection (flag AI tells) and rewriting (remove them) are separable jobs. ATLAS keeps one skill doing both, which is correct at this scale.
- **agentndx.ai "Best Agent Skills for Writing 2026"**: recommends `blader/humanizer` for exactly the role this router assigns it — *"Run it as the last pass before publishing."* Independent confirmation of the humanizer-as-final-pass convention.
- Its other recommended slots map onto skills ATLAS already has: research-assistant → `grounded-citations`; Email Drafter → `email-inbox-triage` + `himalaya`; Document/Meeting Summarizer → `document-to-action-items` + `meeting-action-items`.
- **marketingskills** (coreyhaines31), **claude-skills** (alirezarezvani, 380 skills), **claude-seo** (AgricIDaniel): large marketing/SEO catalogues. Surveyed, not adopted — wrong domain for ATLAS writing.

## 3. Hermes mechanics this router depends on

- `SKILL_PROMPT_DESC_LIMIT = 60` in `agent/skill_utils.py`; `extract_skill_description()` truncates to 57 + `...`. The system-prompt index renders `- <name>: <description>` (`agent/prompt_builder.py::_render_skills_index`).
- `tools/skill_manager_tool.py` **rejects** a new skill whose description exceeds 60 chars — authoring-time enforcement of the routing budget.
- `metadata.hermes.related_skills` unions the user tree with the in-repo tree at load time (see `hermes-agent-skill-authoring`), which is how the cross-links in this router resolve.

## 4. Known gaps this router accepts

- No `copy-editing`-equivalent skill: prose editing that is *not* AI-tell removal has no owner. `humanizer` is the default, and this router's pipeline step 6 is where a future `copy-editing` skill would slot in.
- No generic business-email **composer** (Kimi's `professional-email-composer`): `email-inbox-triage` drafts replies; new outbound email has no owner. Route to `himalaya` + `humanizer` until one exists.
- No SEO/GEO layer (Kimi's `seo-content-writer`). ML Kenya posts are not SEO-driven; revisit only if that changes.
