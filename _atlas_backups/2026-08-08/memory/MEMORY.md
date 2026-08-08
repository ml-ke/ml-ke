Hacker synthesis: Tomnomnom=pipe, jhaddix=surface, zseano=deep. Best=hybrid mass recon + deep chaining + cross-session memory. Full lessons: ~/Dev/ATLAS-LEARNINGS/LESSONS.md via atlas-lesson-bank skill.
§
Tuesday AI Update cron (a658b981983a) PAUSED until Aug 10. Resumer (7ca3294036eb) fires Aug 10 15:00 EAT to unpause it. First live run: Aug 11 (Tue) 12:00 EAT. Workflow: web search 7 regions, write post. Uses blog-drafting skill.
§
Weekly skills-learning cron (3ded7d48e350) Tue 15:00 EAT, notes ~/Dev/ATLAS-LEARNINGS/. skill-quality-audit = 26-smell taxonomy (arxiv 2607.01456). skilldoctor: false-positives on bug-bounty content. Folders: ~/Dev/<target>/ sandbox, ~/Dev/REPORTS/<Target>/ reports.
§
Meta-analysis lesson Jun 2026: When user says conclusion is "lacking" or "not always true," must do 10+ iterations across DIFFERENT source types (CVEs, disclosed reports, top hunters, program rules, live tests) before presenting. Shallow conclusions fit on bumper stickers — nuanced ones have counterexamples. Doc: atlas-continuous-learning/references/meta-analysis-workflow.md
§
AI-detection evasion (Jul 2026): PoC first, capture output, 2-3 narrative paragraphs, no section headers, kill AI vocab. Pipeline: pre-submission-verification Gate U9 → humanizer skill. Details: LESSONS.md §01.
§
OpenCode (Aug 2026): opencode 1.18.11 + Zen key. Paid NOT usable yet. Docs DON'T prove availability — always live-test (deepseek-v4-flash-free documented but disabled). Use ~/.hermes/scripts/opencode_coder.py "TASK" --dir <path> (auto-picks live free model, --fresh forces re-test, auto-fallback). Free models: mimo-v2.5-free (default), nemotron-3-ultra-free, ling-3.0-flash-free, laguna-s-2.1-free, north-mini-code-free. DeepSeek plans; OpenCode heavy coding. Verify generated code. Fail fast on auth walls.
§
Docker env (restart=no): gitlab-local (puma 2), nemoclaw k3s (stopped), yucan (down), supabase draiva (running, 54341-7 0.0.0.0 exposed). NO firewall; pro-g in docker group (=root). pro-g sudo: pw required. Agent admin: hermes-atlas (uid 1001, pw LOCKED, NOPASSWD ALL). Agent sudo reads SUDO_PASSWORD from hermes env. AnyDesk uninstalled. i7-1065G7 throttles — repaste pending.