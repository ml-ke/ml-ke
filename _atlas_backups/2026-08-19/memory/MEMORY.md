Hacker synthesis: Tomnomnom=pipe, jhaddix=surface, zseano=deep. Best=hybrid mass recon + deep chaining + cross-session memory. Full lessons: ~/Dev/ATLAS-LEARNINGS/LESSONS.md via atlas-lesson-bank skill.
§
Crons off-peak (DS peak/off-peak from Aug 16; EAT peak 04-07&09-13): blog-poster 14:05, tue-ai 14:00 Tue. Tue cron a658b981983a PAUSED; resumer 7ca3294036eb Aug 17 15:00 → Aug 18 14:00 EAT.
§
Weekly skills-learning cron (3ded7d48e350) Tue 15:00 EAT, notes ~/Dev/ATLAS-LEARNINGS/. skill-quality-audit = 26-smell taxonomy (arxiv 2607.01456). skilldoctor: false-positives on bug-bounty content. Folders: ~/Dev/<target>/ sandbox, ~/Dev/REPORTS/<Target>/ reports.
§
Meta-analysis lesson Jun 2026: When user says conclusion is "lacking" or "not always true," must do 10+ iterations across DIFFERENT source types (CVEs, disclosed reports, top hunters, program rules, live tests) before presenting. Shallow conclusions fit on bumper stickers — nuanced ones have counterexamples. Doc: atlas-continuous-learning/references/meta-analysis-workflow.md
§
AI-detection evasion (Jul 2026): PoC first, capture output, 2-3 narrative paragraphs, no section headers, kill AI vocab. Pipeline: pre-submission-verification Gate U9 → humanizer skill. Details: LESSONS.md §01.
§
OpenCode (Aug 2026): Zen key, paid unusable, docs lie — live-test. Use ~/.hermes/scripts/opencode_coder.py "TASK" --dir <path>. 18 atlas-* skills bridged; 3 subagents (security-auditor/pentest-recon/code-reviewer). Theory (Aug 16): system-design-theory (master KB: CAP, resilience patterns, WAF pillars, eval framework — load when implementing/evaluating/planning) + opencode/antigravity/atlas-principles v1.1; Antigravity CLI=agy. Details: LESSONS.md §03 / opencode skill. Verify generated code.
§
Docker env (restart=no): gitlab-local (puma 2), nemoclaw k3s (stopped), yucan (down), supabase draiva (running, 54341-7 0.0.0.0 exposed). NO firewall; pro-g in docker group (=root). pro-g sudo: pw required. Agent admin: hermes-atlas (uid 1001, pw LOCKED, NOPASSWD ALL). Agent sudo reads SUDO_PASSWORD from hermes env. AnyDesk uninstalled. i7-1065G7 throttles — repaste pending.