# Bug Bounty Program Case Studies

Real-world program analysis from an audit session (May 2026).

## Vercel Open Source (Recommended Target)

**Program**: HackerOne — "Vercel Open Source"
**Avg Bounty**: $700–$752
**Reports Resolved**: 58
**Features**: Managed, Retesting, Collaboration

### Scope

All Vercel open source projects. The core projects listed on the program page:

| Project | Size | Source Files | Language | Notes |
|---------|------|-------------|----------|-------|
| next.js | 343 MB | 22,223 | TypeScript + Rust | Largest target; compiled webpack, devtools, server |
| ai | 114 MB | 4,394 | TypeScript | Newer SDK, less audited; **primary target** |
| turborepo | 178 MB | 1,599 | TypeScript + Rust | Build orchestration; sparse but powerful |
| swr | 3.2 MB | 261 | TypeScript | Smallest; data fetching hook |

### Why This Program

1. **Source code available** (all repos on GitHub) — no accounts or auth needed
2. **AI SDK** is newer and growing fast — less security review history
3. **Attack surface diversity** — web framework, build tools, SDK, API integrations
4. **Retesting + Collaboration** flags mean the team is responsive and collaborative

### Vulnerability Angles Explored

#### AI SDK (highest ROI)

| Area | Finding | Status |
|------|---------|--------|
| `validate-download-url.ts` | Full SSRF protection (localhost, private IPs, IPv6, .local) | **Protected** — well implemented |
| `secure-json-parse.ts` | `__proto__` stripping | **Protected** |
| `merge-objects.ts` | `__proto__`, `constructor`, `prototype` blocked | **Protected** |
| `extract-reasoning-middleware.ts` | Dynamic `new RegExp` with user-provided tag name | **Low risk** — tagName from developer, not end-user |
| `string.ts` (zod parser) | Dynamic `RegExp` from schema check values | **Low risk** — schema-defined values, not user input |
| Provider API calls (OpenAI, Anthropic, Google providers) | HTTP request construction with user content | **Needs deeper review** — header injection, prompt injection |
| Tool execution pipeline | What happens when AI calls a tool with crafted arguments? | **Needs review** |

#### Next.js

| Area | Finding | Status |
|------|---------|--------|
| `launch-editor.ts` | `child_process.spawn(editor, args)` from env vars | **Dev-only** — REACT_EDITOR/VISUAL/EDITOR env vars |
| `telemetry/storage.ts` | `child_process.spawn` for telemetry | **Hardcoded commands** — safe |
| `trace/upload-trace.ts` | `child_process.spawnSync` | **Hardcoded commands** — safe |

#### Other Programs Considered (from HackerOne directory, May 2026)

| Program | Avg Bounty | Reports | Verdict |
|---------|-----------|---------|---------|
| Anthropic | $1k-$1k | 295 | AI company; very new (launched 05/2026); worth monitoring |
| Notion Labs | $150-$250 | 168 | Proprietary API; needs account for testing |
| Twilio | $200-$250 | 2,987 | Mature; competitive; well-audited |
| Robinhood Markets | $966-$1k | 133 | High payout but complex auth; financial app |
| M-Pesa Africa | $250-$500 | — | Mobile money; African region; invite-only |
| CLEAR | $150-$250 | 100 | Biometric/identity; $100 min bounty |
| Whatnot | $500-$1k | — | Livestream marketplace; newer |
| Audible | $400-$600 | 67 | Amazon-owned; web + mobile |
| Dyson | $200-$400 | 836 | IoT + web apps |

## Program Selection Checklist

- [ ] Is source code available? (Open source >> proprietary)
- [ ] Is the program active (recently responding to reports)?
- [ ] Does the program offer retesting?
- [ ] Is the bounty minimum worth the effort? (>$50)
- [ ] Do I have the right tools/skills for the asset type? (Web, API, Mobile, Source Code)
- [ ] Are there clear rules of engagement and scope boundaries?
