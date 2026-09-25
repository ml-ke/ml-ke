# MCP / Agent Tool-Layer Incident Bank

Verified anchors for the **agent tool layer** class on ml.co.ke: MCP servers, tool
poisoning, rug pulls, over-scoped tool credentials, unauthenticated MCP endpoints.
First used by: `_posts/2026-09-16-mcp-payments-attack-surface.md` (staged Sep 15 2026).
**Re-verify before reuse — CVSS scores, patch versions and adoption stats move.**

## Distinct from the LLM/agent red-team bank

`llm-agent-red-team-incident-bank.md` covers **model-context attacks** (Blue41/Bunq memo
injection, CoSnitch, exfil channels). This bank covers the **tool layer underneath**: what the
agent is plugged into, what credentials that connector holds, and who can change the tool after
approval. The Sep 2 post owns red-teaming, Sep 9 owns exfiltration channels — do not re-use
either's anchors here without a different lens.

## 1. Supply chain: postmark-mcp — first real-world malicious MCP server (Sep 2025)

- npm package `postmark-mcp` uploaded **15 Sep 2025** by user "phanpak"; malicious from
  **version 1.0.16, released 17 Sep 2025**. One-line change: every email the assistant sent
  through it was **BCC'd to `phan@giftshop[.]club`**. Package deleted from npm;
  **1,643 downloads**; maintainer published 31 other packages.
- Koi Security CTO Idan Dardikman: "This is the world's first sighting of a real-world malicious
  MCP server"; "One developer. One line of code. Thousands upon thousands of stolen emails."
- Snyk guidance: assume exposure, rotate credentials sent through it, audit mail logs for BCC
  traffic to the domain.
- Sources: https://thehackernews.com/2025/09/first-malicious-mcp-server-found.html ,
  https://snyk.io/blog/malicious-mcp-server-on-npm-postmark-mcp-harvests-emails/

## 2. Tool poisoning / rug pulls / tool shadowing (Invariant Labs, Apr 2025)

- Poisoned **tool descriptions** carry instructions the model follows using *other* servers'
  tools in the same context. WhatsApp MCP demo exfiltrated message history to the attacker's
  phone number; companion demo read `~/.cursor/mcp.json` (credentials for every connected
  server) plus SSH keys. Confirmation dialog showed a summarised tool name with arguments hidden.
- **Rug pull**: a server changes a tool's description *after* the client approved it.
  **Tool shadowing**: one server's description manipulates how the agent uses another server's tools.
- Source: https://invariantlabs.ai/blog/mcp-security-notification-tool-poisoning-attacks

## 3. Over-scoped credential: Supabase service_role bypasses RLS (mid-2025)

- General Analysis demonstrated a support-ticket indirect injection reaching another tenant's rows
  when the assistant holds the project's **`service_role`** key — which **bypasses RLS by design**
  (Supabase's own docs confirm). Supabase's rebuttal: no customer incident reported, "MCP does not
  bypass our database-level protections like RLS" (`https://supabase.com/blog/defense-in-depth-mcp`).
  **Cite both** — the exposure depends on which credential was wired into the connection.
- Source: https://generalanalysis.com/blog/supabase-mcp-blog
- Related: Datadog Security Labs found SQL injection in Anthropic's reference **Postgres MCP
  server** bypassing its read-only mode (Aug 21 2025); server deprecated Jul 10 2025 but still
  ~21,000 weekly npm downloads; patch in Zed's fork `@zeddotdev/postgres-context-server` v0.1.4.
  https://securitylabs.datadoghq.com/articles/mcp-vulnerability-case-study-SQL-injection-in-the-postgresql-mcp-server/

## 4. Bolted-on MCP endpoints (no auth, inherited capability)

| CVE | Target | CVSS | Fixed | Note |
|-----|--------|------|-------|------|
| CVE-2025-49596 | MCP Inspector | 9.4 | 0.14.1 | browser-based RCE + DNS rebinding ("0.0.0.0 day"), Oligo Security |
| CVE-2025-6514 | mcp-remote | 9.6 | 0.1.16 (17 Jun 2025) | OS command injection via crafted `authorization_endpoint` URL; 437,000+ npm downloads; affects 0.0.5–0.1.15 |
| CVE-2026-27825 / 27826 | mcp-atlassian ("MCPwnfluence") | 9.1 / 8.2 | — | chainable to unauthenticated RCE from the same LAN |
| CVE-2026-33032 | nginx-ui MCP endpoint | 9.8 | 2.3.4 (15 Mar 2026) | `/mcp_message` had no auth; ~2,689 Shodan instances; in Recorded Future's 31 actively-exploited March 2026 list; workaround `middleware.AuthRequired()` |

- Pluto Security's Perkal, worth quoting: "When you bolt MCP onto an existing application, the MCP
  endpoints inherit the application's full capabilities but not necessarily its security controls.
  The result is a backdoor that bypasses every authentication mechanism the application was
  carefully built with."
- Sources: https://thehackernews.com/2026/04/critical-nginx-ui-vulnerability-cve.html ,
  https://thehackernews.com/2025/07/critical-mcp-remote-vulnerability.html ,
  https://www.oligo.security/blog/critical-rce-vulnerability-in-anthropic-mcp-inspector-cve-2025-49596

## 5. Cross-tenant logic bug: Asana MCP (2025)

- Server released **1 May 2025**; on **4 Jun 2025** Asana found a bug that "could have potentially
  exposed certain information from your Asana domain to other Asana MCP users", took the server
  offline and fixed the code. Asana: "This was not a result of a hack or malicious activity."
  **~1,000 customers** may have been affected (spokesperson to BleepingComputer); notifications
  from **16 Jun**.
- Framing that worked: an agent asked for "everything the user can see" is a cross-tenant sweep
  waiting for one authorization bug.
- Source: https://www.upguard.com/blog/asana-discloses-data-exposure-bug-in-mcp-server

## Authoritative guidance

- **NSA CSI, "Model Context Protocol (MCP): Security Design Considerations for AI-Driven
  Automation"**, May 2026, U/OO/6030316-26 — Recommendations: choose supported projects; design for
  boundaries (egress filtering proxy/DLP); validate parameters; constrain and sandbox tool
  execution; sign and verify MCP messages; filter and monitor output pipelines; instrument for
  logging and detection (SIEM); track and patch MCP CVEs; scan networks for open MCP servers
  (names MCP Scanner, Ramparts, CyberMCP, Proximity). Quotes used: "MCP itself cannot enforce these
  security principles at the protocol level"; "a change in capability or data access for an MCP
  server that is already trusted or connected often can be made without approval".
  **Fetch tip:** `media.defense.gov` returns a 470-byte HTML stub to curl; use
  `curl -sL https://r.jina.ai/<PDF URL>` then read the markdown, or retry the PDF with a browser UA.
  URL: https://media.defense.gov/2026/Jun/02/2003943289/-1/-1/0/CSI_MCP_SECURITY.PDF
- **OWASP MCP Top 10** (beta; next release planned Oct 2026): MCP01 token mismanagement, MCP02
  scope creep, MCP03 tool poisoning, MCP04 supply chain attacks, MCP05 command injection, MCP06
  intent flow subversion, MCP07 insufficient authn/authz (confused deputy), MCP08 lack of audit and
  telemetry, MCP09 shadow MCP servers, MCP10 context injection and over-sharing.
  https://owasp.org/www-project-mcp-top-10/

## Statistics (quote WITH the counterweight)

43% command injection (Equixly) · 82% path traversal across 2,614 implementations (Endor Labs) ·
36.7% SSRF of 7,000+ servers (BlueRock) · 33% of 1,000 scanned servers critical (Enkrypt AI) ·
~5.5% tool poisoning of 1,899 servers (Hasan et al.) · 66% of 1,808 servers with some finding
(AgentSeal) · 86% of servers run locally on developer machines, 5% in production (Clutch Security)
· 28% of Fortune 500 on MCP in production by early 2026 (Truto, vendor estimate).
**Caution to state in-post:** one independent audit measured ~78% false-positive rates from
YARA-based MCP scanners — use these to prioritise, never as proof a given server is safe.
Source: https://www.practical-devsecops.com/mcp-security-statistics-2026-report/

## Reusable demo recipe (verified Sep 15 2026)

Stdlib-only, deterministic (no seed needed — literal manifests):
fingerprint each tool as `sha256(json.dumps({name, desc, params}, sort_keys=True))[:12]` pinned at
approval; at each connect, diff fingerprints, flag unreviewed new tools, scan descriptions for
instruction-shaped text (`do not tell`, `before using`, points-at-credentials, routes-data-outward),
and score blast radius from credential scopes (`secrets.read`/`admin`/`ledger.post` weighted 5).
Verdicts: drift or suspect → BLOCK; unreviewed or blast ≥ 6 → NEEDS APPROVAL; else ALLOW.
Verified output on three servers: email-mcp `ALLOW` (blast 3), crm-mcp `BLOCK` (1 drift, blast 9,
three flags), reports-mcp `NEEDS APPROVAL` (1 unreviewed tool).
**Gotcha found while building it:** compare `fingerprint(live[t]) != fingerprint(approved[t])`, not
`fingerprint(live[t]) != approved[t]` — comparing a hash to a dict makes every tool look drifted.
