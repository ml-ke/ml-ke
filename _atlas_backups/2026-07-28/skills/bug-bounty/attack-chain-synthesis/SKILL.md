---
title: Multi-Layer Attack Chain Synthesis
name: attack-chain-synthesis
description: Synthesizes knowledge from 15 security books, CVEs, and real findings into a multi-layer attack chain framework. Covers web → API → network protocol → kernel → boot layers and how to chain vulnerabilities across them.
---

## Reference Files

- `references/book-analysis.md` — Full TOC, key chapters, and cross-book synthesis for the first 10 books
- `references/all-books-toc.md` — Complete 15-book TOC organized by layer with recommended study path
- `references/hidden-feature-bypass.md` — UI-only flag bypass pattern, methodology, and real-world examples (Fireblocks vault enumeration)
# Multi-Layer Attack Chain Synthesis

Synthesized from: Bug Bounty Bootcamp, Hacking APIs, Web Hacker's Handbook, Linux Basics for Hackers, RTFM, Real-World Bug Hunting, Attacking Network Protocols, Practical Reverse Engineering, Rootkits and Bootkits, Hacking Wireless Networks, Fuzzing: Brute Force Discovery, Shellcoder's Handbook, A Bug Hunter's Diary, Google Browser Security Handbook, The Ghidra Book

## The Five-Layer Attack Model

Every vulnerability exists at one of five layers. The most impactful findings chain ACROSS layers.

```
Layer 1: Web Application     ← Our current strongest layer
    ↓
Layer 2: API / Service
    ↓
Layer 3: Network Protocol
    ↓
Layer 4: Kernel / OS
    ↓
Layer 5: Boot / Firmware
```

### Layer 1: Web Application

**Our current expertise.** Vulnerability classes we've found:

| Class | Our Examples | Source Books |
|-------|-------------|--------------|
| SSRF | Vercel CGNAT, Anthropic WebFetch, Kibana webhook, Vercel download.ts | Real-World BH Ch10, Bug Bounty Bootcamp Ch14 |
| Auth Bypass | Supabase IS_PLATFORM, ~40 endpoints | Real-World BH Ch16, Web Hacker's Handbook |
| IDOR | Discourse AI audit log | Real-World BH Ch16, Bug Bounty Bootcamp Ch11 |
| Path Traversal | Next.js launch-editor | Real-World BH (common theme) |
| SSTI/Template Injection | (Checked Vercel AI SDK) | Real-World BH Ch8, Bug Bounty Bootcamp Ch19 |

**Layer 1 → Layer 2 chaining:**
- SSRF → probe internal services (Redis, Elasticsearch, internal dashboards)
- Auth bypass → extract credentials (API keys, JWT secrets) → use at service layer
- IDOR → enumerate users → find privileged accounts

### Layer 2: API / Service

Internal services that become accessible after Layer 1 exploitation:

| Service | Port | What It Exposes | Source |
|---------|------|----------------|--------|
| Redis | 6379 | In-memory data, session tokens, cache poisoning | Real-World BH Ch18 |
| Elasticsearch | 9200 | Full data access, cluster manipulation | Kibana research |
| PostgreSQL | 5432 | Database access, COPY TO PROGRAM RCE | SupaPwn, Supabase |
| Kong | 8000 | API gateway, route manipulation | Supabase test files |
| RabbitMQ | 5672 | Message queue access | Attacking Network Protocols |
| Docker API | 2375 | Container management, host RCE | Real-World BH Ch12 |
| Kubernetes API | 6443 | Pod/namespace control | Web research |
| Prometheus | 9090 | Monitoring data, potential RCE | Hacking APIs |

**Layer 2 → Layer 3 chaining:**
- Service exploitation → understand its binary protocol → fuzz for memory corruption
- Internal API access → protocol-level manipulation (packet crafting)

### Layer 3: Network Protocol

When web/API testing hits a wall, drop to the protocol level. From **Attacking Network Protocols** (James Forshaw, Google Project Zero):

**Protocol Analysis Tools:**
- Wireshark / tshark — capture and analyze traffic
- strace / DTrace — syscall-level tracing
- Custom proxies (Burp, mitmproxy) — for HTTP; Canape — for custom binary protocols
- Network spoofing: ARP poisoning, DHCP spoofing, NAT configuration

**Protocol Structures to Know:**
- **TLV (Type-Length-Value)** — Binary protocols, LDAP, SNMP
- **ASN.1** — Encoding rules (BER, DER, PER), TLS certificates, SNMP
- **Text-based** — HTTP, SMTP, FTP, SIP
- **Endianness** — Network byte order (big-endian) vs host byte order

**Fuzzing Methodology** (ANP Ch 10):
1. **Mutation fuzzing** — take valid input, randomly modify bytes
2. **Generation fuzzing** — create inputs from protocol specification
3. **Triaging crashes** — debugger analysis, root cause identification
4. **Exploit dev** — from crash to controlled execution

**Common Protocol Vulnerabilities:**
- Buffer overflow (fixed-length buffers, variable-length)
- Integer overflow (allocation size arithmetic)
- Format string vulnerabilities
- Command injection via protocol fields
- Authentication bypass via checksum comparison weaknesses

**Layer 3 → Layer 4 chaining:**
- Protocol-level memory corruption → kernel-mode access
- Format string in protocol → arbitrary read/write
- Buffer overflow in network service → RCE as SYSTEM/root

### Layer 4: Kernel / OS

From **Practical Reverse Engineering** (Dang, Gazet, Bachaalany):

**Windows Kernel Attack Surface:**
- IOCTL codes — user-mode to kernel-mode communication
- IRP handlers — driver dispatch routines (MajorFunction table)
- System calls — SSDT entries, KiSystemService
- Memory Descriptor Lists (MDLs) — DMA/buffer access
- Pool memory — paged vs non-paged, lookaside lists

**Key Kernel Primitives:**
| Primitive | What It Allows | Source |
|-----------|---------------|--------|
| CR0.WP bit toggle | Write to read-only kernel memory (x86 only) | PRE Ch 3 |
| SSDT hook | Intercept system calls | PRE Ch 3, RKB Ch 1 |
| IRP hook | Intercept driver communication | PRE Ch 3 |
| DKOM | Direct Kernel Object Manipulation | RKB Ch 3 |
| IOCTL handler bug | Buffer overflow in kernel | PRE Ch 3 |

**Modern Protections:**
- PatchGuard (KPP) — kernel patch protection on x64
- Driver signature enforcement — kernel-mode code must be signed
- VBS / Hypervisor-protected Code Integrity — hardware-isolated

**Layer 4 → Layer 5 chaining:**
- Kernel access → modify boot configuration
- Driver write → UEFI runtime service manipulation
- Kernel hooks → persistent bootkit installation

### Layer 5: Boot / Firmware

From **Rootkits and Bootkits** (Matrosov, Rodionov, Bratus):

**Boot Process Infection Vectors:**
| Vector | Technique | Example Malware | Persistence |
|--------|-----------|----------------|-------------|
| MBR | Overwrite MBR code, preserve partition table | TDL4 | Before OS loads |
| VBR/IPL | Modify IPL to load driver before OS | Rovnix, Carberp | Before kernel |
| GPT | UEFI boot script manipulation | — | Firmware level |
| UEFI NVRAM | Variable storage attacks | — | Survives reinstall |
| SMM | System Management Mode exploit | — | Ring -2 persistence |

**Analysis Methodology for Bootkits:**
1. Load MBR at segment 0x7C00 in IDA Pro
2. Decrypt MBR by emulating BIOS interrupt 0x13
3. Detect modified partition table (invalid type 0x1C for TDL4)
4. Analyze IPL with x86 real-mode disassembly
5. Use emulation/virtualization for dynamic analysis (QEMU, Bochs)

**UEFI Vulnerabilities (RKB Ch 16):**
- UEFI runtime service exploits — arbitrary code in SMM
- Boot script variable manipulation
- NVRAM variable storage attacks — persistent firmware compromise
- Secure Boot bypass — db/dbx database tampering

## Cross-Layer Attack Chains

### Chain A: Web SSRF → Network Pivot → Kernel RCE (Our Most Likely Path)

```
1. SSRF on web app                    [Layer 1 - We've found this 3 times]
   ↓
2. Probe internal ports via SSRF      [Layer 1→2]
   - 6379 (Redis), 9200 (ES), etc.
   ↓
3. Identify vulnerable internal service  [Layer 2]
   - Redis with no auth → RCE via cron
   - Elasticsearch → Groovy/script injection
   ↓
4. Protocol-level exploit             [Layer 3]
   - Craft Redis RESP protocol payload via SSRF
   - Encode shell command in Redis SET
   ↓
5. OS-level access via service RCE     [Layer 4]
   - Reverse shell as web/nobody user
   - Look for SUID binaries (wal-g pattern from SupaPwn)
   ↓
6. Persistence via boot manipulation   [Layer 5]
   - Install MBR/VBR bootkit if physical access
   - Modify UEFI boot variables if firmware access
```

### Chain B: Auth Bypass → Service Key → Protocol → Kubernetes

```
1. Auth bypass on dashboard           [Layer 1 - Supabase finding]
   ↓
2. Extract API keys / JWT secrets     [Layer 1→2]
   ↓
3. Use keys to access internal APIs   [Layer 2]
   - PostgREST, GoTrue, etc.
   ↓
4. COPY TO PROGRAM → shell on DB      [Layer 2→4]
   ↓
5. Access orchestration credentials   [Layer 4 - SupaPwn pattern]
   - S3 buckets, config archives
   ↓
6. Kubernetes API access               [Layer 2→4]
   - Create privileged pods
   - Escape to host via container breakout
```

### Chain D: Fintech API Access → Mass Assignment → Financial Theft

**Source**: Rapyd OpenAPI + TypeScript audit

```... (snip: Chain D content is too long to match inline) ...```

### Chain E: UI-Only Flag Bypass → Hidden Resource Exposure → Workspace Mapping

**⚠️ REJECTED AS FALSE POSITIVE — June 2026, Fireblocks. Do NOT replicate.**

**Source**: Fireblocks vault enumeration + hiddenOnUI bypass

**What went wrong**: The finding claimed IDOR because vaults with `hiddenOnUI: true` were accessible via API. Three fatal flaws:
1. **No boundary crossed** — Vaults are workspace-scoped, not user-scoped. IDOR requires cross-resource-owner access (User A accessing User B's data).
2. **"Hidden" ≠ "Protected"** — `hiddenOnUI` is a UI toggle, not server-enforced access control. No restriction existed to bypass.
3. **No demonstrated impact** — PoC showed "can see hidden vaults" but never demonstrated cross-workspace access.

**Lesson encoded in**: `pre-submission-verification` skill (Gate C1), `crowdstream-techniques` (IDOR technique #5).

**References**: Intigriti Hackademy — "Always ask yourself: is this really an issue or intended behaviour?" PortSwigger IDOR — "resources **belonging to other customers**." RW BH Ch16 — "register **two different accounts** and test them simultaneously."

```
1. Observe unique flag in API response     [Layer 1→2 - API recon]
   `hiddenOnUI: true` — named for UI, enforced nowhere
   
2. Enumerate IDs                           [Layer 1→2]
   Sequential integer IDs (0, 1, 2, 3...)  
   No rate limiting on GET enumeration
   
3. Discover hidden resources               [Layer 2]
   Items with `hiddenOnUI: true` at IDs that don't
   appear in the UI list
   
4. Read full resource data                 [Layer 2]
   Names, balances, metadata — API returns
   complete data regardless of flag
   
5. Full workspace mapping                  [Layer 2]
   Map the entire resource tree including
   intentionally hidden items
```

**Steelman counter-argument**: The flag name `hiddenOnUI` explicitly declares its scope is UI-only. The vendor can argue this is intentional design. To rebut: demonstrate the flag is meant to restrict access (e.g., configuration UI prevents non-admin users from seeing hidden items, showing intent beyond UI cosmetics).

**Cross-reference**: `references/hidden-feature-bypass.md` for full methodology and real-world chain analysis.


```
1. Valid API key (or stolen/leaked)   [Layer 1→2]
   ↓
2. POST /v1/payouts with inline       [Layer 2 - Mass assignment]
   beneficiary object (no pre-reg)
   ↓
3. N parallel requests WITHOUT        [Layer 2 - Missing idempotency]
   idempotency key
   ↓
4. N payouts created to attacker      [Layer 2 - Financial theft]
   beneficiary → wallet drained
   ↓
5. OR: Forge webhook events           [Layer 2 - No auth on webhook]
   POST to /webhook/rapyd without auth
   → Mark orders COMPLETED for free
   ↓
6. OR: Escrow manipulation            [Layer 2]
   PATCH escrow:false on existing
   payments → release funds early
```

### Chain C: IDOR → Full Data → Credential Reuse

```
1. Unscoped find() IDOR               [Layer 1 - Discourse finding]
   ↓
2. Extract full conversation data      [Layer 1 escalation]
   - Includes API keys, tokens in AI prompts
   ↓
3. Credential reuse on other services  [Layer 1→2]
   - API keys, session tokens, email+password
   ↓
4. Lateral movement                    [Layer 2→4]
   - Access other internal services with stolen creds
```

## Tool Stack Per Layer

| Layer | Recon Tools | Analysis Tools | Exploitation Tools |
|-------|-------------|----------------|-------------------|
| Web | subfinder, httpx, katana | Burp Suite, browser devtools | curl, custom Python |
| API | Postman, curl, Wfuzz | Burp Repeater, GraphQL introspection | custom Python, sqlmap |
| Protocol | nmap, rustscan | Wireshark, Canape, strace | AFL, pwntools, Metasploit |
| Kernel | osquery, winpeas | WinDbg, IDA Pro, Ghidra | custom, Metasploit |
| Boot | ChipSec, UEFItool | IDA Pro, QEMU, Bochs | custom, SPI flash programmer |

## Our Vulnerability Blueprint

Based on what we've found, what books teach, and which programs are accessible:

### Target Profile: Self-Hostable TypeScript/Node.js (Supabase, GitLab, Discourse)
**Our formula:** Source code audit → Look for missing auth guards → Extract credentials → Chain to internal services

| Step | Technique | Book Source | Our Track Record |
|------|-----------|-------------|------------------|
| 1 | Find apiWrapper-like guards | Bug Bounty Bootcamp Ch13, Web Hacker's Handbook | ✅ Supabase (IS_PLATFORM) |
| 2 | Check for unscoped find() | Real-World BH Ch16 | ✅ Discourse AI audit log |
| 3 | Look for bare handlers | Hacking APIs Ch10 (BFLA) | ✅ Supabase api-keys.ts |
| 4 | Check SSRF protections | Real-World BH Ch10 | ✅ Vercel, Kibana, Anthropic |

### Target Profile: AI Agent / MCP (Anthropic, Claude Code)
**Our formula:** Permission prompt is the boundary → Find ways to bypass/hide/misrepresent it

| Step | Technique | Source |
|------|-----------|--------|
| 1 | Check if MCP servers bypass canUseTool | Issue #448 (OPEN), skill reason |
| 2 | Check permission prompt → actual execution differential | H1 scope: misrepresentation is in scope |
| 3 | Check hidden tool execution paths | H1 scope: invisible execution is in scope |
| 4 | Do NOT submit agent SSRF as SSRF | ❌ Learned the hard way |

### Target Profile: Internal Service via SSRF (Elasticsearch, Redis, Kafka)
**Our formula:** SSRF → Internal protocol → Command injection → RCE

| Step | Technique | Book Source |
|------|-----------|-------------|
| 1 | Spray internal ports via SSRF | Real-World BH Ch10 |
| 2 | Identify protocol (RESP, HTTP, binary) | Attacking Network Protocols Ch3 |
| 3 | Craft protocol payload | Attacking Network Protocols Ch8 |
| 4 | Fuzz for memory corruption | Attacking Network Protocols Ch10 |

## Cross-Book Escalation Patterns

### Pattern A: SSRF Escalation (4 techniques from 3 books)
1. **Cloud metadata** (169.254.169.254) — Real-World BH Ch10, Bug Bounty Bootcamp Ch14
2. **Internal Redis RCE** (cron via SET) — Real-World BH Ch18
3. **Elasticsearch script injection** — Hacking APIs, Kibana research
4. **Internal HTTP service SSRF** → Jenkins/Kibana at known CVEs — Attacking Network Protocols Ch9

### Pattern B: Auth Bypass Escalation (5 techniques from 4 books)
1. **API key extraction** (bare handlers) — Hacking APIs Ch10, our Supabase finding
2. **JWT secret → forged tokens** — Real-World BH Ch17, Web Hacker's Handbook
3. **SQL access via auth bypass** → COPY TO PROGRAM → RCE — Attacking Network Protocols Ch9, our Supabase research
4. **Kubernetes API from inside** — Real-World BH Ch12
5. **Rootkit persistence** from SQL RCE (boot process modification) — Rootkits and Bootkits Ch7

### Pattern C: IDOR Escalation (3 techniques from 2 books)
1. **403 vs 404 oracle** for ID enumeration — Real-World BH Ch16
2. **show_debug_info endpoint** for full payload — Bug Bounty Bootcamp Ch11, our Discourse finding
3. **Mass assignment via IDOR** for privilege escalation — Bug Bounty Bootcamp Ch11, Hacking APIs Ch10

### Chain F: MPC Crypto Layer — Multi-Finding Cryptographic Key Extraction (Fireblocks mpc-lib)

**Source**: Fireblocks MPC bug bounty (`fireblocks-mbb-og2`), mpc-lib C++ library analysis, Jun 2026

**CORRECTED (PoC-Proven)**: The initial hypothesis about a multi-prime CRT attack is BLOCKED by Blum ZKP in CMP. Proven attack path uses version downgrade + Paillier oracle amplification.

```  
1. Protocol version downgrade via missing lower-bound check     [Protocol logic gap]
   cmp_setup_service.cpp:145 — No min version check  
   → version < 11 accepted silently, weakens all ZK proofs
   PoC: cmp_version_downgrade_poc (27/27 pass)
   ↓
2. Weakened Fiat-Shamir proof binding                            [Crypto gap]
   mta.cpp:552-559 — at v<11, FS seed omits Paillier/RP key binding
   → MTA range proofs not bound to specific key context
   PoC: cmp_malicious_key_poc (37/37 pass)
   ↓
3. Paillier CCA oracle via distinguishable error gates          [Implementation oracle]
   bam_well_formed_proof.cpp:372-486 — 5 verification gates
   → Each gate produces different error for crafted ciphertexts
   PoC: bam_attack_poc (original Finding 004)
   ↓
4. CRT λ extraction (21 probes, 256-bit λ for 3072-bit Paillier)  [Key recovery]
   → 3072-bit Paillier Commitment key has λ of only ~256 bits
   → λ MATCHES ORIGINAL: YES ✓
   PoC: bam_full_extraction_poc (48/48 pass)
   ↓
5. Full ECDSA key recovery                                       [Protocol break]
   Decrypt server's encrypted key share with recovered Paillier key
   → Sign arbitrary transactions with victim's authority
```

**Key insight**: This chain exploits the **difference between protocol specification and implementation**. The spec says version designates capability; the implementation uses version to gate cryptographic strength. Claiming a lower version weakens the crypto without changing the protocol semantics. This pattern is unique to versioned ZKP libraries.

**Unbalanced key finding**: 3072-bit Paillier Commitment key has λ of ~256 bits, NOT ~1536. Means CRT needs only 21 probes (vs 192+ theoretical). The Blum ZKP correctly blocks multi-prime moduli — this attack path is closed in CMP.

**Real-world precedent**: CVE-2023-33241 / CVE-2023-33242 (Makriyannis et al., ACM CCS 2024). 15+ wallet providers broken by similar missing-proof combinations and oracle attacks. BitGo exploited TWICE.

**Cross-reference**: `fireblocks-api-toolkit` skill for full source analysis, Paillier oracle PoC, version downgrade details, build pitfalls.

## Quick Reference: Which Book for Which Problem

| Problem | Best Book | Chapter |
|---------|-----------|---------|
| Found SSRF, what next? | Real-World Bug Hunting | Ch10, Ch18 |
| Need to test custom protocol | Attacking Network Protocols | Ch2, Ch5, Ch8 |
| Analyzing binary service | Attacking Network Protocols | Ch6, Ch9 |
| Need kernel-level exploit | Practical Reverse Engineering | Ch3 |
| Rootkit persistence needed | Rootkits and Bootkits | Ch7, Ch14-16 |
| Report writing | Real-World Bug Hunting | Ch20 |
| API fuzzing | Hacking APIs | Ch9 |
| Memory corruption basics | Attacking Network Protocols | Ch9-10 |
| IDOR → privilege escalation | Real-World Bug Hunting | Ch16 |
| Boot process understanding | Rootkits and Bootkits | Ch5-6, 14 |
| **Start fuzzing a target** | **Fuzzing: Brute Force Vuln Discovery** | **Ch4-7 (file, network, env, in-memory)** |
| **Write shellcode for an exploit** | **Shellcoder's Handbook** | **Ch3 (shellcode), Ch11-13 (bypasses)** |
| **Exploit a stack overflow** | **Shellcoder's Handbook** | **Ch2 (stack), Ch8 (Windows)** |
| **Understand browser security model** | **Google Browser Security Handbook** | **Part 2 (same-origin policy, cookies)** |
| **See a real bug from find to fix** | **A Bug Hunter's Diary** | **Any chapter — each is a full case study** |
| **Heap overflow exploitation** | **Shellcoder's Handbook** | **Ch5 (intro), Ch8 (Windows)** |
| **Format string exploits** | **Shellcoder's Handbook** | **Ch4** |
| **URL parsing attacks / IDN bypasses** | **Google Browser Security Handbook** | **Part 1 (URLs, Unicode)** |

## Cross-Book Knowledge Synthesis

### From Bug Hunter's Diary — Real-World Bug Discovery Methodology
- **Trace user input**: Follow untrusted data from entry point to sink
- **Reverse engineer patches**: Compare patched vs unpatched binaries to find the fix, then bypass
- **NULL pointer dereferences as exploits**: On many kernels, mmap(NULL) maps page zero, making NULL ptr deref exploitable
- **Type conversion flaws**: Integer casts between signed/unsigned, different sizes create logic gaps
- **Think about edge cases**: What happens at boundary conditions? (Ch7: a bug from 4.4BSD era found in modern OS)

### From Shellcoder's Handbook — Exploit Development
- **Stack overflows**: Overflow saved EIP → control execution flow
- **Shellcode**: Write position-independent assembly, avoid null bytes
- **Format strings**: %x, %n to leak memory and write arbitrary values
- **Heap overflows**: Overwrite heap metadata → arbitrary write primitive
- **Bypass techniques**: ret2libc, ROP chains, ASLR brute force, stack canary leak

### From Fuzzing Book — Systematic Vulnerability Discovery
- **Mutation fuzzing**: Take valid input, randomly modify bytes
- **Generation fuzzing**: Create inputs from protocol specification
- **Environment variable fuzzing**: Long strings, special chars in env vars
- **File format fuzzing**: Bit flips, block boundary mutations on file inputs
- **Network protocol fuzzing**: Block-based, protocol-aware mutations
- **In-memory fuzzing**: Fuzz API calls directly without file/network I/O
- **Fuzzer design**: Choose between coverage-guided (AFL), grammar-based, or dumb mutation

### From Browser Security Handbook — Web Platform Deep Dive
- **Same-origin policy**: Protocol+host+port must match; exceptions exist
- **URL parsing attacks**: Unicode normalization differences between browsers and servers
- **Content sniffing**: Browsers ignore Content-Type under certain conditions
- **Port access restrictions**: Some ports (25, 110, etc.) blocked by browsers
- **IDN spoofing**: Homoglyph attacks possible without proper display checks
- **Cookie security**: Path/max-domain restrictions can leak cookies
