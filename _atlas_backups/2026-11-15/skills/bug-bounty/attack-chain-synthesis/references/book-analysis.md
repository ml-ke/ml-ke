# Book Analysis — 15 Security Books

## Core Books (10 original)

### Bug Bounty Bootcamp (Vickie Li)
- Ch10: IDOR, Ch14: SSRF, Ch16: SSTI, Ch17: Logic, Ch19: Template Injection
- Ch21: Info Disclosure, Ch22: Methodology
- **Best for**: Web vulnerability fundamentals, bug bounty workflow

### Hacking APIs (Corey Ball)
- Ch8: Auth Testing, Ch9: Fuzzing, Ch10: BOLA/BFLA, Ch11: Mass Assignment
- Ch13: WAF Evasion
- **Best for**: API endpoint enumeration, auth bypass patterns

### Web Application Hacker's Handbook (Stuttard & Pinto)
- Chaining vulnerabilities, auth bypass patterns, attacking access controls
- **Best for**: Understanding how low-severity bugs combine into critical chains

### Linux Basics for Hackers (OccupyTheWeb)
- Networking, scripting, file system, permissions
- **Best for**: Post-exploitation fundamentals

### RTFM (Ben Clark)
- Quick reference for *NIX, Windows, networking commands
- **Best for**: Post-exploitation command reference

## Set 2 (Added June 1, 2026 — 5 books)

### Real-World Bug Hunting (Peter Yaworski)
- Ch10: SSRF (AWS metadata, blind SSRF, internal port scanning)
- Ch12: RCE (shell commands, ImageMagick, SSH RCE)
- Ch13: Memory Vulnerabilities (buffer overflow, integer overflow)
- Ch15: Race Conditions (TOCTOU, Keybase, payments)
| **16** | **IDOR** | **3 case studies: Binary.com ($300), Moneybird ($100), Twitter Mopub ($5,040) — all use TWO-ACCOUNT methodology** |
- Ch18: Application Logic (privilege bypass, S3 misconfig, 2FA bypass)
- **Best for**: Real-world bounty case studies, escalation methodology

### Attacking Network Protocols (James Forshaw, Google Project Zero)
- Ch6: Application RE (x86 ISA, IDA, static/dynamic RE)
- Ch9: Root Causes (memory corruption, command injection, format strings)
- Ch10: Finding & Exploiting Vulns (fuzzing, triaging, shellcode, ROP, ASLR bypass)
- **Best for**: Protocol fuzzing, binary exploit development, memory corruption

### Practical Reverse Engineering (Dang, Gazet, Bachaalany)
- Ch3: Windows Kernel (system calls, IRQL, pool memory, MDLs, IRPs, IOCTL, SSDT hooks)
- Ch4: Debugging & Automation (WinDbg, scripting, extensions)
- Ch5: Obfuscation (control-flow flattening, symbolic execution)
- **Best for**: Kernel-level vulnerability understanding, driver analysis

### Rootkits and Bootkits (Matrosov, Rodionov, Bratus)
- Ch7-9: Bootkit Infection (MBR/VBR/IPL modification, static/dynamic analysis)
- Ch14-16: UEFI (boot flow, bootkits, firmware vulnerabilities — SMM, NVRAM)
- **Best for**: Boot-level persistence, firmware security

### The Ghidra Book (Chris Eagle, Kara Nance)
- Part II (Ch4-10): Basic usage — setup, data displays, disassembly, data types
- Part III (Ch11-16): Advanced — scripting Java/Python, Eclipse, headless mode
- Part IV (Ch17-20): Custom loaders, processors, decompiler
- Part V (Ch21-23): Obfuscated code, binary patching, binary diffing
- **Best for**: Ghidra for binary analysis of native Node.js addons

## Set 3 (Added June 1, 2026 — 4 books)

### The Shellcoder's Handbook (Anley et al.)
- Ch2: Stack Overflows — overflow saved EIP, control execution
- Ch3: Shellcode — position-independent assembly, null byte avoidance
- Ch4: Format String Bugs — %x/%n to leak memory, write arbitrary values
- Ch5: Heap Overflows — overwrite heap metadata → arbitrary write
- Ch6-8: Windows exploitation — SEH, exceptions, heap
- Ch11-13: Bypass techniques — ret2libc, ROP, ASLR brute force, stack canary leak
- **Best for**: Memory corruption exploitation, shellcode development

### A Bug Hunter's Diary (Tobias Klein)
- Ch4: NULL Pointer FTW — exploiting NULL ptr dereferences
- Ch5: Browse and You're Owned — browser exploitation
- Ch6: One Kernel to Rule Them All — OS X kernel bug
- Ch7: A Bug Older Than 4.4BSD — ancient bug in modern OS
- Ch8: The Ringtone Massacre — iOS exploitation
- **Best for**: Real-world vulnerability case studies from discovery to patch

### Fuzzing: Brute Force Vulnerability Discovery (Sutton et al.)
- Ch4: File Format Fuzzing — bit flips, block boundary mutations
- Ch5: Network Protocol Fuzzing — block-based, protocol-aware
- Ch6: Environment Variable Fuzzing — long strings, special chars
- Ch7: In-Memory Fuzzing — fuzz API calls directly
- **Best for**: Systematic fuzzing methodology, choosing fuzzer types

### Google Browser Security Handbook (Michal Zalewski)
- Part 1: URLs, schemes, HTTP, HTML DOM, JavaScript, CSS, plugins
- Part 2: Same-Origin Policy, cookie security, content sniffing, port restrictions, IDN checks
- Part 3: Experimental/legacy security mechanisms
- **Best for**: Browser security model, URL parsing attacks, IDN spoofing

## Cross-Book Escalation Patterns

### SSRF → RCE (4 techniques)
1. Cloud metadata (169.254.169.254) — Real-World BH Ch10, Bug Bounty Bootcamp Ch14
2. Internal Redis RCE (cron via SET) — Real-World BH Ch18
3. Elasticsearch script injection — Hacking APIs
4. HTTP service → Jenkins/Kibana at known CVEs — Attacking Network Protocols Ch9

### Auth Bypass → Full Compromise (5 techniques)
1. API key extraction (bare handlers) — Hacking APIs Ch10
2. JWT secret → forged tokens — Real-World BH Ch17
3. SQL via auth bypass → COPY TO PROGRAM → RCE — Attacking Network Protocols Ch9
4. Kubernetes API from inside — Real-World BH Ch12
5. Rootkit persistence from SQL RCE — Rootkits and Bootkits Ch7

### Memory Corruption → System Control (3 paths)
1. Stack overflow → ROP → ASLR bypass — Shellcoder's Handbook Ch2/11
2. Heap overflow → arbitrary write → code exec — Shellcoder's Handbook Ch5
3. IOCTL buffer overflow → kernel compromise — Practical RE Ch3
