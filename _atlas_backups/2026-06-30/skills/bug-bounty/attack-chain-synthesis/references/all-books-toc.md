# Complete Book Library — Table of Contents Reference

## 15 Books, All Layers Covered

### Layer 1: Web Application
- Bug Bounty Bootcamp (Vickie Li) — SSRF Ch14, IDOR Ch11, SSTI Ch19, Auth Bypass Ch13
- Real-World Bug Hunting (Yaworski) — SSRF Ch10, IDOR→PE Ch16, RCE Ch12, Logic Ch18
- Web App Hacker's Handbook (Stuttard/Pinto) — Comprehensive methodology
- Google Browser Security Handbook (Zalewski) — SOP, URL parsing, cookies, IDN

### Layer 2: API
- Hacking APIs (Corey Ball) — BOLA/BFLA Ch10, API fuzzing Ch9, auth testing

### Layer 3: Network Protocol
- Attacking Network Protocols (Forshaw, Google Project Zero) — RE Ch6, root causes Ch9, fuzzing Ch10
- Fuzzing: Brute Force Discovery (Sutton) — file Ch4, network Ch5, env Ch6, in-memory Ch7
- Hacking Wireless Networks — WiFi attack methodology

### Layer 4: Kernel / OS
- Practical Reverse Engineering (Dang/Gazet) — x86/ARM Ch1-2, kernel Ch3, debugging Ch4, obfuscation Ch5
- Shellcoder's Handbook (Anley) — stack Ch2, shellcode Ch3, format string Ch4, heap Ch5, Windows Ch6-8, ROP Ch11-13
- A Bug Hunter's Diary (Klein) — kernel Ch6, VLC Ch7, iOS Ch8, browser Ch5
- Linux Basics for Hackers (OccupyTheWeb) — Linux fundamentals

### Layer 5: Boot / Firmware
- Rootkits and Bootkits (Matrosov) — MBR/VBR Ch7, IDA analysis Ch8, UEFI Ch14-16
- The Ghidra Book (Eagle/Nance) — disassembly Ch1-2, scripting Ch14, decompiler Ch19, patches Ch22

### Cross-Layer Reference
- RTFM (Ben Clark) — Quick commands for all layers

## Recommended Study Path

### Phase 1: Immediate (what we already do)
- Bug Bounty Bootcamp + Real-World Bug Hunting → web/API methodology
- Hacking APIs → API-specific testing

### Phase 2: Short-term (fuzzing + protocol)
- Attacking Network Protocols Ch6, 9, 10 → protocol RE + fuzzing
- Fuzzing Ch4-7 → systematic fuzzing
- Shellcoder's Handbook Ch2-5 → memory corruption basics

### Phase 3: Medium-term (binary analysis)
- Ghidra Book Ch4-10, 14 → Ghidra workflow
- Practical RE Ch1-2 → x86/ARM reading
- Bug Hunter's Diary → case study methodology

### Phase 4: Long-term (kernel + boot)
- Practical RE Ch3 → Windows kernel
- Rootkits and Bootkits Ch7-9, 14-16 → boot process
- Shellcoder's Handbook Ch11-13 → bypass techniques
