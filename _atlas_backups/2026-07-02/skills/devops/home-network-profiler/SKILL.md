---
title: Home Network Profiler
name: home-network-profiler
description: Scan and profile devices on a home WiFi network — discover hosts, identify device types via port/service fingerprints and hostname conventions, detect misidentifications, and produce a structured device inventory.
---

# Home Network Profiler

Scan a local LAN (typically a /24 subnet), discover all live hosts, profile each device using port scans, service fingerprints, mDNS/Bonjour discovery, and hostname heuristics. Handles common pitfalls like randomized MACs (privacy addresses), device sleep patterns, false OS detections, and intermittent phones/tablets.

## Prerequisites

- `nmap` installed
- `curl` or `nc` (netcat) available
- `nmcli` (optional, for WiFi info)
- Root/sudo for nmap scans (`-sU`, `-O`, raw socket scans)

Install if needed:
```
sudo apt-get install -y nmap
```

## Step-by-step Methodology

### 1. Gather WiFi context

Get your own IP, gateway, interface name, and connected SSID:

```bash
ip route show default
ip addr show | grep -E 'inet '
iwconfig <interface>       # or: nmcli dev wifi list
```

Check nearby networks, signal strength, channel congestion:
```bash
nmcli -f ACTIVE,SSID,BSSID,CHAN,FREQ,SIGNAL,SECURITY,RATE dev wifi list
```

### 2. Discover live hosts (ping sweep)

```bash
sudo nmap -sn <subnet>/24 -oG -
```

Example: `sudo nmap -sn 192.168.1.0/24 -oG -`

This prints a grepable output showing all live hosts with their reverse-DNS hostnames. Run this **multiple times** over several minutes — mobile devices (phones, tablets, laptops) enter sleep mode and may not respond to every sweep.

### 3. Check known hosts in ARP cache

```bash
cat /proc/net/arp
```

Shows IP-to-MAC mappings for devices your machine has recently communicated with. Can reveal hosts that are asleep but recently active.

### 4. Port & service scan each discovered host

Start with a quick scan of common ports:

```bash
sudo nmap -sV -p 22,80,443,5353,8080,8443,3000,5000,9000,62078 <ip>
```

Then do a full 1-65535 port scan (use `--min-rate=5000` for speed):

```bash
sudo nmap -p- --min-rate=5000 <ip>
```

For devices that block ping (`Note: Host seems down. If it is really up...`), use `-Pn`:

```bash
sudo nmap -Pn -p- --min-rate=5000 <ip>
```

> **NOTE:** A full port scan on devices with no open ports will still take time. Cancel it early (Ctrl+C) if nothing appears in the first ~5 seconds.

### 5. Service version & OS detection

On discovered open ports:

```bash
sudo nmap -sV -p <port1,port2,...> <ip>
```

For OS detection (often unreliable — see pitfalls):

```bash
sudo nmap -O <ip>
```

### 6. mDNS/Bonjour discovery (Apple devices, printers, IoT)

Apple devices advertise via mDNS on UDP port 5353:

```bash
sudo nmap -sU -p 5353 --open <ip>
```

With DNS-SD script:
```bash
sudo nmap -sU -p 5353 --script dns-service-discovery <ip>
```

### 7. Direct banner grab

For known open ports, connect directly to grab banners (HTTP request works on many services):

```bash
timeout 5 bash -c 'echo -e "GET / HTTP/1.0\r\n\r\n" | nc <ip> <port>' | head -5
```

### 8. Scan UDP services

Check for DNS, DHCP, mDNS, NTP, SNMP:

```bash
sudo nmap -sU -p 53,67,68,123,161,5353 <ip>
```

## Device Profiling Heuristics

### Hostname Patterns

The `Name-s-Model` pattern (e.g., `Timothy-s-A17`, `Pro-s-A06`) is characteristic of **Apple devices** using Bonjour/mDNS. The `-s-` represents the possessive "'s" (e.g., "Timothy's A17"). The model suffix can be:

| Suffix | Likely Device |
|--------|--------------|
| A17 | iPhone 15 Pro / iPhone 15 Pro Max (A17 chip) |
| MacBook-Pro | MacBook Pro |
| MacBook-Air | MacBook Air |
| Mac-mini | Mac mini |
| iPad | iPad (any model) |
| iPhone | iPhone (any model) |
| Apple-TV | Apple TV |

For non-Apple patterns:
- `Pros-Galaxy-Note9` style → `Name's Model` for Android phones
- `android-xxxxxxxxxxxx` → Generic Android hostname

### Service Fingerprints (key ports)

| Port | Protocol | Likely Service | Device Type |
|------|----------|---------------|-------------|
| **5353/udp** | mDNS (Bonjour) | Apple ecosystem | Any Apple device |
| **32841/tcp** | `_companion-link._tcp` | Apple TV Remote | Apple TV (tvOS). Returns `{"type":"Tier1","version":"1.0"}` on HTTP GET |
| 7000/tcp | AirPlay | Apple TV / HomePod | Apple media streaming |
| 62078/tcp | AirPlay (legacy) | Apple devices | iOS/macOS |
| 3689/tcp | DAAP | iTunes / Home Sharing | Mac, Apple TV |
| **6800/tcp** | Samsung TV Remote | Smart TV remote control | Samsung Smart TV |
| 55000/tcp | Samsung TV API | Samsung Smart TV | Older Samsung TV remote |
| 80/tcp | HTTP | Router admin / web | Varies |
| 443/tcp | HTTPS | Router admin / web | Varies |
| 53/tcp+udp | DNS | Router / DNS server | Gateway |
| 67/udp | DHCP | DHCP server | Router/Gateway |

### Randomized (Private) MAC Address Detection

MAC addresses where the **second hex digit of the first octet** is one of `2, 6, A, E` are **locally administered** (randomized privacy addresses):

```
DE:9B:3E:E0:ED:78 → 0xDE = 0b11011110 → bit 1 = 1 → random/local
B6:63:33:4C:13:11 → 0xB6 = 0b10110110 → bit 1 = 1 → random/local
```

Modern Apple devices (iOS 14+, macOS 11+), Android (10+), and Windows (10+) all use randomized MACs per network by default. This means **you cannot look up the manufacturer OUI** — these are NOT registered.

## Reference Files

This skill includes a reference file with real-world fingerprint data from an actual home network sweep:

- **`references/device-fingerprints.md`** — verified service banners (Apple TV companion-link `Tier1` protocol, Samsung TV remote control), randomized MAC detection CLI snippet, port-to-service quick-reference table, Starlink router fingerprint, and device sleep behavior observations. Consult this when identifying devices that nmap's `-O` or `-sV` gets wrong.

## Additional Verification: Apple TV Companion-Link

The `_companion-link._tcp` service is exclusive to Apple TV (tvOS). When you find an open ephemeral port (32841 or high port) returning `{"type":"Tier1","version":"1.0"}` on HTTP GET, the device is an Apple TV — regardless of what nmap OS detection says.

**Important**: The companion-link port is allocated from the ephemeral range and can change on reboot. Run a full port scan if the previously known port stops responding. The `Tier1` JSON response is the reliable identifier, not the port number.

### Device Sleep Patterns

| Device Type | Network Behavior |
|-------------|-----------------|
| **Apple TV** | Always on — consistently responds to pings |
| **Smart TV** | May respond in standby, often inconsistent |
| **Smartphone** | Intermittent — only responds when screen-on, appears/disappears between scans |
| **Laptop** | Responds while awake; disappears when lid closed/sleep |
| **Desktop/server** | Consistently on if running continuously |

## Common Pitfalls

### 1. nmap OS detection is unreliable — ALWAYS cross-reference with service fingerprints

nmap's `-O` flag uses TCP/IP stack fingerprinting and can produce **wild misidentifications**. **Do NOT report a device type based on nmap OS detection alone.** Always cross-reference at least two independent signals.

**Real-world example from a live session:**

| Device | nmap -O said | Actual identity | How we knew |
|--------|-------------|----------------|-------------|
| Apple TV (tvOS) | "Microsoft Xbox 360 Dashboard" | Apple TV 4K | `companion-link` on port 32841 returned `{"type":"Tier1","version":"1.0"}`, mDNS on 5353/udp, Apple Bonjour hostname |

**Rule**: When nmap says Xbox 360, PlayStation, or any game console — always confirm with service fingerprints before reporting. Apple TV (tvOS) has been misidentified repeatedly.

**Cross-reference signals (use at least two):**

| Signal | What to look for | Example |
|--------|-----------------|---------|
| **mDNS (5353/UDP)** | Bonjour service = Apple ecosystem | Any Apple device |
| **`_companion-link._tcp` (port ~32841)** | Apple TV Remote protocol | Returns `{"type":"Tier1","version":"1.0"}` on HTTP GET |
| **Port 7000** | AirPlay | Apple TV / HomePod |
| **Hostname pattern** | `Name-s-Model` = Apple Bonjour convention | `Timothy-s-A17` = Timothy's A17 (iPhone 15 Pro) |
| **Port 6800** | Samsung Smart TV Remote | Samsung TV control protocol |
| **Always-on behavior** | Apple TV stays on; phones/laptops sleep | Check run-to-run consistency |

**Apple TV companion-link identification method:**
```bash
# Banner grab on the companion-link port (dynamic, scan for it):
timeout 5 bash -c 'echo -e "GET / HTTP/1.0\r\n\r\n" | nc <ip> 32841' | head -5
# Expected: {"type":"Tier1","version":"1.0"}

# Check for mDNS:
sudo nmap -sU -p 5353 --open <ip>
```

The companion-link port is allocated from the ephemeral range and can change on reboot. The `Tier1` JSON response is the reliable identifier, not the port number.

**Known misidentifications seen in real scans:**
- **Apple TV (tvOS)** → nmap -O says **"Microsoft Xbox 360 Dashboard"**
- Modern iOS devices → various embedded Linux systems
- Randomized MACs (iOS 14+) → manufacturer shows as "Unknown"

### 2. Mobile devices go to sleep

Phones and tablets may not respond to ARP/ping during deep sleep. Run the discovery sweep **at least 2-3 times over a 5-minute period** to catch intermittent devices. Check `/proc/net/arp` which retains recently-seen hosts.

### 3. Randomized MACs hide manufacturer identity

Devices using privacy MACs (iOS 14+, Android 10+, Win 10+) will show as "Unknown" in OUI lookups. Use service fingerprints and hostnames instead.

### 4. Port scan timing on mobile devices

A full `-p-` scan can take 10-15 seconds on a supported device but will timeout against sleeping devices. Start with the fast port list (`-p 22,80,443,5353,8080,8443,62078,7000,5000,6800,9400,32841`) which completes in under 3 seconds per host.

### 5. Nmap service inference for unknown ports

nmap's service inference for unknown ports is often wrong. Port 9400 may be labeled `sec-t4net-srv` but actually used by Apple media services. Always confirm via direct banner grab.

### 6. Apple TV vs Xbox 360 — nmap OS Detection False Positive

nmap `-O` can misidentify **Apple TV (tvOS)** as **"Microsoft Xbox 360 Dashboard"**. The TCP/IP fingerprint in tvOS shares characteristics with the Xbox 360 kernel stack, producing this specific false positive in real-world scans.

**How to distinguish:**
1. Check for **mDNS (UDP 5353)** — Apple devices advertise via Bonjour
2. Probe **port 32841** with a simple HTTP GET — Apple TV returns `{"type":"Tier1","version":"1.0"}` (the companion-link protocol)
3. Look for other Apple services: AirPlay (7000/tcp), DAAP (3689/tcp)
4. Apple TV is **always on** (set-top box) — consistent ping responses
5. Hostname patterns like `Name-s-Model` (e.g., `Timothy-s-A17`) follow Apple's Bonjour naming convention (possessive `'s` + model identifier)

**Xbox indicators:** UPnP on UDP 1900, Xbox-specific port ranges, intermittent presence (console sleep mode). Use cross-reference, never trust `-O` alone.

## Compiling the Report

Structure the output as:

```
## Network Overview
- SSID, ISP (from surrounding networks + gateway), gateway IP, channel, signal

## Device Inventory
For each device:
- IP, hostname, MAC (note if randomized)
- Open ports and identified services
- Likely device type (with confidence notes)
- Behavioral notes (always-on vs intermittent)
```
