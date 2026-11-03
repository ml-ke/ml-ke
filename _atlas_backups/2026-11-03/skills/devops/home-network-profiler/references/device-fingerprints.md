# Device Fingerprint Quick Reference

## Port-to-Service Mapping
| Port | Protocol | Service | Device |
|------|----------|---------|--------|
| 5353/udp | mDNS | Bonjour/Zeroconf | Apple devices |
| 32841/tcp | companion-link | Apple TV Remote | Apple TV |
| 7000/tcp | AirPlay | Media streaming | Apple TV/HomePod |
| 62078/tcp | AirPlay legacy | Media streaming | iOS/macOS |
| 3689/tcp | DAAP | iTunes sharing | Mac/Apple TV |
| 6800/tcp | Samsung Remote | Smart TV control | Samsung TV |
| 55000/tcp | Samsung API | TV remote | Older Samsung TV |
| 53/tcp+udp | DNS | Domain resolution | Router/Gateway |
| 67/udp | DHCP | IP assignment | Router/Gateway |

## Apple Hostname Pattern

`Name-s-Model` = possessive "'s" from Apple Bonjour registration
- `Timothy-s-A17` = "Timothy's A17" (iPhone 15 Pro with A17 chip)
- `Name-s-MacBook-Pro` = MacBook Pro
- `Name-s-iPhone` = iPhone

## Samsung Hostname Pattern (Android)
- `Name-s-Galaxy-{Model}` or `Pros-{Model}`
- `Pros-Galaxy-Note9` = "Pro's Galaxy Note 9"

## nmap OS Detection Reliability

nmap -O is NOT reliable for identifying device types. Always cross-reference:
1. Hostname pattern (Apple vs Android naming)
2. Service fingerprints (banner grabs, mDNS)
3. Port profiles (which services are exposed)
4. Behavioral patterns (always-on vs intermittent)
5. MAC address (randomized vs OUI-registered)

## Android hostname patterns (modern)

| Hostname seen | Meaning |
|---------------|---------|
| `SM-N960F`, `Pixel-7`, `A001T` | **Bare `ro.product.model`** — LineageOS/AOSP builds advertise the model as the DHCP/mDNS hostname. Often the single best "which phone is this" signal, since the MAC is usually randomized (locally-administered) and has no OUI. |
| `android-xxxxxxxxxxxx` | Generic/unset Android hostname |
| `Name-s-Galaxy-Note9` | Samsung stock "Name's device" Bonjour/NETBIOS form |

A phone answering as `SM-N960F` with MAC `96:94:9E:33:F2:7E` is a **Galaxy Note 9 running a custom ROM** (stock would say `Name-s-Galaxy-...`), not an Apple device.

## Phones expose NO services — the control channel is the real question

An idle Android phone on WiFi answers ICMP/ARP with **every one of the 65535 TCP ports
closed**. Scanning finds the device; it does not give you control. Before promising
"I'll make the phone ring / play a sound", check what would have to be ON:

| Channel | Port | How it opens |
|---------|------|--------------|
| adb over WiFi (Wireless debugging) | ephemeral 30000-65535 | Settings → System → Developer options → Wireless debugging → ON (no code needed when the host is already paired; pairing persists across laptop reboots but not across phone reboots) |
| Termux `sshd` | 8022 | Termux app opened, `sshd` started, key installed |
| legacy `adb tcpip 5555` | 5555 | needs USB once |
| Miracast/DLNA/HTTP apps | varies | user-installed |

Practical consequence: a watcher that polls for the port and acts on appearance
(`nmap -sT -Pn -p 5555,8022,30000-65535 --min-rate 10000 --open <ip>` every ~15s →
`adb connect <ip>:<port>` → act) turns a required "flip a switch on the phone"
step into a single instruction to the user, with no second round-trip.

## Sleep is not absence

In one 3-round sweep the Note 9 appeared in rounds 1 and 2 and vanished in round 3
while still being pingable minutes later. Also note that a full port scan across a
**sleep/dismiss** transition can report "all ports filtered" for a host that is up —
re-run the scan before concluding a service is off, and prefer `-sT -Pn -T3`
(connect scan, no root needed) when SYN-scan results look uniformly filtered.
