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
