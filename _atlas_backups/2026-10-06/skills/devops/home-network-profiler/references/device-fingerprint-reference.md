# Real-World Device Fingerprints

Fingerprint data collected during an actual home network sweep (Starlink, Kenya, May 2026).

## Apple TV (tvOS) — Companion-Link Protocol

**Confirmed by**: Port 32841 returning `{"type":"Tier1","version":"1.0"}` on HTTP GET
**nmap -O falsely reported**: "Microsoft Xbox 360 Dashboard"
**Hostname**: `Timothy-s-A17` (Apple Bonjour convention)
**Open ports**: 9400, 9500, 32841 (TCP), 5353/udp (mDNS)
**Behavior**: Always-on, consistently responds to pings

### Companion-link port identification
```bash
# The companion-link port is NOT fixed at 32841. It's allocated from the
# ephemeral range and can change on reboot. To find it:
sudo nmap -p- --min-rate=5000 <ip>

# Once found, confirm with banner grab:
timeout 5 bash -c 'echo -e "GET / HTTP/1.0\r\n\r\n" | nc <ip> 32841'
# Expected: {"type":"Tier1","version":"1.0"}
```

## Starlink Router

**Vendor**: SpaceX / Starlink
**MAC prefix**: D8:42:F7
**Open ports**: 53 (DNS — Cloudflare), 80 (HTTP admin), 443 (HTTPS admin), 67 (DHCP/udp)
**No open ports**: SSH, telnet, SNMP — router is locked down
**Firmware**: Custom Starlink firmware, web admin panel present

## Randomized MAC Detection

Use this quick check to determine if a MAC is a privacy address:
```bash
# A MAC is randomized if the second hex digit of the first octet
# is 2, 6, A, or E. Quick check:
echo "DE:9B:3E:E0:ED:78" | grep -qiE '^[0-9a-f]*[26ae]'
```

## Device Sleep Patterns

Observations during sweeps:
- Apple TV responded to every sweep consistently
- Samsung Galaxy Note9 appeared in 2/3 sweeps (phone active then idle)
- Unknown "Pro-s-A06" device appeared intermittently (consistent with smart TV or tablet)
