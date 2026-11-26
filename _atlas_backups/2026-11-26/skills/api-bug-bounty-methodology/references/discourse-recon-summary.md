# Discourse Recon Summary (May 2026)

## Live Instance
- URL: https://try.discourse.org
- Headers: nginx, CSP with nonce + strict-dynamic, `x-discourse-route: list/latest`
- Resets daily — work fast, document everything same session

## CVEs Studied
- **CVE-2025-46813** (GHSA-v3h7-c287-pfg9): Private data leak on login-required sites via routing. Reverting commit 82d84af6 fixed it.
- **CVE-2024-53991**: Backup disclosure via Rails send_file quirk + predictable filenames + local file storage. Condition: FileStore::LocalStore only.
- **GHSA-mq82-7v5x-rhv8**: Discourse Reactions plugin — reaction notification data exposure.
- 246 total CVEs since 2018 (34 in 2024, 24 in 2025). Average CVSS: 4.0.

## Codebase Stats
- 22,581 files in repo clone (~Dev/discourse/discourse/)
- 52 plugins in plugins/
- Stack: Ruby on Rails (server) + Ember.js (client) + PostgreSQL

## Key Source Files
| File | Purpose |
|------|---------|
| `lib/final_destination.rb` (595 lines) | Main SSRF protection + URL fetch |
| `lib/final_destination/ssrf_detector.rb` | IP range validation (32+ private ranges) |
| `lib/final_destination/resolver.rb` | Thread-safe DNS with timeout |
| `lib/final_destination/http.rb` | Custom Net::HTTP with IP filtering |
| `lib/onebox.rb` | URL preview system |
| `lib/onebox/preview.rb` | Onebox fetch logic |
| `lib/retrieve_title.rb` | Title crawler (uses FinalDestination) |
| `lib/file_helper.rb` | File download (uses FinalDestination) |
| `app/controllers/uploads_controller.rb` | File upload handling |
| `app/controllers/webhooks_controller.rb` | Email webhook handlers (Mailgun, SendGrid, etc.) |
| `app/controllers/admin/admin_controller.rb` | Base admin controller |
| `config/routes.rb` (1900+ lines) | Full route map |

## SSRF Protection Analysis
Discourse's SSRFDetector (`lib/final_destination/ssrf_detector.rb`) is comprehensive:

```ruby
PRIVATE_IPV4_RANGES = [
  IPAddr.new("0.0.0.0/8"),
  IPAddr.new("10.0.0.0/8"),
  IPAddr.new("100.64.0.0/10"),  # CGNAT — Vercel missed this
  IPAddr.new("127.0.0.0/8"),
  IPAddr.new("169.254.0.0/16"),
  IPAddr.new("172.16.0.0/12"),
  IPAddr.new("192.0.0.0/24"),
  IPAddr.new("192.0.0.0/29"),
  IPAddr.new("192.0.0.8/32"),
  IPAddr.new("192.0.0.170/32"),
  IPAddr.new("192.0.0.171/32"),
  IPAddr.new("192.0.2.0/24"),
  IPAddr.new("192.168.0.0/16"),
  IPAddr.new("192.175.48.0/24"),
  IPAddr.new("198.18.0.0/15"),
  IPAddr.new("198.51.100.0/24"),
  IPAddr.new("203.0.113.0/24"),
  IPAddr.new("240.0.0.0/4"),
  IPAddr.new("255.255.255.255/32"),
]
```

Plus IPv6 ranges: ::1, ::, 64:ff9b:1::/48, 100::/64, 2001::/23, fc00::/7, fe80::/10

The connection flow:
1. `FinalDestination::HTTP#connect` calls `SSRFDetector.lookup_and_filter_ips(hostname)`
2. DNS resolves to actual IPs via `Addrinfo.getaddrinfo`
3. Each resolved IP is checked against private ranges
4. Only allowed IPs are connected to
5. `self.ipaddr = ip` is set to the resolved IP (not hostname) for the TCP connection
6. Host header is preserved as original hostname

This blocks: DNS rebinding, private IPs, internal services, metadata endpoints.

## Attack Vectors Not Yet Deeply Tested
1. AI plugin audit log IDOR (requires debugging group access)
2. Chat plugin message authorization
3. Discourse AI plugin — MCP OAuth controller (new, complex)
4. Discourse AI plugin — bot controller streaming interaction
5. Email webhook parameter injection (signature verification prevents injection of unverified webhooks, but verified webhooks might have injection vectors in parsed fields)
6. Admin backup/restore path traversal
