# ISP CPE Fingerprinting + LAN Exposure Audit

## Why: "STARLINK" SSID ≠ Starlink uplink
The SSID is a user-editable label. Read the router's own status API instead of trusting it.
A Tozed **ZLT X17M / ZLT W304VA PRO** answering a `POST /cgi-bin/http.cgi` with
`network_type_str: 5G(NSA)` + `network_operator: Airtel Africa` + `sim_status: 1` is a cellular CPE
on a SIM uplink, whatever the SSID says. `wired_link_list:["LAN1"]` = a LAN port is link-up, which
may be a dumb AP in bridge mode (bridges have no IP, so they never appear in an ARP sweep).

## Tozed / ZLT recipe
1. `curl http://<gw>/` → `<title>Router</title>`, `<script src="js/main.js">` etc. Grab `js/main.js`,
   `js/helper.js`, `js/login.js`.
2. In `main.js` find `Url.DEFAULT_CGI = '/cgi-bin/http.cgi'` and the `RequestCmd` table
   (`INIT_PAGE:80`, `SYS_INFO:0`, `DEVICE_VERSION_INFO:43`, `GET_LTE_STATUS:82`, `ARP_BINDING:83`,
   `LOGIN:100`, `CHANGE_PASSWD:102`, `GET_SYS_STATUS:113`, `GET_NEXT_LOGIN_TIME:232`, `LAN_INFO:208`,
   `DHCPCLIENT_INFO:223`).
3. Enumerate without a session:
   ```bash
   for c in $(seq 0 240); do
     r=$(curl -s -X POST -H 'Content-Type: application/json' \
       -d "{\"cmd\":$c,\"method\":\"GET\",\"sessionId\":\"\"}" http://<gw>/cgi-bin/http.cgi)
     case "$r" in *NO_AUTH*|'') ;; *) echo "cmd=$c -> $r";; esac
   done
   ```
   Leaking unauthenticated on X17M fw 30.01.0: **43** (fw/hw/build/config), **80** (region, branding),
   **97** (language), **113** (model, board, SIM, operator, LAN link, WiFi flags), **232** (hands out a
   **login token to anyone**). Everything else is `NO_AUTH`.
4. Login flow: `cmd=232` → `token`; then `cmd=100` with `passwd = sha256(token + password)`,
   `username`. Failures are counted in bands of 3 with a timed lockout — **test at most one default
   credential pair** and report that you did.
5. `fake_version` (e.g. `6.10.9`) is the repackaged/branded version; `real_fwversion`
   (`30.01.0`) is the true one. `aeraId` (e.g. `NG0002`) encodes the operator variant —
   the login page title (`Login to MTN Broadband 5G Router Interface`) reveals the issuing carrier.
6. Dead end check: the legacy `/cgi/xml_action.cgi` referenced by commented-out digest-auth JS
   returned **404** — the path is gone, do not chase it.

## Router hardening checks worth reporting
- `curl -s -o /dev/null -w '%{redirect_url}' http://<gw>/` — empty means the admin UI serves over
  **plaintext HTTP with no HTTPS redirect**; `https://<gw>/` works but is self-signed
  (`ssl_verify_result=18`).
- Login JS may keep the password in `localStorage` **base64-encoded** (= plaintext equivalent).
- Full `-p-` scan of the gateway: only 53/80/443 open on this model. Absence of 23 (telnet),
  7547 (TR-069/CWMP), 7777 (ZLT W51 LeakyTozed) is a finding — say so.
- Vendor-family CVEs (e.g. Tozed **ZLT X300** fw 6.01.3 TR-069 `IPPingDiagnostics` root RCE,
  CVSS 9.8, rogue-LTE-eNodeB vector) are **model/firmware specific** — do not transfer them to a
  sibling model without evidence; label them "monitor only" and flag irregular CVE IDs as unverified.

## The LAN's real risk is usually the scanner host
```bash
docker ps --format '{{.Names}}\t{{.Image}}\t{{.Ports}}'   # 0.0.0.0:PORT-> = LAN-wide
timeout 3 bash -c 'echo > /dev/tcp/<own-LAN-IP>/<port>'  # confirm reachable via the LAN IP, not localhost
sudo ufw status                                          # inactive ⇒ nothing filters them
```
Real case: a Supabase CLI dev stack published `54341 kong / 54342 postgres / 54343 studio /
54344 mailpit / 54347 analytics` on `0.0.0.0`. From the LAN IP, `GET /api/platform/profile`,
`/api/platform/organizations`, `/api/platform/projects` all returned **200 with no auth**, and the
Studio container carried the Supabase CLI **default JWT secret** → forge `service_role` → full DB
read/write for anyone on the WiFi. Remediation: bind `127.0.0.1:<host>:<container>`, `ufw enable`,
rotate the JWT secret. Report this **above** any device-inventory trivia — a clean sweep plus an open
dev stack is a failed audit.
