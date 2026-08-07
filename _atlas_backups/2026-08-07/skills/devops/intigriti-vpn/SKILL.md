---
name: intigriti-vpn
category: devops
description: Manage the Intigriti PWN VPN tunnel. Bring up/down/check the WireGuard connection to the Intigriti test environment.
---

# Intigriti PWN VPN

## Config Location

The WireGuard config is at: `/home/pro-g/.hermes/wireguard/intigriti.conf`

## Usage

### Bring up the VPN
```bash
sudo wg-quick up /home/pro-g/.hermes/wireguard/intigriti.conf
```
Verify with: `sudo wg show` or check IP `ip addr show wgportal`

### Take down the VPN
```bash
sudo wg-quick down /home/pro-g/.hermes/wireguard/intigriti.conf
```

### Check status
```bash
sudo wg show
```

### Bash Aliases (loaded via ~/.hermes/bashrc.d/intigriti-vpn.sh)
```
vpn-up      — bring VPN up
vpn-down    — take VPN down  
vpn-status  — show connection status
```

## Script

Also available at `/home/pro-g/.hermes/scripts/intigriti-vpn.sh`:
```bash
/home/pro-g/.hermes/scripts/intigriti-vpn.sh up     # connect
/home/pro-g/.hermes/scripts/intigriti-vpn.sh down   # disconnect
/home/pro-g/.hermes/scripts/intigriti-vpn.sh status # check
```

## Network Details

- **VPN IP**: `10.0.2.124/32`
- **Server**: `52.51.233.22:33333`
- **Route**: All traffic via VPN (`0.0.0.0/0`)
- **PWN Environment**: `app.pwn.intigriti.rocks`, `login.pwn.intigriti.rocks`, etc.
- **DNS**: `1.1.1.1`

## Verification

After connecting, verify the tunnel:
```bash
curl -s --max-time 5 "https://app.pwn.intigriti.rocks/" | head -c 100
# Should return 200 (was 403 from outside VPN)
```

## Notes

- The config file belongs to a peer on the Intigriti WireGuard Portal
- Interface name will be `wgportal` (derived from config filename)
- Do NOT expose the PrivateKey or PresharedKey
