#!/bin/bash
# Intigriti PWN VPN management
# Source this in ~/.bashrc or call directly
# Usage: intigriti-vpn up|down|status

CONFIG="/home/pro-g/.hermes/wireguard/intigriti.conf"

case "${1:-status}" in
  up)
    echo "Bringing up Intigriti PWN VPN..."
    sudo wg-quick up "$CONFIG"
    echo "VPN IP: $(ip addr show wgportal 2>/dev/null | grep -oP 'inet \K[\d.]+' || echo 'not connected')"
    ;;
  down)
    echo "Taking down Intigriti PWN VPN..."
    sudo wg-quick down "$CONFIG"
    ;;
  status|*)
    echo "=== WireGuard Status ==="
    sudo wg show 2>/dev/null || echo "No WireGuard interfaces active"
    echo ""
    echo "=== VPN IP ==="
    ip addr show wgportal 2>/dev/null | grep -oP 'inet \K[\d.]+' || echo "wgportal interface not found"
    ;;
esac
