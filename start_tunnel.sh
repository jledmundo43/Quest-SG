#!/usr/bin/env bash
# (Re)start a free Cloudflare quick tunnel to the local server. A NEW random URL is issued each time.
cd "$(dirname "$0")"
pkill -f "cloudflared tunnel" 2>/dev/null; sleep 1
nohup ./bin/cloudflared tunnel --no-autoupdate --url http://localhost:8000 > tunnel.log 2>&1 &
for i in $(seq 1 30); do
  url=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' tunnel.log | head -1)
  [ -n "$url" ] && { echo "$url" | tee public_url.txt; exit 0; }
  sleep 1
done
echo "Tunnel URL not found yet; check tunnel.log"
