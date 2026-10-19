#!/usr/bin/env bash
# (Re)start the game server in the background on port 8000. Set OPENAI_API_KEY first to enable the AI guide.
cd "$(dirname "$0")"
pkill -f "uvicorn app:app" 2>/dev/null; sleep 1
nohup .venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &
sleep 2; curl -s localhost:8000/api/health; echo
