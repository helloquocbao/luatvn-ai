#!/usr/bin/env bash
# Khởi động trợ lý pháp luật: Ollama + máy chủ web.
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/Library/Python/3.14/bin:/opt/homebrew/bin:$PATH"

if ! curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  echo "→ khởi động Ollama…"
  OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 nohup ollama serve > data/ollama.log 2>&1 &
  until curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; do sleep 1; done
fi

echo "→ http://localhost:8000  (Ctrl-C để dừng)"
exec uv run uvicorn luatvn.api:app --port 8000 --host 127.0.0.1
