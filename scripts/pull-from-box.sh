#!/bin/bash
set -e
BOX_PATH="${1:-/workspace/voiceagent}"
MSG="${2:-sync from box $(date -Is)}"
HOST_DIR="$(cd "$(dirname "$0")/.." && pwd)"
echo "Pulling box:$BOX_PATH -> $HOST_DIR (excludes weights, keeps runs/logs)"
rsync -avz --exclude-from="$HOST_DIR/.rsync-exclude-pull" --exclude='.git' \
  "box:$BOX_PATH/" "$HOST_DIR/" 
echo "Pull complete. Changed files:"
cd "$HOST_DIR"
git status --short | head -100
echo ""
echo "To commit: git add -A && git commit -m \"$MSG\" && git push"
