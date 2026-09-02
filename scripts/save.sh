#!/bin/bash
set -e
# One-command save: pull from box + commit + push
BOX_PATH="${1:-/workspace/voiceagent}"
MSG="${2:-save $(date -Is)}"
HOST_DIR="$(cd "$(dirname "$0")/.." && pwd)"
"$HOST_DIR/scripts/pull-from-box.sh" "$BOX_PATH" "$MSG"
cd "$HOST_DIR"
if [ -z "$(git status --porcelain)" ]; then
  echo "No changes to commit."
  exit 0
fi
git add -A
git commit -m "$MSG"
git push origin main
echo "Saved and pushed: $MSG"
