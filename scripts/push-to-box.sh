#!/bin/bash
set -e
# Push host repo to box (excluding weights, .git, etc)
# Usage: ./scripts/push-to-box.sh [box_path]
BOX_PATH="${1:-/workspace/voiceagent}"
HOST_DIR="$(cd "$(dirname "$0")/.." && pwd)"
echo "Pushing $HOST_DIR -> box:$BOX_PATH"
ssh box "mkdir -p $BOX_PATH"
rsync -avz --exclude-from="$HOST_DIR/.rsync-exclude" --exclude='.git' --delete \
  "$HOST_DIR/" "box:$BOX_PATH/"
echo "Done. Box path: $BOX_PATH"
ssh box "ls -lh $BOX_PATH | head -20"
