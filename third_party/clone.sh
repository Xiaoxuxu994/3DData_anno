#!/bin/bash
# Clone the third-party repos next to our code at the pinned commits.
# Usage: bash third_party/clone.sh [--all]   (default: only what v1.3 needs: vipe, Pi3, depth-anything-3)
set -euo pipefail
cd "$(dirname "$0")/.."
NEED="vipe Pi3 depth-anything-3"
[ "${1:-}" = "--all" ] && NEED="vipe Pi3 depth-anything-3 vggt-omega MoGe SpatialVID HunyuanWorld-Voyager"
while IFS=$'\t' read -r name url commit _; do
  case "$name" in ''|\#*) continue;; esac
  [[ " $NEED " == *" $name "* ]] || continue
  if [ -d "$name/.git" ]; then echo "skip $name (exists)"; continue; fi
  git clone "$url" "$name" && git -C "$name" checkout -q "$commit"
  echo "ok $name @ ${commit:0:7}"
done < third_party/repos.lock
echo "Next: create each repo's .venv as in envs/README.md"
