#!/usr/bin/env bash
# Sync results/features/ between this machine and a GCS bucket.
#
# Usage:
#   export SCD_GCS_BUCKET=gs://your-bucket/scd-ml
#   ./scripts/sync_features.sh push   # VM  → bucket  (run after extraction)
#   ./scripts/sync_features.sh pull   # Mac → bucket  (run before local analysis)
#
# Requires `gsutil` (installed with the Google Cloud SDK) and `gcloud auth login`.

set -euo pipefail

BUCKET="${SCD_GCS_BUCKET:-}"
if [ -z "$BUCKET" ]; then
  echo "Error: set SCD_GCS_BUCKET first (e.g. export SCD_GCS_BUCKET=gs://my-bucket/scd-ml)"
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL="$REPO_ROOT/results/features/"
REMOTE="$BUCKET/features/"

mkdir -p "$LOCAL"

case "${1:-}" in
  push) gsutil -m rsync -r -x '\.gitkeep$' "$LOCAL" "$REMOTE" ;;
  pull) gsutil -m rsync -r "$REMOTE" "$LOCAL" ;;
  *)    echo "Usage: $0 {push|pull}"; exit 1 ;;
esac
