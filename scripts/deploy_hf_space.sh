#!/usr/bin/env bash
# Sync, commit, push GitHub, and deploy to HF Space (SSH git).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MSG="${1:-feat(hf-space): QVPIC Gradio demo}"

echo "=== 1. Sync HF space bundle ==="
bash scripts/sync_hf_space.sh

echo "=== 2. Git commit (qvpic) ==="
git add docs/HF_SPACE_README.md scripts/sync_hf_space.sh scripts/deploy_hf_space.sh \
  space/qvpic/ tests/test_hf_space.py web/
git status --short
if git diff --cached --quiet; then
  echo "No staged changes"
  GH_SHA="$(git rev-parse HEAD)"
else
  git commit -m "$MSG"
  GH_SHA="$(git rev-parse HEAD)"
fi
echo "GitHub SHA: $GH_SHA"

echo "=== 3. Git push origin main ==="
git push origin main

echo "=== 4. Deploy to HF Space ==="
HF_DIR="/tmp/hf-qvpic"
rm -rf "$HF_DIR"
if ! git clone git@hf.co:spaces/kinaar111/qvpic "$HF_DIR" 2>/dev/null; then
  echo ""
  echo "HF Space git@hf.co:spaces/kinaar111/qvpic not found."
  echo "Create it at https://huggingface.co/new-space (Gradio SDK, name qvpic)"
  echo "or: HF_TOKEN=... hf repos create kinaar111/qvpic --type space --space-sdk gradio"
  exit 1
fi

rsync -av --delete \
  --exclude='.git' \
  --exclude='.venv' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='*.egg-info' \
  "$ROOT/space/qvpic/" "$HF_DIR/"
cd "$HF_DIR"
git add -A
git status --short
if git diff --cached --quiet; then
  echo "No HF changes to commit"
  HF_SHA="$(git rev-parse HEAD)"
  HF_PUSH="no changes"
else
  git commit -m "$MSG"
  HF_SHA="$(git rev-parse HEAD)"
  git push origin main
  HF_PUSH="OK"
fi

echo ""
echo "=== RESULTS ==="
echo "GITHUB_SHA=$GH_SHA"
echo "HF_SHA=$HF_SHA"
echo "HF_PUSH=$HF_PUSH"