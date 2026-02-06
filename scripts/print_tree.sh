#!/usr/bin/env bash
set -euo pipefail
if command -v tree >/dev/null 2>&1; then
  tree -a
else
  echo "Install 'tree' or use: find . -maxdepth 3 -print"
  find . -maxdepth 3 -print
fi
