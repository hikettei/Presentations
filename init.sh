#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: ./init.sh <project_name>" >&2
  exit 1
fi

project_name="$1"

if [[ -e "$project_name" ]]; then
  echo "Error: '$project_name' already exists." >&2
  exit 1
fi

if [[ ! -f "./TEMPLATE.md" ]]; then
  echo "Error: ./TEMPLATE.md not found." >&2
  exit 1
fi

mkdir -p "$project_name"
cp "./TEMPLATE.md" "$project_name/slide.md"

git add "$project_name"
git commit -m "Initialize ${project_name} presentation"
