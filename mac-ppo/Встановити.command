#!/bin/bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"
python_path=""
for candidate in /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 /opt/homebrew/bin/python3.12; do
  if [ -x "$candidate" ]; then python_path="$candidate"; break; fi
done
if [ -z "$python_path" ]; then python_path="$(command -v python3.12 || true)"; fi
if [ -z "$python_path" ]; then
  echo "Потрібен Python 3.12: https://www.python.org/downloads/macos/"
  exit 1
fi
if [ ! -x .venv/bin/python ]; then "$python_path" -m venv .venv; fi
.venv/bin/python -m pip install -r requirements.txt
echo "Готово. Відкрий Почати_навчання.command для живого навчання."
