#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -f "$ROOT_DIR/.venv/bin/activate" ]; then
  # Use project environment so preflight and smoke tests see installed deps.
  source "$ROOT_DIR/.venv/bin/activate"
fi

export PYTHONPATH="${PYTHONPATH:-}:$ROOT_DIR"

if [ -z "${RESULTS_FILE:-}" ]; then
  RESULTS_FILE="$(python3 - <<'PY'
import glob
import json
import os

best_tts = ""
best_tts_mtime = -1.0
best_any = ""
best_any_mtime = -1.0

for p in glob.glob('reports/*.json'):
    try:
        with open(p, 'r', encoding='utf-8') as f:
            d = json.load(f)
        if isinstance(d.get('results'), list) and d.get('schema_version') == 'v2':
            mt = os.path.getmtime(p)
            if mt > best_any_mtime:
                best_any = p
                best_any_mtime = mt
            rows = d.get('results') or []
            has_tts = any(isinstance(r, dict) and ('tts_metrics' in r) for r in rows)
            if has_tts and mt > best_tts_mtime:
                best_tts = p
                best_tts_mtime = mt
    except Exception:
        pass

print(best_tts or best_any)
PY
)"
fi

if [ -z "${RESULTS_FILE}" ]; then
  RESULTS_FILE="reports/vqa_with_tts_results.json"
fi
export RESULTS_FILE

echo "[pre-demo] 1/5 preflight"
python3 scripts/jetson_preflight_check.py

echo "[pre-demo] 2/5 smoke test (text-only, strict config)"
python3 scripts/run_integrated.py \
  --image-path data/eval/images/crosswalk/Crosswalk_1.png \
  --task crosswalk_signal \
  --compression 192 \
  --profile label_only_eval \
  --warmup-images 1 \
  --no-tts \
  --output-dir reports/pre_demo_smoke

echo "[pre-demo] 3/5 optional TTS smoke"
if [ -d "$HOME/vibevoice_test/voices" ]; then
  python3 scripts/run_integrated.py \
    --image-path data/eval/images/crosswalk/Crosswalk_1.png \
    --task crosswalk_signal \
    --compression 192 \
    --profile sentence_demo_fast \
    --tts microsoft/VibeVoice-Realtime-0.5B \
    --voices-dir "$HOME/vibevoice_test/voices" \
    --strict-demo \
    --warmup-images 1 \
    --output-dir reports/pre_demo_tts
else
  echo "[pre-demo] voices directory missing; skipping TTS smoke"
fi

echo "[pre-demo] 4/5 capture manifest"
python3 scripts/capture_run_manifest.py \
  --results "$RESULTS_FILE" \
  --out reports/run_manifest.json

echo "[pre-demo] 5/5 schema sanity"
python3 - <<'PY'
import json
import os
from pathlib import Path

p = Path(os.environ.get('RESULTS_FILE', 'reports/vqa_with_tts_results.json'))
if p.exists():
    d = json.loads(p.read_text())
    print('results_file', str(p))
    print('schema_version', d.get('schema_version', 'missing'))
    print('has_results', isinstance(d.get('results'), list))
else:
    print(f'results file missing: {p}')
PY

echo "[pre-demo] complete"
