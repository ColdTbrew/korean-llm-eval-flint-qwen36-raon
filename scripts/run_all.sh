#!/bin/bash
# Usage: URL=http://127.0.0.1:8082/v1/chat/completions MODEL=qwen3.6-35b-a3b-official OUT=results/qwen36-official bash scripts/run_all.sh
# Flint was run without presence_penalty (EVAL_EXTRA='{}'); Qwen/Raon with presence_penalty 1.5, min_p 0.
set -e
export FLINT_URL=${URL:-http://127.0.0.1:8080/v1/chat/completions} EVAL_MODEL=${MODEL:-flint}
export EVAL_EXTRA=${EVAL_EXTRA:-'{"presence_penalty":1.5,"min_p":0.0}'}
HERE=$(cd "$(dirname "$0")" && pwd); mkdir -p "$OUT"; cd "$OUT"
cp "$HERE/../data/long_ko.txt" .
python3 "$HERE/bench.py" 2>&1 | tee bench.log
python3 "$HERE/run_eval.py" "$HERE/../data/prompts.json" results.jsonl 2>&1 | tee eval.log
