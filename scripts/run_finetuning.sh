#!/bin/bash

# Run fine-tuning using Hugging Face TRL with PEFT (LoRA).

DATASET="./data/processed/corpus.jsonl"
OUT_DIR="./output"
MODEL="Llama-3.2-1B"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Check for required dataset
if [ ! -f "$DATASET" ]; then
  echo "Error: Dataset file $DATASET not found."
  echo "Please run: node scripts/ingest.js"
  exit 1
fi

# Verify required Python packages
python - <<'PY'
import importlib.util
import sys

missing = [pkg for pkg in ("trl", "peft") if importlib.util.find_spec(pkg) is None]
if missing:
    print(f"Missing packages: {', '.join(missing)}")
    sys.exit(1)
PY
if [ $? -ne 0 ]; then
  echo "Hugging Face TRL or PEFT not installed."
  exit 1
fi

python "$SCRIPT_DIR/train.py" \
  --data "$DATASET" \
  --out "$OUT_DIR" \
  --model "$MODEL" \
  --epochs 3 \
  --batch-size 4 \
  --bits 4 \
  --gradient-checkpointing
