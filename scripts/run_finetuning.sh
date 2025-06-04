#!/bin/bash

# Run fine-tuning using Hugging Face TRL with PEFT (LoRA).

DATASET="./data/processed/formatted_dataset.txt"
OUT_DIR="./output"
MODEL="Llama-3.2-1B"

# Check for required dataset
if [ ! -f "$DATASET" ]; then
  echo "Error: Dataset file $DATASET not found."
  echo "Please run scripts/chunk_text.py first."
  exit 1
fi

# Verify required Python packages
python - <<'PY'
import importlib, sys
missing=[pkg for pkg in ("trl","peft") if importlib.util.find_spec(pkg) is None]
if missing:
    print(f"Missing packages: {', '.join(missing)}")
    sys.exit(1)
PY
if [ $? -ne 0 ]; then
  echo "Hugging Face TRL or PEFT not installed."
  exit 1
fi

python scripts/train.py \
  --data "$DATASET" \
  --out "$OUT_DIR" \
  --model "$MODEL" \
  --epochs 3 \
  --batch-size 4 \
  --bits 4 \
  --gradient-checkpointing
