"""Simple script for fine-tuning a Llama-based model using HuggingFace.

The script expects a JSONL file with objects containing a ``text`` field.  It
performs causal language modelling (next-token prediction) on that corpus.

Usage example:

```
python3 train.py --data data/processed/corpus.jsonl --out model/ \
    --model llama-base --epochs 3 --batch-size 2
```
"""

import argparse
from pathlib import Path

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def main(data_path: str, model_dir: str, base_model: str, epochs: int, batch_size: int):
    print(f"Training with {data_path}; base model {base_model}; output to {model_dir}")

    dataset = load_dataset("json", data_files=data_path)["train"]

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_model)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length")

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir=model_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        logging_steps=10,
        save_steps=1000,
        save_total_limit=2,
    )

    trainer = Trainer(model=model, args=args, train_dataset=tokenized, data_collator=collator)
    trainer.train()
    trainer.save_model(model_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="../data/processed/corpus.jsonl")
    parser.add_argument("--out", default="../model")
    parser.add_argument(
        "--model", default="llama-base", help="Base model name or path"
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    main(args.data, args.out, args.model, args.epochs, args.batch_size)
