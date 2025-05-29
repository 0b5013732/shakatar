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
import os
from pathlib import Path
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def select_device(local_rank: int) -> str:
    """Return a torch device string for ``local_rank``.

    Raises ``ValueError`` if ``local_rank`` points to a GPU that does not
    exist in the current environment.
    """

    if local_rank >= 0 and torch.cuda.is_available():
        count = torch.cuda.device_count()
        if local_rank >= count:
            raise ValueError(
                f"Invalid device index {local_rank}; only {count} GPU(s) available."
            )
        return f"cuda:{local_rank}"

    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_model_path(base_model: str) -> str:
    """Return a path or model name usable by HuggingFace APIs.

    Names from the Ollama CLI can include ``:`` which is not accepted by
    HuggingFace.  If ``base_model`` contains ``:`` and the direct path does not
    exist, the function also checks for a sibling path where ``:`` is replaced
    with ``-``.  A ``ValueError`` is raised if neither exists.
    """

    sanitized = base_model
    if ":" in base_model:
        alt = base_model.replace(":", "-")
        if Path(base_model).exists():
            sanitized = base_model
        elif Path(alt).exists():
            sanitized = alt
        else:
            raise ValueError(
                "Model names containing ':' are not valid HuggingFace IDs. "
                "Please provide a path to a local model directory or a valid repo ID."
            )
    return sanitized


def main(
    data_path: str,
    model_dir: str,
    base_model: str,
    epochs: int,
    batch_size: int,
    local_rank: int,
):
    """Run the fine-tuning loop.

    ``local_rank`` should be supplied when launching with ``torchrun`` so that
    each process uses the correct GPU.  When ``local_rank`` is ``-1`` (the
    default) the script behaves as before and simply picks ``cuda`` if
    available.
    """
    device = select_device(local_rank)

    print(
        f"Training with {data_path}; base model {base_model}; output to {model_dir}"
    )
    print(f"Detected device: {device}")

    dataset = load_dataset("json", data_files=data_path)["train"]

    sanitized = resolve_model_path(base_model)

    tokenizer = AutoTokenizer.from_pretrained(sanitized)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(sanitized)
    model.to(device)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length")

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    fp16 = torch.cuda.is_available()

    args = TrainingArguments(
        output_dir=model_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        logging_steps=10,
        save_steps=1000,
        save_total_limit=2,
        fp16=fp16,
        local_rank=local_rank,
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
    parser.add_argument(
        "--local_rank",
        type=int,
        default=int(os.environ.get("LOCAL_RANK", -1)),
        help="Provided by torchrun for distributed training",
    )
    args = parser.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    main(
        args.data,
        args.out,
        args.model,
        args.epochs,
        args.batch_size,
        args.local_rank,
    )
