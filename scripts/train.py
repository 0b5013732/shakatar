"""Placeholder script for fine-tuning a Llama-based model.
Requires the `transformers` library and appropriate hardware.
"""

import argparse
from pathlib import Path


def main(data_path: str, model_dir: str):
    # TODO: Implement fine-tuning using HuggingFace Transformers or similar.
    print(f"Training with {data_path}; output to {model_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="../data/processed/corpus.jsonl")
    parser.add_argument("--out", default="../model")
    args = parser.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    main(args.data, args.out)
