import argparse
import modal
from . import train

app = modal.App("shakatar-train")

@app.function()
def run_training(data_path: str, model_dir: str, base_model: str,
                  epochs: int, batch_size: int, bits: int,
                  gradient_checkpointing: bool = False) -> None:
    train.main(
        data_path,
        model_dir,
        base_model,
        epochs,
        batch_size,
        -1,
        gradient_checkpointing,
        bits,
    )


@app.local_entrypoint()
def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune model on Modal")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bits", type=int, default=16)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    args = parser.parse_args()

    run_training.remote(
        args.data,
        args.out,
        args.model,
        args.epochs,
        args.batch_size,
        args.bits,
        args.gradient_checkpointing,
    )
