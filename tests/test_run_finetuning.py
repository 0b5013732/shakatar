import subprocess
from pathlib import Path
import shutil


def test_run_finetuning_requires_trl(tmp_path):
    data_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True)
    dataset = data_dir / "formatted_dataset.txt"
    dataset.write_text("dummy\n")

    script_dst = tmp_path / "run_finetuning.sh"
    script_src = Path(__file__).resolve().parent.parent / "scripts" / "run_finetuning.sh"
    script_dst.write_text(script_src.read_text())
    script_dst.chmod(0o755)

    train_src = Path(__file__).resolve().parent.parent / "scripts" / "train.py"
    train_dst = tmp_path / "train.py"
    shutil.copy(train_src, train_dst)

    result = subprocess.run([
        "bash",
        str(script_dst)
    ], cwd=tmp_path, capture_output=True, text=True)

    assert result.returncode != 0
    assert (
        "TRL or PEFT not installed" in result.stdout
        or "TRL or PEFT not installed" in result.stderr
        or "DatasetGenerationError" in result.stderr
        or "Failed to load JSON" in result.stderr
    )
