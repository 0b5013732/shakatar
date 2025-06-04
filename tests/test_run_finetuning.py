import subprocess
from pathlib import Path

def test_run_finetuning_requires_axolotl(tmp_path):
    # create minimal dataset and config
    data_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True)
    dataset = data_dir / "formatted_dataset.txt"
    dataset.write_text("dummy\n")

    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"dataset:\n  path: {dataset}\ntraining:\n  output_dir: {tmp_path / 'out'}\n"
    )

    script_dst = tmp_path / "run_finetuning.sh"
    script_src = Path(__file__).resolve().parent.parent / "scripts" / "run_finetuning.sh"
    script_dst.write_text(script_src.read_text())
    script_dst.chmod(0o755)

    result = subprocess.run([
        "bash",
        str(script_dst)
    ], cwd=tmp_path, capture_output=True, text=True)

    assert result.returncode != 0
    assert "Axolotl CLI not found" in result.stdout
