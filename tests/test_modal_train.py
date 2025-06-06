import os
import subprocess
import sys

def test_modal_train_invokes_train(tmp_path):
    stub_dir = tmp_path / "scripts"
    stub_dir.mkdir()
    (stub_dir / "train.py").write_text(
        "def main(*a, **k):\n    print('CALLED')\n"
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{stub_dir}:{os.getcwd()}"

    result = subprocess.run(
        [sys.executable, "-m", "scripts.modal_train", "--data", "d", "--out", "o", "--model", "m"],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert "CALLED" in result.stdout
    assert result.returncode == 0

