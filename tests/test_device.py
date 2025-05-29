import os
import sys
import types
import importlib

import pytest

# Ensure scripts package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Provide minimal stubs for optional dependencies
if "torch" not in sys.modules:
    torch_stub = types.SimpleNamespace()
    cuda_stub = types.SimpleNamespace(
        is_available=lambda: False, device_count=lambda: 0
    )
    torch_stub.cuda = cuda_stub
    sys.modules["torch"] = torch_stub

if "datasets" not in sys.modules:
    datasets_stub = types.ModuleType("datasets")
    datasets_stub.load_dataset = lambda *_, **__: None
    sys.modules["datasets"] = datasets_stub

if "transformers" not in sys.modules:
    transformers_stub = types.ModuleType("transformers")
    for attr in [
        "AutoTokenizer",
        "AutoModelForCausalLM",
        "DataCollatorForLanguageModeling",
        "Trainer",
        "TrainingArguments",
    ]:
        setattr(transformers_stub, attr, object)
    sys.modules["transformers"] = transformers_stub

train = importlib.import_module("scripts.train")
select_device = train.select_device


def test_select_device_valid_gpu(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.device_count", lambda: 2)
    assert select_device(1) == "cuda:1"


def test_select_device_invalid_rank(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    with pytest.raises(ValueError):
        select_device(1)


def test_select_device_cpu(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    assert select_device(-1) == "cpu"
