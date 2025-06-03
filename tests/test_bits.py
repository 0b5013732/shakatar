import importlib
import os
import sys
import types
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class DummyArgs:
    def __init__(self, *a, **k):
        DummyArgs.kwargs = k


class DummyTrainer:
    def __init__(self, *a, **k):
        pass

    def train(self):
        pass

    def save_model(self, *a, **k):
        pass


def setup_modules(called):
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)

    ds_stub = types.SimpleNamespace(map=lambda *a, **k: [])
    fake_ds = types.ModuleType("datasets")
    fake_ds.load_dataset = lambda *a, **k: {"train": ds_stub}

    def fake_from_pretrained(*a, **k):
        called.update(k)
        return types.SimpleNamespace(to=lambda *_: None)

    fake_tf = types.ModuleType("transformers")
    fake_tf.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: types.SimpleNamespace(pad_token=None, eos_token="</s>")
    )
    fake_tf.AutoModelForCausalLM = types.SimpleNamespace(from_pretrained=fake_from_pretrained)
    fake_tf.DataCollatorForLanguageModeling = lambda tokenizer=None, mlm=False: object()
    fake_tf.Trainer = DummyTrainer
    fake_tf.TrainingArguments = DummyArgs

    return {"torch": fake_torch, "datasets": fake_ds, "transformers": fake_tf}


def test_load_in_8bit(tmp_path):
    called = {}
    modules = setup_modules(called)
    with mock.patch.dict(sys.modules, modules):
        train = importlib.import_module("scripts.train")
        importlib.reload(train)
        train.main("data", str(tmp_path), "model", 1, 1, -1, False, 8)
        assert called.get("load_in_8bit") is True
    sys.modules.pop("scripts.train", None)


def test_load_in_4bit(tmp_path):
    called = {}
    modules = setup_modules(called)
    with mock.patch.dict(sys.modules, modules):
        train = importlib.import_module("scripts.train")
        importlib.reload(train)
        train.main("data", str(tmp_path), "model", 1, 1, -1, False, 4)
        assert called.get("load_in_4bit") is True
    sys.modules.pop("scripts.train", None)


def test_bitsandbytes_importable():
    try:
        import bitsandbytes
        # If the import succeeds, the test passes.
        # You could add an assertion here like assert True, but it's implicit.
    except ImportError:
        # If bitsandbytes is not installed, an ImportError will be raised.
        # We re-raise an AssertionError to make the test failure more explicit
        # about the missing dependency.
        raise AssertionError(
            "The 'bitsandbytes' library is not installed, but it is required when "
            "using 4-bit or 8-bit quantization options (e.g., --bits 4 or --bits 8). "
            "Please install it by running 'pip install bitsandbytes'."
        )
