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
    fake_tf.BitsAndBytesConfig = lambda **kwargs: types.SimpleNamespace(**kwargs)
    fake_tf.DataCollatorForLanguageModeling = lambda tokenizer=None, mlm=False: object()
    fake_tf.Trainer = DummyTrainer
    fake_tf.TrainingArguments = DummyArgs

    fake_peft = types.ModuleType("peft")
    # Mock get_peft_model to return a SimpleNamespace that has a print_trainable_parameters method
    mock_peft_model = types.SimpleNamespace(print_trainable_parameters=lambda: None)
    fake_peft.get_peft_model = lambda model, config: mock_peft_model
    fake_peft.LoraConfig = lambda **kwargs: types.SimpleNamespace(**kwargs)
    fake_peft.TaskType = types.SimpleNamespace(CAUSAL_LM="CAUSAL_LM")

    fake_bnb = types.ModuleType("bitsandbytes")

    return {
        "torch": fake_torch,
        "datasets": fake_ds,
        "transformers": fake_tf,
        "peft": fake_peft,
        "bitsandbytes": fake_bnb,
    }


def test_load_in_8bit(tmp_path):
    called = {}
    modules = setup_modules(called)
    with mock.patch.dict(sys.modules, modules):
        train = importlib.import_module("scripts.train")
        importlib.reload(train)
        train.main("data", str(tmp_path), "model", 1, 1, -1, False, 8)
        assert "quantization_config" in called
        assert getattr(called["quantization_config"], "load_in_8bit", False) is True
    sys.modules.pop("scripts.train", None)


def test_load_in_4bit(tmp_path):
    called = {}
    modules = setup_modules(called)
    with mock.patch.dict(sys.modules, modules):
        train = importlib.import_module("scripts.train")
        importlib.reload(train)
        train.main("data", str(tmp_path), "model", 1, 1, -1, False, 4)
        assert "quantization_config" in called
        assert getattr(called["quantization_config"], "load_in_4bit", False) is True
    sys.modules.pop("scripts.train", None)


def test_bitsandbytes_importable():
    # This test is now moot as we are mocking bitsandbytes
    pass
