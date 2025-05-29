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

def test_auto_device_map_multi_gpu(tmp_path):
    called = {}
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 2)

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

    modules = {"torch": fake_torch, "datasets": fake_ds, "transformers": fake_tf}
    with mock.patch.dict(sys.modules, modules):
        train = importlib.import_module("scripts.train")
        importlib.reload(train)
        train.main("data", str(tmp_path), "model", 1, 1, -1, False, 16)
        assert called.get("device_map") == "auto"
    sys.modules.pop("scripts.train", None)
