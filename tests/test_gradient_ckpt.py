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

def test_gradient_checkpointing_flag(tmp_path):
    fake_torch = types.ModuleType('torch')
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)

    ds_stub = types.SimpleNamespace(map=lambda *a, **k: [])
    fake_ds = types.ModuleType('datasets')
    fake_ds.load_dataset = lambda *a, **k: {'train': ds_stub}

    fake_tf = types.ModuleType('transformers')
    fake_tf.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: types.SimpleNamespace(
            pad_token=None,
            eos_token='</s>'
        )
    )
    fake_tf.AutoModelForCausalLM = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: types.SimpleNamespace(to=lambda *_: None)
    )
    fake_tf.DataCollatorForLanguageModeling = lambda tokenizer=None, mlm=False: object()
    fake_tf.TrainingArguments = DummyArgs
    fake_tf.Trainer = DummyTrainer

    modules = {'torch': fake_torch, 'datasets': fake_ds, 'transformers': fake_tf}
    with mock.patch.dict(sys.modules, modules):
        train = importlib.import_module('scripts.train')
        importlib.reload(train)
        train.main('data', str(tmp_path), 'model', 1, 1, -1, True)
        assert DummyArgs.kwargs['gradient_checkpointing'] is True
