import importlib
import os
import sys
import types
from unittest import mock

# Ensure scripts package is importable
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


def test_destroy_process_group_called(tmp_path):
    destroy_mock = mock.Mock()

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: False, device_count=lambda: 0
    )
    fake_torch.distributed = types.SimpleNamespace(
        is_initialized=lambda: True,
        destroy_process_group=destroy_mock,
    )

    ds_stub = types.SimpleNamespace(map=lambda *a, **k: [])
    fake_ds = types.ModuleType("datasets")
    fake_ds.load_dataset = lambda *a, **k: {"train": ds_stub}

    fake_tf = types.ModuleType("transformers")
    fake_tf.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: types.SimpleNamespace(
            pad_token=None, eos_token="</s>"
        )
    )
    fake_tf.AutoModelForCausalLM = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: types.SimpleNamespace(to=lambda *_: None)
    )
    fake_tf.BitsAndBytesConfig = types.SimpleNamespace
    fake_tf.DataCollatorForLanguageModeling = lambda tokenizer=None, mlm=False: object()
    fake_tf.Trainer = DummyTrainer
    fake_tf.TrainingArguments = DummyArgs

    integ_stub = types.ModuleType("transformers.integrations")
    bnb_stub = types.ModuleType("transformers.integrations.bitsandbytes")
    bnb_stub.validate_bnb_backend_availability = lambda raise_exception=True: None

    fake_peft = types.ModuleType("peft")
    fake_peft.get_peft_model = lambda model, config: model
    fake_peft.LoraConfig = lambda **kwargs: types.SimpleNamespace(**kwargs)
    fake_peft.TaskType = types.SimpleNamespace(CAUSAL_LM="CAUSAL_LM")

    modules = {
        "torch": fake_torch,
        "datasets": fake_ds,
        "transformers": fake_tf,
        "transformers.integrations": integ_stub,
        "transformers.integrations.bitsandbytes": bnb_stub,
        "peft": fake_peft,
    }

    with mock.patch.dict(sys.modules, modules):
        train = importlib.import_module("scripts.train")
        importlib.reload(train)
        train.main("data", str(tmp_path), "model", 1, 1, -1)
        assert destroy_mock.called
