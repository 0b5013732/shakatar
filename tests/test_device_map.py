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
    finally:
        # More robust cleanup
        sys.modules.pop("scripts.train", None)
        # Ensure critical mocks are removed if they were added to sys.modules directly
        # though mock.patch.dict should handle its own entries.
        # This is more about ensuring 'scripts.train' is gone.
        if "torch" in modules: # if fake_torch was patched in
            sys.modules.pop("torch", None)
        if "datasets" in modules:
            sys.modules.pop("datasets", None)
        if "transformers" in modules:
            sys.modules.pop("transformers", None)

def test_auto_device_map_single_gpu(tmp_path):
    called_kwargs_in_from_pretrained = {} 
    fake_torch = types.ModuleType("torch")
    # Simulate single CUDA GPU
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 1)

    ds_stub = types.SimpleNamespace(map=lambda *a, **k: [])
    fake_ds = types.ModuleType("datasets")
    fake_ds.load_dataset = lambda *a, **k: {"train": ds_stub}

    def fake_from_pretrained_capture_kwargs(*a, **k):
        called_kwargs_in_from_pretrained.update(k)
        # Return a mock model that has a 'to' method
        mock_model = types.SimpleNamespace()
        mock_model.to = lambda device: None # Mock the .to() method
        return mock_model

    fake_tf = types.ModuleType("transformers")
    fake_tf.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: types.SimpleNamespace(pad_token=None, eos_token="</s>")
    )
    fake_tf.AutoModelForCausalLM = types.SimpleNamespace(from_pretrained=fake_from_pretrained_capture_kwargs)
    fake_tf.DataCollatorForLanguageModeling = lambda tokenizer=None, mlm=False: object()
    fake_tf.Trainer = DummyTrainer
    fake_tf.TrainingArguments = DummyArgs

    modules_to_patch = {"torch": fake_torch, "datasets": fake_ds, "transformers": fake_tf}
    
    # Clean up scripts.train from sys.modules if it exists from a previous test run (pytest usually isolates but good practice)
    if "scripts.train" in sys.modules:
        del sys.modules["scripts.train"]

    # Store original sys.modules state to restore it properly
    original_sys_modules = sys.modules.copy()

    try:
        # Apply mocks. Using clear=True ensures only our mocks are present during the critical section.
        with mock.patch.dict(sys.modules, modules_to_patch, clear=True):
            train_module = importlib.import_module("scripts.train")
            importlib.reload(train_module) 
            
            train_module.main(
                data_path="dummy_data.jsonl",
                model_dir=str(tmp_path), 
                base_model="dummy_model",
                epochs=1, 
                batch_size=1, 
                local_rank=-1, 
                gradient_checkpointing=False, 
                bits=16 # Key condition: 16-bit training
            )
            
            assert called_kwargs_in_from_pretrained.get("device_map") == "auto", \
                f"Expected device_map='auto', got {called_kwargs_in_from_pretrained.get('device_map')}"
    finally:
        # Restore sys.modules to its original state before this test
        sys.modules.clear()
        sys.modules.update(original_sys_modules)
        # Explicitly remove scripts.train again to be absolutely sure, as import_module caches it.
        if "scripts.train" in sys.modules:
            del sys.modules["scripts.train"]
