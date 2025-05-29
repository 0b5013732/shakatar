import os
import sys
import types
import importlib
import tempfile
import unittest

# Ensure scripts package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Minimal stubs for optional dependencies to avoid heavy imports
if "torch" not in sys.modules:
    torch_stub = types.SimpleNamespace()
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)
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
resolve_model_path = train.resolve_model_path


class ModelPathTest(unittest.TestCase):
    def test_exact_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = resolve_model_path(tmp)
            self.assertEqual(result, tmp)

    def test_colon_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            alt = os.path.join(tmp, "model")
            os.mkdir(alt)
            colon_name = alt.replace("-", ":")
            result = resolve_model_path(colon_name)
            self.assertEqual(result, alt)

    def test_missing(self):
        with self.assertRaises(ValueError):
            resolve_model_path("foo:bar")


if __name__ == "__main__":
    unittest.main()
