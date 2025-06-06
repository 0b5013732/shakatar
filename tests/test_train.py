import importlib
import os
import sys
import types
import unittest
from unittest import mock

# Ensure scripts package is importable when tests are run from the tests folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Create a minimal fake torch module for testing
fake_torch = types.ModuleType('torch')
fake_cuda = types.SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)
fake_torch.cuda = fake_cuda

fake_datasets = types.ModuleType('datasets')
fake_datasets.load_dataset = lambda *a, **k: {'train': []}

fake_transformers = types.ModuleType('transformers')
fake_transformers.AutoTokenizer = object
fake_transformers.AutoModelForCausalLM = object
fake_transformers.BitsAndBytesConfig = object
fake_transformers.DataCollatorForLanguageModeling = object
fake_transformers.Trainer = object
fake_transformers.TrainingArguments = object
integrations_stub = types.ModuleType('transformers.integrations')
bnb_stub = types.ModuleType('transformers.integrations.bitsandbytes')
bnb_stub.validate_bnb_backend_availability = lambda raise_exception=True: None

fake_peft = types.ModuleType('peft')
fake_peft.get_peft_model = lambda *a, **k: object()
fake_peft.LoraConfig = object
fake_peft.TaskType = types.SimpleNamespace(CAUSAL_LM='CAUSAL_LM')

with mock.patch.dict(sys.modules, {
    'torch': fake_torch,
    'datasets': fake_datasets,
    'transformers': fake_transformers,
    'transformers.integrations': integrations_stub,
    'transformers.integrations.bitsandbytes': bnb_stub,
    'peft': fake_peft,
}):
    train = importlib.import_module('scripts.train')


class DeviceSelectionTest(unittest.TestCase):
    def test_cpu_no_cuda(self):
        with mock.patch.object(train.torch.cuda, 'is_available', return_value=False):
            self.assertEqual(train.select_device(-1), 'cpu')

    def test_default_cuda(self):
        with mock.patch.object(train.torch.cuda, 'is_available', return_value=True):
            self.assertEqual(train.select_device(-1), 'cuda')

    def test_rank_zero(self):
        with mock.patch.object(train.torch.cuda, 'is_available', return_value=True), \
             mock.patch.object(train.torch.cuda, 'device_count', return_value=2):
            self.assertEqual(train.select_device(0), 'cuda:0')

    def test_rank_invalid(self):
        with mock.patch.object(train.torch.cuda, 'is_available', return_value=True), \
             mock.patch.object(train.torch.cuda, 'device_count', return_value=1):
            with self.assertRaises(ValueError):
                train.select_device(1)


if __name__ == '__main__':
    unittest.main()
