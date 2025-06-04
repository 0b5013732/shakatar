import importlib
import sys
from unittest import mock
import types

class DummyTokenizer:
    def __init__(self):
        self.pad_token = None
        self.eos_token = "</s>"

    def __call__(self, text, return_overflowing_tokens=True, truncation=True,
                 max_length=512, stride=0, padding="longest", return_tensors="pt"):
        if self.pad_token is None:
            raise ValueError("pad_token is None")
        return {"input_ids": [[0, 1]]}

    def batch_decode(self, ids, skip_special_tokens=True):
        return ["decoded"]

def test_chunk_text_sets_pad_token():
    dummy = DummyTokenizer()
    tf_stub = types.ModuleType("transformers")
    tf_stub.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *a, **k: dummy)
    with mock.patch.dict(sys.modules, {"transformers": tf_stub}):
        if "scripts.chunk_text" in sys.modules:
            del sys.modules["scripts.chunk_text"]
        ct = importlib.import_module("scripts.chunk_text")
    assert dummy.pad_token == dummy.eos_token
    chunks = ct.chunk_text("text", dummy, 10, 2)
    assert chunks == ["decoded"]

