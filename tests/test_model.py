import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from pannello import model


class ModelCacheTests(unittest.TestCase):
    def setUp(self):
        model._MODEL = None
        model._PANEL_CLASS_IDS = None
        model._MODEL_CACHE.clear()

    def test_load_model_caches_per_weights_path(self):
        calls = []

        class FakeYOLO:
            def __init__(self, weights):
                self.weights = weights
                self.names = {0: 'panel'}
                calls.append(weights)

            def to(self, device):
                self.device = device
                return self

        torch_mod = types.ModuleType('torch')
        torch_mod.cuda = types.SimpleNamespace(is_available=lambda: False)
        ultralytics_mod = types.ModuleType('ultralytics')
        ultralytics_mod.YOLO = FakeYOLO

        with tempfile.TemporaryDirectory() as td, \
             patch.dict(sys.modules, {'torch': torch_mod, 'ultralytics': ultralytics_mod}):
            weights_a = Path(td) / 'a.pt'
            weights_b = Path(td) / 'b.pt'
            weights_a.write_text('a', encoding='utf-8')
            weights_b.write_text('b', encoding='utf-8')

            first = model.load_model(str(weights_a))
            second = model.load_model(str(weights_b))
            third = model.load_model(str(weights_a))

        self.assertIs(first, third)
        self.assertIsNot(first, second)
        self.assertEqual(calls, [str(weights_a), str(weights_b)])


if __name__ == '__main__':
    unittest.main()
