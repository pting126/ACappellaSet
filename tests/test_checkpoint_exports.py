"""Safe inference-export loading contracts; no research weights required."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from models import conditioned


class TinySeparator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2))
        self.role_embedding = None


class SafeConditionedExports(unittest.TestCase):
    def fixture(self, directory, **changes):
        root = Path(directory)
        encoder = root / 'encoder.yaml'
        encoder.write_text('encoder: test\n')
        config = root / 'config.json'
        config.write_text(json.dumps({'source_mode': 'target_residual',
                                     'encoder_config': 'encoder.yaml',
                                     'encoder_config_sha256': self.sha(encoder)}))
        payload = {'format': 'acappellaset-conditioned-inference-v1', 'step': 48000,
                   'model_state': {'weight': torch.tensor([1., 2.])}}
        payload.update(changes)
        checkpoint = root / 'weights.pt'
        torch.save(payload, checkpoint)
        spec = {'checkpoint_sha256': 'original-research-checkpoint', 'config_sha256': self.sha(config),
                'step': 48000, 'source_order': ['target', 'residual'],
                'portable_checkpoint': {'sha256': self.sha(checkpoint),
                                        'format': 'acappellaset-conditioned-inference-v1'}}
        return checkpoint, config, spec

    @staticmethod
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_safe_load_preserves_tensors_and_uses_weights_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint, config, spec = self.fixture(tmp)
            with patch.object(conditioned, 'ROOT', Path(tmp)), \
                 patch.object(conditioned, 'build_model', return_value=TinySeparator()), \
                 patch.object(torch, 'load', wraps=torch.load) as load:
                model, _ = conditioned.load_model(checkpoint, config, spec, torch.device('cpu'))
                self.assertTrue(load.call_args.kwargs['weights_only'])
                torch.testing.assert_close(model.weight, torch.tensor([1., 2.]), rtol=0, atol=0)
                self.assertFalse(model.training)

    def test_unregistered_bytes_are_rejected_before_deserialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint, config, spec = self.fixture(tmp)
            with checkpoint.open('ab') as stream:
                stream.write(b'changed')
            with patch.object(torch, 'load') as load, self.assertRaisesRegex(ValueError, 'checksum'):
                conditioned.load_model(checkpoint, config, spec, torch.device('cpu'))
            load.assert_not_called()

    def test_wrong_step_or_extra_training_arguments_are_rejected(self):
        for change in ({'step': 131000}, {'args': {'private': '/not-for-release'}}):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                checkpoint, config, spec = self.fixture(tmp, **change)
                with patch.object(conditioned, 'ROOT', Path(tmp)), self.assertRaises(ValueError):
                    conditioned.load_model(checkpoint, config, spec, torch.device('cpu'))

    def test_training_export_remains_safe_and_compatible(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint, config, spec = self.fixture(tmp)
            spec.pop('portable_checkpoint')
            spec.update(checkpoint_sha256=self.sha(checkpoint), safe_training_export=True)
            with patch.object(conditioned, 'ROOT', Path(tmp)), \
                 patch.object(conditioned, 'build_model', return_value=TinySeparator()), \
                 patch.object(torch, 'load', wraps=torch.load) as load:
                conditioned.load_model(checkpoint, config, spec, torch.device('cpu'))
                self.assertTrue(load.call_args.kwargs['weights_only'])


if __name__ == '__main__':
    unittest.main()
