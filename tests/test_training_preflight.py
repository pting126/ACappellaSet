"""Preflight must reject unusable inputs before creating a training run."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from models.registry import ROOT, resolve
from prepare_data import sha256


class TrainingPreflightTests(unittest.TestCase):
    def fixture(self, root, model='K5'):
        spec = resolve(model)
        init = root / 'initializer.th'
        init.write_bytes(b'fixture; never deserialized')
        paths = []
        names = ('train_sk', 'valid_jc', 'valid_sk')
        for name in names:
            directory = root / name
            directory.mkdir()
            (directory / 'DATA_PROTOCOL.json').write_text('{}')
            receipt = {'set': name, 'metadata_only': False,
                       'outputs': {'DATA_PROTOCOL.json': sha256(directory / 'DATA_PROTOCOL.json')}}
            (directory / 'preparation.json').write_text(json.dumps(receipt))
            paths.append(directory)
        options = SimpleNamespace(deterministic=False, initialization=init,
            speaker_initialization=None, manifest=paths[0], valid_jc=paths[1],
            valid_sk=paths[2], cpu=True, device=0, max_updates=1,
            output=root / 'run', resume=None, dry_run=True, workers=0)
        return options, spec

    def run_preflight(self, options, spec):
        from training.conditioned import run
        official = json.loads((ROOT / 'third_party/initialization.json').read_text())['official_sha256']
        def fixture_hash(path):
            return official if Path(path) == options.initialization else sha256(path)
        # Only the original initializer identity is substituted. No model is built
        # or loaded: these tests exercise input validation, not training math.
        with patch('training.conditioned.sha256', side_effect=fixture_hash), \
                patch('training.conditioned.initialize') as initialize, \
                contextlib.redirect_stdout(io.StringIO()):
            try:
                run(options, spec)
            finally:
                initialize.assert_not_called()
                self.assertFalse(options.output.exists())

    def test_missing_or_wrong_singer_initializer_rejected_by_dry_run(self):
        for supplied in (False, True):
            with self.subTest(supplied=supplied), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                options, spec = self.fixture(root)
                if supplied:
                    options.speaker_initialization = root / 'wrong.pt'
                    options.speaker_initialization.write_bytes(b'wrong encoder')
                with self.assertRaisesRegex(ValueError, 'ReDimNet2 initializer'):
                    self.run_preflight(options, spec)

    def test_part_only_dry_run_does_not_require_singer_initializer(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.run_preflight(*self.fixture(Path(tmp), 'K1'))

    def test_wrong_training_or_validation_set_rejected(self):
        for field in ('manifest', 'valid_jc', 'valid_sk'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                options, spec = self.fixture(Path(tmp), 'K1')
                path = getattr(options, field) / 'preparation.json'
                receipt = json.loads(path.read_text())
                receipt['set'] = 'test_sk'
                path.write_text(json.dumps(receipt))
                with self.assertRaisesRegex(ValueError, 'Expected prepared set'):
                    self.run_preflight(options, spec)


if __name__ == '__main__':
    unittest.main()
