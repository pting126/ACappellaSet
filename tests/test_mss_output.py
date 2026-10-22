"""Exercise the patched native writer without importing unrelated MSS models."""
import ast
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
import soundfile as sf


@unittest.skipUnless(os.environ.get('ACAPPELLASET_MSS_CHECKOUT'),
                     'Set ACAPPELLASET_MSS_CHECKOUT to the patched, pinned MSS checkout')
class MSSOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(os.environ['ACAPPELLASET_MSS_CHECKOUT']) / 'valid.py'
        module = ast.parse(path.read_text())
        function = next(node for node in module.body
                        if isinstance(node, ast.FunctionDef) and node.name == 'write_estimate')
        namespace = {'np': np, 'sf': sf}
        # The actual pure writer from the installed patch; no model or writer mock.
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
        cls.write = staticmethod(namespace['write_estimate'])

    def test_explicit_float_is_exact_even_below_unit_peak(self):
        with tempfile.TemporaryDirectory() as tmp:
            samples = np.full((2, 31), .1234567, dtype='float32')
            path = self.write(Path(tmp) / 'float', samples, 44100, 'FLOAT')
            self.assertEqual(Path(path).suffix, '.wav')
            self.assertEqual(sf.info(path).subtype, 'FLOAT')
            np.testing.assert_array_equal(sf.read(path, dtype='float32')[0].T, samples)

    def test_explicit_pcm_and_legacy_auto_formats(self):
        with tempfile.TemporaryDirectory() as tmp:
            for index, (mode, peak, extension, subtype) in enumerate([
                ('PCM_16', .2, '.wav', 'PCM_16'), ('auto', .2, '.flac', 'PCM_16'),
                ('auto', 1.2, '.wav', 'FLOAT')]):
                with self.subTest(mode=mode, peak=peak):
                    path = self.write(Path(tmp) / str(index), np.full((2, 31), peak, dtype='float32'), 44100, mode)
                    self.assertEqual(Path(path).suffix, extension)
                    self.assertEqual(sf.info(path).subtype, subtype)

    def test_unknown_format_rejected_without_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                self.write(Path(tmp) / 'bad', np.ones((2, 31)), 44100, 'unknown')
            self.assertEqual(list(Path(tmp).iterdir()), [])


if __name__ == '__main__': unittest.main()
