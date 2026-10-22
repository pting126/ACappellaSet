"""Selection, query, enrollment, and overlap-add contracts without model weights."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf

from inference import read_enrollment
from models.query import validate_query
from models.registry import entries, resolve, select_checkpoint
from models.windowing import window_starts


class ConditionedContracts(unittest.TestCase):
    def test_selection_is_explicit_and_deduplicated(self):
        conditioned = [s for s in entries() if s['family'] == 'conditioned-htdemucs']
        self.assertEqual(len(conditioned), 9)
        self.assertEqual(sum(len(s['selections']) for s in conditioned), 14)
        with self.assertRaises(ValueError):
            select_checkpoint(resolve('K1'))
        self.assertEqual(select_checkpoint(resolve('K1'), 'sk_shared')['step'], 198000)
        self.assertEqual(select_checkpoint(resolve('K1'), 'jc_inclusive')['step'], 4000)
        self.assertEqual(select_checkpoint(resolve('M3'), 'jc_shared')['checkpoint_sha256'],
                         select_checkpoint(resolve('M3'), 'combined_shared')['checkpoint_sha256'])
        with self.assertRaises(ValueError):
            select_checkpoint(resolve('M3'), 'combined_inclusive')

    def test_query_requirements_and_roles(self):
        self.assertEqual(validate_query(resolve('K1'), 'alto')['role'], 'A')
        self.assertEqual(validate_query(resolve('K5'), enrollment='singer.wav')['role'], None)
        for name in ('Singer_conditioned_htdemucs_SK_full_remix', 'Singer_conditioned_htdemucs_SK_single_clone'):
            spec = resolve(name)
            self.assertEqual(select_checkpoint(spec)['selection'], 'combined_inclusive')
            self.assertEqual(validate_query(spec, enrollment='singer.wav')['role'], None)
            with self.assertRaises(ValueError):
                validate_query(spec)
        for model, kwargs in [('K1', {}), ('K1', {'role': 'banana'}),
                              ('K1', {'role': 'A', 'enrollment': 'x.wav'}),
                              ('K5', {}), ('K5', {'role': 'A', 'enrollment': 'x.wav'}),
                              ('K7', {'role': 'A'}), ('K7', {'enrollment': 'x.wav'})]:
            with self.subTest(model=model, kwargs=kwargs), self.assertRaises(ValueError):
                validate_query(resolve(model), **kwargs)

    def test_unknown_roles_require_explicit_diagnostic(self):
        with self.assertRaises(ValueError):
            validate_query(resolve('K7'), 'Lead', 'x.wav')
        query = validate_query(resolve('K7'), 'Lead', 'x.wav', allow_untrained_role=True)
        self.assertEqual(query['mechanism'], 'enrollment_plus_existing_zero_role')
        with self.assertRaises(ValueError):
            validate_query(resolve('L1'), 'Bar')
        self.assertEqual(validate_query(resolve('L4'), 'Bar', 'x.wav',
                                       allow_untrained_role=True)['role_index'], 3)

    def test_enrollment_crop_then_pad(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'enrollment.wav'
            audio = np.linspace(-.5, .5, 44100, dtype=np.float32)
            sf.write(path, audio, 44100, subtype='FLOAT')
            segment, details = read_enrollment(path, 44100, .5, 1.)
            self.assertEqual(segment.shape, (2, 44100))
            np.testing.assert_array_equal(segment[0, :22050], audio[22050:])
            np.testing.assert_array_equal(segment[0], segment[1])
            self.assertFalse(np.any(segment[:, 22050:]))
            self.assertEqual(details['zero_padded_frames_at_original_rate'], 22050)
            for start in (-1, float('nan'), 1, 3):
                with self.subTest(start=start), self.assertRaises(ValueError):
                    read_enrollment(path, 44100, start, 1.)

    def test_window_coverage_and_final_alignment(self):
        for length in (1, 9, 10, 11, 18, 19, 1001):
            coverage = np.zeros(length)
            starts = window_starts(length, 10, 4)
            for start in starts:
                coverage[start:min(start + 10, length)] += 1
            self.assertTrue(np.all(coverage > 0))
            self.assertEqual(starts[-1], max(0, length - 10))
        for args in ((0, 10, 4), (20, 10, 11), (20, 10, 0)):
            with self.assertRaises(ValueError):
                window_starts(*args)


if __name__ == '__main__':
    unittest.main()
