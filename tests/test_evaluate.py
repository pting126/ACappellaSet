"""Full CLI scoring checks with analytically known signals."""

import csv
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf

from evaluate import evaluate


class EvaluationTests(unittest.TestCase):
    def test_known_sisdr_improvement_silence_and_excluded_composition(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = np.tile([1., -1., 1., -1.], 100).astype(np.float32)
            noise = np.tile([1., 1., -1., -1.], 100).astype(np.float32)
            signals = {'ref': target, 'mix': target + noise, 'pred': target + .5 * noise,
                       'silent': target * 10 ** (-46 / 20)}
            for name, signal in signals.items():
                sf.write(root / (name + '.wav'), np.stack([signal, signal], axis=1), 44100, subtype='FLOAT')
            base = dict(model='fixture', corpus='JC', protocol='mixture_only', song_id='one',
                        track_id='one', role='S', mixture='mix.wav', reference='ref.wav', estimate='pred.wav')
            rows = [base, dict(base, role='A', reference='silent.wav'),
                    dict(base, corpus='AB', song_id='PC01', track_id='renamed_variant',
                         mixture='missing.wav', reference='missing.wav', estimate='missing.wav'),
                    dict(base, corpus='AB', song_id='PC02', track_id='PC02')]
            manifest = root / 'pairs.csv'
            with manifest.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(base))
                writer.writeheader(); writer.writerows(rows)
            report = evaluate(manifest, root / 'scores')
            self.assertEqual(report['retained_pairs'], 2)
            self.assertEqual(len(report['excluded_pairs']), 2)
            for summary in report['summaries']:
                self.assertAlmostEqual(summary['overall_si_sdri'], 10 * math.log10(4), places=5)
            self.assertTrue((root / 'scores/metrics_per_item.csv').is_file())
            with self.assertRaises(FileExistsError):
                evaluate(manifest, root / 'scores')


if __name__ == '__main__':
    unittest.main()
