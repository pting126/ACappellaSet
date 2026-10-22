"""Exercise documented entry points as separate processes with real WAV files."""
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf
from models.registry import ROOT
from prepare_specialists import STEMS, SPLITS


class CLIWorkflowTests(unittest.TestCase):
    def invoke(self, script, *arguments, success=True):
        result = subprocess.run([sys.executable, str(ROOT / script), *map(str, arguments)],
                                cwd=ROOT, text=True, capture_output=True, timeout=90)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_evaluation_cli_known_score_and_overwrite_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = np.tile([1., -1., 1., -1.], 100).astype('float32') * .1
            noise = np.tile([1., 1., -1., -1.], 100).astype('float32') * .1
            for name, samples in [('ref', target), ('mix', target + noise), ('pred', target + .5 * noise)]:
                sf.write(root / (name + '.wav'), np.stack([samples, samples], axis=1), 44100, subtype='FLOAT')
            row = dict(model='synthetic', corpus='JC', protocol='mixture_only', song_id='fixture',
                       track_id='fixture', role='S', mixture='mix.wav', reference='ref.wav', estimate='pred.wav')
            manifest = root / 'pairs.csv'
            with manifest.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=row)
                writer.writeheader(); writer.writerow(row)
            args = ('--manifest', manifest, '--output', root / 'scores')
            self.invoke('evaluate.py', *args)
            output = root / 'scores/metrics_overall.json'
            before = output.read_bytes()
            report = json.loads(before)
            self.assertEqual(report['retained_pairs'], 1)
            for summary in report['summaries']:
                self.assertAlmostEqual(summary['overall_si_sdri'], 10 * math.log10(4), places=5)
            self.invoke('evaluate.py', *args, success=False)
            self.assertEqual(output.read_bytes(), before)

    def test_specialist_preparation_cli_both_corpus_recipes(self):
        for corpus in ('SK', 'JC'):
            with self.subTest(corpus=corpus), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); source = root / 'input'; output = root / 'output'
                for split in SPLITS[corpus]: (source / split).mkdir(parents=True)
                track = source / SPLITS[corpus][0] / 'fixture'; track.mkdir()
                stems = []
                for index, name in enumerate(STEMS[corpus]):
                    signal = np.full((64, 2), .01 * (index + 1), dtype='float32')
                    stems.append(signal)
                    sf.write(track / (name + '.wav'), signal, 44100, subtype='FLOAT')
                sf.write(track / 'mixture.wav', np.full((64, 2), .42, dtype='float32'), 44100, subtype='FLOAT')
                self.invoke('prepare_specialists.py', '--corpus', corpus, '--input', source,
                            '--output', output, '--targets', 'alto', '--allow-subset')
                prepared = output / 'alto' / SPLITS[corpus][0] / 'fixture'
                mixture = sf.read(prepared / 'mixture.wav', dtype='float32')[0]
                residual = sf.read(prepared / 'nonalto.wav', dtype='float32')[0]
                total = np.zeros_like(stems[0])
                for name, signal in zip(STEMS[corpus], stems):
                    if corpus == 'SK' or name != 'alto': total += signal
                if corpus == 'SK':
                    np.testing.assert_array_equal(mixture, total)
                    total -= stems[STEMS[corpus].index('alto')]
                else:
                    np.testing.assert_array_equal(mixture, np.full((64, 2), .42, dtype='float32'))
                np.testing.assert_array_equal(residual, total)


if __name__ == '__main__':
    unittest.main()
