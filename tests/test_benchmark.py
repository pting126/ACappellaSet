"""Benchmark output-role semantics and unavailable-query handling."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import soundfile as sf

class BenchmarkTests(unittest.TestCase):
    def fixture(self,root,role):
        target=(.1*np.sin(np.arange(1000)*.13)).astype('float32')
        target=np.stack([target,target])
        mixture=target+.03*np.cos(np.arange(1000)*.07).astype('float32')[None]
        sf.write(root/'mix.wav',mixture.T,44100,subtype='FLOAT')
        sf.write(root/'target.wav',target.T,44100,subtype='FLOAT')
        row=dict(split='test',song_id='song',track_id='song',part=role,singer_id='singer',
                 mixture_path=str(root/'mix.wav'),target_path=str(root/'target.wav'))
        with (root/'eval_pairs_test_correct.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=row);writer.writeheader();writer.writerow(row)
        return target,mixture

    def test_specialist_scores_target_index_one(self):
        import torch
        from benchmark import evaluate_fixed
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);target,mixture=self.fixture(root,'A')
            spec={'id':'specialist','source_order':['nonalto','alto']}
            with patch('models.fixed.separate',return_value=np.stack([mixture-target,target])):
                rows,excluded=evaluate_fixed(None,None,spec,root,'JC',torch.device('cpu'))
            self.assertEqual(len(rows),1);self.assertEqual(rows[0]['role'],'A')
            self.assertGreater(rows[0]['si_sdr'],60);self.assertEqual(excluded,[])

    def test_jc_head_never_relabels_to_bar(self):
        import torch
        from benchmark import evaluate_fixed
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);target,mixture=self.fixture(root,'Bar')
            spec={'id':'Fixed_6stem_htdemucs_JC','source_order':['lead_vocal']}
            with patch('models.fixed.separate',return_value=target[None]):
                rows,excluded=evaluate_fixed(None,None,spec,root,'SK',torch.device('cpu'))
            self.assertEqual(rows,[]);self.assertEqual(excluded[0]['reason'],'no_matching_output')

    def test_unavailable_conditioned_query_skips_before_inference(self):
        import torch
        from training.validation import evaluate_model
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.fixture(root,'Lead')
            (root/'enrollment_index.csv').write_text('singer_id\n')
            with patch('training.validation.separate') as separate:
                rows,excluded=evaluate_model(None,{'conditioning':'role'},{'id':'part'},root,'JC',torch.device('cpu'),
                                              split='test',allowed_roles=['S','A','T','B','VP','Bar'])
                separate.assert_not_called()
            self.assertEqual(rows,[]);self.assertEqual(excluded[0]['reason'],'unavailable_part_query')

if __name__=='__main__':unittest.main()
