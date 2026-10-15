"""Training contracts that do not require research weights or a GPU."""
import json
from pathlib import Path
import tempfile
import unittest
from models.registry import resolve
from models.trained import resolve_export


class TrainingContracts(unittest.TestCase):
    def test_new_run_identity_and_manifest_paths(self):
        spec=resolve('K5')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);manifest=root/'inference.json';checkpoint=root/'inference.pt'
            data={'format':'acappellaset-conditioned-training-v1','model':spec['id'],
                  'config_sha256':spec['config_sha256'],'step':4,'file':'inference.pt','sha256':'0'*64}
            manifest.write_text(json.dumps(data))
            resolved=resolve_export(manifest,checkpoint)
            self.assertEqual(resolved['selection'],'new_training_run')
            self.assertTrue(resolved['safe_training_export'])
            with self.assertRaises(ValueError):resolve_export(manifest,checkpoint,'jc_shared')
            with self.assertRaises(ValueError):resolve_export(manifest,root/'elsewhere.pt')
            for update in [{'file':'../inference.pt'},{'config_sha256':'incorrect'},{'step':0}]:
                manifest.write_text(json.dumps(dict(data,**update)))
                with self.assertRaises(ValueError):resolve_export(manifest,checkpoint)

    def test_pooled_selection_weights_pairs_not_domains_or_roles(self):
        from training.validation import scope_scores
        rows=[{'corpus':'JC','role':'S','si_sdri':0},
              {'corpus':'JC','role':'S','si_sdri':0},
              {'corpus':'JC','role':'Lead','si_sdri':20},
              {'corpus':'SK','role':'A','si_sdri':12},
              {'corpus':'SK','role':'Bar','si_sdri':8}]
        result=scope_scores(rows)
        self.assertEqual(result['combined_shared'],{'n_pairs':3,'si_sdri':4})
        self.assertEqual(result['combined_inclusive'],{'n_pairs':5,'si_sdri':8})
        self.assertEqual(result['jc_shared']['si_sdri'],0)
        self.assertEqual(result['sk_inclusive']['si_sdri'],10)


class SelectionExportTests(unittest.TestCase):
    def test_best_exports_are_loadable_and_ties_preserve_first_step(self):
        from unittest.mock import patch
        import torch
        from training.validation import validate_and_select
        from models.fixed import sha256
        spec=resolve('K5')
        def evaluate(model,config,spec,manifest,corpus,device):
            roles=['S','A','T','B','VP']*10+(['Lead']*10 if corpus=='JC' else ['Bar']*2)
            return [dict(corpus=corpus,role=role,si_sdri=1.) for role in roles],[]
        with tempfile.TemporaryDirectory() as temp,patch('training.validation.evaluate_model',side_effect=evaluate):
            root=Path(temp);model=torch.nn.Linear(2,2)
            for step in (1000,2000):
                validate_and_select(model,{},spec,root,root,root,step,torch.device('cpu'))
            selections=json.loads((root/'selection.json').read_text())
            self.assertEqual(len(selections),6)
            for name,record in selections.items():
                self.assertEqual(record['step'],1000)
                checkpoint=root/record['file']
                resolved=resolve_export(root/('best_'+name+'.json'),checkpoint)
                self.assertEqual(resolved['checkpoint_sha256'],sha256(checkpoint))
                payload=torch.load(checkpoint,weights_only=True)
                self.assertEqual(payload['step'],1000)

if __name__=='__main__':unittest.main()
