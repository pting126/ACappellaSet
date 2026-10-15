"""Check known audited results, explicit scope and provenance rejection."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from reproduce import ROOT,load_artifacts,build_tables,reproduce

class ReproductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan,cls.groups,cls.summaries=load_artifacts()
        cls.tables=build_tables(cls.plan,cls.summaries)

    def test_audited_conditioning_and_development_results(self):
        expected=[6.1,6.4,7.3,2.6,5.0,2.9]
        self.assertEqual([float(f"{r['Shared']:.1f}") for r in self.tables['target_conditioning']],expected)
        a=self.tables['development_a']
        self.assertEqual([r['step'] for r in a],[179000,12000,106000,195000,198000,195000])
        self.assertAlmostEqual(a[3]['JC'],14.88953532487154,places=10)
        self.assertAlmostEqual(a[4]['SK'],13.404679947,places=9)
        self.assertTrue(all(r['scope']=='shared' for r in a))
        b=self.tables['development_b']
        self.assertTrue(all(r['scope']=='inclusive' and r['validation_selection']=='combined_inclusive' for r in b))
        self.assertEqual([[float(f"{r[c]:.1f}") for c in ['SK','JC','AB']] for r in b],
                         [[13.6,5.4,2.0],[11.9,4.4,2.8],[11.9,6.0,5.4]])

    def test_scopes_and_checkpoint_identities(self):
        self.assertTrue(all(not(r['corpus']=='AB' and r['song_id']=='PC01') for rows in self.groups.values() for r in rows))
        self.assertTrue(all(len(r['checkpoint_sha256'])==64 for rows in self.groups.values() for r in rows))
        row=next(r for r in self.tables['cross_corpus'] if r['artifact']=='K1_198000_SK')
        self.assertAlmostEqual(row['Shared'],12.3630337868,places=9)
        for r in self.tables['target_conditioning']:
            if r['condition']=='Part':self.assertIsNone(r['Lead'] if r['test']=='JC' else r['Bar'])

    def test_changed_data_rejected_and_output_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'input';shutil.copytree(ROOT/'reproduction',root)
            with (root/'per_pair.csv.gz').open('ab') as stream:stream.write(b'changed')
            with self.assertRaises(ValueError):load_artifacts(root)
            draft=Path(temp)/'draft.json';draft.write_text(json.dumps([dict(model='JC_ep321',dataset='JC',role='Shared',draft_value=999)]))
            output=Path(temp)/'output';reproduce(output,draft_cells=draft)
            summary=json.loads((output/'summary.json').read_text())
            self.assertEqual(summary['manuscript_status'],'REQUIRES_RECONCILIATION')
            self.assertGreater(summary['historical_draft_discrepancies'],0)
            with self.assertRaises(FileExistsError):reproduce(output)

if __name__=='__main__':unittest.main()
