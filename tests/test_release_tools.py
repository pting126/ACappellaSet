"""Behavioral tests for portable preparation, discovery and reporting."""
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import soundfile as sf
from evaluation.reporting import summarize, validate_identity
from models.registry import ROOT, entries, resolve
from prepare_specialists import prepare, STEMS, SPLITS

class RegistryTests(unittest.TestCase):
    def test_config_hashes_aliases_and_contracts(self):
        self.assertEqual(len(entries()),15)
        names=set()
        for spec in entries():
            self.assertEqual(hashlib.sha256((ROOT/spec['config']).read_bytes()).hexdigest(),spec['config_sha256'])
            for name in [spec['id'],*spec.get('aliases',[])]:
                self.assertNotIn(name,names);names.add(name)
                self.assertEqual(resolve(name),spec)
            if spec['family']=='part-specific-htdemucs':
                self.assertEqual(spec['source_order'],['non'+spec['target_role'],spec['target_role']])
                self.assertEqual(spec['output_contract'],{'residual_index':0,'target_index':1})
        self.assertEqual(resolve('combined-ep333')['id'],'Fixed_6stem_htdemucs_SKnJC')

    def test_cli_discovery_and_unpublished_download(self):
        for script in ['inference.py','download.py']:
            result=subprocess.run([sys.executable,str(ROOT/script),'--list-models'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('specialists-sk-alto-ep237',result.stdout)
        result=subprocess.run([sys.executable,str(ROOT/'download.py'),'--model','Fixed_6stem_htdemucs_SK'],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('no published portable checkpoint',result.stderr)

class PreparationTests(unittest.TestCase):
    def fixture(self,root,corpus):
        for split in SPLITS[corpus]:(root/split).mkdir(parents=True)
        track=root/SPLITS[corpus][0]/'song1';track.mkdir()
        rng=np.random.default_rng(36);stems={}
        for stem in STEMS[corpus]:
            stems[stem]=rng.normal(0,.05,(53,2)).astype(np.float32)
            sf.write(track/(stem+'.wav'),stems[stem],44100,subtype='FLOAT')
        sf.write(track/'mixture.wav',np.full((53,2),.42,dtype=np.float32),44100,subtype='FLOAT')
        return track,stems

    def test_residual_recipes_and_relocatable_outputs(self):
        for corpus in STEMS:
            with self.subTest(corpus=corpus),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);track,stems=self.fixture(root/'input',corpus)
                output=root/'prepared';receipt=prepare(corpus,root/'input',output,['alto'],allow_subset=True)
                moved=root/'moved';output.rename(moved)
                result=moved/'alto'/SPLITS[corpus][0]/'song1'
                mixture=sf.read(result/'mixture.wav',dtype='float32',always_2d=True)[0]
                actual=sf.read(result/'nonalto.wav',dtype='float32',always_2d=True)[0]
                expected=np.zeros_like(stems['alto'])
                if corpus=='JC':
                    for role,audio in stems.items():
                        if role!='alto':expected+=audio
                    np.testing.assert_array_equal(mixture,np.full((53,2),.42,dtype=np.float32))
                else:
                    for audio in stems.values():expected+=audio
                    np.testing.assert_array_equal(mixture,expected);expected=expected-stems['alto']
                np.testing.assert_array_equal(actual,expected)
                self.assertEqual((track/'alto.wav').read_bytes(),(result/'alto.wav').read_bytes())
                self.assertEqual(len(receipt['tracks']),1)
                with self.assertRaises(FileExistsError):prepare(corpus,root/'input',moved,allow_subset=True)

    def test_missing_audio_alignment_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);track,_=self.fixture(root/'input','SK')
            with self.assertRaises(ValueError):prepare('SK',root/'input',root/'out')
            (track/'bass.wav').unlink()
            with self.assertRaises((OSError,RuntimeError)):prepare('SK',root/'input',root/'out',allow_subset=True)
            sf.write(track/'bass.wav',np.ones((7,2)),44100,subtype='FLOAT')
            with self.assertRaises(ValueError):prepare('SK',root/'input',root/'out',allow_subset=True)
            self.assertFalse((root/'out').exists())

class ReportingTests(unittest.TestCase):
    def row(self,track,role,value):
        return dict(model='example',corpus='JC',protocol='mixture_only',song_id=track,track_id=track,
                    role=role,si_sdr=value-2,mix_si_sdr=-2,si_sdri=value)

    def test_pair_weighting_shared_and_inclusive(self):
        rows=[self.row('one','S',0),self.row('two','S',0),self.row('one','A',12),self.row('one','Lead',20)]
        report={r['scope']:r for r in summarize(rows)}
        self.assertEqual(report['shared']['overall_si_sdri'],4)
        self.assertEqual(report['inclusive']['overall_si_sdri'],8)

    def test_duplicates_nonfinite_and_decomposition(self):
        row=self.row('one','S',3)
        with self.assertRaises(ValueError):summarize([row,row])
        for value in [float('nan'),12]:
            with self.assertRaises(ValueError):summarize([dict(row,si_sdri=value)])

    def test_pc01_and_exact_matching(self):
        row=dict(self.row('custom_variant','S',3),corpus='AB',song_id='PC01')
        with self.assertRaises(ValueError):summarize([row])
        for track in ['PC02','PC010','PC01ish']:validate_identity(dict(row,song_id=track,track_id=track))
        with self.assertRaises(ValueError):validate_identity(dict(row,song_id='PC02',track_id='PC01_rough_DM'))

    def test_protocols_remain_separate(self):
        row=self.row('one','S',3)
        reports=summarize([dict(row,protocol='strict_enrollment'),dict(row,protocol='fallback_inclusive')])
        self.assertEqual(len(reports),4)

if __name__=='__main__':unittest.main()
