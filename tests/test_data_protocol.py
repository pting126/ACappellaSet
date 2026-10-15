"""Historical membership, leakage, checksum and preparation contract checks."""
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from prepare_data import ROOT, materialize, parse_roots


def rows(name, filename):
    raw=gzip.decompress((ROOT/'manifests/splits'/name/(filename+'.gz')).read_bytes())
    return list(csv.DictReader(io.StringIO(raw.decode())))


class DataProtocolTests(unittest.TestCase):
    def test_original_compositions_do_not_cross_splits(self):
        for corpus in ('sk','jc'):
            populations=[]
            for split,filename in [('train','target_examples.csv'),('valid','eval_pairs_valid_correct.csv'),
                                   ('test','eval_pairs_test_correct.csv')]:
                populations.append({r['song_id'] for r in rows(split+'_'+corpus,filename)})
            for i in range(3):
                for j in range(i):self.assertFalse(populations[i]&populations[j],corpus)
        self.assertNotIn('PC01',{r['song_id'] for r in rows('test_ab','eval_pairs_test_correct.csv')})

    def test_portable_hashes_and_scientific_rows_survive_materialization(self):
        with tempfile.TemporaryDirectory() as temp:
            roots={k:Path(temp)/k for k in ('SK','JC','AB')}
            index=json.loads((ROOT/'manifests/splits/index.json').read_text())
            for name,entry in index['sets'].items():
                destination=Path(temp)/name
                receipt=materialize(name,roots,destination,metadata_only=True)
                self.assertTrue(receipt['metadata_only'])
                self.assertEqual(receipt['audio'],{})
                for filename,info in entry['files'].items():
                    self.assertEqual(hashlib.sha256((ROOT/info['file']).read_bytes()).hexdigest(),info['sha256'])
                    if not filename.endswith('.csv'):continue
                    expected=rows(name,filename)
                    with (destination/filename).open(newline='') as stream: actual=list(csv.DictReader(stream))
                    for row in actual:
                        for key,value in row.items():
                            for tag,root in roots.items():
                                if value.startswith(str(root)+'/'):row[key]=tag+'/'+value[len(str(root))+1:]
                    self.assertEqual(len(actual),len(expected))
                    for actual_row,expected_row in zip(actual,expected):
                        self.assertEqual(actual_row,expected_row)
                with self.assertRaises(ValueError):materialize(name,roots,destination,True)

    def test_root_validation(self):
        for values in [['SK=a','SK=b'],['OTHER=a'],['SK='],['missing_separator']]:
            with self.assertRaises(ValueError):parse_roots(values)

    def test_specialist_filter_membership_and_output(self):
        import numpy as np
        import soundfile as sf
        from prepare_specialists import prepare,STEMS
        inventory=json.loads((ROOT/'manifests/splits/track_inventory.json').read_text())
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'input'; output=Path(temp)/'output'
            for split,tracks in inventory['corpora']['SK'].items():
                for track in tracks:
                    directory=source/split/track;directory.mkdir(parents=True)
                    for part in STEMS['SK']:
                        sf.write(directory/(part+'.wav'),np.full((32,2),.01,dtype='float32'),44100,subtype='FLOAT')
            receipt=prepare('SK',source,output,['soprano','baritone'],historical_role_filter=True)
            self.assertTrue(receipt['historical_role_filter'])
            for part,by_split in inventory['specialist_filters']['SK'].items():
                for split,expected in by_split.items():
                    self.assertEqual(sorted(p.name for p in (output/part/split).iterdir()),expected)
            self.assertTrue((output/'soprano/train_flat'/inventory['specialist_filters']['SK']['soprano']['train_flat'][0]/'mixture.wav').is_file())


if __name__=='__main__':unittest.main()
