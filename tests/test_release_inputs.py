"""Offline release mapping, input integrity, and downstream preparation contracts."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
from data import release
from prepare_specialists import prepare


def csv_file(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


class ReleaseInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.source = self.base / 'release'
        self.repo = self.base / 'repo'
        self.out = self.base / 'prepared'
        (self.repo / 'manifests/splits').mkdir(parents=True)
        self.metadata, self.files = [], []
        inventory = {'corpora': {'SK': {'train_flat': [], 'valid_flat': [], 'test_flat': []}}}
        self.tracks = ['C001_edited_SP', 'C001_produced_SP', 'C002_rough_DM', 'PC01_rough_DM']
        for index, track in enumerate(self.tracks):
            song, variant, spatial = track.split('_')
            primary = variant != 'edited'
            folder = Path('audio/primary') / song if primary else Path('audio/alternates') / song / variant
            active = set(release.STEMS) - ({'baritone'} if song != 'PC01' else set())
            row = dict(song_id=song, Group='Annappella' if song=='PC01' else 'Seagull-K',
                       is_primary=str(int(primary)), variant=variant,
                       spatial_type='stereo_spatial' if spatial=='SP' else 'dual_mono',
                       stem_count=str(len(active)))
            row.update({role: 'Singer' if role in active else '0' for role in release.STEMS})
            self.metadata.append(row)
            if song != 'PC01': inventory['corpora']['SK']['valid_flat' if song=='C002' else 'train_flat'].append(track)
            for role in sorted(active | {'mixture'}):
                path = self.source / folder / (role+'.wav'); path.parent.mkdir(parents=True, exist_ok=True)
                # Deliberately preserve a non-additive mixture above 1.0.
                values = np.full((53, 2), 1.5 if role=='mixture' else .125, dtype=np.float32)
                sf.write(path, values, 44100, subtype='FLOAT' if role=='mixture' else 'PCM_16')
                self.files.append(dict(path=(folder/(role+'.wav')).as_posix(),recording_id=track,role=role,
                    sha256=release.digest(path),bytes=path.stat().st_size,sample_rate=44100,channels=2,
                    frames=53,subtype='FLOAT' if role=='mixture' else 'PCM_16'))
        csv_file(self.source/'metadata/metadata.csv',self.metadata)
        csv_file(self.source/'metadata/splits.csv',[dict(song_id='C001',split='train'),
                  dict(song_id='C002',split='valid'),dict(song_id='PC01',split='external_test')])
        self.save_manifest()
        (self.repo/'manifests/splits/track_inventory.json').write_text(json.dumps(inventory))
        # Real compressed historical contract, including retained exclusion metadata.
        index = {'sets': {'train_sk': {'files': {}}}}
        row = {'target_path':'SK/train_flat/C001_edited_SP/soprano.wav',
               'source_audio_sha256':next(f['sha256'] for f in self.files if f['recording_id']==self.tracks[0] and f['role']=='soprano')}
        plain = self.base/'targets.csv'; csv_file(plain,[row]); raw=plain.read_bytes()
        compressed=gzip.compress(raw); relative='manifests/splits/targets.csv.gz'
        (self.repo/relative).write_bytes(compressed)
        index['sets']['train_sk']['files']['target_examples.csv']={
            'file':relative,'sha256':hashlib.sha256(compressed).hexdigest(),
            'content_sha256':hashlib.sha256(raw).hexdigest()}
        (self.repo/'manifests/splits/index.json').write_text(json.dumps(index))
        self.patcher=patch.object(release,'ROOT',self.repo);self.patcher.start();self.addCleanup(self.patcher.stop)

    def pin(self):
        (self.repo/'manifests/dataset.json').write_text(json.dumps({'revision':'fixture', 'files':[
            {'path':name,'sha256':release.digest(self.source/name)} for name in release.DOCUMENTS]}))

    def save_manifest(self):
        csv_file(self.source/'file_manifest.csv', self.files);self.pin()

    def test_symlinks_padding_splits_and_specialist_input(self):
        before={f['path']:release.digest(self.source/f['path']) for f in self.files}
        receipt=release.prepare_release(self.source,self.out)
        self.assertEqual(receipt['historical_hashes_matched'],1)
        self.assertEqual(sum(x['kind']=='generated_silence' for x in receipt['outputs'].values()),3)
        track=self.out/'SK/train_flat/C001_edited_SP'
        self.assertTrue((track/'soprano.wav').is_symlink())
        np.testing.assert_array_equal(sf.read(track/'baritone.wav')[0],np.zeros((53,2)))
        self.assertEqual(sf.info(track/'baritone.wav').subtype,'PCM_16')
        self.assertEqual(float(sf.read(track/'mixture.wav')[0].max()),1.5)
        self.assertTrue((self.out/'AB/PC01_rough_DM/mixture.wav').exists())
        self.assertEqual(before,{f['path']:release.digest(self.source/f['path']) for f in self.files})
        # Empty test split for this small fixture; complete release has all splits.
        (self.out/'SK/test_flat').mkdir()
        prepared=prepare('SK',self.out/'SK',self.base/'specialists',['alto'],allow_subset=True)
        self.assertEqual(len(prepared['tracks']),3)
        moved=self.base/'moved';moved.mkdir()
        self.source.rename(moved/'release');self.out.rename(moved/'prepared')
        self.assertTrue((moved/'prepared/SK/train_flat/C001_edited_SP/soprano.wav').is_file())

    def test_copy_output_is_self_contained_and_cannot_overwrite(self):
        receipt=release.prepare_release(self.source,self.out,'copy')
        with self.assertRaises(FileExistsError): release.prepare_release(self.source,self.out)
        self.source.rename(self.base/'away')
        for rel, entry in receipt['outputs'].items():
            path=self.out/rel
            self.assertFalse(path.is_symlink())
            self.assertEqual(release.digest(path),entry['sha256'])
        with self.assertRaises(ValueError): release.prepare_release(self.base/'away',self.base/'away/inside')

    def test_missing_active_audio_never_becomes_padding(self):
        (self.source/self.files[0]['path']).unlink()
        with self.assertRaisesRegex(ValueError,'Missing extracted audio'):
            release.prepare_release(self.source,self.out)
        self.assertFalse(self.out.exists())

    def test_audio_corruption_rolls_back_and_metadata_pin_is_enforced(self):
        path=self.source/self.files[-1]['path']; path.write_bytes(path.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError,'checksum/size mismatch'):
            release.prepare_release(self.source,self.out)
        self.assertFalse(self.out.exists())
        path=self.source/'metadata/metadata.csv';path.write_text(path.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'metadata checksum mismatch'):
            release.plan_release(self.source)

    def test_split_drift_duplicate_and_unsafe_paths_are_rejected(self):
        csv_file(self.source/'metadata/splits.csv',[dict(song_id='C001',split='test'),
                 dict(song_id='C002',split='valid'),dict(song_id='PC01',split='external_test')]);self.pin()
        with self.assertRaisesRegex(ValueError,'split mismatch'):release.plan_release(self.source)
        csv_file(self.source/'metadata/splits.csv',[dict(song_id='C001',split='train'),
                 dict(song_id='C002',split='valid'),dict(song_id='PC01',split='external_test')]);self.pin()
        self.files.append(dict(self.files[0]));self.save_manifest()
        with self.assertRaisesRegex(ValueError,'Duplicate or mismatched'):release.plan_release(self.source)
        self.files.pop();self.files[0]['path']='../outside.wav';self.save_manifest()
        with self.assertRaisesRegex(ValueError,'Unsafe'):release.plan_release(self.source)

    def test_historical_hash_mismatch_is_not_silently_accepted(self):
        next(f for f in self.files if f['recording_id']==self.tracks[0] and f['role']=='soprano')['sha256']='0'*64
        self.save_manifest()
        with self.assertRaisesRegex(ValueError,'Historical source hash differs'):release.plan_release(self.source)

    def test_alignment_and_nonfinite_audio(self):
        self.files[0]['frames']=54;self.save_manifest()
        with self.assertRaisesRegex(ValueError,'Misaligned'):release.plan_release(self.source)
        self.files[0]['frames']=53
        entry=next(f for f in self.files if f['role']=='mixture')
        path=self.source/entry['path'];sf.write(path,np.full((53,2),np.nan),44100,subtype='FLOAT')
        entry['sha256']=release.digest(path);entry['bytes']=path.stat().st_size;self.save_manifest()
        with self.assertRaisesRegex(ValueError,'Non-finite'):release.prepare_release(self.source,self.out)
        self.assertFalse(self.out.exists())

    def test_sources_changed_during_preparation_are_rejected(self):
        original=release.verify_audio
        count=0
        def verify(path,entry):
            nonlocal count
            original(path,entry);count+=1
            if count==len(self.files):
                first=self.source/self.files[0]['path'];first.write_bytes(first.read_bytes()+b'changed')
        with patch.object(release,'verify_audio',side_effect=verify):
            with self.assertRaisesRegex(ValueError,'changed during'):release.prepare_release(self.source,self.out)
        self.assertFalse(self.out.exists())

    def test_actual_cli_and_invalid_modes(self):
        real_root=Path(__file__).resolve().parents[1]
        (self.repo/'data').mkdir()
        shutil.copyfile(real_root/'prepare_data.py',self.repo/'prepare_data.py')
        shutil.copyfile(real_root/'data/release.py',self.repo/'data/release.py')
        (self.repo/'data/__init__.py').touch()
        command=[sys.executable,str(self.repo/'prepare_data.py'),'--release-root',str(self.source),
                 '--output',str(self.out)]
        result=subprocess.run(command,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads(result.stdout)['recordings'],4)
        for extra in [ ['--set','train_sk'], ['--metadata-only'], ['--root','SK=/anything'] ]:
            result=subprocess.run(command+extra,capture_output=True,text=True)
            self.assertEqual(result.returncode,2,result.stderr)


if __name__=='__main__':unittest.main()
