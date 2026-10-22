"""Offline download integrity, selection, and private-access error contracts."""
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

import download
from models.registry import entries, select_checkpoint


class DownloadTests(unittest.TestCase):
    def invoke(self, output, spec):
        with patch('sys.argv', ['download.py', '--model', 'fixture', '--output', str(output)]), \
             patch('models.registry.resolve', return_value=spec), redirect_stdout(io.StringIO()):
            download.main()

    def fixture(self):
        data = b'verified-checkpoint-bytes'
        return data, {'id': 'fixture', 'portable_checkpoint': {
            'filename': 'fixture.pt', 'bytes': len(data),
            'sha256': hashlib.sha256(data).hexdigest(),
            'download_url': 'https://huggingface.co/fixture/resolve/revision/fixture.pt'}}

    def test_all_selections_have_distinct_download_names_and_complete_metadata(self):
        artifacts = []
        for spec in entries():
            selections = spec.get('selections', [spec])
            for item in selections:
                selected = select_checkpoint(spec, item['selection_labels'][0]) if 'selections' in spec else spec
                artifacts.append(selected['portable_checkpoint'])
        self.assertEqual(len(artifacts), 29)
        self.assertEqual(len({a['filename'] for a in artifacts}), 29)
        for artifact in artifacts:
            self.assertTrue(artifact['download_url'].startswith('https://huggingface.co/'))
            self.assertEqual(len(artifact['sha256']), 64)
            self.assertGreater(artifact['bytes'], 0)
            self.assertTrue(artifact['weights_only'])

    def test_download_and_cached_identity_are_verified(self):
        data, spec = self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            with patch('urllib.request.urlopen', return_value=io.BytesIO(data)) as request:
                self.invoke(output, spec)
                request.assert_called_once()
            self.assertEqual((output / 'fixture.pt').read_bytes(), data)
            with patch('urllib.request.urlopen') as request:
                self.invoke(output, spec)
                request.assert_not_called()

    def test_corrupted_response_is_not_published(self):
        _, spec = self.fixture()
        with tempfile.TemporaryDirectory() as tmp, \
             patch('urllib.request.urlopen', return_value=io.BytesIO(b'corrupt')):
            with self.assertRaisesRegex(ValueError, 'size and SHA-256'):
                self.invoke(Path(tmp), spec)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_private_repository_error_explains_authenticated_download(self):
        _, spec = self.fixture()
        error = urllib.error.HTTPError(spec['portable_checkpoint']['download_url'], 401, 'private', {}, None)
        with tempfile.TemporaryDirectory() as tmp, \
             patch('urllib.request.urlopen', side_effect=error), redirect_stderr(io.StringIO()) as stderr:
            with self.assertRaises(SystemExit):
                self.invoke(Path(tmp), spec)
            self.assertIn('hf auth login', stderr.getvalue())
            self.assertEqual(list(Path(tmp).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
