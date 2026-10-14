"""Behavioral tests; real-checkpoint equivalence is recorded separately."""

import tempfile
from pathlib import Path
import unittest

import numpy as np
from omegaconf import OmegaConf
import soundfile as sf
import torch

from inference import read_audio
from models.fixed import load_model, separate


class IdentityParts(torch.nn.Module):
    def forward(self, x):
        return torch.stack((x, -x), dim=1)


class FixedInferenceTests(unittest.TestCase):
    def setUp(self):
        self.config = OmegaConf.create({
            "training": {"samplerate": 8, "segment": 2, "channels": 2,
                         "instruments": ["positive", "negative"], "use_amp": False},
            "inference": {"num_overlap": 4, "batch_size": 4},
        })

    def test_overlap_preserves_short_exact_and_partial_tail(self):
        for length in (1, 15, 16, 17, 37):
            with self.subTest(length=length):
                mix = np.arange(2 * length, dtype=np.float32).reshape(2, length)
                result = separate(IdentityParts(), self.config, mix, torch.device("cpu"))
                np.testing.assert_array_equal(result, np.stack((mix, -mix)))

    def test_invalid_audio_fails(self):
        for mix in (np.empty((2, 0)), np.full((2, 4), np.nan), np.zeros((3, 4))):
            with self.assertRaises(ValueError):
                separate(IdentityParts(), self.config, mix, torch.device("cpu"))

    def test_nonfinite_predictions_fail(self):
        class Bad(torch.nn.Module):
            def forward(self, x):
                return torch.stack((x, x), dim=1) * float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            separate(Bad(), self.config, np.ones((2, 19)), torch.device("cpu"))

    def test_unregistered_checkpoint_rejected_before_deserialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "untrusted.ckpt"
            path.write_bytes(b"not a checkpoint")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_model(path, path, {"checkpoint_sha256": "0" * 64}, torch.device("cpu"))

    def test_mono_resampling_and_multichannel_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.wav"
            sf.write(path, np.ones(4800, dtype=np.float32) * 0.1, 48000, subtype="FLOAT")
            audio, info = read_audio(path, 44100)
            self.assertEqual(audio.shape, (2, 4410))
            np.testing.assert_array_equal(audio[0], audio[1])
            self.assertTrue(info["mono_duplicated"] and info["resampled"])
            sf.write(path, np.zeros((100, 3)), 44100, subtype="FLOAT")
            with self.assertRaises(ValueError):
                read_audio(path, 44100)


if __name__ == "__main__":
    unittest.main()
