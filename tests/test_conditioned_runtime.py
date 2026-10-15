"""Numerical overlap-add behavior with a known separator, no checkpoints needed."""
import unittest
import numpy as np
import torch
from models.conditioned import separate


class KnownSeparator(torch.nn.Module):
    embedding_encoder = None

    def forward(self, mixture, enrollment, role):
        return torch.stack((mixture * .25, mixture * .75), dim=1)


class OverlapAddTests(unittest.TestCase):
    def test_gain_output_order_and_complete_coverage(self):
        config = {'sample_rate': 10, 'seconds': 1, 'hop_seconds': .4, 'enroll_seconds': 1}
        for length in (1, 9, 10, 11, 18, 19, 101):
            with self.subTest(length=length):
                mixture = np.random.default_rng(length).normal(0, .1, (2, length)).astype('float32')
                actual = separate(KnownSeparator(), config, mixture, torch.device('cpu'))
                np.testing.assert_allclose(actual[0], mixture * .25, atol=1e-7, rtol=1e-6)
                np.testing.assert_allclose(actual[1], mixture * .75, atol=1e-7, rtol=1e-6)

    def test_silence_and_invalid_input(self):
        config = {'sample_rate': 10, 'seconds': 1, 'hop_seconds': .4, 'enroll_seconds': 1}
        actual = separate(KnownSeparator(), config, np.zeros((2, 19)), torch.device('cpu'))
        self.assertTrue(np.isfinite(actual).all())
        self.assertFalse(actual.any())
        for waveform in (np.zeros((2, 0)), np.zeros((3, 10)), np.full((2, 10), np.nan)):
            with self.assertRaises(ValueError):
                separate(KnownSeparator(), config, waveform, torch.device('cpu'))
