"""Research ReDimNet2 wrapper, constructed without downloading pretrained weights.

Inference loads every encoder tensor from the verified separation checkpoint.
The vendored WeSpeaker modules retain their Apache-2.0 notices.
"""
from pathlib import Path

import torch
from torch import Tensor, nn


class WeSpeakerReDimNetEmbedding(nn.Module):
    def __init__(self, config_path, emb_dim=192, freeze_encoder=True,
                 input_sample_rate=44100):
        super().__init__()
        import torchaudio
        import yaml
        from .vendor.wespeaker.redimnet2 import ReDimNet2Wrap
        from .vendor.wespeaker.tfmel import TFMelFrontend

        config = yaml.safe_load(Path(config_path).read_text())
        if config['model'] != 'ReDimNet2Wrap':
            raise ValueError('Only the audited ReDimNet2Wrap encoder is supported')
        if int(config['model_args']['embed_dim']) != emb_dim:
            raise ValueError('Encoder embedding dimension mismatch')
        data = config['dataset_args']
        if data['frontend'] != 'tfmel':
            raise ValueError('The released encoder requires the tfmel frontend')
        self.torchaudio = torchaudio
        self.input_sample_rate = input_sample_rate
        self.freeze_encoder = freeze_encoder
        self.sample_rate = int(data['resample_rate'])
        self.model = ReDimNet2Wrap(**config['model_args'])
        self.model.add_module('frontend', TFMelFrontend(
            **data['tfmel_args'], sample_rate=self.sample_rate))
        if hasattr(self.model, 'prepare_for_frontend'):
            self.model.prepare_for_frontend('tfmel')
        self.model.frontend_type = 'tfmel'
        if freeze_encoder:
            self.model.eval()
            for parameter in self.model.parameters():
                parameter.requires_grad = False

    def embedding(self, waveform: Tensor) -> Tensor:
        if waveform.dim() == 3:
            waveform = waveform.mean(dim=1)
        if self.sample_rate != self.input_sample_rate:
            waveform = self.torchaudio.functional.resample(
                waveform, self.input_sample_rate, self.sample_rate
            )
        lengths = torch.full(
            (waveform.shape[0],),
            waveform.shape[-1],
            dtype=torch.long,
            device=waveform.device,
        )
        was_training = self.model.training
        if self.freeze_encoder:
            self.model.eval()
            with torch.no_grad():
                features, _ = self.model.frontend(waveform, lengths)
                embedding = self.model(features)
        else:
            features, _ = self.model.frontend(waveform, lengths)
            embedding = self.model(features)
        if was_training and self.freeze_encoder:
            self.model.train()
        if isinstance(embedding, tuple):
            embedding = embedding[-1]
        if embedding.ndim != 2:
            embedding = embedding.reshape(embedding.shape[0], -1)
        return embedding


def build_embedding_encoder(backend, emb_dim, layers, bundle, speechbrain_source,
                            speechbrain_savedir, freeze, projection,
                            wespeaker_model_path, wespeaker_vendor_path):
    if backend != 'wespeaker_redimnet2':
        raise ValueError('Only the released WeSpeaker ReDimNet2 backend is supported')
    return WeSpeakerReDimNetEmbedding(wespeaker_model_path, emb_dim, freeze)
