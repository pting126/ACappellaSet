"""Audited K/L/M FiLM separator. Forward computation retained from research source.

The embedded Demucs implementation retains its MIT license and notices.
"""
from __future__ import annotations
from fractions import Fraction
from pathlib import Path
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from .conditioned_encoder import build_embedding_encoder

ROLE_VOCAB = ("S", "A", "T", "Bar", "B", "VP", "Lead", "None")
ROLE_TO_INDEX = {role: index for index, role in enumerate(ROLE_VOCAB)}

class DualConditionedHTDemucs(nn.Module):
    def __init__(
        self,
        demucs_root: Path,
        source_mode: str = "target_residual",
        audio_channels: int = 2,
        emb_dim: int = 32,
        embedding_num_layers: int = 2,
        embedding_backend: str = "netsv",
        torchaudio_bundle: str = "WAVLM_BASE_PLUS",
        speechbrain_source: str = "speechbrain/spkrec-ecapa-voxceleb",
        speechbrain_savedir: str = "checkpoints/pretrained_download/speechbrain/spkrec-ecapa-voxceleb",
        freeze_external_embedding: bool = True,
        speechbrain_projection: str = "linear",
        wespeaker_model_path: str = "checkpoints/pretrained_download/wespeaker/wespeaker-voxceleb-redimnet2-B6-LM",
        wespeaker_vendor_path: str = ".vendor/wespeaker",
        channels: int = 48,
        depth: int = 4,
        growth: int = 2,
        bottom_channels: int = 0,
        dconv_mode: int = 1,
        t_layers: int = 5,
        nfft: int = 4096,
        segment: float = 8.0,
        sample_rate: int = 44100,
        role_embedding_dim: int = 16,
        conditioning: str = "dual",
    ) -> None:
        super().__init__()
        from .vendor.demucs.htdemucs import HTDemucs

        if source_mode not in {"one_source", "target_residual", "residual_target"}:
            raise ValueError(f"unsupported source_mode={source_mode}")
        if conditioning not in {"none", "role", "enrollment", "dual"}:
            raise ValueError(f"unsupported V2-A conditioning={conditioning}")
        self.source_mode = source_mode
        self.conditioning = conditioning
        self.embedding_backend = embedding_backend
        self.role_vocab = ROLE_VOCAB
        if source_mode == "one_source":
            sources = ["target"]
        elif source_mode == "target_residual":
            sources = ["target", "residual"]
        else:
            sources = ["residual", "target"]
        self.embedding_encoder = None
        if conditioning in {"enrollment", "dual"}:
            self.embedding_encoder = build_embedding_encoder(
                embedding_backend,
                emb_dim,
                embedding_num_layers,
                torchaudio_bundle,
                speechbrain_source,
                speechbrain_savedir,
                freeze_external_embedding,
                speechbrain_projection,
                wespeaker_model_path,
                wespeaker_vendor_path,
            )
        self.demucs = HTDemucs(
            sources=sources,
            audio_channels=audio_channels,
            channels=channels,
            growth=growth,
            depth=depth,
            bottom_channels=bottom_channels,
            dconv_mode=dconv_mode,
            t_layers=t_layers,
            nfft=nfft,
            samplerate=sample_rate,
            segment=segment,
            use_train_segment=True,
        )
        bottleneck_channels = channels * (growth ** (depth - 1))
        self.spec_film = None
        self.time_film = None
        if conditioning in {"enrollment", "dual"}:
            self.spec_film = nn.Linear(emb_dim, 2 * bottleneck_channels)
            self.time_film = nn.Linear(emb_dim, 2 * bottleneck_channels)
            nn.init.zeros_(self.spec_film.weight)
            nn.init.zeros_(self.spec_film.bias)
            nn.init.zeros_(self.time_film.weight)
            nn.init.zeros_(self.time_film.bias)
        self.role_embedding = None
        self.role_spec_film = None
        self.role_time_film = None
        if conditioning in {"role", "dual"}:
            self.role_embedding = nn.Embedding(len(ROLE_VOCAB), role_embedding_dim)
            nn.init.normal_(self.role_embedding.weight, std=0.02)
            with torch.no_grad():
                self.role_embedding.weight[ROLE_TO_INDEX["None"]].zero_()
                self.role_embedding.weight[ROLE_TO_INDEX["Bar"]].zero_()
            self.role_spec_film = nn.Linear(
                role_embedding_dim, 2 * bottleneck_channels, bias=False
            )
            self.role_time_film = nn.Linear(
                role_embedding_dim, 2 * bottleneck_channels, bias=False
            )
            nn.init.zeros_(self.role_spec_film.weight)
            nn.init.zeros_(self.role_time_film.weight)

    def _film(self, x: Tensor, context: tuple[Tensor | None, Tensor | None], branch: str) -> Tensor:
        emb, role_emb = context
        gamma_beta = None
        singer_layer = self.spec_film if branch == "spec" else self.time_film
        role_layer = self.role_spec_film if branch == "spec" else self.role_time_film
        if singer_layer is not None and emb is not None:
            gamma_beta = singer_layer(F.normalize(emb, dim=-1))
        if role_layer is not None and role_emb is not None:
            role_term = role_layer(F.normalize(role_emb, dim=-1))
            gamma_beta = role_term if gamma_beta is None else gamma_beta + role_term
        if gamma_beta is None:
            return x
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        while gamma.dim() < x.dim():
            gamma = gamma.unsqueeze(-1)
            beta = beta.unsqueeze(-1)
        return x * (1.0 + gamma) + beta

    def encode_enrollment(self, enrollment: Tensor) -> Tensor:
        """Encode enrollment audio without running the separator."""
        if self.embedding_encoder is None:
            raise RuntimeError("this model has no enrollment encoder")
        enrollment_mono = enrollment.mean(dim=1) if enrollment.dim() == 3 else enrollment
        return self.embedding_encoder.embedding(enrollment_mono)

    def forward_with_embedding(
        self,
        mix: Tensor,
        enrollment_embedding: Tensor | None,
        role_index: Tensor,
    ) -> Tensor:
        """Run separation from an explicit singer embedding.

        This is the authoritative path for exact-zero query diagnostics: callers
        inject a literal zero tensor here instead of encoding silent audio.
        """
        if self.embedding_encoder is None and enrollment_embedding is not None:
            raise RuntimeError("cannot inject enrollment into an unconditioned model")
        role_emb = self.role_embedding(role_index) if self.role_embedding is not None else None
        return self._forward_demucs_conditioned(
            mix, (enrollment_embedding, role_emb)
        )

    def forward(self, mix: Tensor, enrollment: Tensor, role_index: Tensor) -> Tensor:
        emb = (
            self.encode_enrollment(enrollment)
            if self.embedding_encoder is not None
            else None
        )
        return self.forward_with_embedding(mix, emb, role_index)

    def _forward_demucs_conditioned(
        self,
        mix: Tensor,
        emb: tuple[Tensor | None, Tensor | None],
    ) -> Tensor:
        d = self.demucs
        length = mix.shape[-1]
        length_pre_pad = None
        if d.use_train_segment:
            if d.training:
                d.segment = Fraction(mix.shape[-1], d.samplerate)
            else:
                training_length = int(d.segment * d.samplerate)
                if mix.shape[-1] < training_length:
                    length_pre_pad = mix.shape[-1]
                    mix = F.pad(mix, (0, training_length - length_pre_pad))
        z = d._spec(mix)
        mag = d._magnitude(z).to(mix.device)
        x = mag
        B, _C, Fq, T = x.shape
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)

        xt = mix
        meant = xt.mean(dim=(1, 2), keepdim=True)
        stdt = xt.std(dim=(1, 2), keepdim=True)
        xt = (xt - meant) / (1e-5 + stdt)

        saved, saved_t, lengths, lengths_t = [], [], [], []
        for idx, encode in enumerate(d.encoder):
            lengths.append(x.shape[-1])
            inject = None
            if idx < len(d.tencoder):
                lengths_t.append(xt.shape[-1])
                tenc = d.tencoder[idx]
                xt = tenc(xt)
                if not tenc.empty:
                    saved_t.append(xt)
                else:
                    inject = xt
            x = encode(x, inject)
            if idx == 0 and d.freq_emb is not None:
                frs = torch.arange(x.shape[-2], device=x.device)
                freq_emb = d.freq_emb(frs).t()[None, :, :, None].expand_as(x)
                x = x + d.freq_emb_scale * freq_emb
            saved.append(x)

        x = self._film(x, emb, "spec")
        xt = self._film(xt, emb, "time")

        if d.crosstransformer:
            if d.bottom_channels:
                b, c, f, t = x.shape
                x = rearrange(x, "b c f t-> b c (f t)")
                x = d.channel_upsampler(x)
                x = rearrange(x, "b c (f t)-> b c f t", f=f)
                xt = d.channel_upsampler_t(xt)
            x, xt = d.crosstransformer(x, xt)
            if d.bottom_channels:
                x = rearrange(x, "b c f t-> b c (f t)")
                x = d.channel_downsampler(x)
                x = rearrange(x, "b c (f t)-> b c f t", f=f)
                xt = d.channel_downsampler_t(xt)

        for idx, decode in enumerate(d.decoder):
            skip = saved.pop(-1)
            x, pre = decode(x, skip, lengths.pop(-1))
            offset = d.depth - len(d.tdecoder)
            if idx >= offset:
                tdec = d.tdecoder[idx - offset]
                length_t = lengths_t.pop(-1)
                if tdec.empty:
                    assert pre.shape[2] == 1, pre.shape
                    pre = pre[:, :, 0]
                    xt, _ = tdec(pre, None, length_t)
                else:
                    skip = saved_t.pop(-1)
                    xt, _ = tdec(xt, skip, length_t)

        assert len(saved) == 0 and len(lengths_t) == 0 and len(saved_t) == 0
        S = len(d.sources)
        x = x.view(B, S, -1, Fq, T)
        x = x * std[:, None] + mean[:, None]
        x_is_mps = x.device.type == "mps"
        if x_is_mps:
            x = x.cpu()
        zout = d._mask(z, x)
        if d.use_train_segment:
            if d.training:
                x = d._ispec(zout, length)
            else:
                x = d._ispec(zout, training_length)
        else:
            x = d._ispec(zout, length)
        if x_is_mps:
            x = x.to("mps")
        if d.use_train_segment:
            if d.training:
                xt = xt.view(B, S, -1, length)
            else:
                xt = xt.view(B, S, -1, training_length)
        else:
            xt = xt.view(B, S, -1, length)
        xt = xt * stdt[:, None] + meant[:, None]
        out = xt + x
        if length_pre_pad:
            out = out[..., :length_pre_pad]
        return out
