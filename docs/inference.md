# Fixed-output inference

The Combined ep333 model has a working inference entry point. Its architecture
and configuration match the archived research model. **Weights are not yet
publicly hosted**; these instructions require an existing copy of the registered
checkpoint. Zenodo download instructions will follow publication.

## Installation: tested Linux CPU environment

Tested on Linux x86_64 with Python 3.10.19 and a fresh virtual environment that
does not inherit research site-packages. From the repository root:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install pip==25.2
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.10.0 torchaudio==2.10.0
python -m pip install -r requirements-linux-cpu-lock.txt
python -m pip check
python -m unittest discover -s tests -v
```

The lock contains complete tested package versions. The shorter
`requirements-inference.txt` lists direct dependencies. Demucs **4.0.0** is used
for this fixed-model adapter; the conditioned-family source commit is a separate
dependency decision. No full MSS checkout or Demucs fork is needed for inference.

## Checkpoint

The registered original file is:

```text
model_htdemucs_ep_333_sdr_-1.2322_si_sdr_-0.3075_k_sdr_87.1730.ckpt
SHA-256: cbfa201798f7e7f4766bacd3aa8622b3ae40321729bd7f0b3555ac4348c57879
Size: 333986969 bytes
```

You may rename the local file; the loader checks its content hash and the
configuration hash before deserializing the legacy checkpoint, then loads all
model weights strictly. Other checkpoint hashes are rejected. This version does
not accept arbitrary checkpoints or change output heads automatically.

## Separate an audio file

```bash
python inference.py --model combined-ep333 --checkpoint checkpoints/combined-ep333.ckpt --input /path/to/mixture.wav --output outputs/my-mixture --device cpu
```

The output directory must not already exist. It contains:

```text
vp.wav
bass.wav
soprano.wav
alto.wav
tenor.wav
other.wav
inference.json
```

`other` retains the combined model's JC Lead / SK Baritone training semantics.
It is not two independent Lead and Baritone outputs. The receipt records input,
checkpoint, configuration and output hashes, software versions, and settings.

Input may be mono or stereo in a format supported by SoundFile. Mono is
duplicated to stereo. Other sample rates use SciPy polyphase resampling to
44.1 kHz. Output is stereo 44.1 kHz Float32 WAV at original model gain, without
peak normalization or clipping; values can exceed ±1. Output duration follows
the resampled input. Inputs with more than two channels are rejected.

Inference retains 10-second chunks, a 2.5-second hop, batch size 4, uniform
overlap-add, and right zero padding. CPU uses float32 without autocast. A full
song can require more memory and time than the validation clips.

## Validation and limits

On 2026-10-13, all six outputs were **bit-for-bit identical** to the existing
research implementation on a 12.6-second music excerpt and a 0.3-second synthetic
input using the actual checkpoint. Five behavioral tests passed. See the
[validation report](validation/combined-ep333-linux-cpu.json).

This establishes the tested Linux CPU path, not separation quality or a replay
of historical GPU paper results. The CLI retains a CUDA option and original
CUDA AMP behavior, but CUDA, macOS, and MPS were not validated in this release.
The CPU lock is not a CUDA installation guide. Training and other model families
are still pending.
