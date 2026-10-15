# Fixed-output inference

The Fixed_6stem_htdemucs_SKnJC model has a working inference entry point and a published
110 MB portable checkpoint on [Hugging Face](https://huggingface.co/pting126/ACappellaSet-models).
The checkpoint is licensed under Apache-2.0. Training-data terms and third-party
rights are separate from the rights granted by the project; see [checkpoints](checkpoints.md).

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

The published model-only portable export is:

```text
Fixed_6stem_htdemucs_SKnJC.pt
SHA-256: 8f87f3a1224de680085fea0c186600edc7cfbb91b8f56f8862da4772565e0cda
Size: 109822110 bytes
```

It contains all 525 model tensors with their original dtype and values, and
omits optimizer state. The loader uses `torch.load(weights_only=True)` for this
format. The receipt records the hash of the actual file used and its original
research-checkpoint identity separately.

You may rename either local file; the loader checks its content hash and the
configuration hash before deserializing the legacy checkpoint, then loads all
model weights strictly. Both registered formats load strictly; other checkpoint hashes are rejected. This version does
not accept arbitrary checkpoints or change output heads automatically.

## Download the checkpoint

```bash
python download.py --model Fixed_6stem_htdemucs_SKnJC
```

No Hugging Face account or token is needed. The standard-library downloader uses
an immutable release revision and verifies size and SHA-256 before publishing
the local file. Existing files are reused only if their hash matches.

## Separate an audio file

```bash
python inference.py --model Fixed_6stem_htdemucs_SKnJC --checkpoint checkpoints/Fixed_6stem_htdemucs_SKnJC.pt --input /path/to/mixture.wav --output outputs/my-mixture --device cpu
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
The CPU lock is not a CUDA installation guide. The additional original fixed/specialist adapters and training smoke checks
have separate reports; see [conditioned inference](conditioned-inference.md) for the shared query adapters.

The portable export was additionally tested on 2026-10-14 against fresh
original-checkpoint inference and the frozen research outputs. All six stems
were bit-identical on both clips; six behavioral tests passed. See the
[portable validation report](validation/combined-ep333-portable-linux-cpu.json).

## Original fixed and part-specific models

```bash
python inference.py --list-models
python inference.py --model Fixed_6stem_htdemucs_SK --checkpoint /path/to/original-sk.ckpt --input mixture.wav --output outputs/sk --device cpu
python inference.py --model specialists-jc-alto-ep223 --checkpoint /path/to/original-alto.ckpt --input mixture.wav --output outputs/alto --device cpu
```

Each command requires the exact registered checkpoint hash. Downloading SK/JC
or specialist weights is still pending. No custom or arbitrary pickle checkpoint
is accepted. The specialist command writes `nonalto.wav` and `alto.wav`;
`nonalto` is the residual head, and `alto` is the extracted target. Neither
requires an enrollment recording. The same rule applies to each bank member.
All outputs preserve model gain. Do not interpret a specialist's residual as
another named vocal part.

The added adapters have a separate CPU parity report in
[additional-model validation](validation/additional-models-linux-cpu.json).
It uses a short synthetic clip and is not a separation-quality evaluation.
