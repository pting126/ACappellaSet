# Conditioned training

`train.py` supports the original natural-data K1/K5/K7, L1/L3/L4, and M3
recipes. They train part-, singer-, or dual-conditioned target/residual
extractors. The newer fixed-six and enrollment-specialist experiments and the
AI-augmented recipes are outside this code release.

## Environment

Use Python 3.10 in a new environment on Linux. The validated CUDA installation
uses PyTorch 2.10.0 and CUDA 12.8 wheels:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install pip==25.3 setuptools==80.9.0 wheel==0.45.1
python -m pip install torch==2.10.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-training-conditioned.txt
python -m pip check
```

For CPU diagnostics, substitute the PyTorch `/whl/cpu` index and pass `--cpu`.
Full training requires suitable GPU resources. Runtime verification is recorded
in the training validation report; it does not establish convergence to the
paper's scores on a different software/hardware stack.

## Inputs

Prepare the exact manifests using [data preparation](preparation.md). The
trainer requires verified audio preparation, not `--metadata-only` output.
Provide the official PT6S initializer identified in
[`initialization.json`](../third_party/initialization.json). Singer and dual
recipes also require the original WeSpeaker ReDimNet2 B6-LM `avg_model.pt`,
whose SHA-256 is `b9314cd0184d3823c70a2518d354397bf049832f90fb1d7584ff6c0b0d8b152a`.
The trainer verifies these identities before loading them; pretrained files are
not distributed here. Part-only recipes do not load a singer encoder.

The separator copies 521 compatible PT6S tensors and seeds its four two-output
head tensors. This is distinct from the original MSS specialists' positional
head mapping. K/L/M use the same seeded backbone initialization and audited
role-token initialization. The complete encoder is trained for singer and dual
recipes; inference subsequently loads it from the separation checkpoint.

## Train and resume

```bash
python train.py --model K5 \
  --initialization /weights/pt_6s.th \
  --speaker-initialization /weights/avg_model.pt \
  --manifest datasets/manifests/train_sk \
  --valid-jc datasets/manifests/valid_jc \
  --valid-sk datasets/manifests/valid_sk \
  --output runs/K5
```

The recipe fixes 200,000 optimizer updates, batch size 2 with four accumulated
microbatches, seed 1337, ten-second mixture/enrollment windows, learning rates
1e-5 (separator) and 1e-6 (encoder), and the original augmentation and loss.
There is no learning-rate scheduler. `--max-updates 4` permits a short diagnostic
run without changing the recorded 200k budget. `--dry-run` verifies inputs and
prints the run identity. `--workers` controls data-loading workers.

To continue, repeat the same command and output directory with
`--resume runs/K5/latest.pt`. Resume checks the checkpoint hash, recipe, manifests,
code, and Torch version, then restores model, optimizer, random generators and
the next data position. Checkpoints are saved at optimizer boundaries. Use
`--deterministic` from the beginning of a run when exact replay on the same
runtime/device is required; the resume command must retain that option.
Ordinary CUDA training can differ numerically between otherwise identical runs.
The verified resume check covers a deliberate stop and continuation, not recovery
from every possible interruption during filesystem writes.

Full-song validation occurs every 1,000 updates, with six separate selectors:
JC, SK, and pooled, each under shared and inclusive scopes. Shared pools retained
S/A/T/B/VP pairs; inclusive adds JC Lead or SK Bar. Pooled scores weight every
pair equally, not each corpus equally. First occurrence wins exact ties. No test
scores enter selection. Unsupported part queries retain the explicitly audited
zero-role diagnostic; Lead and Bar are never relabeled as one another.

The run writes `latest.pt` (resume state), permanent 5k snapshots, per-pair
validation records, `selection.json`, and best weights for each selector.
Each best checkpoint has an adjacent `best_<criterion>.json` inference manifest;
pass that manifest and its `.pt` file to use the validation-selected model.
The scientific loss/data functions and their source hashes are listed in
[`conditioned_training_sources.json`](../third_party/conditioned_training_sources.json).

## Use your trained model

A completed requested run also writes `inference.pt` and `inference.json`.
The export contains tensor weights and step metadata, without optimizer state.
It loads with PyTorch's restricted `weights_only=True` path.

```bash
python inference.py --run-manifest runs/K5/inference.json \
  --checkpoint runs/K5/inference.pt \
  --input mixture.wav --enrollment singer.wav --output separated --device cuda
```

For part-only training use `--role S`; dual training requires both a role and
enrollment. The manifest identifies the recipe automatically. The output is
labeled a new training run, never a published paper checkpoint.
