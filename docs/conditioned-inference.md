# Shared query-conditioned extraction

Part conditioning specifies a vocal role. Singer conditioning specifies a singer
using enrollment audio. Dual conditioning uses both. Each returns `target.wav`
and `residual.wav` as stereo 44.1 kHz Float32 audio at the model's original gain.
The conditioned output order is **target, residual**; the original specialists
use the opposite order.

Use the existing [Linux CPU installation](inference.md). The required Demucs and
ReDimNet2 inference source is included with pinned provenance and upstream
licenses. A separate Demucs checkout, WeSpeaker installation, or original
speaker pretrained checkpoint is unnecessary for inference: the separation
checkpoint contains its matching encoder state.

## Choose a model and validation selection

```bash
python inference.py --list-models

python inference.py --model Part_conditioned_htdemucs_SK --selection sk_shared \
  --checkpoint /path/to/registered-checkpoint.pt --role S \
  --input mixture.wav --output outputs/soprano --device cpu

python inference.py --model Singer_conditioned_htdemucs_SK --selection combined_inclusive \
  --checkpoint /path/to/registered-checkpoint.pt --enrollment singer.wav \
  --enrollment-start 5 --input mixture.wav --output outputs/singer --device cpu

python inference.py --model Dual_conditioned_htdemucs_JC --selection sk_inclusive \
  --checkpoint /path/to/registered-checkpoint.pt --role A --enrollment singer.wav \
  --input mixture.wav --output outputs/alto-singer --device cuda
```

The 14 released selections from nine model recipes are listed in [the conditioned registry](../manifests/conditioned.json). `combined_*` means
pooled JC+SK **validation**, while `SKnJC` in a model name means joint training.
Selection is required when multiple checkpoint identities are registered.
The loader accepts only the exact registered checkpoint hash and loads every
separator and encoder tensor strictly. All 14 weight files are uploaded to the currently private [model repository](https://huggingface.co/AcappellaSet/ACappellaSet-Models). Use `hf auth login` and `hf download` with repository access, then pass the downloaded path with `--checkpoint`. Released exports contain only `format`, `step`, and all `model_state` tensors, including the matching fine-tuned singer encoder. They load with `weights_only=True`; arbitrary or modified files are rejected.

## Singer-augmentation models

The augmentation models use the same singer-conditioned architecture and enrollment preprocessing. Both use pooled-inclusive validation selection; `combined_inclusive` is their CLI selection name.

| Public model ID | Training data | Selected step |
| --- | --- | ---: |
| `Singer_conditioned_htdemucs_SK_full_remix` | SK full-remix singer augmentation | 131,000 |
| `Singer_conditioned_htdemucs_SK_single_clone` | SK single-clone singer augmentation | 48,000 |

```bash
hf auth login
hf download AcappellaSet/ACappellaSet-Models conditioned/enrollment/train-sk-full-remix/sisdr-v1/redimnet2-finetuned/step-0131000.pt --local-dir checkpoints
python inference.py --model Singer_conditioned_htdemucs_SK_full_remix --selection combined_inclusive \
  --checkpoint checkpoints/conditioned/enrollment/train-sk-full-remix/sisdr-v1/redimnet2-finetuned/step-0131000.pt \
  --enrollment singer.wav --input mixture.wav --output outputs/full-remix --device cpu
```

Use the single-clone model ID and its corresponding checkpoint to run that selection. Synthetic augmentation audio is not included in the dataset release; the two augmentation models have inference adapters, while the portable training recipes cover the seven natural-data models.

## Enrollment and role behavior

Enrollment is a ten-second crop beginning at `--enrollment-start` (default zero).
Short crops are padded with zeros. Cropping occurs before resampling; mono is
duplicated to stereo, and the encoder averages channels and resamples to 16 kHz
internally. The adapter applies the native enrollment peak normalization.
Empty, nonfinite, multichannel, and out-of-range crops fail clearly.

Use an isolated recording of the desired singer. The convenience CLI cannot
establish singer identity or benchmark independence from arbitrary audio files.
Exact benchmark enrollment selection belongs to the evaluation manifests.
Singer-only models require no role label and can query singers outside the
training part vocabulary.

Part and dual queries normally accept only supervised roles. To reproduce the
paper's explicit zero-token cross-vocabulary diagnostic, add
`--allow-untrained-role` for SK-trained Lead or JC-trained Bar. The loader checks
that this historical token is exactly zero and the receipt labels the diagnostic.
This does not make that role supervised or claim a meaningful part-only query.
There is no automatic Lead/Bar mapping.

## Inference and validation

Native inference uses ten-second windows, a four-second hop, an end-aligned
final window, per-window mixture peak scaling, and Hann overlap-add with weights
clamped to 0.001. No output peak normalization or mixture-consistency correction
is applied. The receipt records the selection, query, enrollment crop and hash,
input/output hashes, and runtime.

The [validation report](validation/conditioned-inference.json) distinguishes
short-input CPU checkpoint comparisons from full-song CUDA target comparisons.
It establishes implementation parity for the tested inputs, not separation
quality or cross-device bitwise equality. macOS/MPS remain unvalidated.

The [inference-export checks](validation/inference-exports-2026-10-22.json) cover exact tensor preservation and strict loading of every new export, plus native CPU and actual CLI comparisons for both augmentation models. These additional checks use synthetic audio and do not measure separation quality.
