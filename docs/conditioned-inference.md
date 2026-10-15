# Conditioned extraction

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

The 12 active selections are listed in [the conditioned registry](../manifests/conditioned.json).
K1/K5/K7 and L1/L3/L4/M3 remain accepted research aliases. `combined_*` means
pooled JC+SK **validation**, while `SKnJC` in a model name means joint training.
Selection is required when multiple checkpoint identities are registered.
The loader accepts only the exact registered checkpoint hash and loads every
separator and encoder tensor strictly. These weights are currently local-only;
the download command reports their unpublished status without substituting a model.

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
