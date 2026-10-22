# Runnable examples

Install the [inference environment](../docs/inference.md) first. The following
example uses generated tones, so no dataset download or recording permission is
needed for the generated audio. Download a registered checkpoint from the
[public model repository](https://huggingface.co/AcappellaSet/ACappellaSet-Models);
no login or access request is required for model weights.
This is an installation check, **not an a cappella quality demonstration**.

```bash
python examples/make_demo_audio.py --output outputs/demo-input
python download.py --model Fixed_6stem_htdemucs_SKnJC
python inference.py --model Fixed_6stem_htdemucs_SKnJC \
  --checkpoint checkpoints/Fixed_6stem_htdemucs_SKnJC.pt \
  --input outputs/demo-input/mixture.wav --output outputs/demo-separated --device cpu
```

Use a new output directory on each invocation. The six Float32 output WAVs and
`inference.json` record exact file hashes, sample rate, channels, source order,
configuration and checkpoint identity. Replace the synthetic input with your own
recording for a meaningful separation example.

For conditioned inference, supply the exact registered local checkpoint and a
real enrollment recording of the target singer:

```bash
# Singer query: no role label.
python inference.py --model Singer_conditioned_htdemucs_SK --selection combined_shared \
  --checkpoint /weights/singer-sk-step106000.pt --input mixture.wav \
  --enrollment target-singer.wav --output outputs/singer --device cuda

# Part query: no enrollment recording.
python inference.py --model Part_conditioned_htdemucs_SK --selection sk_shared \
  --checkpoint /weights/part-sk-step198000.pt --input mixture.wav \
  --role S --output outputs/soprano --device cuda

# Dual query: singer and role.
python inference.py --model Dual_conditioned_htdemucs_SK --selection jc_inclusive \
  --checkpoint /weights/dual-sk-step7000.pt --input mixture.wav \
  --enrollment target-singer.wav --role A --output outputs/dual --device cuda
```

All conditioned selections are publicly downloadable. Use `python download.py --model MODEL --selection CRITERION` or `hf download` without login, then pass the downloaded file to inference. Inference does not download weights automatically. `python inference.py --list-models` lists supported selections.
See [training](../docs/conditioned-training.md) for your own model and
[reproduction](../docs/reproduction.md) for frozen-checkpoint benchmark runs and
archive-derived tables. The generated tone enrollment is useful only for CLI
smoke tests; it does not represent a singer identity.
