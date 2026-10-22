# Runnable examples

Install the [inference environment](../docs/inference.md) first. The following
example uses generated tones, so no dataset download or recording permission is
needed for the generated audio. You still need an authorized local checkpoint;
the model repository is private and public weight downloads are pending.
This is an installation check, **not an a cappella quality demonstration**.

```bash
python examples/make_demo_audio.py --output outputs/demo-input
python inference.py --model Fixed_6stem_htdemucs_SKnJC \
  --checkpoint /path/to/Fixed_6stem_htdemucs_SKnJC.pt \
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
python inference.py --model K5 --selection combined_shared \
  --checkpoint /weights/K5_step106000.pt --input mixture.wav \
  --enrollment target-singer.wav --output outputs/singer --device cuda

# Part query: no enrollment recording.
python inference.py --model K1 --selection sk_shared \
  --checkpoint /weights/K1_step198000.pt --input mixture.wav \
  --role S --output outputs/soprano --device cuda

# Dual query: singer and role.
python inference.py --model K7 --selection jc_inclusive \
  --checkpoint /weights/K7_step7000.pt --input mixture.wav \
  --enrollment target-singer.wav --role A --output outputs/dual --device cuda
```

Conditioned weights are not downloaded automatically; their public release
remains pending. `python inference.py --list-models` lists supported selections.
See [training](../docs/conditioned-training.md) for your own model and
[reproduction](../docs/reproduction.md) for frozen-checkpoint benchmark runs and
archive-derived tables. The generated tone enrollment is useful only for CLI
smoke tests; it does not represent a singer identity.
