# Preparing original part-specific training data

`prepare_specialists.py` consumes an **already prepared, already split** six-stem
corpus. It does not download audio, resample files, decide splits, or create
silent placeholders. Missing source files fail, including genuinely absent
parts that must already have explicitly prepared zero WAVs.

```bash
python prepare_specialists.py --corpus JC --input /path/to/jc/6stem --output datasets/jc-specialists
python prepare_specialists.py --corpus SK --input /path/to/sk/6_stem --output datasets/sk-specialists
```

| Corpus | Input splits | Expected track counts | Mixture | Residual |
| --- | --- | --- | --- | --- |
| JC | train / valid / test | 30 / 10 / 10 | Existing mixture, unchanged | Float32 sum of other five stems |
| SK | train_flat / valid_flat / test_flat | 69 / 11 / 20 | Float32 sum of six stems | Additive mixture minus target |

These counts describe the historical six-stem inputs, not a guarantee about the
future Zenodo layout. All inputs must be aligned, finite, 44.1 kHz stereo.
Summation order matches the original preparation scripts. There is no peak
normalization or clipping; generated files are Float32 WAVs.

Use `--targets alto tenor` to prepare selected banks. `--allow-subset` permits
custom data or synthetic smoke tests; the receipt records that historical
counts were not enforced. It does not establish benchmark comparability.

The output has `mixtures/<split>/<track>/mixture.wav` and
`<target>/<split>/<track>/{mixture,<target>,non<target>}.wav`.
Targets are copied, residuals are generated, and mixture symlinks point within
the output tree. Moving the entire output directory preserves those links.
`preparation.json` records source and generated-file checksums and the recipe.
The output must be a new directory outside the input tree.

Historical SK Soprano and Baritone specialist runs used **real-target filtered
subsets** after this preparation. Those eligibility manifests are not yet
published. Do not describe the unfiltered six-stem preparation as an exact
reproduction of their training populations. Track counts alone also do not
verify composition-level split integrity.
