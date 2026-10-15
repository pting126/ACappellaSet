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
subsets**. Use the explicit filtering option described below for those recipes.

## Exact conditioned populations and splits

The compressed CSVs in [`manifests/splits`](../manifests/splits/index.json)
retain the historical rows, order, singer identities, composition IDs, source
hashes, and enrollment candidates. Audio is not included. Filesystem prefixes
are portable; they do not depend on a research server checkout.

Provide the historical 44.1 kHz stereo layout:

```text
SK/  train_flat/<track>/{mixture,soprano,alto,tenor,baritone,bass,vp}.wav
     valid_flat/<track>/...  test_flat/<track>/...
JC/  train/<song>/{mixture,soprano,alto,tenor,bass,vp,lead_vocal}.wav
     valid/<song>/...  test/<song>/...
AB/  <track>/... (the historical test/6stem root)
```

For example:

```bash
python prepare_data.py --set train_sk --root SK=/data/SK --output datasets/manifests/train_sk
python prepare_data.py --set valid_jc --root JC=/data/JC --output datasets/manifests/valid_jc
python prepare_data.py --set valid_sk --root SK=/data/SK --output datasets/manifests/valid_sk
```

| Recipe | Training set | Validation sets |
| --- | --- | --- |
| K1 / K5 / K7 | `train_sk` | `valid_jc`, `valid_sk` |
| L1 / L3 / L4 | `train_jc` | `valid_jc`, `valid_sk` |
| M3 | `train_sknjc` | `valid_m_jc`, `valid_m_sk` |

Combined preparation needs both `--root JC=...` and `--root SK=...`.
`test_jc`, `test_sk`, and `test_ab` preserve the held-out evaluation cohorts.
SK's test manifest has 100 candidate pairs; the scorer removes two silent
references. AB excludes PC01 as a target. Enrollment galleries retain the
historical independent-song rules and explicit fallback metadata.

Preparation verifies input hashes where historical hashes exist, verifies audio
rate/channels/alignment, and records every referenced audio hash. It writes a new
`preparation.json` and never overwrites an existing output. `--metadata-only`
is useful for inspecting paths but does not qualify a manifest for training.
Do not modify the audio after preparation; regenerate the manifest receipt if
files are replaced. `ARCHIVED_AUG_LINK` appears only in unused historical
`prepared_link` metadata; these natural-data recipes do not read that audio.

The exact six-stem memberships and SK specialist filters are recorded in
[`track_inventory.json`](../manifests/splits/track_inventory.json). To reproduce
the original SK specialist populations, add `--historical-role-filter` to
`prepare_specialists.py`: Soprano retains 50/9/14 and Baritone 22/2/6 tracks
across train/validation/test; other roles retain the full inventory.
Without `--allow-subset`, preparation now checks exact membership as well as
counts. Composition IDs are disjoint across train, validation, and test.

These manifests describe the historical experiments. A final Zenodo archive
identifier and mapping from its published layout remain pending the dataset
release; the scripts do not assume that an unpublished archive is available.
