# Preparing local training and evaluation data

## From a downloaded ACappellaSet release

Obtain dataset approval and download/extract both audio archives yourself, as
shown in the [dataset guide](dataset.md). All commands below operate on local
files. They do not connect to Hugging Face, download audio, or require tokens.
Use the Python environment described in the [inference guide](inference.md).

Pass the extracted release root, containing `file_manifest.csv`, `metadata/`,
and `audio/`, and a **new output directory outside the release root**:

```bash
python prepare_data.py --release-root /path/to/AcappellaSet --output datasets/release-inputs
```

This verifies the metadata against the dataset revision in
[`manifests/dataset.json`](../manifests/dataset.json), verifies every audio file's
SHA-256, size, header, alignment and finite samples, and creates:

```text
datasets/release-inputs/
├── SK/train_flat/<recording_id>/{mixture,soprano,alto,tenor,baritone,bass,vp}.wav
├── SK/valid_flat/<recording_id>/...
├── SK/test_flat/<recording_id>/...
├── AB/<recording_id>/...
└── release_preparation.json
```

The SK split counts remain **69 / 11 / 20 recording versions**. All seven AB
versions are retained, including PC01; existing benchmark manifests still
exclude PC01 targets and choose their historical recording versions. Song-level
split assignments are checked against the historical inventory, never regenerated.

By default the 647 released audio files become relative symlinks. Their bytes,
mixtures, gain, and role names remain unchanged. The command creates 102 aligned,
16-bit PCM silent WAVs only for roles explicitly marked `0` in the release
metadata: 97 SK and five AB padding files. This consumes additional disk space
for silence, but does not duplicate the real recordings. Padding reproduces
absent-role waveforms, not historical WAV container bytes. Historical target and
enrollment hashes are checked wherever recorded. Missing or corrupted active
stems cause an error; they are never silently replaced with zeros.

Keep the source release available at the same relative location. To make a
self-contained output, or when symlinks are unavailable, add `--link-mode copy`;
this requires enough disk space for another audio copy. Both modes leave the
source release untouched and refuse to overwrite existing output directories.
The receipt records the source revision, mappings, hashes, and generated padding.

For original fixed SK training, pass `datasets/release-inputs/SK/train_flat`
and `datasets/release-inputs/SK/valid_flat` as `--train-data` and `--valid-data`
to the [existing training command](training.md). For specialists, pass
`datasets/release-inputs/SK` to `prepare_specialists.py` as shown below.

For conditioned training or the benchmark CLI, prepare the existing manifests:

```bash
python prepare_data.py --set train_sk --root SK=datasets/release-inputs/SK --output datasets/manifests/train_sk
python prepare_data.py --set valid_sk --root SK=datasets/release-inputs/SK --output datasets/manifests/valid_sk
python prepare_data.py --set test_sk --root SK=datasets/release-inputs/SK --output datasets/manifests/test_sk
python prepare_data.py --set test_ab --root AB=datasets/release-inputs/AB --output datasets/manifests/test_ab
```

These commands retain the original target rows, singer identities, enrollment
rules, and exclusions. Updated descriptive release metadata does not silently
rewrite historical experimental annotations. Training and benchmark invocations
then use the resulting manifest directories, following the guides below.

JaCappella (JC) is a separate dataset: obtain and prepare it separately for recipes
that require it, including K-family JC validation. Supply its existing root with
`--root JC=/path/to/JC`; the release conversion neither supplies nor changes JC.
Custom datasets or modified metadata are not the pinned v1.0 release; use the
existing input-layout workflow instead of disabling integrity checks.

## Original part-specific inputs

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

These counts describe the historical six-stem inputs. The published
[ACappellaSet v1.0 archives](https://huggingface.co/datasets/AcappellaSet/ACappellaSet)
use a different layout and omit absent-role padding WAVs. Run the local release
conversion above before invoking this specialist preparation command. All inputs must be
aligned, finite, 44.1 kHz stereo.
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

These manifests describe the historical experiments. The official dataset is
published; its version, revision, file URLs, and checksums are recorded in
[`manifests/dataset.json`](../manifests/dataset.json). The `--release-root` mode above converts its primary/alternate layout to these
input paths and generates explicitly absent-role padding. JaCappella (JC) is a
separate corpus and is not included in ACappellaSet.

The [release-input validation report](validation/release-inputs-2026-10-22.json)
records the complete 647-file conversion check, historical manifest preparation,
actual training-loader reads, and regression tests.
