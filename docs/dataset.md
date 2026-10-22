# Dataset

**ACappellaSet v1.0 is published at
[AcappellaSet/ACappellaSet on Hugging Face](https://huggingface.co/datasets/AcappellaSet/ACappellaSet).**
The official release contains audio, recording metadata, song-level reference
splits, documentation, and checksums. The dataset card is the authoritative
source for its description and access conditions.

## Access and download

Access requires accepting the dataset's custom
[Non-commercial Research Use Terms](https://huggingface.co/datasets/AcappellaSet/ACappellaSet#full-terms-of-use)
and receiving manual approval for your own Hugging Face account. Submit the
request on the dataset page. These Terms are separate from this code
repository's MIT license and from separately released model-weight terms.

Install the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli#install-with-pip)
if needed, separately from the pinned model environment. After approval:

```bash
hf auth login
hf download AcappellaSet/ACappellaSet --repo-type dataset --local-dir ./AcappellaSet
```

Use the approved account when authenticating. The download includes both audio
archives and the standalone release files. For an immutable snapshot, add
`--revision` with the `revision` value in
[`manifests/dataset.json`](../manifests/dataset.json).

This is an archive-based distribution. The automatic Dataset Viewer is disabled;
direct `datasets.load_dataset("AcappellaSet/ACappellaSet")` loading is not configured.

## Files and extraction

| File | Contents |
| --- | --- |
| `audio/AcappellaSet_v1.0_primary.zip` | 53 primary recording versions |
| `audio/AcappellaSet_v1.0_alternates.zip` | 54 alternate recording versions |
| `metadata/metadata.csv` | Curator-maintained recording metadata, 20 fields |
| `metadata/splits.csv` | One `song_id,split` assignment per song |
| `file_manifest.csv` | Per-WAV path, format, size, and SHA-256 |
| `README.md`, `METADATA_SCHEMA.md`, `PROVENANCE.md`, `SPLITS.md`, `LICENSE` | Dataset description, definitions, provenance, split protocol, and Terms |
| `SHA256SUMS`, `checksums/audio_SHA256SUMS`, `checksums/metadata_SHA256SUMS` | Release-file, extracted-audio, and documentation/metadata checksums |

Extract both ZIPs into the downloaded dataset root:

```bash
unzip ./AcappellaSet/audio/AcappellaSet_v1.0_primary.zip -d ./AcappellaSet
unzip ./AcappellaSet/audio/AcappellaSet_v1.0_alternates.zip -d ./AcappellaSet
```

The archives already contain the `audio/` prefix. Extracting into
`./AcappellaSet/audio/` would incorrectly create `audio/audio/`. There is no
separate documentation ZIP. The resulting audio paths are:

```text
AcappellaSet/audio/primary/<song_id>/*.wav
AcappellaSet/audio/alternates/<song_id>/<variant>/*.wav
```

From the dataset root, optionally check downloaded files with
`shasum -a 256 -c SHA256SUMS` on macOS, or `sha256sum -c SHA256SUMS` on Linux.
After extraction, use the same command with `checksums/audio_SHA256SUMS` to
check the WAVs. Retain the release revision and checksums with your experiment.

## Audio and annotations

The release contains 53 songs and 107 recording versions from Seagull-K (47 songs),
Annappella (3), and REAL A Cappella (3). The latter two groups form the AB
experimental corpus; the metadata retains their actual group identities.
Primary selection prefers produced, then edited, then rough. Primary recordings
provide approximately 2.5 hours; counting all versions gives approximately 5 hours
with shared musical material.

All 647 WAVs are 44.1 kHz and two-channel. The 540 vocal stems are **16-bit PCM**;
mixtures comprise 60 16-bit PCM files and 47 Float32 files. Two-channel audio
may be dual-mono. Absent vocal parts have no supplied WAV and are marked `0` in
metadata. Some stems contain layered takes or multiple singers.

Some floating-point mixtures exceed ±1. Preserve their values when loading and
processing them; direct integer conversion can clip them. Any gain change must
be shared by the mixture and its stems. The published
[PROVENANCE.md](https://huggingface.co/datasets/AcappellaSet/ACappellaSet/blob/main/PROVENANCE.md)
describes mixture reconstruction and the 106-file mixture-consistency audit;
PC01 is outside that audit's coverage.

## Splits

Join `metadata/splits.csv` to `metadata/metadata.csv` on `song_id`. All recording
versions of a song must stay in the same split. The reference assignments are
34 training, 6 validation, and 7 test songs from SK, plus 6 AB songs assigned to
`external_test`. The splits are not singer-disjoint.

The reference AB singer-specific benchmark excludes PC01, leaving five eligible
songs. PC01 remains in the released dataset; `external_test` alone does not imply
benchmark eligibility. See the official
[SPLITS.md](https://huggingface.co/datasets/AcappellaSet/ACappellaSet/blob/main/SPLITS.md)
and the code's [evaluation guide](evaluation.md) for the precise protocol.

## Training and evaluation inputs

After downloading and extracting the approved dataset, create compatible local
inputs without copying the real recordings:

```bash
python prepare_data.py --release-root /path/to/AcappellaSet --output datasets/release-inputs
```

This offline command verifies release hashes, maps recordings to historical
SK/AB paths and splits, and generates silent padding for explicitly absent roles.
It preserves released mixtures and leaves the source dataset untouched. The
default output uses symlinks; `--link-mode copy` creates an independent copy.
See [preparation](preparation.md#from-a-downloaded-acappellaset-release) for the
resulting training paths and subsequent manifest commands. Dataset access and
manual downloading remain the user's responsibility.

Exact historical composition assignments, target rows, enrollment galleries,
and specialist filters are preserved in
[`manifests/splits`](../manifests/splits/README.md). These retain historical
recording identifiers and query-level populations; they are not replacements
for the release's song-level split CSV. See [preparation](preparation.md) for
required input layouts. JaCappella (JC), used by some recipes, is a separate
corpus and is not included in ACappellaSet.

## Citation

Acknowledge ACappellaSet, identify version 1.0, and cite its
[official dataset repository](https://huggingface.co/datasets/AcappellaSet/ACappellaSet).
Dataset credits are Ting-Yu Pan and Kexin Phyllis Ju. Use the accompanying paper
citation when it becomes available. The root `CITATION.cff` describes the software.
