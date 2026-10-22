# ACappellaSet

Dataset resources, model weights, and code for a cappella source separation.

**ACappellaSet v1.0 is published on [Hugging Face](https://huggingface.co/datasets/AcappellaSet/ACappellaSet).**
The release includes audio, annotations, reference splits, documentation, and
checksums. Access requires accepting the dataset Terms and manual approval.

**Model weights:** [Hugging Face](https://huggingface.co/AcappellaSet/ACappellaSet-Models)
— all 29 inference-only checkpoints are publicly downloadable under CC BY-NC 4.0. Original fixed/specialist scripts, data preparation, and canonical
scoring are included.
Conditioned inference, portable training recipes, and exact historical split manifests are included.

## Availability

| Resource | Status |
| --- | --- |
| Fixed_6stem_htdemucs_SKnJC inference | Implemented; actual-checkpoint Linux CPU validation passed |
| Checkpoint inventory | 29 validated inference exports, including both singer-augmentation models; external baseline linked |
| Model weights | [Hugging Face](https://huggingface.co/AcappellaSet/ACappellaSet-Models) — 29 public inference-only checkpoints |
| Dataset and annotations | [ACappellaSet v1.0 published on Hugging Face](https://huggingface.co/datasets/AcappellaSet/ACappellaSet); manual access approval |
| Fixed/specialist inference code | All 3 fixed models and 12 original specialists registered; publicly downloadable inference exports |
| Fixed/specialist training | Pinned MSS patch and portable launcher; see validation limits in the training guide |
| Specialist data preparation | Runnable; preserves the different JC and SK residual recipes |
| Canonical scoring | Saved-WAV scoring and exact-cohort benchmark driver |
| Table reproduction | Four tables from 3,293 archived per-pair scores with explicit selections and scope |
| Shared conditioned inference | Nine inference recipes / 14 selections, including full-remix and single-clone singer augmentation |
| Shared conditioned training | Seven recipes; clean CUDA environment, native initialization/data parity, and deterministic resume verified |
| Exact historical splits | Ten manifest sets and original SK specialist filters; audio verified on research storage |

## Choose a workflow

Start from a local clone of this repository and run commands from its root.
Dataset preparation accepts files you already have; it does not request access
or download restricted audio automatically.

| Task | Guide | Required inputs |
| --- | --- | --- |
| Separate your own recording | [Fixed/specialist inference](docs/inference.md) or [conditioned inference](docs/conditioned-inference.md) | Mixture, registered checkpoint; role and/or singer enrollment for conditioned models |
| Prepare the published dataset | [Local release preparation](docs/preparation.md#from-a-downloaded-acappellaset-release) | Extracted primary/alternate audio and release metadata |
| Train an existing recipe | [Fixed/specialist training](docs/training.md) or [conditioned training](docs/conditioned-training.md) | Prepared training/validation data and verified initializers |
| Score saved estimates | [Evaluation](docs/evaluation.md) | Aligned mixture, reference, and estimate WAVs in a pairs manifest |
| Evaluate a registered checkpoint | [Benchmarking](docs/reproduction.md#run-a-frozen-checkpoint-on-the-historical-test-cohort) | Prepared historical test manifest and checkpoint |
| Regenerate archived result tables | [Artifact reproduction](docs/reproduction.md#regenerate-the-archived-tables) | Python only; no audio or checkpoint downloads |

Training covers the 15 original fixed/specialist and seven natural-data
conditioned recipes. It starts from their documented initializers; adapting a
released ACappellaSet checkpoint to a new dataset is not an exposed workflow.
The two augmentation models support inference, but their training recipes and
augmentation audio are not included.

## Run inference

Follow the [installation and inference guide](docs/inference.md) to create the
tested Python 3.10 environment. Download a registered checkpoint from Hugging Face
without an access request and supply its local path. The loader verifies its SHA-256:

```bash
python inference.py --model Fixed_6stem_htdemucs_SKnJC --checkpoint /path/to/Fixed_6stem_htdemucs_SKnJC.pt --input /path/to/mixture.wav --output outputs/my-mixture --device cpu
```

The command writes stereo 44.1 kHz Float32 WAVs named `vp`, `bass`, `soprano`,
`alto`, `tenor`, and `other`, plus an inference receipt. The sixth output retains
the combined model's JC Lead / SK Baritone semantics. No output peak
normalization is applied. Checkpoint and configuration hashes are verified
before strict loading, and existing output directories are not overwritten.

On two validation inputs, all six outputs matched the research implementation
**bit-for-bit** in a fresh Linux CPU environment. Six behavioral tests passed; the portable export also matches exactly.
See the [validation report](docs/validation/combined-ep333-portable-linux-cpu.json).
The [inference-export checks](docs/validation/inference-exports-2026-10-22.json) additionally cover all reduced exports and both singer-augmentation adapters.
The [October 21 workflow audit](docs/validation/workflow-audit-2026-10-21.json)
additionally covers macOS inference, fresh Linux CPU/CUDA environments, short
training runs for all 22 recipes, WAV scoring, and conditioned resume. These
checks do not establish full-training convergence or reproduce paper scores.

## Other script workflows

```bash
# Discover adapters without loading Torch or downloading weights.
python inference.py --list-models

# Prepare already split, 44.1 kHz stereo six-stem inputs for original specialists.
python prepare_specialists.py --corpus JC --input /path/to/jc/6stem --output datasets/jc-specialists

# Score saved estimates, with shared/inclusive pair-weighted summaries.
python evaluate.py --manifest /path/to/pairs.csv --output outputs/scores
```

See [specialist preparation](docs/preparation.md), [evaluation](docs/evaluation.md),
and [training](docs/training.md) for input formats and supported recipes.
[Runnable examples](examples/README.md) cover fixed, part, singer and dual queries;
[paper reproduction](docs/reproduction.md) covers archived tables and fresh benchmarks.
For part-, singer-, and dual-conditioned queries, see [conditioned inference](docs/conditioned-inference.md).
`train.py --dry-run` verifies the original fixed/specialist training setup before
launching it. Model file paths are listed on Hugging Face. Dataset audio requires separate access approval.

## Models and weights

The released model families are fixed-output separation trained on JC, SK, and
their combination; per-part specialist banks; and part-, singer-, and
dual-conditioned extraction. Singer conditioning requires enrollment audio and
the matching encoder parameters.

The [checkpoint checklist](docs/checkpoints.md) lists 29 checkpoint identities
(all included), with explicit epoch/step and validation selections. Exact hashes are
in [the candidate manifest](manifests/checkpoint_candidates.json), which records
the current release inventory. The three historical fixed models and 12 historical
specialists are confirmed for release inclusion. Later singer-enrollment
specialists and newer fixed-six training runs are excluded from the ICASSP
release. Shared conditioned checkpoints are listed with their validation criteria.
Both full-remix and single-clone singer-augmentation checkpoints are included for inference.
The [inference registry](manifests/models.json) separately records models the
public code can load. See [model documentation](docs/models.md).

**Model weights:** [Hugging Face](https://huggingface.co/AcappellaSet/ACappellaSet-Models)
— publicly downloadable under CC BY-NC 4.0. Large checkpoints are not committed to this GitHub
repository.

## Dataset and training

Download the official [ACappellaSet v1.0 release](https://huggingface.co/datasets/AcappellaSet/ACappellaSet)
after manual access approval. It contains 53 songs in 107 recording versions,
distributed as primary and alternate audio ZIPs with standalone metadata,
song-level reference splits, documentation, and checksums. See the
[dataset guide](docs/dataset.md) for access, download, extraction, and format details.

Prepare the downloaded and extracted release locally:

```bash
python prepare_data.py --release-root /path/to/AcappellaSet --output datasets/release-inputs
```

This verifies the release, preserves the recorded splits and mixtures, and
creates the required SK/AB input layout using links plus absent-role silence.
It does not download audio or require Hugging Face credentials. See
[data preparation](docs/preparation.md) for the next training/evaluation commands;
recipes requiring JaCappella still need that separate corpus.
See the [training guide](docs/training.md) and [runtime validation report](docs/validation/training-runtime.json).

## Repository layout

```text
ACappellaSet/
├── download.py                   # Revision-pinned download with SHA-256 verification
├── inference.py                  # Fixed, specialist, and conditioned inference
├── train.py                      # MSS and conditioned training recipes
├── prepare_data.py               # Local release conversion and historical manifests
├── prepare_specialists.py        # Corpus-specific two-stem preparation
├── evaluate.py                   # Canonical scoring of saved estimates
├── benchmark.py                  # Frozen-checkpoint test cohorts
├── reproduce.py                  # Tables from archived per-pair scores
├── reproduction/                 # Scoped score artifacts and provenance
├── evaluation/                   # Metrics and pair-weighted reporting
├── models/                       # Architecture and inference adapter
├── configs/                      # Fixed, specialist, conditioned configs
├── requirements-inference.txt    # Direct dependencies
├── requirements-linux-cpu-lock.txt
├── environment.yml               # Optional interpreter bootstrap
├── data/                         # Audited K/L/M training loaders
├── training/                     # Conditioned training, validation, resume
├── scripts/                      # Verified MSS patch installer
├── manifests/                    # Models, dataset, candidate checkpoints, splits
├── third_party/                  # Source provenance and retained licenses
├── examples/                     # Runnable commands and synthetic smoke input
├── tests/                        # Inference behavioral tests
└── docs/                         # Guides, checkpoint checklist, validation report
```

Inference does not require a Demucs fork or a full MSS checkout. Original
fixed/specialist training uses a separate checkout at the exact documented
MSS commit plus the included reviewed patch. Shared conditioned training uses
the included [K/L/M recipes](docs/conditioned-training.md) and vendored architecture.

## License and citation

Original code and documentation use the [MIT license](LICENSE). Vendored and
adapted code retains its upstream notices; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Model weights have separate release terms; the code license does not license
them. Model terms and upstream notices will accompany the
[model weights](https://huggingface.co/AcappellaSet/ACappellaSet-Models).
Dataset audio is governed by the published
[ACappellaSet Non-commercial Research Use Terms](https://huggingface.co/datasets/AcappellaSet/ACappellaSet#full-terms-of-use),
separately from this repository's code license and any model-weight terms.

[CITATION.cff](CITATION.cff) currently describes this software repository. For the
dataset, cite ACappellaSet, version 1.0, and its
[official repository](https://huggingface.co/datasets/AcappellaSet/ACappellaSet).
The accompanying paper citation will be added when available.
