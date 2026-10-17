# ACappellaSet

Dataset resources, model weights, and code for a cappella source separation.

**Fixed_6stem_htdemucs_SKnJC weights and Linux CPU inference are available.** Download the
110 MB portable checkpoint from [Hugging Face](https://huggingface.co/AcappellaSet/ACappellaSet-Models)
using the verified download command below. The dataset is being prepared for
Zenodo. Original fixed/specialist scripts, data preparation, and canonical scoring are included.
Conditioned inference, portable training recipes, and exact historical split manifests are included.

## Availability

| Resource | Status |
| --- | --- |
| Fixed_6stem_htdemucs_SKnJC inference | Implemented; actual-checkpoint Linux CPU validation passed |
| Checkpoint inventory | 27 active checkpoints, 2 deferred AI-augmented checkpoints; external baseline linked |
| Public checkpoint downloads | Fixed_6stem_htdemucs_SKnJC available on [Hugging Face](https://huggingface.co/AcappellaSet/ACappellaSet-Models) |
| Dataset and annotations | Zenodo dataset record pending |
| Fixed/specialist inference code | All 3 fixed models and 12 original specialists registered; unpublished weights must be supplied locally |
| Fixed/specialist training | Pinned MSS patch and portable launcher; see validation limits in the training guide |
| Specialist data preparation | Runnable; preserves the different JC and SK residual recipes |
| Canonical scoring | Saved-WAV scoring and exact-cohort benchmark driver |
| Table reproduction | Four tables from 3,293 archived per-pair scores with explicit selections and scope |
| Shared conditioned inference | Seven recipes / 12 selections; CPU checkpoint and CUDA full-song parity passed; weights supplied locally |
| Shared conditioned training | Seven recipes; clean CUDA environment, native initialization/data parity, and deterministic resume verified |
| Exact historical splits | Ten manifest sets and original SK specialist filters; audio verified on research storage |

## Run inference

Follow the [installation and inference guide](docs/inference.md) to create the
tested Python 3.10 environment. Download and verify the registered checkpoint:

```bash
python download.py --model Fixed_6stem_htdemucs_SKnJC
python inference.py --model Fixed_6stem_htdemucs_SKnJC --checkpoint checkpoints/Fixed_6stem_htdemucs_SKnJC.pt --input /path/to/mixture.wav --output outputs/my-mixture --device cpu
```

The command writes stereo 44.1 kHz Float32 WAVs named `vp`, `bass`, `soprano`,
`alto`, `tenor`, and `other`, plus an inference receipt. The sixth output retains
the combined model's JC Lead / SK Baritone semantics. No output peak
normalization is applied. Checkpoint and configuration hashes are verified
before strict loading, and existing output directories are not overwritten.

On two validation inputs, all six outputs matched the research implementation
**bit-for-bit** in a fresh Linux CPU environment. Six behavioral tests passed; the portable export also matches exactly.
See the [validation report](docs/validation/combined-ep333-portable-linux-cpu.json).
CUDA and macOS have not been validated, and these tests are not quality scores.

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
launching it. These scripts do not provide missing checkpoint downloads or dataset audio.

## Models and weights

The planned model families are fixed-output separation trained on JC, SK, and
their combination; per-part specialist banks; and part-, singer-, and
dual-conditioned extraction. Singer conditioning requires enrollment audio and
the matching encoder parameters.

The [checkpoint checklist](docs/checkpoints.md) lists 29 checkpoint identities
(27 active, 2 deferred), including updated K1/M3 epoch/step choices. Exact hashes are
in [the candidate manifest](manifests/checkpoint_candidates.json), which records
the current release inventory. The three historical fixed models and 12 historical
specialists are confirmed for release inclusion. Later singer-enrollment
specialists and newer fixed-six training runs are excluded from the ICASSP
release. Shared conditioned checkpoints are listed with their validation criteria.
The AI-augmented K6 and K6_no_aug releases are deferred.
The [inference registry](manifests/models.json) separately records models the
public code can load. See [model documentation](docs/models.md).

Weights are hosted in the [Hugging Face model repository](https://huggingface.co/AcappellaSet/ACappellaSet-Models),
with verified checksums and applicable licenses. Fixed_6stem_htdemucs_SKnJC is available;
other checkpoints remain in preparation. A final paper snapshot may also
be archived on Zenodo. Large checkpoints are not committed to this GitHub repository.

## Dataset and training

Audio, annotations, splits, format documentation, and applicable terms will be
published in a Zenodo dataset record linked from this repository. See the
[dataset guide](docs/dataset.md). The current specialist preparation script consumes the historical pre-split
six-stem layout; it does not create the forthcoming Zenodo release or its splits.
See the [training guide](docs/training.md) and [runtime validation report](docs/validation/training-runtime.json).

## Repository layout

```text
ACappellaSet/
├── download.py                   # Revision-pinned download with SHA-256 verification
├── inference.py                  # Fixed, specialist, and conditioned inference
├── train.py                      # MSS and conditioned training recipes
├── prepare_data.py               # Exact historical manifests against local audio
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
**Project model-weight license: TBD.** Further weight uploads are paused pending
upstream rights clarification. Combined/SKnJC was previously published under
Apache-2.0; this correction does not revoke valid prior grants or establish
upstream rights. See the [weight licensing status and historical record](https://huggingface.co/AcappellaSet/ACappellaSet-Models/blob/main/WEIGHTS_NOTICE.md).
Dataset audio and third-party rights remain separate.

[CITATION.cff](CITATION.cff) currently describes this repository. Paper metadata
and Zenodo DOIs will be added when available.
