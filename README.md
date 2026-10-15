# ACappellaSet

Dataset resources, model weights, and code for a cappella source separation.

**Fixed_6stem_htdemucs_SKnJC weights and Linux CPU inference are available.** Download the
110 MB portable checkpoint from [Hugging Face](https://huggingface.co/pting126/ACappellaSet-models)
using the verified download command below. The dataset is being prepared for
Zenodo. Other model adapters and training workflows remain in preparation.

## Availability

| Resource | Status |
| --- | --- |
| Fixed_6stem_htdemucs_SKnJC inference | Implemented; actual-checkpoint Linux CPU validation passed |
| Checkpoint inventory | 27 active checkpoints, 2 deferred AI-augmented checkpoints; external baseline linked |
| Public checkpoint downloads | Fixed_6stem_htdemucs_SKnJC available on [Hugging Face](https://huggingface.co/pting126/ACappellaSet-models) |
| Dataset and annotations | Zenodo dataset record pending |
| Training scripts | Being prepared |
| Specialist and conditioned inference | Being prepared |

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

Weights are hosted in the [Hugging Face model repository](https://huggingface.co/pting126/ACappellaSet-models),
with verified checksums and applicable licenses. Fixed_6stem_htdemucs_SKnJC is available;
other checkpoints remain in preparation. A final paper snapshot may also
be archived on Zenodo. Large checkpoints are not committed to this GitHub repository.

## Dataset and training

Audio, annotations, splits, format documentation, and applicable terms will be
published in a Zenodo dataset record linked from this repository. See the
[dataset guide](docs/dataset.md). Dataset preparation and training scripts will
follow the published schema and fixed splits; see the [training guide](docs/training.md).

## Repository layout

```text
ACappellaSet/
├── download.py                   # Revision-pinned download with SHA-256 verification
├── inference.py                  # Working fixed-model CLI
├── models/                       # Architecture and inference adapter
├── configs/                      # Fixed, specialist, conditioned configs
├── requirements-inference.txt    # Direct dependencies
├── requirements-linux-cpu-lock.txt
├── environment.yml               # Optional interpreter bootstrap
├── data/                         # Dataset loaders: pending
├── scripts/                      # Download/preparation tools: pending
├── manifests/                    # Models, dataset, candidate checkpoints, splits
├── third_party/                  # Source provenance and retained licenses
├── examples/                     # Public audio example: pending
├── tests/                        # Inference behavioral tests
└── docs/                         # Guides, checkpoint checklist, validation report
```

`train.py` and `evaluate.py` will be added with their working implementations.
No Demucs fork or full MSS checkout is required for the current fixed-model
inference package. Required MSS training modifications will be released separately.

## License and citation

Original code and documentation use the [MIT license](LICENSE). Vendored and
adapted code retains its upstream notices; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Fixed_6stem_htdemucs_SKnJC weights use Apache-2.0; see the [weight license](https://huggingface.co/pting126/ACappellaSet-models/blob/main/fixed/SKnJC/legacy/LICENSE).
The 13 natural SK-only weights are assigned Apache-2.0. JC-trained weight
terms require clarification; the existing Combined release retains its published
license. Dataset audio and third-party rights are separate from project licensing.

[CITATION.cff](CITATION.cff) currently describes this repository. Paper metadata
and Zenodo DOIs will be added when available.
