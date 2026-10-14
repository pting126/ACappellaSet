# ACappellaSet

Dataset resources, model weights, and code for a cappella source separation.

**Combined-model inference is available and validated on Linux CPU.** The dataset
is being prepared for Zenodo, and weights will be published on
[Hugging Face](https://huggingface.co/pting126/ACappellaSet-models). Running inference
currently requires an existing copy of the registered Combined ep333 checkpoint.
Training, specialist, and conditioned-model workflows are still being packaged.

## Availability

| Resource | Status |
| --- | --- |
| Combined ep333 inference | Implemented; actual-checkpoint Linux CPU validation passed |
| Checkpoint inventory | 29 project candidates and one external baseline documented |
| Public checkpoint downloads | [Hugging Face repository](https://huggingface.co/pting126/ACappellaSet-models) created; weights pending |
| Dataset and annotations | Zenodo dataset record pending |
| Training scripts | Being prepared |
| Specialist and conditioned inference | Being prepared |

## Run inference

Follow the [installation and inference guide](docs/inference.md) to create the
tested Python 3.10 environment. With the registered checkpoint available locally:

```bash
python inference.py --model combined-ep333 --checkpoint checkpoints/combined-ep333.ckpt --input /path/to/mixture.wav --output outputs/my-mixture --device cpu
```

The command writes stereo 44.1 kHz Float32 WAVs named `vp`, `bass`, `soprano`,
`alto`, `tenor`, and `other`, plus an inference receipt. The sixth output retains
the combined model's JC Lead / SK Baritone semantics. No output peak
normalization is applied. Checkpoint and configuration hashes are verified
before strict loading, and existing output directories are not overwritten.

On two validation inputs, all six outputs matched the research implementation
**bit-for-bit** in a fresh Linux CPU environment. Five behavioral tests passed.
See the [validation report](docs/validation/combined-ep333-linux-cpu.json).
CUDA and macOS have not been validated, and these tests are not quality scores.

## Models and weights

The planned model families are fixed-output separation trained on JC, SK, and
their combination; per-part specialist banks; and part-, singer-, and
dual-conditioned extraction. Singer conditioning requires enrollment audio and
the matching encoder parameters.

The [checkpoint checklist](docs/checkpoints.md) lists all 29 project candidates
from the audited paper snapshot, including epoch/step choices. Exact hashes are
in [the candidate manifest](manifests/checkpoint_candidates.json), which records
the snapshot inventory. The three historical fixed models and 12 historical
specialists are confirmed for release inclusion; newer-model selections and
weight terms remain pending.
The [inference registry](manifests/models.json) separately records models the
public code can load. See [model documentation](docs/models.md).

Weights will be hosted in the [Hugging Face model repository](https://huggingface.co/pting126/ACappellaSet-models),
with verified checksums and applicable licenses. It currently contains metadata
and a configuration, with no weights uploaded. A final paper snapshot may also
be archived on Zenodo. Large checkpoints are not committed to this GitHub repository.

## Dataset and training

Audio, annotations, splits, format documentation, and applicable terms will be
published in a Zenodo dataset record linked from this repository. See the
[dataset guide](docs/dataset.md). Dataset preparation and training scripts will
follow the published schema and fixed splits; see the [training guide](docs/training.md).

## Repository layout

```text
ACappellaSet/
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
Dataset and model-weight terms are separate. A code license does not grant
rights to the dataset or model weights.

[CITATION.cff](CITATION.cff) currently describes this repository. Paper metadata
and Zenodo DOIs will be added when available.
