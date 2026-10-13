# ACappellaSet

Dataset resources, model weights, and code for a cappella source separation.
This repository will provide inference and training support for fixed-output,
part-specific, and conditioned separation models.

## Release status

The public release is being prepared. This initial repository contains the
project layout, release documentation, and dependency and artifact manifests.
Inference and training implementations, tested model environments, checkpoints,
and dataset downloads are not yet available here.

| Resource | Location | Availability |
| --- | --- | --- |
| Model implementations and scripts | This repository | Being prepared |
| Training and inference configurations | `configs/` | Being prepared |
| Dataset, annotations, and splits | Zenodo; linked from this repository | Record pending |
| Model checkpoints | Separate Zenodo model record; linked from `manifests/models.json` | Record pending |
| Tested inference and training environment | This repository | Pending checkpoint validation |

## Models

The planned release covers:

- **Fixed-output separation:** models trained on JC, SK, and their combination.
- **Part-specific extraction:** a bank of specialist models for individual parts.
- **Conditioned extraction:** part-, singer-, and dual-conditioned models. Singer
  conditioning requires enrollment audio and the matching encoder weights.

The checkpoint registry will identify each released model's configuration,
training corpus, output order, dependencies, and SHA-256 checksum. Final model
selections and download links will be added when the artifacts are published.
See [model documentation](docs/models.md).

## Repository layout

```text
ACappellaSet/
├── README.md
├── LICENSE
├── THIRD_PARTY_NOTICES.md
├── CITATION.cff
├── environment.yml              # Repository tooling only, initially
├── models/                      # Custom architectures and adapters
├── configs/
│   ├── fixed/
│   ├── specialists/
│   └── conditioned/
├── data/                        # Dataset loaders and preprocessing
├── scripts/                     # Artifact and dataset utilities
├── manifests/
│   ├── models.json
│   ├── dataset.json
│   └── splits/
├── third_party/
│   ├── sources.json
│   └── patches/
├── examples/
├── tests/
└── docs/
    ├── dataset.md
    ├── models.md
    ├── inference.md
    ├── training.md
    └── evaluation.md
```

The planned top-level entry points are `inference.py`, `train.py`, and
`evaluate.py`. They will be added with their working implementations; this
initial structure does not provide those commands.

## Dataset and weights

Dataset audio will be hosted on Zenodo with annotations, split manifests,
documentation, and its applicable license. Model weights will use a separate
record so that models can be versioned independently of the dataset.

Large audio files and checkpoints are not stored in Git history. Download
instructions will link to specific artifact versions and verified checksums.
See [dataset documentation](docs/dataset.md).

## Inference and training

The release will include a tested inference example and instructions for
starting training with the published data and configurations. See the
[inference guide](docs/inference.md) and [training guide](docs/training.md)
for their current availability and required artifacts.

`environment.yml` currently provides Python for repository tooling. It is not
the model runtime, and does not install PyTorch, Demucs, or MSS. Model runtime
instructions and exact tested dependencies will accompany the model code.

## License and citation

See [LICENSE](LICENSE) for the current code licensing status and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream dependencies.
Dataset and model-weight terms will be documented separately with their
artifacts. A code license does not grant rights to the dataset or model weights.

[CITATION.cff](CITATION.cff) currently describes this repository. Paper metadata
and Zenodo DOIs will be added when available.
