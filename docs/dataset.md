# Dataset

The dataset will be uploaded to Zenodo. The record URL, DOI, exact version,
archive checksums, and license will be added to
[`manifests/dataset.json`](../manifests/dataset.json).
No public dataset download is provided by this repository yet.

The dataset record is intended to contain audio, annotations, exact splits,
format documentation, and citation information. The documentation will describe
sample rates, channels, file encoding, mixture/stem relationships, part labels,
composition IDs, recording variants, and singer metadata that can be published.

Dataset preparation code will consume the published schema. It will preserve
composition-level splits and distinguish intentionally absent parts from
missing files. Keep original downloads separate from derived training inputs.

The dataset license is separate from the code license. Any augmented or
converted material must have its own documented provenance and applicable
terms. The initial repository makes no claim that those assets are available.

Exact historical composition splits, target rows, and enrollment galleries are
already included in [`manifests/splits`](../manifests/splits/README.md).
`prepare_data.py` resolves user-provided historical audio roots and verifies
source files. These metadata do not imply that the Zenodo audio release is
already available. See [preparation](preparation.md).
