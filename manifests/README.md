# Artifact manifests

`models.json` registers the three original fixed models and 12 original
specialists, including exact checkpoint/config hashes and output contracts.
`conditioned.json` adds nine shared query-conditioned recipes with 14 checkpoint
selections. Together, the registries provide all 29 inference exports, each with
a revision-pinned `portable_checkpoint.download_url`, size, and SHA-256.
All 29 exports are publicly downloadable without login or an access request. Adapters also accept exact registered original checkpoints.
A null legacy `url` does not override a populated portable download URL.
`dataset.json` records the published ACappellaSet v1.0 dataset on Hugging Face,
including an immutable repository revision, file URLs, sizes, SHA-256 checksums,
and the manual access-approval requirement.

`checkpoint_candidates.json` records the 29 selected project checkpoint
identities and one separately linked external baseline. It retains original
research hashes alongside inference-export hashes, release paths, and revisions.
Use `models.json` and `conditioned.json` for CLI IDs and download metadata.
See [the readable checklist](../docs/checkpoints.md).

Before a model entry is added, record a stable ID, model family, training
corpus, configuration path, dependency revisions, checkpoint URL and SHA-256,
checkpoint format, sample rate, input channels, exact source order, supported
roles, and license. Include enrollment and encoder requirements when relevant.
Record the training step or epoch and the validation selection criterion.

Dataset entries identify the official release version, archive URLs and
SHA-256 checksums, metadata paths, split files, and the published Terms.
Do not store private machine paths or server credentials in public manifests.
