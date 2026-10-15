# Artifact manifests

`models.json` registers the three original fixed models and 12 original
specialists, including exact checkpoint/config hashes and output contracts.
Only the Combined model has a published `portable_checkpoint.download_url`;
the other adapters accept the registered local original checkpoint. A null
legacy `url` does not override a populated portable download URL.
`dataset.json` remains a pending dataset registry.

`checkpoint_candidates.json` lists the 29 deduplicated project candidates and
one external baseline in the audited paper snapshot. It is a release-planning
inventory, not a download registry or a finalized checkpoint selection.
See [the readable checklist](../docs/checkpoints.md).

Before a model entry is added, record a stable ID, model family, training
corpus, configuration path, dependency revisions, checkpoint URL and SHA-256,
checkpoint format, sample rate, input channels, exact source order, supported
roles, and license. Include enrollment and encoder requirements when relevant.
Record the training step or epoch and the validation selection criterion.

Dataset entries will identify the exact Zenodo version, archive URLs and
SHA-256 checksums, metadata version, split files, and applicable terms.
Do not store private machine paths or server credentials in public manifests.
