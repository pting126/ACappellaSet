# Artifact manifests

`models.json` contains the registered Fixed_6stem_htdemucs_SKnJC checkpoint identity and
inference contract. Its null download URL means the weight is not publicly
hosted yet. `dataset.json` remains a pending dataset registry.

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
