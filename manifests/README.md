# Artifact manifests

`models.json` and `dataset.json` are empty release registries. Their null
locations indicate unpublished artifacts, not valid download endpoints.

Before a model entry is added, record a stable ID, model family, training
corpus, configuration path, dependency revisions, checkpoint URL and SHA-256,
checkpoint format, sample rate, input channels, exact source order, supported
roles, and license. Include enrollment and encoder requirements when relevant.
Record the training step or epoch and the validation selection criterion.

Dataset entries will identify the exact Zenodo version, archive URLs and
SHA-256 checksums, metadata version, split files, and applicable terms.
Do not store private machine paths or server credentials in public manifests.
