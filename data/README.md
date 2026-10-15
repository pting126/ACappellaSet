# Data and enrollment policies

`strict_k.py`, `strict_l.py`, and `strict_m.py` preserve the original natural-SK,
JC, and combined conditioned-training populations and enrollment rules. Their
scientific helper functions retain source provenance in
[`conditioned_training_sources.json`](../third_party/conditioned_training_sources.json).

Use root-level `prepare_data.py` for exact historical conditioned manifests and
`prepare_specialists.py` for original mixture-only two-stem inputs, including the
optional exact SK real-target filters. See [preparation](../docs/preparation.md).
No audio is stored here. The final Zenodo archive layout remains a separate
release input; the current manifests describe the historical experiment layout.
