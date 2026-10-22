# Data and enrollment policies

`strict_k.py`, `strict_l.py`, and `strict_m.py` preserve the original natural-SK,
JC, and combined conditioned-training populations and enrollment rules. Their
scientific helper functions retain source provenance in
[`conditioned_training_sources.json`](../third_party/conditioned_training_sources.json).

Use root-level `prepare_data.py` for exact historical conditioned manifests and
`prepare_specialists.py` for original mixture-only two-stem inputs, including the
optional exact SK real-target filters. See [preparation](../docs/preparation.md).
No audio is stored here. The official
[ACappellaSet v1.0 dataset](https://huggingface.co/datasets/AcappellaSet/ACappellaSet)
is published separately. The current manifests describe the historical experiment
layout. `prepare_data.py --release-root` uses `data/release.py` to convert the
locally extracted release into verified SK/AB inputs with absent-role padding. See the [dataset guide](../docs/dataset.md).
