# Models and checkpoints

The release registry is [`manifests/models.json`](../manifests/models.json).
It contains the exact Combined ep333 checkpoint identity and inference settings.
Weights are not downloadable yet. The [Hugging Face model repository](https://huggingface.co/pting126/ACappellaSet-models)
contains release metadata and will host the prepared checkpoints.
The [checkpoint checklist](checkpoints.md) covers all candidates identified in
the audited paper snapshot.

| Family | Planned support | Required artifact contract |
| --- | --- | --- |
| Fixed-output HTDemucs | JC, SK, combined | Exact configuration and ordered outputs |
| Part-specific HTDemucs | Per-part specialist banks | One checkpoint per part; residual/target ordering |
| Conditioned HTDemucs | Part, singer, dual | Conditioning mode, role vocabulary, enrollment requirements, encoder state |

Soprano, Alto, Tenor, Bass, and Vocal Percussion are shared part labels. Lead
and Baritone are distinct roles. A combined model's sixth output may represent
a corpus-dependent `other` slot; it must not be advertised as two independent
Lead and Baritone estimates.

Historical MSS specialist checkpoints use residual output index 0 and target
index 1. The release loader must verify the contract instead of assuming all
two-output models follow it.

For singer-conditioned models, the matching encoder parameters are part of
the model. Replacing a fine-tuned encoder with its original pretrained weights
does not reproduce the checkpoint.

Model cards will record training data, initialization, selection criterion,
source order, preprocessing, inference settings, limitations, and applicable
weight terms. The three historical fixed models and 12 historical specialists are confirmed
for release inclusion. The later singer-enrollment specialist experiments are
excluded from this release; the original specialist banks are mixture-only.
Newer fixed-model and shared conditioned-model selections remain pending.

## Public inference support

Only Combined ep333 is currently registered and validated by the public CLI.
JC ep321, SK ep279, and the original part-specific specialist banks still need
their model-specific registration/adapters and validation. Shared part-, singer-,
and dual-conditioned adapters are also pending. A model listed for release is
not necessarily supported by the current inference command.
