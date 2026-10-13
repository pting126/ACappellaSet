# Models and checkpoints

The release registry is [`manifests/models.json`](../manifests/models.json).
It currently contains no released checkpoints. Weights are planned for a
separate Zenodo record linked from that registry.

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
weight terms. Public checkpoint choices have not yet been finalized here.
