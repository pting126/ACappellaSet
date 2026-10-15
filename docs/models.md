# Models and checkpoints

The release registry is [`manifests/models.json`](../manifests/models.json).
It contains the exact Fixed_6stem_htdemucs_SKnJC checkpoint identity and inference settings.
Combined portable weights are available. The [Hugging Face model repository](https://huggingface.co/pting126/ACappellaSet-models)
contains the Combined checkpoint and metadata for the remaining planned weights.
The [checkpoint checklist](checkpoints.md) covers all candidates identified in
the updated release plan.

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
Newer fixed-six training runs are also excluded from the ICASSP release. Shared
conditioned checkpoints, including singer-conditioned AI augmentation, are
listed with their validation selectors in the registry. K1 and M3 now use
final-200k selections with matching updated evaluations; historical checkpoint
identities remain recorded.

## Public inference support

Only Fixed_6stem_htdemucs_SKnJC is currently registered and validated by the public CLI.
Fixed_6stem_htdemucs_JC, Fixed_6stem_htdemucs_SK, and the original part-specific specialist banks still need
their model-specific registration/adapters and validation. Shared part-, singer-,
and dual-conditioned adapters are also pending. A model listed for release is
not necessarily supported by the current inference command.

The active release plan has 27 checkpoints; K6 and K6_no_aug are deferred.
Natural SK-only weight licensing is Apache-2.0; JC-trained weight terms need
clarification. The existing Combined Apache release remains recorded.

Public fixed-model names are `Fixed_6stem_htdemucs_SK`,
`Fixed_6stem_htdemucs_JC`, and `Fixed_6stem_htdemucs_SKnJC`.
`SKnJC` means joint SK+JC training. Epochs remain in provenance metadata,
not in the public names. The legacy `combined-ep333` CLI alias still works.
