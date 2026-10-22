# Models and checkpoints

The release registry is [`manifests/models.json`](../manifests/models.json).
It contains fixed and original specialist identities and inference settings.
The [conditioned registry](../manifests/conditioned.json) adds seven recipes covering 12 active selections.
Public checkpoint downloads are pending. The currently private
[Hugging Face model repository](https://huggingface.co/AcappellaSet/ACappellaSet-Models)
contains the SKnJC portable checkpoint and release metadata. All inference
examples require an authorized local checkpoint.
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

All three fixed models and the 12 original mixture-only specialists are registered.
Use `python inference.py --list-models` to see IDs and ordered outputs. The
`Checkpoint download` column reflects a registered URL, not a live access check;
the registered portable URL currently requires private repository access. The same strict loader verifies each model's checkpoint/config
hashes. Specialists additionally enforce residual index 0 and target index 1.
The SKnJC portable checkpoint is uploaded privately; other adapters require
the exact original local checkpoint. Their original filenames/hashes remain in
the registry. Shared part-, singer-, and dual-conditioned adapters are also available
with verified local checkpoints; see [conditioned inference](conditioned-inference.md).

The active release plan has 27 checkpoints; K6 and K6_no_aug are deferred.
Weight terms and upstream notices are maintained separately in the model
repository. This GitHub repository's MIT code license does not license weights;
historical publication receipts retain their original terms and dates.

Public fixed-model names are `Fixed_6stem_htdemucs_SK`,
`Fixed_6stem_htdemucs_JC`, and `Fixed_6stem_htdemucs_SKnJC`.
`SKnJC` means joint SK+JC training. Epochs remain in provenance metadata,
not in the public names. The legacy `combined-ep333` CLI alias still works.
