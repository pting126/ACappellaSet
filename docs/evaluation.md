# Evaluation

A basic evaluation entry point will accompany the model implementation.
It is not included in this initial repository.

Evaluation must retain the selected model's preprocessing and inference
settings, compare corresponding source roles, and document the aggregation
scope. Shared-role evaluation and native-role evaluation are different scopes;
Lead and Baritone must not be relabeled as each other.

Singer enrollment protocols must state whether enrollment comes from a
different composition and whether any fallback is allowed. Strict enrollment
and fallback-inclusive results must remain separate. Checkpoints are selected
using validation criteria, not held-out test scores.

For future AB evaluations, exclude targets whose authoritative parent song ID
is `PC01`, including `PC01_rough_DM` and any other variant of that composition.
Keep `PC02`. This is an evaluation cohort decision, not a claim of corrupted
audio, and does not automatically exclude PC01 as enrollment for other songs.
Preserve historical results and label the new cohort explicitly.

The public implementation will document the exact metrics, silence handling,
source eligibility, and aggregation rules used for its reported scores.
