# Fixed-output models

`combined_ep333.yaml` is byte-identical to the archived combined model config.
The inference adapter overrides batch size to 4, matching the research backend.
JC and SK configurations are pending migration. The sixth
output has corpus-specific semantics; do not rename Lead, Baritone, or a mixed
`other` output as though they were interchangeable.
