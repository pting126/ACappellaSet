# Model implementations

`fixed.py` supports the original three fixed-output models and twelve mixture-only
specialists. `conditioned.py` and `conditioned_core.py` support nine shared
part-, singer-, and dual-conditioned inference recipes, including the two
singer-augmentation models. `conditioned_encoder.py` restores
the complete matching singer encoder from the separation checkpoint.

`registry.py` defines public names, aliases, checkpoint selections, hashes and
output contracts. `query.py` validates part/enrollment inputs; untrained role
queries require explicit opt-in. `trained.py` resolves restricted tensor exports
from new public training runs without presenting them as paper checkpoints.

Required Demucs and WeSpeaker modules are vendored with retained licenses and
source provenance. Inference does not require a project Demucs fork or MSS
checkout. Original fixed/specialist training uses the separately pinned and
patched MSS checkout. See [models](../docs/models.md) and
[conditioned inference](../docs/conditioned-inference.md).
