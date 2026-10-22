# Conditioned models

The nine JSON configurations cover the seven natural-data K/L/M recipes and
two singer-augmentation models, retaining their audited architecture and inference
settings. Role vocabulary indices and supervised roles are distinct; Lead and
Bar are never mapped to each other. `redimnet2.yaml` is the byte-identical encoder
configuration. Inference strictly loads the matching fine-tuned encoder tensors
from each separation checkpoint. See the conditioned inference guide and
`third_party/conditioned_sources.json` for provenance.
