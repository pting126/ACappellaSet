# Conditioned models

The seven K/L/M JSON configurations retain the audited architecture and inference
settings. Role vocabulary indices and supervised roles are distinct; Lead and
Bar are never mapped to each other. `redimnet2.yaml` is the byte-identical encoder
configuration. Inference strictly loads the matching fine-tuned encoder tensors
from each separation checkpoint. See the conditioned inference guide and
`third_party/conditioned_sources.json` for provenance.
