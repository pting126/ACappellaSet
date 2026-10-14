# Configurations

Release configurations are grouped by model family. The first inference
configuration is `fixed/combined_ep333.yaml`. Each configuration must match the
weights, architecture, source order, preprocessing, and inference protocol.

The archived combined YAML retains its original training fields for provenance;
this does not provide a runnable public training workflow yet.
Public training configurations must record the initialization, data/split version,
objective, augmentation, seed, validation criterion, and required encoder.
Use repository-relative paths or explicit user-supplied paths.
