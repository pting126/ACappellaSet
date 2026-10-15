# Configurations

`fixed/` and `specialists/` contain the exact original MSS configurations.
`conditioned/` contains K/L/M architecture and inference settings plus the exact
ReDimNet2 encoder configuration. `training/` contains the seven natural-data
conditioned recipes, including initialization, loss, augmentation, optimizer,
seed, update budget, and validation cadence.

Configuration hashes are checked by the loaders and launchers. Output order is
part of the contract: original specialists use residual/target; conditioned
models use target/residual. Do not exchange configurations across families.

Use [MSS training](../docs/training.md) or
[conditioned training](../docs/conditioned-training.md), with the exact prepared
split manifests. Archived epochs/steps identify research checkpoints; a fresh
training run is not a guarantee of recreating their exact scores.
