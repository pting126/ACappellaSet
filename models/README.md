# Model implementation

`fixed.py` loads registered fixed-output and original part-specific HTDemucs
checkpoints and uses native MSS overlap-add. `registry.py` provides discovery
without importing Torch. `demucs4ht.py` retains its original upstream source.

Shared conditioned-model implementations remain in preparation. Inference
support and published-weight availability are separate; use `--list-models`.
