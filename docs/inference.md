# Inference

The inference implementation and downloadable checkpoints are being prepared.
There is no runnable `inference.py` in this initial repository.

The supported workflow will be:

1. Install the tested runtime for the selected model family.
2. Download the model and its configuration using the registry, verifying hashes.
3. Supply a mixture and, for singer-conditioned models, enrollment audio.
4. Run separation and save outputs using the checkpoint's recorded stem names.

The release instructions will specify audio preprocessing, supported devices,
segment length, overlap, precision, output encoding, and enrollment processing.
Different model families must retain their verified inference behavior.

A small, permitted example and its exact command will be published after a
clean-environment test with the actual released weights. The tooling-only
`environment.yml` is not an inference environment.
