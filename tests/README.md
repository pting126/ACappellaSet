# Inference validation

Run `python -m unittest discover -s tests -v` in the documented environment.
The five tests cover short/overlapping/partial-tail chunk reconstruction,
invalid input, non-finite predictions, checkpoint hash rejection before
deserialization, and mono/resampling/multichannel handling.

The full Combined ep333 checkpoint was separately compared with the archived
MSS inference implementation on two inputs in a fresh Linux CPU environment.
All six outputs matched bit-for-bit. The
[validation report](../docs/validation/combined-ep333-linux-cpu.json) includes
settings, package versions, output hashes, and tested source hashes.

CUDA, other checkpoints, training, and resume are not covered by these results.
