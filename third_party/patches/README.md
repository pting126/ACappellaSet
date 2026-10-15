# MSS training patch

`mss-training.patch` applies to MSS commit
`c6a02267a2969b94dc2b261b6701a32ad9b7f118`. Use
`python scripts/prepare_mss.py --checkout /path/to/mss` from this repository.
The helper verifies source hashes and normalizes the six affected files to LF
before applying the patch. Direct `git apply` against CRLF files may fail.

`mss-training.json` records upstream, audited working-source, and final hashes.
The patch preserves upstream MIT attribution. It removes private path injection
and the later optional validation-portal integration, retaining the original
MSS fixed/specialist training path. It does not include the shared conditioned
model training recipes. See [training](../../docs/training.md) for validation
limits and required data.
