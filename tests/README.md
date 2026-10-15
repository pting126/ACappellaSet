# Tests

`test_release_tools.py` covers model discovery, config hashes, specialist output
contracts, corpus-specific residual preparation, movable output links, and
pair-weighted reporting/exclusions. `test_evaluate.py` scores synthetic WAVs
with analytically known SI-SDRi and exercises silence/PC01 handling.
`test_fixed_inference.py` covers overlap-add, audio conversion, and strict loading.

Run from the repository root in the inference environment:

```bash
python -m unittest discover -s tests -v
```

Actual-checkpoint comparisons and their limits are recorded in `docs/validation`.
Training smoke checks are separate from a clean-environment full training run.
