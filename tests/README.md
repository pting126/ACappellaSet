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

`test_data_protocol.py` checks exact scientific CSV preservation, composition split
integrity, PC01 target exclusion, and original specialist-filter membership.
`test_training.py` checks pair-weighted selection, tie handling, restricted new-run
exports, and manifest validation. Real-data CUDA training, native initialization
and sample parity, and exact deterministic resume are recorded in
`docs/validation/training-runtime.json`.

`test_reproduction.py` checks audited table values, checkpoint steps, shared and
inclusive scopes, and artifact tamper rejection. `test_benchmark.py` checks target
output routing, no Lead/Bar relabeling, and unavailable-query exclusion.
`python scripts/check_release.py` verifies hashes and local documentation links.

`test_release_inputs.py` exercises offline release conversion through the CLI,
relative links, independent copies, explicit absent-role silence, source identity
and split checks, corrupted/missing/non-finite audio, rollback, and downstream
specialist inputs. The full real-release check is recorded separately in
`docs/validation/release-inputs-2026-10-22.json`.
