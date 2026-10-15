# Historical dataset splits

`index.json` identifies ten exact historical CSV sets and their checksums.
CSV files are compressed with gzip; row order, identities, and scientific
fields are preserved. Only filesystem prefixes are portable. `track_inventory.json`
records the original six-stem memberships and SK Soprano/Baritone real-target filters.

Use `prepare_data.py` to resolve local audio roots and verify the audio.
See [preparation](../../docs/preparation.md) for layouts and enrollment policies.
The final Zenodo version and archive mapping remain pending the dataset release.
Do not regenerate random splits or infer enrollment eligibility from filenames.
