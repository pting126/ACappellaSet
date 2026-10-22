# Historical dataset splits

`index.json` identifies ten exact historical CSV sets and their checksums.
CSV files are compressed with gzip; row order, identities, and scientific
fields are preserved. Only filesystem prefixes are portable. `track_inventory.json`
records the original six-stem memberships and SK Soprano/Baritone real-target filters.

Use `prepare_data.py` to resolve local audio roots and verify the audio.
See [preparation](../../docs/preparation.md) for layouts and enrollment policies.
The official [ACappellaSet v1.0 dataset](https://huggingface.co/datasets/AcappellaSet/ACappellaSet)
is published, with one `song_id,split` assignment per song in `metadata/splits.csv`.
These query/enrollment manifests preserve the historical experiment inputs;
`prepare_data.py --release-root` maps the locally extracted release to their
expected SK/AB paths without rewriting their scientific fields. See the
[dataset registry](../dataset.json) for the release revision and checksums.
Do not regenerate random splits or infer enrollment eligibility from filenames.
