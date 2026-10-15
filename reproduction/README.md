# Frozen result artifacts

`plan.json` maps table rows to exact checkpoint selections and source artifacts.
`per_pair.csv.gz` contains the retained numeric scores and identities; no audio,
weight binaries, private server paths, or manuscript text are included.

Run `python reproduce.py --output outputs/paper-tables` from the repository root.
See [reproduction](../docs/reproduction.md) for scopes, historical inference
provenance, AB enrollment limits, and optional manuscript-cell comparison.
