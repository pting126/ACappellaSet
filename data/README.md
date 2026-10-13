# Dataset loading and preprocessing

Dataset loaders and preparation code will live here. Audio itself will be
distributed through Zenodo, not this directory.

Preparation must use the published layout and composition-level splits,
preserve source labels, and write derived artifacts to a user-selected output
directory. Missing real audio must fail explicitly. Any intentionally absent
part must be declared by the metadata rather than inferred from a missing file.

See [dataset documentation](../docs/dataset.md). The portable public data
schema and preparation implementation are still being prepared.
