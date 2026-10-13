# Model implementations

This directory is reserved for the custom architectures and adapters in the
public release. Implementations have not yet been migrated here.

The planned families are fixed-output HTDemucs, part-specific HTDemucs, and
part-, singer-, and dual-conditioned HTDemucs. Custom conditioning belongs in
this repository; upstream Demucs is a dependency rather than a copied checkout.

Each implementation must be paired with its configuration, strict checkpoint
loader, and explicit input/output contract. See [model documentation](../docs/models.md).
