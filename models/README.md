# Model implementations

`demucs4ht.py` contains the unchanged audited MSS HTDemucs architecture.
`fixed.py` provides strict registered-checkpoint loading and the fixed-model
overlap-add implementation. The first supported configuration is Fixed_6stem_htdemucs_SKnJC.

Specialist and conditioned adapters are still pending. Upstream Demucs is a
dependency rather than a copied checkout; see the retained third-party notices.

Each implementation must be paired with its configuration, strict checkpoint
loader, and explicit input/output contract. See [model documentation](../docs/models.md).
