# Checkpoints proposed for release

This inventory covers the audited paper snapshot: **29 project checkpoints**, plus one external baseline to link to. It is a proposed release set, not a claim that weights are already downloadable or that every final paper selection is frozen. The [Hugging Face repository](https://huggingface.co/pting126/ACappellaSet-models)
is created, but weights and their terms remain pending. On 2026-10-14, the
three historical fixed models and 12 historical specialists were confirmed
for release inclusion. Newer-model release selections remain under review.

Exact filenames, hashes, and recorded selection labels are in [`manifests/checkpoint_candidates.json`](../manifests/checkpoint_candidates.json). Repeated table uses of the same checkpoint are deduplicated by SHA-256. Epochs below are the stored checkpoint epoch numbers.

## Fixed-output models (3)

| Training corpus | Epoch | Status |
| --- | ---: | --- |
| JC_ep321 | 321 | Confirmed for release inclusion |
| SK_ep279 | 279 | Confirmed for release inclusion |
| Combined_ep333 | 333 | Confirmed for release inclusion |

## Specialists (12)

Each bank contains six distinct checkpoints. The source order is residual index 0, target index 1.

| Target | JC epoch | SK epoch |
| --- | ---: | ---: |
| S | 162 | 220 |
| A | 223 | 237 |
| T | 119 | 214 |
| B | 110 | 182 |
| VP | 45 | 205 |
| Lead | 149 | — |
| Bar | — | 43 |

## Conditioned and augmentation models (14)

| Model | Description | Selected steps |
| --- | --- | --- |
| K1 | SK part-conditioned | 4,000, 184,000 |
| K5 | SK singer-conditioned | 12,000, 106,000, 179,000 |
| K7 | SK dual-conditioned | 7,000 |
| L1 | JC part-conditioned | 119,000, 141,000 |
| L3 | JC singer-conditioned | 184,000 |
| L4 | JC dual-conditioned | 188,000 |
| M3 | JC+SK singer-conditioned | 159,000, 173,000 |
| K6 | SK full-remix singer-conditioned | 131,000 |
| K6_no_aug | SK single-clone singer-conditioned | 48,000 |

K1 at 184k is an earlier snapshot selection: the completed-200k SK shared/inclusive
winner is 198k. The listed K1 4k JC-inclusive selection remained unchanged.
This inventory retains the original snapshot selections.

Different steps correspond to different validation scopes; they are not interchangeable “best” checkpoints. The registry preserves the recorded labels. M3 is the audited snapshot through 179,000 updates, not a final 200,000-update selection. `K6_no_aug` is a historical identifier for **single-clone** training; ordinary waveform augmentation remains enabled.

Singer-conditioned and dual models require their matching encoder state. The public export must retain fine-tuned encoder tensors or supply a verified companion artifact. Raw training checkpoints may include optimizer state and private configuration paths; prepare a portable model-only export and validate it before uploading. Exported files receive new hashes.

## Additional dependencies and optional releases

- **SepACap JC:** link to the authors’ checkpoint and code; it is not one of the 29 project weights to redistribute. Permission to redistribute was not established.
- **Training initialization:** document and link the exact HTDemucs initialization (including the PT6S checkpoint where used) and speaker-encoder initialization. These are additional dependencies, not substitutes for the fine-tuned release weights. Their final public artifact references still need to be pinned.
- **Historical fixed baselines:** JC epoch 199 and SK epoch 178 are distinct from JC321/SK279. Add them if their corresponding comparisons remain in the public scope; do not silently substitute one pair for the other.
- Frozen-encoder, additional K/L/M variants, later fixed-model selections, and matched specialist studies are optional extensions unless retained in the final paper. This snapshot inventory does not claim to cover every experiment.

## Distribution

Host project weights on [Hugging Face](https://huggingface.co/pting126/ACappellaSet-models). An optional final paper model archive may be deposited on Zenodo. Keep configurations, model code, registry entries, and download utilities in GitHub. Before publishing each weight, freeze its intended use, verify loading and inference, and record applicable weight/data terms. No checkpoint has been uploaded by this code-packaging change.
