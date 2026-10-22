# Checkpoints proposed for release

This inventory covers **29 released project checkpoints**, plus one external baseline linked separately. All project weights are uploaded to the currently private [Hugging Face repository](https://huggingface.co/AcappellaSet/ACappellaSet-Models) as inference-only exports under CC BY-NC 4.0. Downloading requires repository access until publication. All fixed, original part-specific, and shared query-conditioned selections have public inference adapters, including full-remix and single-clone singer augmentation.

Exact filenames, hashes, and recorded selection labels are in [`manifests/checkpoint_candidates.json`](../manifests/checkpoint_candidates.json). Repeated table uses of the same checkpoint are deduplicated by SHA-256. Epochs below are the stored checkpoint epoch numbers.

## Fixed-output models (3)

| Training corpus | Epoch | Status |
| --- | ---: | --- |
| Fixed_6stem_htdemucs_JC | 321 | Confirmed for release inclusion |
| Fixed_6stem_htdemucs_SK | 279 | Confirmed for release inclusion |
| Fixed_6stem_htdemucs_SKnJC | 333 | Confirmed for release inclusion |

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

## Shared query-conditioned models (14 selections)

| Model | Description | Selected steps |
| --- | --- | --- |
| Vocal-part-conditioned HTDemucs / SK | SK part-conditioned | 4,000, 198,000 |
| Singer-conditioned HTDemucs / SK | SK singer-conditioned | 12,000, 106,000, 179,000 |
| Dual-conditioned HTDemucs / SK | SK dual-conditioned | 7,000 |
| Vocal-part-conditioned HTDemucs / JC | JC part-conditioned | 119,000, 141,000 |
| Singer-conditioned HTDemucs / JC | JC singer-conditioned | 184,000 |
| Dual-conditioned HTDemucs / JC | JC dual-conditioned | 188,000 |
| Singer-conditioned HTDemucs / JC + SK | JC+SK singer-conditioned | 195,000, 198,000 |
| Singer-conditioned HTDemucs / SK full-remix | SK full-remix singer-conditioned | 131,000 |
| Singer-conditioned HTDemucs / SK single-clone | SK single-clone singer-conditioned | 48,000 |

The SK part-conditioned model uses 198k for SK-shared selection in the cross-corpus table; its 4k
JC-inclusive selection in the target-conditioning table is unchanged. The jointly trained singer-conditioned model uses
195k for JC/pooled-shared selection and 198k for SK-shared selection in
Table 4(a). These selections were checked through 200,000 updates and have
matching updated test evaluations. The registry retains superseded identities;
older scores must not be attached to the replacement checkpoints.

Table 4(a) uses shared selection and shared evaluation. Table 4(b) uses
pooled-inclusive selection and inclusive evaluation. Shared scores pool all
retained S/A/T/B/VP pairs; they are pair-weighted, not part-macro averages.
Inclusive additionally retains JC Lead or SK Bar. Registry labels beginning
with `combined_` mean pooled JC+SK validation, not combined training.

Single-clone refers to the singer-augmentation dataset; ordinary waveform augmentation remains enabled. Selection corpus, scope, candidate
cutoff, and selected step are separate metadata fields.

Singer-conditioned and dual models include their matching fine-tuned encoder state. Released exports retain every separator and encoder tensor and omit optimizer state, training arguments, and internal paths. The export hash differs from the original research-checkpoint hash; both are recorded. Registered loaders accept the safe export and the exact original research identity.

## Additional dependencies and optional releases

- **SepACap JC:** link to the authors’ checkpoint and code; it is not one of the 29 project weights to redistribute. Permission to redistribute was not established.
- **Training initialization:** document and link the exact HTDemucs initialization (including the PT6S checkpoint where used) and speaker-encoder initialization. These are additional dependencies, not substitutes for the fine-tuned release weights. Pinned identities and artifact references are recorded in `third_party/initialization.json` and `third_party/encoder_initialization.json`.
- **Historical fixed baselines:** JC epoch 199 and SK epoch 178 are distinct from JC321/SK279. Add them if their corresponding comparisons remain in the public scope; do not silently substitute one pair for the other.
- Newer fixed-six and singer-enrollment specialist experiments are excluded from this ICASSP release. Frozen-encoder and additional K/L/M variants remain outside the paper-focused inventory unless needed by a reported experiment. This inventory does not cover every research experiment.

## Distribution

Host project weights on [Hugging Face](https://huggingface.co/AcappellaSet/ACappellaSet-Models). An optional final paper model archive may be deposited on Zenodo. Keep configurations, model code, registry entries, and download utilities in GitHub. Original research-checkpoint identities, inference-export hashes, selection criteria, and configuration hashes are retained separately. All 29 weights use CC BY-NC 4.0 as selected by the project owner; upstream notices remain in the model repository. This GitHub repository's MIT code license does not relicense model weights or dataset audio.
