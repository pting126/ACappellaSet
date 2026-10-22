# Checkpoints proposed for release

This inventory covers the updated release plan: **29 project checkpoints (27 active, 2 deferred)**, plus one external baseline to link to. It is a proposed release set, not a claim that weights are already downloadable or that every final paper selection is frozen. The [Hugging Face repository](https://huggingface.co/AcappellaSet/ACappellaSet-Models)
is currently private: SKnJC portable weights are uploaded there, but public
checkpoint downloads remain pending. On 2026-10-14, the
three historical fixed models and 12 historical specialists were confirmed
for release inclusion. The ICASSP release excludes newer fixed-six training
runs and later singer-enrollment specialists. The conditioned candidates below
cover the paper experiments with updated K1 and M3 evaluations. Public loader
validation has passed for the 12 active conditioned selections; their weight downloads remain pending.

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

## Conditioned models (12 active, 2 deferred)

| Model | Description | Selected steps |
| --- | --- | --- |
| K1 | SK part-conditioned | 4,000, 198,000 |
| K5 | SK singer-conditioned | 12,000, 106,000, 179,000 |
| K7 | SK dual-conditioned | 7,000 |
| L1 | JC part-conditioned | 119,000, 141,000 |
| L3 | JC singer-conditioned | 184,000 |
| L4 | JC dual-conditioned | 188,000 |
| M3 | JC+SK singer-conditioned | 195,000, 198,000 |
| K6 | SK full-remix singer-conditioned — deferred | 131,000 |
| K6_no_aug | SK single-clone singer-conditioned — deferred | 48,000 |

K1 uses 198k for SK-shared selection in the cross-corpus table; its 4k
JC-inclusive selection in the target-conditioning table is unchanged. M3 uses
195k for JC/pooled-shared selection and 198k for SK-shared selection in
Table 4(a). These selections were checked through 200,000 updates and have
matching updated test evaluations. The registry retains superseded identities;
older scores must not be attached to the replacement checkpoints.

Table 4(a) uses shared selection and shared evaluation. Table 4(b) uses
pooled-inclusive selection and inclusive evaluation. Shared scores pool all
retained S/A/T/B/VP pairs; they are pair-weighted, not part-macro averages.
Inclusive additionally retains JC Lead or SK Bar. Registry labels beginning
with `combined_` mean pooled JC+SK validation, not combined training.

`K6_no_aug` is a historical identifier for **single-clone** training; ordinary
waveform augmentation remains enabled. Selection corpus, scope, candidate
cutoff, and selected step are separate metadata fields.

Singer-conditioned and dual models require their matching encoder state. The public export must retain fine-tuned encoder tensors or supply a verified companion artifact. Raw training checkpoints may include optimizer state and private configuration paths; prepare a portable model-only export and validate it before uploading. Exported files receive new hashes.

## Additional dependencies and optional releases

- **SepACap JC:** link to the authors’ checkpoint and code; it is not one of the 29 project weights to redistribute. Permission to redistribute was not established.
- **Training initialization:** document and link the exact HTDemucs initialization (including the PT6S checkpoint where used) and speaker-encoder initialization. These are additional dependencies, not substitutes for the fine-tuned release weights. Their final public artifact references still need to be pinned.
- **Historical fixed baselines:** JC epoch 199 and SK epoch 178 are distinct from JC321/SK279. Add them if their corresponding comparisons remain in the public scope; do not silently substitute one pair for the other.
- Newer fixed-six and singer-enrollment specialist experiments are excluded from this ICASSP release. Frozen-encoder and additional K/L/M variants remain outside the paper-focused inventory unless needed by a reported experiment. This inventory does not cover every research experiment.

## Distribution

Host project weights on [Hugging Face](https://huggingface.co/AcappellaSet/ACappellaSet-Models). An optional final paper model archive may be deposited on Zenodo. Keep configurations, model code, registry entries, and download utilities in GitHub. Before publishing each weight, freeze its intended use, verify loading and inference, and record applicable weight/data terms. Weight terms and upstream notices are maintained separately in the model
repository; this GitHub repository does not grant model-weight rights.
K6 and K6_no_aug remain deferred. Code licensing, technical validation,
upstream weight rights, and training-data rights are separate.
