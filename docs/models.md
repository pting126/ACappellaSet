# Models and checkpoints

The [fixed/specialist registry](../manifests/models.json) and [shared query-conditioned registry](../manifests/conditioned.json) cover **29 released inference checkpoints**: 3 fixed-output, 12 original part-specific, and 14 shared query-conditioned selections from nine recipes. All are uploaded to the currently private [Hugging Face model repository](https://huggingface.co/AcappellaSet/ACappellaSet-Models); access requires repository permission until publication. See the [checkpoint inventory](checkpoints.md) for selections.

## Architecture

| Family | Architecture and inputs | Outputs |
| --- | --- | --- |
| Fixed-output HTDemucs | [MSS HTDemucs](../models/demucs4ht.py); mixture audio | Six ordered vocal parts |
| Part-specific HTDemucs | Same MSS backbone with a two-output head; one mixture-only checkpoint per target part | Residual, target |
| Shared query-conditioned HTDemucs | [Query-conditioned HTDemucs](../models/conditioned_core.py); mixture plus part embedding, singer enrollment, or both | Target, residual |

Both backbones have waveform and spectrogram branches coupled by a cross-domain Transformer. Shared query-conditioned models add FiLM modulation to both branches before that Transformer. Singer and dual models include the matching fine-tuned ReDimNet2 enrollment encoder; part-only models use a learned role embedding. Full-remix and single-clone augmentation change the singer-conditioned training data, not its architecture.

Exact dimensions, source order, conditioning, and preprocessing are recorded in the registered [fixed](../configs/fixed), [specialist](../configs/specialists), and [conditioned](../configs/conditioned) configurations.

Soprano, Alto, Tenor, Bass, and Vocal Percussion are shared part labels. Lead
and Baritone are distinct roles. A combined model's sixth output may represent
a corpus-dependent `other` slot; it must not be advertised as two independent
Lead and Baritone estimates.

Historical MSS specialist checkpoints use residual output index 0 and target
index 1. The release loader must verify the contract instead of assuming all
two-output models follow it.

For singer-conditioned models, the matching encoder parameters are part of
the model. Replacing a fine-tuned encoder with its original pretrained weights
does not reproduce the checkpoint.

## Checkpoints and inference

The model card records training data, selection criterion, source order, preprocessing, and applicable weight terms. All released files are inference-only exports. Fixed/specialist exports are tensor state dictionaries; shared query-conditioned exports contain only `format`, `step`, and `model_state`, including the matching fine-tuned encoder tensors. Safe loading uses `weights_only=True`; registered hashes are checked first and model loading is strict. Original research identities remain recorded separately from export hashes.

Use `python inference.py --list-models` to see all 24 model recipes and their 29 selections. The download column reports registered artifacts; private repository access is separate from availability in the registry. Use the authenticated Hugging Face CLI while the model repository is private. After public release, `download.py --model MODEL --selection CRITERION` can fetch a registered artifact anonymously and verify its size and hash. See [fixed/specialist inference](inference.md) and [shared query-conditioned inference](conditioned-inference.md).

Seven natural-data conditioned models have portable training recipes. Full-remix and single-clone augmentation models additionally have released inference weights and adapters; a portable training recipe for their synthetic datasets is not included.

All project weights use CC BY-NC 4.0, with upstream notices preserved in the model repository. Code/documentation use MIT. Dataset access and dataset Terms are separate.

Public fixed-model names are `Fixed_6stem_htdemucs_SK`,
`Fixed_6stem_htdemucs_JC`, and `Fixed_6stem_htdemucs_SKnJC`.
`SKnJC` means joint SK+JC training. Epochs remain in provenance metadata,
not in the public names. The legacy `combined-ep333` CLI alias still works.
