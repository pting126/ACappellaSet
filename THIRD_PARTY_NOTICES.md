# Third-party dependencies

`models/demucs4ht.py` is vendored unchanged from the audited MSS checkout.
The HTDemucs inference branch in `models/fixed.py` is adapted from MSS.
MSS and Demucs license texts are retained in `third_party/licenses/`.
Exact source hashes and modifications are recorded in
[fixed inference provenance](third_party/fixed_inference_provenance.json).
No model weights are bundled. Other dependency references are recorded in
[third_party/sources.json](third_party/sources.json).

| Project | Role | Upstream code license |
| --- | --- | --- |
| [Demucs](https://github.com/facebookresearch/demucs) | HTDemucs backbone and helper modules | [MIT](https://github.com/facebookresearch/demucs/blob/main/LICENSE) |
| [Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training) | Fixed-output and specialist training/inference implementation | [MIT](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/LICENSE) |
| [WeSpeaker](https://github.com/wenet-e2e/wespeaker) | Speaker embedding support for relevant conditioned models | [Apache-2.0](https://github.com/wenet-e2e/wespeaker/blob/master/LICENSE) |

Original license notices must accompany any code copied or adapted from these
projects. Dependency code licenses do not automatically establish permission
to redistribute pretrained weights or training data. Exact encoder revisions,
weight provenance, and applicable weight terms will be recorded before release.

The reviewed MSS training patch in `third_party/patches/mss-training.patch`
also derives from the MIT-licensed MSS project at the pinned base commit.
Its manifest records upstream and modified source hashes; the patch retains
original author notices. The canonical metric and preparation provenance
manifests identify extracts/adaptations from this project's research scripts.

# Conditioned model dependencies

Conditioned inference includes the required Demucs modules at commit
`e976d93ecc3865e5757426930257e200846a520a` under MIT, and the ReDimNet2,
pooling, and tfmel modules from WeSpeaker commit
`dfa741957e5c11f477623b6e583d67d0af25ee88` under Apache-2.0. Their copyright
headers and licenses are retained under `models/vendor/`. WeSpeaker imports
were changed to package-relative imports; the computations are unchanged.
The conditioned forward pass adapts Demucs internals and retains that upstream
attribution. Exact sources and transformations are recorded in
`third_party/conditioned_sources.json`. These code licenses do not assign terms
to separately distributed model weights or audio.
