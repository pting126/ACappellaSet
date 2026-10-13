# Third-party dependencies

No upstream implementation or model weights are bundled in this initial
repository. The intended dependencies and audited source revisions are recorded
in [third_party/sources.json](third_party/sources.json).

| Project | Role | Upstream code license |
| --- | --- | --- |
| [Demucs](https://github.com/facebookresearch/demucs) | HTDemucs backbone and helper modules | [MIT](https://github.com/facebookresearch/demucs/blob/main/LICENSE) |
| [Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training) | Fixed-output and specialist training/inference implementation | [MIT](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/LICENSE) |
| [WeSpeaker](https://github.com/wenet-e2e/wespeaker) | Speaker embedding support for relevant conditioned models | [Apache-2.0](https://github.com/wenet-e2e/wespeaker/blob/master/LICENSE) |

Original license notices must accompany any code copied or adapted from these
projects. Dependency code licenses do not automatically establish permission
to redistribute pretrained weights or training data. Exact encoder revisions,
weight provenance, and applicable weight terms will be recorded before release.
