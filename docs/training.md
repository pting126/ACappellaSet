# Training

`train.py` launches the original MSS fixed-output or mixture-only part-specific
recipe using a dedicated, patched MSS checkout. It does not train the shared
part-, singer-, or dual-conditioned models; those recipes are still being packaged.

## Prepare the training implementation

Use a separate environment and checkout for training. The inference environment
is not a complete training environment. Start with Python 3.10 and the pinned
MSS checkout's requirements; a clean training environment lock is still pending.
The short training smoke test uses the existing research environment, not a new
installation or a full training reproduction.

```bash
git clone https://github.com/ZFTurbo/Music-Source-Separation-Training.git /path/to/mss
git -C /path/to/mss checkout c6a02267a2969b94dc2b261b6701a32ad9b7f118
python scripts/prepare_mss.py --checkout /path/to/mss --check-only
python scripts/prepare_mss.py --checkout /path/to/mss
python -m pip install -r /path/to/mss/requirements.txt
```

The installer requires a clean checkout, verifies the upstream commit and six
source hashes, applies the patch in a temporary directory, and verifies its
outputs before updating the checkout. The [patch manifest](../third_party/patches/mss-training.json)
records its exact provenance. It includes source-aware specialist initialization,
checkpoint handling, and the project's loss options. Private server paths and
the later optional validation-portal bridge are removed. A Demucs fork is not
required for this path.

## Inputs and initialization

Provide pre-split 44.1 kHz stereo data with `mixture.wav` and the source filenames
listed by `python inference.py --list-models`, one directory per track. Preserve
composition-level splits: every recording variant of a composition belongs to
the same split. This launcher does not create or certify dataset splits.

For specialists, see [data preparation](preparation.md). Historical SK Soprano
and Baritone training used additional real-target filtering. The general
preparation script does not reproduce those filtered inventories; use the
verified role-specific inventories when reproducing those models. Their public
split/eligibility manifests are still pending.

Use the original Demucs `htdemucs_6s` initializer, not a trained ACappellaSet
checkpoint. Both the official file and the tensor-equivalent research copy are
identified in [initialization.json](../third_party/initialization.json). Obtain
it from the official URL there; the launcher verifies the file hash before MSS
loads it. It does not download weights automatically.

Historical specialist initialization copies positional PT6S output blocks into
the target head and sums the remaining blocks into the residual head. The old
labels supplied to that initializer describe the project's positional mapping;
they do not mean the original Demucs model was trained on vocal-part labels.

## Launch a new run

First inspect the verified command:

```bash
python train.py --model Fixed_6stem_htdemucs_JC \
  --mss-root /path/to/mss --initialization /path/to/pt_6s.th \
  --train-data /path/to/jc/train --valid-data /path/to/jc/valid \
  --output runs/jc-fixed --dry-run
```

Remove `--dry-run` to execute. To train an original specialist, choose its
registered ID and its two-stem data, for example `specialists-jc-alto-ep223`.
The epoch in that historical ID identifies the reference checkpoint, not the
stopping epoch of this new run.

The wrapper uses the original masked-loss recipe, seed 0, native configuration,
and SI-SDR validation scheduler. Specialist head initialization is `pt6s`.
Logging is offline. Existing output directories are rejected; `launch.json`
records the command and code/config/initializer hashes. Resume, DDP, and
multi-corpus checkpoint selection are not exposed by this wrapper.

The included configurations are byte-identical research configs, including
training duration and augmentation settings. A new run is not guaranteed to
recover a historical checkpoint: exact splits, sample order, hardware, software,
and continuation history also matter. Native MSS validation is distinct from
the canonical paper scorer in [evaluation.md](evaluation.md).

## Validation record

The [training smoke report](validation/mss-training-smoke.json) records exact-base
patch application, a training-launcher dry run, verified specialist head mapping,
and one finite optimizer update for each of the fixed and specialist families
on two short synthetic examples. This does not validate a fresh training
environment, complete dataset loading, resume, or convergence to paper scores.
