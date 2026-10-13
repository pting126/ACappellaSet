# Training

Training implementations and portable configurations are being prepared.
There is no runnable `train.py` in this initial repository.

The release will document:

- The dataset version, composition-level splits, and required preprocessing.
- Model architecture, role/output definitions, and initialization weights.
- Objectives, augmentation, optimization, random seeds, and validation rules.
- Enrollment sampling and encoder initialization for conditioned models.
- Hardware/software requirements, checkpoint saving, and resume behavior.

The required MSS project modifications must be supplied with an exact upstream
base through a public fork or versioned patch. Installing unmodified MSS alone
does not reproduce the project training recipes.

Historical environments differ between model families. The candidate Demucs
revision in `third_party/sources.json` is not yet a verified universal runtime
pin. The final environment will be tested with real checkpoints and training
steps before runnable training commands are advertised.
