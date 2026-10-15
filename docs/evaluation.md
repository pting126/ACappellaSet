# Evaluation

`evaluate.py` scores already generated full-song estimates against explicitly
paired references and mixtures. It does not run inference or select checkpoints.
The SI-SDR functions are unchanged extracts from the canonical research snapshot;
[function hashes](../third_party/metric_provenance.json) record their provenance.

```bash
python evaluate.py --manifest /path/to/pairs.csv --output outputs/scores
```

The CSV requires these columns:

```csv
model,corpus,protocol,song_id,track_id,role,mixture,reference,estimate
my-model,JC,mixture_only,example,example,S,audio/mixture.wav,audio/soprano.wav,predictions/soprano.wav
```

Paths are relative to the CSV directory or absolute. This example describes the
schema; no example corpus audio is bundled. `song_id` is the authoritative
composition ID; `track_id` identifies a recording/mix variant. Roles are
`S/A/T/B/VP`, plus `Lead` for JC or `Bar` for SK and AB. Match outputs to their
actual roles: Lead and Bar are not interchangeable. A combined fixed model's
`other` slot can supply the appropriate corpus-specific sixth role; it does
not provide two independent outputs.

`protocol` must be `mixture_only`, `part_query`, `strict_enrollment`, or
`fallback_inclusive`. Reports are grouped by model, corpus, and protocol, so
strict and fallback-inclusive results are never averaged together. The caller
must establish the enrollment protocol and source correspondence; the evaluator
cannot infer them from output WAVs.

All three WAVs must be finite stereo 44.1 kHz with exactly equal lengths.
There is no resampling, gain adjustment, permutation search, truncation, or
alignment correction. Scores use float32 CPU tensors and two Torch threads:
subtract each channel's temporal mean, compute channel-wise SI-SDR with the
canonical epsilon, then average channel dB values. SI-SDRi subtracts the mixture
baseline for that same reference. Complete references below -45 dBFS are
excluded before reporting.

`metrics_per_item.csv` records full-precision scores and all three audio hashes.
`metrics_overall.json` records exclusions, runtime, and both scopes:

- **Shared:** all retained S/A/T/B/VP track–role pairs.
- **Inclusive:** retained shared pairs plus the corpus-specific Lead or Bar.

Each pair has weight one. Overall is not the average of per-role means, not a
duration-weighted mean, and not a mean over inference windows. Counts and roles
present are reported; incomplete input CSVs are not certified full benchmarks.
Duplicate pairs, conflicting composition IDs, inconsistent comparison audio,
and non-finite scores fail explicitly. The output directory must be new.

For AB, targets whose authoritative `song_id` is `PC01` are excluded before
audio loading, including arbitrarily named recording variants. Obvious
`PC01_*` tracks assigned another composition ID are rejected. `PC02` remains.
This is an evaluation-cohort decision; original data, historical results, and
enrollment eligibility are unchanged. The evaluator does not rewrite paper tables.
