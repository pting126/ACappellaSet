# Reproducing tables and evaluations

Two workflows are available: regenerate tables from the archived score artifacts,
or run a supplied checkpoint on a prepared test corpus. They are deliberately
identified separately in output receipts.

## Regenerate the archived tables

This command needs only Python's standard library; it does not download audio or
weights, retrain models, or select checkpoints using test scores:

```bash
python reproduce.py --output outputs/paper-tables
```

It produces full-precision CSVs and one-decimal LaTeX rows for cross-corpus,
target-conditioning, and development panels (a) and (b). The package contains
3,293 per-pair rows across 47 source artifacts with input/checkpoint hashes,
retained population counts, and inference provenance. Integrity, duplicate pairs,
SI-SDRi decomposition, common mixture baselines, and selector identities are
checked before tables are written. An existing output directory is rejected.

The snapshot includes the updated K1 198k and M3 shared 195k/198k selections.
Panel (a) uses shared validation selection and shared test scores. Panel (b)
uses pooled-inclusive selection and inclusive test scores. Shared pools all
retained S/A/T/B/VP recording–part pairs; it is not an unweighted mean of five
part scores or an equally weighted average of corpora. AB PC01 is excluded as
a target. AB singer results in panel (b) use the explicitly labeled historical
fallback-inclusive protocol; they do not establish independent enrollment for
all 25 pairs.

K6 and K6_no_aug score rows are included only to reproduce the augmentation
comparison. Their weights, training audio, and training recipes remain deferred.
SepACap is an external baseline; its JC train/test overlap cannot be excluded.

The cross-corpus artifacts retain their historical inference protocols, including
metric-only rescoring of archived fixed-model predictions. These artifact-derived
tables are **not certification that every cell in a manuscript matches**. The
older draft contains scope/protocol differences. Reconcile those explicitly
before claiming exact paper reproduction; no manuscript is modified here.
Optional `--draft-cells /path/to/cells.json` compares user-supplied cells (fields:
`model`, `dataset`, `role`, `draft_value`) and writes a separate discrepancy CSV.
The repository does not include private manuscript drafts.

## Run a frozen checkpoint on the historical test cohort

Prepare the corpus using [data preparation](preparation.md), install the
[inference environment](inference.md), and supply an exact registered checkpoint:

```bash
python prepare_data.py --set test_sk --root SK=/data/SK --output datasets/manifests/test_sk
python benchmark.py --model K5 --selection combined_shared \
  --checkpoint /weights/K5_step106000.pt --manifest datasets/manifests/test_sk \
  --corpus SK --device cuda --output outputs/K5-SK
```

`benchmark.py` accepts fixed, original specialist, part-, singer-, and dual-conditioned
models. It fixes the checkpoint before testing, applies the original full-song
inference adapter and canonical metric, and writes per-pair scores, shared and
inclusive summaries, exclusions, and provenance. It does not save audio estimates
or choose a checkpoint from the resulting test scores. A specialist reports its
own target only; bank overalls must pool the retained source rows across the bank,
not average specialist means.

Part and dual queries outside the training vocabulary are excluded by default.
For the explicitly reported dual zero-role diagnostic, add
`--allow-untrained-role`. There is no Lead/Bar relabeling. Singer queries need no
part token; the archived activity-ranked enrollment gallery supplies the query.
AB defaults to different-song enrollment only. Add
`--allow-enrollment-fallback` to reproduce the historical fallback-inclusive AB
protocol, including its same-recording cases. Always report which protocol was used.

The training validation engine passed a complete 112-pair JC/SK event, six selector
streams, and native inference/data parity checks. The benchmark wrapper's role
routing and exclusions have additional synthetic contract tests; it has not rerun
every archived table on every hardware backend. Small numerical differences across
hardware/runtime versions are possible. For saved predictions, use
[`evaluate.py`](evaluation.md), which verifies explicit mixture/reference/estimate
alignment and computes the same pair-weighted summaries.
