# Hard Drive Analysis v4

Can the metrics `smartctl` reports (SMART attributes) predict how much life a hard drive has
left? A survival model trained on Backblaze's public drive-stats panel, compared with the
same model without SMART.

The pipeline runner (`data_pipeline`) and the pre-training data gate (`data_verify`) live in
`src/`. Nothing outside this repository is needed.

## Quick start

```bash
uv sync                          # Python 3.13
uv run pytest                    # all tests: project, runner, gate
uv run ruff check src steps tests
uv run data-pipeline -n          # what would run, and why
uv run data-pipeline             # run everything that is out of date
```

The panel goes in `dataset/dataset.parquet` (21.8 GiB, not in git, checked by sha256 in
`pipeline.toml`). If it is missing, the first two steps build it from Backblaze's CSVs; see
[Raw data](#raw-data). Results land in `build/report/results.md` and
`build/report/analysis.md`. Long steps log progress to `build/logs/<step>.log`.

## Scales

Set `scale` in `config/sample.toml`. Changing it reruns only what depends on it. The first
run also scans the whole panel once (`drive_index`, about 6 minutes), cached for every scale.

| Scale | Drives kept (failing / healthy) | Time after `drive_index` | Memory |
|---|---|---|---|
| small | 4% / 0.4% | ~7 min | any machine |
| medium | 20% / 2% | ~22 min | 3.7 GiB training estimate |
| full | 100% / 10% | ~66 min | 11.0 GiB peak |

Sampled drives and snapshots carry inverse-probability weights, so every scale estimates the
same numbers; smaller scales give wider intervals.

`HDA_MEMORY_FRACTION` (default 0.6) and `HDA_THREADS` tune DuckDB and LightGBM resources
without invalidating checkpoints.

## Pipeline

```
panel_download ─► panel_convert ─► dataset.parquet ─► drive_index ─► cohort ─► extract ─► snapshots ─► verify ─► explore
                                                                                                         │
                                                                                                         └─► train_* (7 models + seeds) ─► predict ─► evaluate ─► report
```

| Step | Output | What it does |
|---|---|---|
| `panel_download` | `/srv/downloads/backblaze-drive-stats/zips/` | Once only: Backblaze's 40 quarterly zips, 2016 Q1 to 2025 Q4 (28 GiB) |
| `panel_convert` | `dataset/dataset.parquet` | Once only: their 3,653 daily CSVs to one parquet file, typed by `config/panel-schema.toml` |
| `drive_index` | `build/drives.parquet` | Checks the dataset's sha256, then writes one row per drive |
| `cohort` | `build/cohort.parquet` | HDDs only. Splits by drive (70/10/20) and draws the weighted sample |
| `extract` | `build/panel/` | The cohort's daily rows, deduplicated and cut at the first failure |
| `snapshots` | `build/snapshots.parquet` | Sampled snapshots with trend features and right-censored labels |
| `verify` | `build/verify-report.json` | The gate: recomputes every label and checks splits, weights, schema and leaks |
| `explore` | `build/explore/` | Failure rate per model, single-attribute signal, correlations |
| `train_*` | `build/models/` | LightGBM models. `hazard_smart` is the model under test; the others are ablations and baselines |
| `predict` | `build/predictions.parquet` | Survival curves, failure probability per horizon, median remaining life |
| `evaluate` | `build/metrics.json`, `*.csv` | Weighted, censoring-aware metrics with bootstrap intervals |
| `report` | `build/report/` | `results.md` with the headline numbers, `analysis.md` with the explanation |

The runner runs each step in a fresh subprocess and skips it when its code, params and
inputs are unchanged. `uv run data-verify checks` lists every gate check.

## Raw data

`panel_download` and `panel_convert` rebuild `dataset/dataset.parquet` from the source:
Backblaze's [drive-stats CSVs](https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data),
one file per day (3,653 of them), about 250 GB for 2016-2025. They exist so the panel can be traced back to
and rebuilt from the public data. They are meant to run once. These choices are deliberate:

- **Kept, not rerun.** Both are `once` steps. While the file in `[pipeline].fixed` exists
  with its pinned sha256, the runner reports them as `kept` and never runs them: not on a
  code or param change, not with `-f`, `-r` or `--only`. A full rebuild takes hours and
  reproduces data that does not change, so a fresh run of the analysis must not trigger it.
  Only a missing panel is rebuilt. A panel with the wrong sha256 stops any run that
  includes them instead: it may be a new panel waiting for its pin, and overwriting it
  would cost hours. (A run that leaves them out, such as `--only report`, is not checked
  here; `drive_index` checks the same pin before it reads the panel.)
- **Everything big on the HDD.** The 250 GB of CSVs do not fit on the SSD, so downloads and
  scratch live under `/srv/downloads/backblaze-drive-stats/` on the download HDD: the zips
  (kept), each quarter's CSVs while it converts (up to ~12 GB), the staged per-quarter
  parquet files, and DuckDB's spill folder. DuckDB asks for a very large spill cache on
  this job; on the SSD it would fill the disk, so its `temp_directory` is on the HDD too.
  Only the finished panel (about 23 GB) is written to `dataset/`, first as a hidden
  `.dataset.parquet.part` that is renamed into place when complete. The absolute paths in
  `pipeline.toml` are machine-specific on purpose: this project runs on one machine.
- **Slow is fine.** The HDD makes the job IO-bound: hours for the download and the
  conversion. That is the expected cost of a step that runs once.
- **Faithful copy.** CSV values are copied as they are, quirks included (an empty
  2017-01-30, a CR-only 2019-06-17 with `M/D/YY` dates, a 2022-12-10 file holding
  2022-12-11's rows). Later steps already handle them. Column types are pinned in
  `config/panel-schema.toml` to the ones of the panel all results so far came from, so the
  rebuild is a drop-in replacement and no type depends on DuckDB's guessing. A CSV column
  missing from that file, or a value that does not fit its type, stops the conversion.
- **Same rows, not same bytes.** The same zips give the same rows in the same order, but
  DuckDB's parallel writers cut row groups differently on every run, so the file's sha256
  changes. After any rebuild, pin the new sha256. Making the bytes repeat would mean
  single-threaded CSV parsing, hours more for no change in the data.
- **Resumable.** Complete zips are not downloaded again, and converted quarters are reused,
  so an interrupted run picks up where it stopped.
- **Complete.** Every day of each quarter must have its CSV (a header-only day counts), so
  a zip missing a day stops the conversion instead of shrinking the panel.

Unlike the v1 panel, rows are in file order: quarter by quarter, then day by day, each
file's rows as written (v1 had every Q1 first, then every Q2, ...). Also,
`filename` no longer nests Q1 2022 one folder deeper. Neither changes a result.

To rebuild: move the panel away, run the two steps, then pin the sha256 that
`panel_convert` prints in both places in `pipeline.toml` (`[pipeline].fixed` and
`drive_index`):

```bash
mv dataset/dataset.parquet dataset/dataset-v1.parquet
uv run data-pipeline --only panel_download panel_convert
```

## Layout

```
pipeline.toml        steps and models
config/study.toml    the shared contract: split, label bins, features, evaluation
config/sample.toml   scales
config/panel-schema.toml  column order and types of dataset.parquet
src/data_pipeline/   the runner (config, checkpoint, orchestrator, worker, CLI)
src/data_verify/     the data gate (checks, spec, CLI)
steps/               the code of each step, plus shared helpers
tests/               synthetic panel, unit tests, end-to-end run, gate tamper tests
tests/runner/        runner tests
tests/gate/          gate tests
```

## Rules

- **Never load the panel into pandas.** Whole-panel work is DuckDB, in month chunks.
- **Steps talk only through declared files.**
- **Outputs are deterministic** (sorted, seeded), so unchanged results keep later steps cached.
- **One contract.** A rule that two steps share goes in `config/study.toml`.
- **Data reaches a model only through the gate** (`depends_on = ["verify"]`).

## Licence

MIT, see [LICENSE](LICENSE).
