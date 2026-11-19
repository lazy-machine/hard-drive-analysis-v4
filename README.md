# Hard Drive Analysis v4

Can the health counters a hard drive reports about itself (SMART attributes, what `smartctl`
prints) predict how much life the drive has left?

This project trains a survival model on Backblaze's public
[drive-stats data](https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data):
one row per drive per day, 2016 to 2025, 647 million rows. It then compares that model with
the same model trained without SMART.

## What it found

Full-scale run of 2026-11-18, on test drives from 2024-2025. ROC-AUC measures how well a
score ranks drives that fail within the horizon above those that don't: 0.5 is random, 1 is
perfect.

| Fails within | With SMART | Without SMART |
|---|---|---|
| 7 days | 0.905 | 0.603 |
| 30 days | 0.877 | 0.603 |
| 365 days | 0.787 | 0.585 |

- SMART carries real failure signal, strongest close to the failure.
- Failures are rare (about 0.1% of drive-days fail within 30 days), so most alerts are still
  false alarms: with the alert threshold set to flag 1% of healthy drive-days, 3.8% of the
  flagged drive-days fail within 30 days.
- The model seldom names a failure day months ahead.

A run writes the full numbers to `build/report/results.md` and the explanation to
`build/report/analysis.md`.

## What you need

| | |
|---|---|
| OS | Linux (developed and tested there; CI runs Ubuntu) |
| Tools | `git` and [uv](https://docs.astral.sh/uv/getting-started/installation/). uv installs Python 3.13+ and every Python dependency |
| System library | GNU OpenMP (`libgomp.so.1`), which LightGBM loads. Most desktops have it; on a minimal Debian or Ubuntu: `sudo apt install libgomp1` |
| Memory | 16 GB for `small` and `medium`, a 32 GB machine for `full` |
| Disk | About 90 GiB to build the panel (28 GiB of zips kept, 60 GiB free for scratch), 30 GiB for the panel, 2 GiB for `build/` |
| Time | About 3 hours to build the panel once, then 15 minutes (`small`) to under 2 hours (`full`) per run |

Times in this file were measured on a 16-thread machine with 31 GB of memory, an SSD for the
project and an HDD for the downloads.

## Setup

### 1. Install and test

No data needed. The tests use a small synthetic panel and take about 4 minutes.

```bash
git clone https://github.com/lazy-machine/hard-drive-analysis-v4.git
cd hard-drive-analysis-v4
uv sync
uv run pytest -q
```

### 2. Choose where the downloads go

The two panel build steps keep their big files under `/srv/downloads/backblaze-drive-stats/`.
The path is hard-coded in `pipeline.toml`, because this project was built for one machine.
Do one of the following.

Create that folder, if `/srv/downloads` is (or can be) on a disk with 90 GiB free:

```bash
sudo mkdir -p /srv/downloads/backblaze-drive-stats && sudo chown "$USER" /srv/downloads/backblaze-drive-stats
```

Or point the steps at a folder of your own, one you can write to. Replace `/mnt/big` with
it:

```bash
sed -i 's#/srv/downloads#/mnt/big#g' pipeline.toml
```

After this change one test, `test_pipeline_pins_agree`, fails: it checks that the paths
still start with `/srv/downloads/`. That is expected on your machine.

### 3. Build the panel

```bash
uv run data-pipeline --only panel_download panel_convert
```

This downloads Backblaze's 40 quarterly zips (about 40 minutes here) and converts their
3,653 daily CSVs to `dataset/dataset.parquet` (about 2 hours here). It can be interrupted:
finished zips and converted quarters are reused. Follow it from a second terminal:

```bash
tail -F build/logs/panel_download.log build/logs/panel_convert.log
```

### 4. Pin your panel's sha256

`pipeline.toml` holds the sha256 of the panel in two places, and the pipeline refuses a
panel that does not match. Your panel has the same rows as ours but different bytes (see
[Raw data](#raw-data)), so its sha256 differs. Once step 3 has ended with `✓ completed`,
this hashes your panel (about a minute) and writes the result into both places. It changes
nothing if the panel is missing:

```bash
sha=$(sha256sum dataset/dataset.parquet | cut -d' ' -f1) && [ ${#sha} -eq 64 ] && sed -i -E "s/sha256 = \"[0-9a-f]{64}\"/sha256 = \"$sha\"/" pipeline.toml
```

`git diff pipeline.toml` should now show the same new sha256 on two lines.

### 5. Run

```bash
uv run data-pipeline -n
uv run data-pipeline
```

The first line is a dry run: it shows what would run, and why. The second runs everything
that is out of date. Results land in `build/report/`. Each long step logs its progress to
`build/logs/<step>.log`.

## Scales

The scale decides how many drives a run uses. Set `scale` in `config/sample.toml`. It is
committed as `small`; use `full` for real results.

| Scale | Drives kept (failing / healthy) | Time after `drive_index` | Memory |
|---|---|---|---|
| small | 4% / 0.4% | ~7 min | any machine |
| medium | 20% / 2% | ~22 min | 3.7 GiB training estimate |
| full | 100% / 10% | ~105 min | 11.4 GiB training estimate |

- The first run also scans the whole panel once (`drive_index`, about 5 minutes). Every
  scale reuses that scan.
- Changing the scale reruns only the steps that depend on it.
- Sampled drives and snapshots carry weights that undo the sampling, so every scale
  estimates the same numbers. Smaller scales give wider intervals.

Two environment variables tune resources without causing any step to rerun:
`HDA_MEMORY_FRACTION` (share of the machine's memory DuckDB may use, default 0.6) and
`HDA_THREADS` (threads for DuckDB and LightGBM, worker processes for the bootstrap; default:
one per core, and for LightGBM whatever OpenMP picks).

## Commands

| Command | What it does |
|---|---|
| `uv run data-pipeline -n` | Dry run: what would run, and why |
| `uv run data-pipeline` | Run everything that is out of date |
| `uv run data-pipeline -F` | The same, printing each step's output live |
| `uv run data-pipeline -r STEP` | Run `STEP` again even if nothing changed |
| `uv run data-pipeline -o STEP` | Look at `STEP` only: it runs if it is out of date (its inputs must exist) |
| `uv run data-pipeline -f` | Run every step again even if nothing changed |
| `uv run data-verify checks` | List every check of the data gate |
| `uv run pytest -q` | All tests |
| `uv run ruff check src steps tests` | Lint |

No flag reruns the two panel build steps while the panel exists and matches its pin, and
any run that includes them builds a missing panel. See [Raw data](#raw-data).

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

The runner (`data_pipeline`) and the data gate (`data_verify`) live in `src/`; nothing
outside this repository is needed. The runner runs each step in a fresh subprocess and
skips it when its code, params and inputs are unchanged.

## Raw data

`panel_download` and `panel_convert` build `dataset/dataset.parquet` from the source:
Backblaze's drive-stats CSVs, one file per day (3,653 of them), about 250 GB for 2016-2025.
They exist so the panel can be traced back to and rebuilt from the public data. They are
meant to run once. These choices are deliberate:

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

To rebuild an existing panel, move it away, then repeat [Setup](#setup) steps 3 and 4:

```bash
mv dataset/dataset.parquet dataset/dataset-old.parquet
```

## Layout

```
pipeline.toml             steps and models
config/study.toml         the shared contract: split, label bins, features, evaluation
config/sample.toml        scales
config/panel-schema.toml  column order and types of dataset.parquet
src/data_pipeline/        the runner (config, checkpoint, orchestrator, worker, CLI)
src/data_verify/          the data gate (checks, spec, CLI)
steps/                    the code of each step, plus shared helpers
tests/                    synthetic panel, unit tests, end-to-end run, gate tamper tests
tests/runner/             runner tests
tests/gate/               gate tests
dataset/                  the panel (not in git)
build/                    the last run's data, logs and report (not in git)
```

## Rules

- **Never load the panel into pandas.** Whole-panel work is DuckDB, in month chunks.
- **Steps talk only through declared files.**
- **Outputs are deterministic** (sorted, seeded), so unchanged results keep later steps cached.
- **One contract.** A rule that two steps share goes in `config/study.toml`.
- **Data reaches a model only through the gate** (`depends_on = ["verify"]`).

## Licence

MIT, see [LICENSE](LICENSE).
