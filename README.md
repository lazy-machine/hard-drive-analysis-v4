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
`pipeline.toml`). Results land in `build/report/results.md` and `build/report/analysis.md`.
Long steps log progress to `build/logs/<step>.log`.

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
dataset.parquet ─► drive_index ─► cohort ─► extract ─► snapshots ─► verify ─► explore
                                                                      │
                                                                      └─► train_* (7 models) ─► predict ─► evaluate ─► report
```

| Step | Output | What it does |
|---|---|---|
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

## Layout

```
pipeline.toml        steps and models
config/study.toml    the shared contract: split, label bins, features, evaluation
config/sample.toml   scales
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
