# hard-drive-analysis-v4

Survival model on Backblaze's drive-stats panel: do SMART attributes predict remaining drive
life? `pipeline.toml` lists the steps. The runner runs each one in a fresh subprocess,
checkpoints it, and skips it when nothing it depends on changed.

## Commands
- Setup: `uv sync`
- Tests: `uv run pytest -q` (project, runner and gate suites, ~4 min)
- Lint: `uv run ruff check src steps tests`
- Dry run: `uv run data-pipeline -n`
- Run: `uv run data-pipeline` (scale in `config/sample.toml`: `small` by default, `full` for
  real results)
- Gate reference: `uv run data-verify checks`, `uv run data-verify params`

CI (`.github/workflows/test.yml`) runs lint and tests on every push and pull request.

## Layout
- `src/data_pipeline`: the runner (TOML config, content-hash checkpoints, subprocess worker).
- `src/data_verify`: the pre-training data gate, run as the `verify` step.
- `steps`: pipeline code, one module per group of steps.
- `tests`: project tests (synthetic panel, end-to-end run, gate tamper tests).
  `tests/runner` and `tests/gate` test the two packages.
- `config/study.toml`: rules shared by every scale. `config/sample.toml`: run scale.
- `dataset/dataset.parquet`: input panel (gitignored, checked by sha256 in `pipeline.toml`).
  `panel_download` and `panel_convert` (`steps/raw.py`) build it from Backblaze's CSVs.
- `build`: the last run's data; `build/report` holds its results.

## Code rules
- Never load the panel into pandas. Whole-panel work is DuckDB, in month chunks.
- A step reads only its declared `inputs` and `params` and writes only its `outputs`.
  Two exceptions, neither read by another step nor checkpointed: progress logs in
  `build/logs/<step>.log` (`common.Progress`) and scratch in `build/tmp/<step>/` (DuckDB
  spill and staging, `common.connect`).
- Outputs are deterministic (sorted, seeded, no timestamps), so caching stays correct.
- A rule two steps share goes in `config/study.toml`.
- Data reaches a model only through the gate (`depends_on = ["verify"]`). A test for a
  gate check injects the fault it guards against.
- `src/` is on the pipeline's `python_path`, so a change to the gate reruns `verify`.
- Style: `from __future__ import annotations`, type hints, short docstrings that say why.
- The panel build steps are `once` steps by the owner's decision: never rerun while the
  panel is healthy, even with `-f`. All their big files (downloads, CSVs, staging, DuckDB
  spill) stay under `/srv/downloads` on the HDD; only the panel goes to `dataset/`. They
  are IO-bound and take hours. Don't propose making them rerunnable, moving their scratch
  to the SSD, or speeding them up. README.md, "Raw data", gives the reasons.

## Workflow
- Every major change goes on a branch and a pull request, reviewed before merge.
- Report issues in this repository's GitHub issues.

## Talking to the user
- Be direct. Break down chains of complex terms.
- No flattery.
- Ask when instructions contradict each other.
