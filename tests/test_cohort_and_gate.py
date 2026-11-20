"""Cohort selection rules, contract validation, and the gate catching injected bugs."""

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from common import load_sample, load_study, stable_bucket, stable_unit, stride_offset
from drives import select_cohort

ROOT = Path(__file__).resolve().parent.parent
STUDY = load_study(ROOT / "config" / "study.toml")
SCALE = {"failing_fraction": 1.0, "healthy_fraction": 1.0, "stride_days": 7, "dense_days": 30,
         "scale": "t"}


def drives_frame(rows):
    cols = ["serial_number", "model", "first_date", "last_date", "first_failure",
            "capacity_bytes"]
    return pd.DataFrame(rows, columns=cols).assign(n_models_per_month=1)


def test_stable_hashes_are_stable():
    # fixed values: a change here would silently reshuffle every split, sample and schedule
    serials = ("Z305B2QN", "ZA1FLE1P", "A", "B", "C")
    assert [stable_bucket(s) for s in serials] == [1, 32, 49, 94, 59]
    assert [stride_offset(s, 7) for s in serials] == [6, 5, 6, 0, 0]
    assert [round(stable_unit(s, "sample"), 6) for s in serials] == [
        0.701468, 0.375672, 0.601464, 0.829172, 0.52053]


def test_cohort_rules():
    serials = [f"S{i}" for i in range(400)]
    rows = [(s, "ST4000DM000", "2020-01-01", "2025-12-31", None, 4e12) for s in serials]
    rows.append(("SSD1", "CT250MX500SSD1", "2020-01-01", "2025-12-31", None, 2.5e11))
    # test-bucket drive that failed before test_start has nothing to test
    early = next(s for s in (f"E{i}" for i in range(1000)) if stable_bucket(s) >= 80)
    rows.append((early, "ST4000DM000", "2020-01-01", "2023-06-01", "2023-06-01", 4e12))
    # train-bucket drive that only appears after train_end
    late = next(s for s in (f"L{i}" for i in range(1000)) if stable_bucket(s) < 70)
    rows.append((late, "ST4000DM000", "2024-03-01", "2025-12-31", None, 4e12))
    cohort, counts = select_cohort(drives_frame(rows), STUDY, SCALE)

    assert "SSD1" not in set(cohort["serial_number"])
    assert early not in set(cohort["serial_number"])
    assert late not in set(cohort["serial_number"])
    assert counts["excluded_ssd"] == 1 and counts["excluded_outside_window"] == 2
    expected = {"train": range(0, 70), "val": range(70, 80), "test": range(80, 100)}
    for s, split in zip(cohort["serial_number"], cohort["split"], strict=True):
        assert stable_bucket(s) in expected[split]


def test_sampling_weights_are_inverse_fractions():
    rows = [(f"S{i}", "ST4000DM000", "2020-01-01", "2025-12-31",
             "2021-05-05" if i % 3 == 0 else None, 4e12) for i in range(3000)]
    scale = dict(SCALE, failing_fraction=0.5, healthy_fraction=0.1)
    cohort, _ = select_cohort(drives_frame(rows), STUDY, scale)
    w = cohort.groupby("stratum")["drive_weight"].unique()
    assert list(w["failing"]) == [2.0] and list(w["healthy"]) == [10.0]
    kept_healthy = (cohort["stratum"] == "healthy").sum()
    assert 140 < kept_healthy < 260  # about 10% of 2000


def test_contract_validation(tmp_path):
    bad = (ROOT / "config" / "study.toml").read_text().replace(
        "horizons = [7, 30, 90, 180, 365]", "horizons = [7, 45]")
    (tmp_path / "study.toml").write_text(bad)
    with pytest.raises(ValueError, match="not bin edges"):
        load_study(tmp_path / "study.toml")
    # 14 is a bin edge but not a horizon: evaluate would find no hazard p14 column
    bad = (ROOT / "config" / "study.toml").read_text().replace(
        "binary_horizon = 30", "binary_horizon = 14")
    (tmp_path / "study.toml").write_text(bad)
    with pytest.raises(ValueError, match="not in label.horizons"):
        load_study(tmp_path / "study.toml")
    (tmp_path / "sample.toml").write_text('scale = "nope"\n[scales.small]\n')
    with pytest.raises(ValueError, match="not defined"):
        load_sample(tmp_path / "sample.toml")


def _checks(project, tmp_path, tamper_sql=None):
    """Run the pipeline's data-verify step, optionally on tampered snapshots."""
    import tomllib

    from data_verify import run

    build = project / "build"
    pipeline = tomllib.loads((project / "pipeline.toml").read_text())
    step = next(s for s in pipeline["steps"] if s["name"] == "verify")
    inputs = {k: project / v for k, v in step["inputs"].items()}
    if tamper_sql:
        con = duckdb.connect()
        con.sql(f"CREATE TABLE snap AS SELECT * FROM read_parquet('{build}/snapshots.parquet')")
        con.sql(f"CREATE VIEW cohort AS SELECT * FROM read_parquet('{build}/cohort.parquet')")
        changed = con.sql(tamper_sql).fetchall()[0][0]
        assert changed > 0, "the tamper must change at least one row"
        inputs["snapshots"] = tmp_path / "snapshots.parquet"
        con.sql(f"COPY (SELECT * FROM snap) TO '{inputs['snapshots']}' (FORMAT parquet)")
    report = run(step["params"], inputs, log=lambda *a: None, workdir=project)
    return {c["id"]: c for c in report["checks"]}


def _gate(project, tmp_path, tamper_sql=None):
    checks = _checks(project, tmp_path, tamper_sql)
    return {cid for cid, c in checks.items()
            if c["status"] in ("fail", "error") and c["severity"] == "error"}


def test_gate_passes_clean_data(pipeline_run, tmp_path):
    project, _ = pipeline_run
    assert _gate(project, tmp_path) == set()


def test_gate_catches_v2_label_bug(pipeline_run, tmp_path):
    """Rows inside a visible failure window relabelled as healthy (review §3.2)."""
    project, _ = pipeline_run
    failed = _gate(project, tmp_path,
                   "UPDATE snap SET event = 0 WHERE event = 1 AND duration_days < 10")
    assert {"assert.labels_match_cohort", "assert.no_visible_failure_labelled_censored"} <= failed


def test_gate_catches_split_leak(pipeline_run, tmp_path):
    project, _ = pipeline_run
    failed = _gate(project, tmp_path, """UPDATE snap SET split = 'train'
                               WHERE (serial_number, date) =
                               (SELECT (serial_number, date) FROM snap WHERE split = 'test'
                                ORDER BY serial_number, date LIMIT 1)""")
    assert {"split.group_overlap", "assert.snapshot_split_matches_cohort"} <= failed


def test_gate_catches_train_rows_in_test_period(pipeline_run, tmp_path):
    project, _ = pipeline_run
    failed = _gate(project, tmp_path, """UPDATE snap SET date = DATE '2024-02-01'
                               WHERE split = 'train' AND date = (SELECT max(date) FROM snap
                                                                 WHERE split = 'train')""")
    assert {"temporal.bounds", "temporal.order"} <= failed


def test_gate_catches_stride_rows_off_stride_days(pipeline_run, tmp_path):
    """Stride rows moved off the drive's stride day keep their weight but not the density."""
    project, _ = pipeline_run
    stride = load_sample(project / "config" / "sample.toml")["stride_days"]
    failed = _gate(project, tmp_path, f"""UPDATE snap SET date = snap.date + 1 FROM cohort c
                               WHERE snap.serial_number = c.serial_number
                                 AND abs(snap.weight - c.drive_weight * {stride}) < 1e-6""")
    assert "custom.stride_rows_on_stride_days" in failed


def test_gate_catches_dense_weight_far_from_failure(pipeline_run, tmp_path):
    project, _ = pipeline_run
    failed = _gate(project, tmp_path, """UPDATE snap SET weight = c.drive_weight FROM cohort c
                               WHERE snap.serial_number = c.serial_number
                                 AND (snap.serial_number, snap.date) =
                                     (SELECT (s.serial_number, s.date) FROM snap s
                                      JOIN cohort c2 USING (serial_number)
                                      WHERE c2.failure_date IS NULL
                                      ORDER BY s.serial_number, s.date LIMIT 1)""")
    assert "assert.dense_rows_before_visible_failure" in failed
    assert "assert.snapshot_weights_valid" not in failed  # the weight itself is a valid value


def test_gate_warns_on_cohort_drive_without_snapshots(pipeline_run, tmp_path):
    project, _ = pipeline_run
    name = "assert.cohort_drives_have_snapshots"
    before = _checks(project, tmp_path)[name]
    after = _checks(project, tmp_path, """DELETE FROM snap WHERE serial_number =
                                          (SELECT min(serial_number) FROM snap)""")[name]
    assert after["status"] == "fail" and after["severity"] == "warn"
    assert not after["blocking"]
    missing = before.get("details", {}).get("violations", 0)
    assert after["details"]["violations"] == missing + 1



def test_gate_counts_all_null_features_as_sparse(pipeline_run, tmp_path):
    """Features with no value at all in train are the sparsest, not exempt."""
    import json

    project, _ = pipeline_run
    features = json.loads((project / "build" / "feature-schema.json").read_text())["features"]
    nulled = sorted(features[-(len(features) // 4 + 1):])
    sets = ", ".join(f'"{f}" = NULL' for f in nulled)
    name = "custom.few_sparse_features"
    check = _checks(project, tmp_path, f"UPDATE snap SET {sets} WHERE split = 'train'")[name]
    assert check["status"] == "fail" and check["severity"] == "warn"
    assert set(nulled) <= set(check["details"]["sparse"])


def _gate_on(project, tmp_path, snapshots_sql=None, cohort_sql=None):
    """Blocking check ids, with the snapshot or cohort table rewritten by a SELECT."""
    import tomllib

    from data_verify import run

    build = project / "build"
    step = next(s for s in tomllib.loads((project / "pipeline.toml").read_text())["steps"]
                if s["name"] == "verify")
    inputs = {k: project / v for k, v in step["inputs"].items()}
    con = duckdb.connect()
    for name, sql in (("snapshots", snapshots_sql), ("cohort", cohort_sql)):
        if sql:
            con.sql(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{build}/{name}.parquet')")
            inputs[name] = tmp_path / f"{name}.parquet"
            con.sql(f"COPY ({sql}) TO '{inputs[name]}' (FORMAT parquet)")
    report = run(step["params"], inputs, log=lambda *a: None, workdir=project)
    return set(report["blocking"])


def test_gate_catches_non_float_feature(pipeline_run, tmp_path):
    project, _ = pipeline_run
    blocking = _gate_on(project, tmp_path, snapshots_sql=(
        "SELECT * REPLACE (smart_9_raw::DOUBLE AS smart_9_raw) FROM snapshots"))
    assert "custom.features_are_float" in blocking


def test_gate_catches_cohort_drive_in_wrong_split(pipeline_run, tmp_path):
    """A cohort drive without snapshots in the wrong split: only the cohort check sees it."""
    project, _ = pipeline_run
    blocking = _gate_on(project, tmp_path, cohort_sql="""
        SELECT * REPLACE (CASE WHEN serial_number = (SELECT min(serial_number) FROM cohort)
                          THEN CASE split WHEN 'train' THEN 'test' ELSE 'train' END
                          ELSE split END AS split)
        FROM cohort""")
    assert "custom.cohort_split_matches_stable_bucket" in blocking


def _dense_rows(project):
    """Snapshot rows at the dense weight, with whether they fall on the drive's stride day."""
    build = project / "build"
    stride = load_sample(project / "config" / "sample.toml")["stride_days"]
    rows = duckdb.sql(f"""SELECT s.serial_number, s.date
                          FROM read_parquet('{build}/snapshots.parquet') s
                          JOIN read_parquet('{build}/cohort.parquet') c USING (serial_number)
                          WHERE abs(s.weight - c.drive_weight) < 1e-6
                          ORDER BY s.serial_number, s.date""").df()
    days = (pd.to_datetime(rows["date"]) - pd.Timestamp("2000-01-01")).dt.days
    off = rows["serial_number"].map(lambda s: int(stable_unit(s, "stride") * stride))
    return rows.assign(on_stride=(days + off) % stride == 0), stride


def test_gate_catches_stride_weight_in_dense_window(pipeline_run, tmp_path):
    """A dense-window row on a stride day weighted as a stride row: a valid weight, wrong place."""
    project, _ = pipeline_run
    rows, stride = _dense_rows(project)
    serial, day = rows.loc[rows["on_stride"], ["serial_number", "date"]].iloc[0]
    failed = _gate(project, tmp_path, f"""UPDATE snap SET weight = weight * {stride}
                               WHERE serial_number = '{serial}' AND date = DATE '{day:%Y-%m-%d}'""")
    assert "assert.dense_window_complete" in failed
    assert not failed & {"assert.snapshot_weights_valid", "custom.stride_rows_on_stride_days",
                         "assert.dense_rows_before_visible_failure"}


def test_gate_catches_deleted_dense_rows(pipeline_run, tmp_path):
    project, _ = pipeline_run
    rows, _ = _dense_rows(project)
    serial = rows["serial_number"].iloc[0]
    failed = _gate(project, tmp_path, f"""DELETE FROM snap USING cohort c
                               WHERE snap.serial_number = c.serial_number
                                 AND snap.serial_number = '{serial}'
                                 AND abs(snap.weight - c.drive_weight) < 1e-6""")
    assert "assert.dense_window_complete" in failed


def test_gate_catches_null_cohort_weight(pipeline_run, tmp_path):
    project, _ = pipeline_run
    blocking = _gate_on(project, tmp_path, cohort_sql="""
        SELECT * REPLACE (CASE WHEN serial_number = (SELECT min(serial_number) FROM cohort)
                          THEN NULL ELSE drive_weight END AS drive_weight)
        FROM cohort""")
    assert "assert.cohort_weights_match_sampling" in blocking


FIRST_FAILING = "SELECT min(serial_number) FROM cohort WHERE failure_date IS NOT NULL"


def test_gate_catches_wrong_stratum(pipeline_run, tmp_path):
    """A failing drive sampled as healthy, its weight consistent with the wrong stratum."""
    project, _ = pipeline_run
    healthy = load_sample(project / "config" / "sample.toml")["healthy_fraction"]
    blocking = _gate_on(project, tmp_path, cohort_sql=f"""
        SELECT * REPLACE (
            CASE WHEN serial_number = ({FIRST_FAILING}) THEN 'healthy' ELSE stratum END AS stratum,
            CASE WHEN serial_number = ({FIRST_FAILING}) THEN 1 / {healthy} ELSE drive_weight END
                AS drive_weight)
        FROM cohort""")
    assert "custom.cohort_stratum_matches_rule" in blocking
    assert "assert.cohort_weights_match_sampling" not in blocking


@pytest.mark.parametrize("column", ["first_date", "last_date", "failure_date"])
def test_gate_catches_cohort_dates_off_drive_index(pipeline_run, tmp_path, column):
    project, _ = pipeline_run
    blocking = _gate_on(project, tmp_path, cohort_sql=f"""
        SELECT * REPLACE (CASE WHEN serial_number = ({FIRST_FAILING}) THEN {column} + 1
                          ELSE {column} END AS {column})
        FROM cohort""")
    assert "assert.cohort_dates_match_drives" in blocking
