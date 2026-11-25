"""Project checks for the data-verify gate (the `custom` list of the verify step).

The generic split, time, schema and leakage checks and the SQL rules on labels, censoring
and weights are declared in pipeline.toml. These are the checks that need Python. They
use the same helpers as the steps that built the data, so they cannot drift from them.
"""

from __future__ import annotations

import numpy as np

from common import load_sample, load_study, on_stride, stable_bucket
from drives import assign_split, first_seen_after_train, select_cohort, stratum


def stride_rows_on_stride_days(v):
    """Rows weighted drive_weight * stride fall on the drive's stride days.

    common.on_stride is the schedule's definition; this checks the snapshot step's SQL
    against it, so it shares the definition rather than keeping a copy.
    """
    stride = int(v.var("stride"))
    if stride == 1:
        v.add("stride_rows_on_stride_days", True,
              "stride_days = 1: dense and stride weights coincide, rows cannot be told apart",
              category="weights")
        return
    cohort = v.dataset("cohort")[["serial_number", "drive_weight"]]
    bad = 0
    for d in v.splits.values():
        m = d[["serial_number", "date", "weight"]].merge(cohort, on="serial_number", how="left")
        stride_weight = np.abs(m["weight"] - m["drive_weight"] * stride) < 1e-6
        bad += int((stride_weight & ~on_stride(m, stride)).sum())
    v.add("stride_rows_on_stride_days", bad == 0,
          f"{bad} rows with weight drive_weight * {stride} off the drive's stride day",
          category="weights", details={"rows": bad},
          hint="The snapshot step and the cohort disagree on the stride offset.")


def cohort_split_matches_stable_bucket(v):
    """Every cohort drive, including drives without snapshots, is in its rule's split: its
    bucket's, or test if it was first seen after train_end.

    split.rule only sees snapshot rows; this covers the whole cohort, as drives.cohort
    assigns it. cohort_dates_match_drives pins the first dates it reads.
    """
    cfg = {k: v.var(k) for k in ("train_buckets", "val_buckets", "test_buckets", "train_end")}
    cohort = v.dataset("cohort")
    new = first_seen_after_train(cohort["first_date"], cfg)
    expected = [assign_split(stable_bucket(s), n, cfg)
                for s, n in zip(cohort["serial_number"], new, strict=True)]
    wrong = cohort.loc[cohort["split"].to_numpy() != expected, "serial_number"]
    v.add("cohort_split_matches_stable_bucket", wrong.empty,
          f"{len(wrong)} of {len(cohort)} cohort drives in the wrong split", category="split",
          details={"drives": len(wrong), "examples": sorted(wrong)[:5]},
          hint="The cohort step and the split rule in config/study.toml disagree.")


def cohort_matches_drive_index(v):
    """The cohort is the one drives.select_cohort draws from the drive index: same drives,
    splits and weights. The other cohort checks only see the drives that are in it, so a
    drive left out of the cohort (and of the snapshots) would pass them."""
    want, _ = select_cohort(v.dataset("drives"), load_study(v.inputs["study"]),
                            load_sample(v.inputs["sample"]))
    have = v.dataset("cohort")[["serial_number", "split", "drive_weight"]]
    m = want[["serial_number", "split", "drive_weight"]].merge(
        have, on="serial_number", how="outer", suffixes=("", "_cohort"), indicator=True)
    bad = m[(m["_merge"] != "both") | (m["split"] != m["split_cohort"])
            | ~np.isclose(m["drive_weight"], m["drive_weight_cohort"], rtol=1e-12, atol=0)]
    kinds = bad["_merge"].map({"left_only": "missing", "right_only": "extra",
                               "both": "split or weight"}).value_counts()
    v.add("cohort_matches_drive_index", bad.empty,
          f"{len(bad)} drive(s) differ from the cohort the drive index gives: "
          f"{kinds.to_dict()}", category="split",
          details={"drives": len(bad), "examples": sorted(bad["serial_number"])[:5]},
          hint="The cohort is not select_cohort's output for this drive index and config.")


def cohort_stratum_matches_rule(v):
    """Every cohort drive's stratum is what drives.stratum gives, so its weight follows.

    cohort_weights_match_sampling checks drive_weight against stratum, split and first date;
    this pins stratum to the drive's failure date and split, with the rule the cohort step
    uses.
    """
    cohort = v.dataset("cohort")
    expected = stratum(cohort, v.var("train_end"))
    wrong = cohort.loc[cohort["stratum"].to_numpy() != expected, "serial_number"]
    v.add("cohort_stratum_matches_rule", wrong.empty,
          f"{len(wrong)} of {len(cohort)} cohort drives in the wrong sampling stratum",
          category="weights", details={"drives": len(wrong), "examples": sorted(wrong)[:5]},
          hint="The cohort step wrote a stratum that drives.stratum does not give.")


def features_are_float(v):
    """Every feature is stored as FLOAT (float32), the type the snapshot step writes."""
    train = v.train
    other = {f: str(train[f].dtype) for f in v.features if str(train[f].dtype) != "float32"}
    v.add("features_are_float", not other, f"{len(other)} non-float32 feature(s): "
          f"{dict(list(other.items())[:5])}", category="schema", details={"columns": other})


def few_sparse_features(v):
    """At most a quarter of the features are less than 1% filled in train."""
    train = v.train
    fill = train[v.features].notna().mean()
    sparse = sorted(fill[fill < 0.01].index)
    v.add("few_sparse_features", len(sparse) <= len(v.features) // 4,
          f"{len(sparse)} feature(s) < 1% filled in train: {sparse}", severity="warn",
          category="features", details={"sparse": sparse})
