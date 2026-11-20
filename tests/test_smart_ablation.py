"""The with- vs without-SMART comparison must differ in SMART features only."""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# extra outputs that do not change the trees
POST_TRAINING = {"recalibrate_recent"}


def _training(name: str) -> dict:
    """The step without what may differ: its name, outputs, and features."""
    steps = tomllib.loads((ROOT / "pipeline.toml").read_text())["steps"]
    step = dict(next(s for s in steps if s["name"] == name))
    for key in ("name", "description", "outputs"):
        step.pop(key)
    step["params"] = {k: v for k, v in step["params"].items()
                      if k not in {"features", *POST_TRAINING}}
    return step


def test_smart_and_nosmart_share_training_setup():
    steps = {s["name"]: s for s in tomllib.loads((ROOT / "pipeline.toml").read_text())["steps"]}
    assert steps["train_hazard_smart"]["params"]["features"] == "all"
    assert steps["train_hazard_nosmart"]["params"]["features"] == "context"
    assert _training("train_hazard_smart") == _training("train_hazard_nosmart")


def test_history_spans_are_context_not_smart():
    from common import load_study
    from panel import feature_schema

    study = load_study(ROOT / "config" / "study.toml")
    schema = feature_schema(study, ["MODEL A"])
    context, smart = schema["groups"]["context"], schema["groups"]["smart"]
    spans = [f"window_days_{w}" for w in study["features"]["trend_windows"]]
    assert spans and set(spans) <= set(context)
    assert not any(f.startswith("smart_") for f in context)
    assert all(f.startswith("smart_") for f in smart)
    assert sorted(schema["features"]) == sorted(context + smart)
