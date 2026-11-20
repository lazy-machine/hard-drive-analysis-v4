"""The with- vs without-SMART comparison must differ in SMART features only."""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# extra outputs that do not change the trees
POST_TRAINING = {"recalibrate_recent"}


def _params(name: str) -> dict:
    steps = tomllib.loads((ROOT / "pipeline.toml").read_text())["steps"]
    return next(s for s in steps if s["name"] == name)["params"]


def test_smart_and_nosmart_share_training_params():
    smart, nosmart = _params("train_hazard_smart"), _params("train_hazard_nosmart")
    assert (smart["features"], nosmart["features"]) == ("all", "context")
    strip = {"features", *POST_TRAINING}
    assert ({k: v for k, v in smart.items() if k not in strip}
            == {k: v for k, v in nosmart.items() if k not in strip})


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
