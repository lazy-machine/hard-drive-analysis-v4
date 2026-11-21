"""The saved model text leaves out the thread count and the histogram fallback.

Row- and column-wise histograms can still round leaf values differently (see HISTOGRAM in
model.py); this only removes the parameter block's differences.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np

import model


def _booster(threads: int, histogram: str) -> lgb.Booster:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5000, 8))
    y = (X[:, 0] + rng.normal(size=5000) > 1).astype(float)
    p = {**model.BASE_PARAMS, **model.HISTOGRAM[histogram], "seed": 1, "num_threads": threads}
    return lgb.train(p, lgb.Dataset(X, y, params=p), num_boost_round=20)


def test_model_text_ignores_threads_and_histogram_fallback():
    texts = {model.model_text(_booster(t, h), 20) for t, h in [(1, "col"), (4, "col"), (4, "row")]}
    assert len(texts) == 1
    text = texts.pop()
    assert "num_threads" not in text and "force_" not in text


def test_model_text_loads_and_predicts_like_the_trained_booster():
    booster = _booster(2, "col")
    X = np.random.default_rng(1).normal(size=(100, 8))
    loaded = lgb.Booster(model_str=model.model_text(booster, 20))
    np.testing.assert_array_equal(loaded.predict(X), booster.predict(X))
