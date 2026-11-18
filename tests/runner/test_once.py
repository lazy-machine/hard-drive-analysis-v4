"""Once steps and fixed files: built one time, then never rerun while the file is healthy."""

from __future__ import annotations

import hashlib
import textwrap
from pathlib import Path

import pytest

from data_pipeline import config

from test_runner_pipeline import make, run, statuses

BUILD = """
[[steps]]
name = "fetch"
function = "fns:write_param"
outputs = { o = "raw/fetched.txt" }
params = { v = "zips" }
once = true

[[steps]]
name = "convert"
function = "fns:copy"
inputs = { src = "raw/fetched.txt" }
outputs = { dst = "data/panel.txt" }
once = true

[[steps]]
name = "use"
function = "fns:count"
inputs = { src = "data/panel.txt" }
outputs = { n = "out/n.txt" }
"""


def fixed(sha: str = "") -> str:
    return f'fixed = [{{ path = "data/panel.txt", sha256 = "{sha}" }}]'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_missing_fixed_file_is_built_then_kept_even_when_forced(tmp_path):
    cfg = make(tmp_path, BUILD, fixed())
    r, _ = run(cfg)
    assert r.ok and statuses(r) == {"fetch": "ran", "convert": "ran", "use": "ran"}
    panel = tmp_path / "data" / "panel.txt"
    assert panel.read_text() == "ZIPS"

    panel.write_text("trusted")  # anything that exists is healthy without a pin
    for kw in ({}, {"force": True}, {"rerun": ["fetch", "convert"]}, {"only": ["convert"]}):
        r, out = run(cfg, **kw)
        assert r.ok and r.outcomes["convert"].status == "kept", kw
        assert "kept" in out
        assert panel.read_text() == "trusted"
    assert statuses(run(cfg, force=True)[0]) == {"fetch": "kept", "convert": "kept", "use": "ran"}


def test_matching_sha256_keeps_and_code_changes_do_not_rerun(tmp_path):
    cfg = make(tmp_path, BUILD, fixed())
    run(cfg)
    panel = tmp_path / "data" / "panel.txt"
    cfg.write_text(cfg.read_text().replace(fixed(), fixed(sha(panel))))
    (tmp_path / "steps" / "fns.py").write_text((tmp_path / "steps" / "fns.py").read_text() + "\n# edit\n")
    r, _ = run(cfg, force=True)
    assert r.ok and r.outcomes["fetch"].status == "kept" and r.outcomes["convert"].status == "kept"


def test_wrong_sha256_stops_the_run_without_rebuilding(tmp_path):
    cfg = make(tmp_path, BUILD, fixed("0" * 64))
    panel = tmp_path / "data" / "panel.txt"
    panel.parent.mkdir()
    panel.write_text("someone else's panel")
    r, out = run(cfg, force=True)
    assert not r.ok and r.halted_by == "fetch"
    assert "pin its sha256; delete it only if it is damaged" in out
    assert panel.read_text() == "someone else's panel"
    assert not (tmp_path / "raw").exists()


def test_broken_symlink_stops_the_run(tmp_path):
    cfg = make(tmp_path, BUILD, fixed())
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "panel.txt").symlink_to(tmp_path / "gone" / "panel.txt")
    r, out = run(cfg)
    assert not r.ok and "broken symlink" in out and not (tmp_path / "gone").exists()


def test_dry_run_reports_kept(tmp_path):
    cfg = make(tmp_path, BUILD, fixed())
    run(cfg)
    r, _ = run(cfg, dry_run=True, force=True)
    assert r.outcomes["convert"].status == "kept" and r.outcomes["use"].status == "would run"


def test_kept_step_with_a_missing_requirement_does_not_block(tmp_path):
    steps = BUILD.replace('params = { v = "zips" }', 'params = { v = "zips" }\nrequires = ["no-such-package-xyz"]')
    assert "no-such-package-xyz" in steps
    cfg = make(tmp_path, steps, fixed())
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "panel.txt").write_text("trusted")
    r, _ = run(cfg)
    assert r.ok and statuses(r) == {"fetch": "kept", "convert": "kept", "use": "ran"}
    (tmp_path / "data" / "panel.txt").unlink()  # now it must run, and the missing package blocks
    r, _ = run(cfg)
    assert not r.ok and r.halted_by == "fetch"


def test_only_does_not_warn_about_kept_once_steps(tmp_path):
    cfg = make(tmp_path, BUILD, fixed())
    run(cfg)
    (tmp_path / "steps" / "fns.py").write_text((tmp_path / "steps" / "fns.py").read_text() + "\n# edit\n")
    r, out = run(cfg, only=["use"])
    assert r.ok and not r.warnings, out


@pytest.mark.parametrize("header, steps, message", [
    ("", BUILD, "lists no file to keep them by"),
    (fixed(), BUILD.replace("once = true\n", ""), "only a once step may build a fixed file"),
    (fixed("abc"), BUILD, "64 lowercase hex digits"),
    ('fixed = [{ path = "data/panel.txt" }, { path = "data/panel.txt" }]', BUILD, "lists a file twice"),
    ('fixed = [{ sha256 = "" }]', BUILD, "with a path"),
    (fixed(), BUILD.replace("once = true", 'once = "yes"', 1), "once must be true or false"),
])
def test_config_rejects(tmp_path, header, steps, message):
    cfg = make(tmp_path, steps, header)
    with pytest.raises(config.ConfigError, match=message):
        config.load(cfg)


def test_fixed_is_optional(tmp_path):
    cfg = make(tmp_path, textwrap.dedent("""
        [[steps]]
        name = "upper"
        function = "fns:copy"
        inputs = { src = "in.txt" }
        outputs = { dst = "out/upper.txt" }
    """))
    assert config.load(cfg).fixed == []
