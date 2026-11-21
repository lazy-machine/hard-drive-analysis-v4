"""The worker runs a step's current source, never a stale ``.pyc`` that still looks valid."""

from __future__ import annotations

import importlib.util
import os
import py_compile
import sysconfig
from pathlib import Path

from data_pipeline import worker
from test_runner_pipeline import make, run, statuses

STEP = """
[[steps]]
name = "s"
function = "{module}:main"
outputs = {{ o = "o.txt" }}
"""

CODE = "import helper\ndef main(ctx):\n    ctx.outputs['o'].write_text('{tag}' + helper.TAG)\n"
HELPER = "TAG = '{tag}'\n"


def plant_stale_pyc(path: Path, old: str, new: str) -> None:
    """Compile ``old`` into the cache, then write ``new`` with the same size and mtime."""
    assert len(old) == len(new)
    path.write_text(old)
    stat = path.stat()
    py_compile.compile(str(path), cfile=importlib.util.cache_from_source(str(path)),
                       invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP)
    path.write_text(new)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))


def test_same_size_same_mtime_edit_runs_new_source(tmp_path):
    cfg = make(tmp_path, STEP.format(module="fns2"))
    steps = tmp_path / "steps"
    plant_stale_pyc(steps / "fns2.py", CODE.format(tag="A"), CODE.format(tag="B"))
    plant_stale_pyc(steps / "helper.py", HELPER.format(tag="A"), HELPER.format(tag="B"))
    r, _ = run(cfg)
    assert r.ok
    assert (tmp_path / "o.txt").read_text() == "BB"


def test_package_module_runs_new_source(tmp_path):
    cfg = make(tmp_path, STEP.format(module="pkg.mod"))
    pkg = tmp_path / "steps" / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (tmp_path / "steps" / "helper.py").write_text(HELPER.format(tag="B"))
    plant_stale_pyc(pkg / "mod.py", CODE.format(tag="A"), CODE.format(tag="B"))
    assert run(cfg)[0].ok
    assert (tmp_path / "o.txt").read_text() == "BB"


def test_worker_writes_no_bytecode_for_step_code(tmp_path):
    cfg = make(tmp_path, STEP.format(module="fns2"))
    (tmp_path / "steps" / "fns2.py").write_text(CODE.format(tag="A"))
    (tmp_path / "steps" / "helper.py").write_text(HELPER.format(tag="A"))
    r, _ = run(cfg)
    assert statuses(r) == {"s": "ran"}
    assert not (tmp_path / "steps" / "__pycache__").exists()


def test_symlinked_package_submodule_runs_new_source(tmp_path):
    cfg = make(tmp_path, STEP.format(module="pkg.mod"))
    pkg = tmp_path / "elsewhere" / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (tmp_path / "steps" / "pkg").symlink_to(pkg, target_is_directory=True)
    (tmp_path / "steps" / "helper.py").write_text(HELPER.format(tag="B"))
    plant_stale_pyc(pkg / "mod.py", CODE.format(tag="A"), CODE.format(tag="B"))
    assert run(cfg)[0].ok
    assert (tmp_path / "o.txt").read_text() == "BB"


def test_project_venv_under_root_keeps_bytecode_cache(tmp_path):
    site = tmp_path / ".venv" / "lib" / "site-packages"
    (site / "numpy").mkdir(parents=True)
    (tmp_path / "steps").mkdir()
    roots, installed = {tmp_path}, {site}
    assert worker.is_step_code_dir(str(tmp_path / "steps"), roots, installed)
    assert not worker.is_step_code_dir(str(site), roots, installed)
    assert not worker.is_step_code_dir(str(site / "numpy"), roots, installed)
    assert not worker.is_step_code_dir(str(tmp_path / "steps" / "x.zip"), roots, installed)
    assert Path(sysconfig.get_path("purelib")).resolve() in worker._installed_dirs()


def test_explicit_root_inside_installed_dir_is_step_code(tmp_path):
    site = tmp_path / "site-packages"
    root = site / "steps"
    (root / "pkg").mkdir(parents=True)
    for roots in ({root}, {site}):
        assert worker.is_step_code_dir(str(root / "pkg"), roots, {site})


def test_symlink_then_dotdot_is_matched_after_resolving(tmp_path):
    (tmp_path / "steps" / "a").mkdir(parents=True)
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "link").symlink_to(tmp_path / "steps" / "a", target_is_directory=True)
    entry = str(tmp_path / "other" / "link" / "..")
    assert worker.is_step_code_dir(entry, {tmp_path / "steps"}, set())
