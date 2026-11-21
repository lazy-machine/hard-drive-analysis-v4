"""Worker process: runs exactly one step function, then exits.

Invoked by the orchestrator as ``python -m data_pipeline.worker SPEC RESULT``. Each step
gets a fresh interpreter, so no memory, globals or imported modules leak between steps.
"""

from __future__ import annotations

import importlib
import importlib.machinery as machinery
import json
import os
import signal
import site
import sys
import sysconfig
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from .context import StepContext


def _summarize(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        parts = []
        for k, v in value.items():
            parts.append(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}")
        return " ".join(parts)
    return str(value)


def _stop_with_runner(runner: int) -> None:
    """End the step, and what it started, if the runner dies.

    A worker leads its own session, so a signal that ends the runner never reaches it, and
    nothing else would stop it writing outputs that no checkpoint will record. This relies on
    the worker being the runner's direct child: a launcher that forks would look like a
    runner that died.
    """
    if not hasattr(os, "killpg"):
        return

    def watch() -> None:
        while os.getppid() == runner:
            time.sleep(0.5)
        os.killpg(os.getpgid(0), signal.SIGKILL)

    threading.Thread(target=watch, daemon=True).start()


class _SourceLoader(machinery.SourceFileLoader):
    """Compiles from source every time.

    The runner keys a checkpoint on the source's hash, but a ``.pyc`` is trusted on the
    source's mtime and size, so an edit that keeps both would run the old code.
    """

    def get_code(self, fullname: str):
        path = self.get_filename(fullname)
        return self.source_to_code(self.get_data(path), path)


def _forms(path: str) -> set[Path]:
    """A path as given and with symlinks resolved: either may be the one under a root."""
    return {Path(os.path.abspath(path)), Path(path).resolve()}


def _installed_dirs() -> set[Path]:
    dirs = {sysconfig.get_path(k) for k in ("stdlib", "platstdlib", "purelib", "platlib")}
    dirs.update(site.getsitepackages())
    dirs.add(site.getusersitepackages())
    return set().union(*(_forms(d) for d in dirs if d))


def is_step_code_dir(entry: str, roots: set[Path], installed: set[Path]) -> bool:
    """Under a ``python_path`` root, unless also under an installed-packages directory
    that lies inside that root (a project ``.venv`` under the default root ``.``)."""
    if not os.path.isdir(entry):
        return False
    return any(path.is_relative_to(r) and not any(
                   path.is_relative_to(d) and d != r and d.is_relative_to(r) for d in installed)
               for path in _forms(entry) for r in roots)


def _compile_from_source(python_path: list[str]) -> None:
    """Import step code from source; installed packages keep their cache."""
    roots = set().union(*(_forms(r) for r in python_path))
    installed = _installed_dirs()
    finder = machinery.FileFinder.path_hook(
        (machinery.ExtensionFileLoader, machinery.EXTENSION_SUFFIXES),
        (_SourceLoader, machinery.SOURCE_SUFFIXES),
        (machinery.SourcelessFileLoader, machinery.BYTECODE_SUFFIXES),
    )

    def hook(entry: str) -> machinery.FileFinder:
        if not is_step_code_dir(entry or ".", roots, installed):
            raise ImportError("not a pipeline path")
        return finder(entry)

    sys.path_hooks.insert(0, hook)
    sys.path_importer_cache.clear()


def main(argv: list[str]) -> int:
    spec_path, result_path = argv
    spec = json.loads(Path(spec_path).read_text())
    _stop_with_runner(spec["runner"])  # from the runner: it may die before this line runs

    for p in reversed(spec["python_path"]):
        if p not in sys.path:
            sys.path.insert(0, p)
    _compile_from_source(spec["python_path"])

    try:
        module = importlib.import_module(spec["module"])
        func = getattr(module, spec["attr"])
        ctx = StepContext(
            name=spec["name"],
            inputs={k: Path(v) for k, v in spec["inputs"].items()},
            outputs={k: Path(v) for k, v in spec["outputs"].items()},
            params=spec["params"],
            workdir=Path(spec["workdir"]),
        )
        summary = _summarize(func(ctx))
    except BaseException as exc:  # report everything, including SystemExit/KeyboardInterrupt
        sys.stdout.flush()
        tb = exc.__traceback__.tb_next or exc.__traceback__  # hide this module's frame
        traceback.print_exception(type(exc), exc, tb, file=sys.stdout)
        return 1

    sys.stdout.flush()
    Path(result_path).write_text(json.dumps({"summary": summary}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
