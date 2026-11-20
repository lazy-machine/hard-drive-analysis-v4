"""Every package a step's code imports is in its ``requires``.

The runner puts only ``requires`` versions in a step's fingerprint, so a package missing
there can be upgraded under a cached result.
"""

from __future__ import annotations

import ast
import sys
from importlib import metadata
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from data_pipeline import checkpoint, config

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = config.load(ROOT / "pipeline.toml")
DISTS = metadata.packages_distributions()

# pandas reads and writes Parquet through pyarrow.
IMPLIED = {"pandas": {"pyarrow"}}
# Imported only for a file type the pipeline does not use (data_verify's .yaml references).
OPTIONAL = {"yaml"}


def top_level_imports(path: Path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_bytes())):
        if isinstance(node, ast.Import):
            names.update(a.name.partition(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.partition(".")[0])
    return names


def module_name(path: Path) -> str | None:
    for root in PIPELINE.python_path:
        if path.is_relative_to(root):
            return ".".join(path.relative_to(root).with_suffix("").parts).removesuffix(".__init__")
    return None


def needed(step: config.Step) -> set[str]:
    """Distributions imported anywhere in the code the runner hashes for ``step``."""
    files = {checkpoint.module_file(PIPELINE, step.module)}
    modules = [step.module]
    for rel in step.watch:  # watched modules may be imported dynamically
        if (path := PIPELINE.resolve(rel)).suffix == ".py":
            files.add(path)
            if module := module_name(path):
                modules.append(module)
    for module in modules:
        files.update(checkpoint.local_imports(PIPELINE, module))
    names = set().union(*(top_level_imports(f) for f in files))
    dists = set()
    for name in names - set(sys.stdlib_module_names) - OPTIONAL:
        found = checkpoint.module_file(PIPELINE, name)
        if found and any(found.is_relative_to(r) for r in PIPELINE.python_path):
            continue  # local code: fingerprinted by its source
        assert name in DISTS, f"{step.name}: {name} is neither local nor installed"
        dists.update(canonicalize_name(d) for d in DISTS[name])
    for pkg, implied in IMPLIED.items():
        if pkg in dists:
            dists |= implied
    return dists


@pytest.mark.parametrize("step", [s for s in PIPELINE.steps if not s.once], ids=lambda s: s.name)
def test_requires_lists_every_imported_package(step):
    declared = {canonicalize_name(Requirement(r).name) for r in step.requires}
    assert needed(step) - declared == set()


def test_specifiers_agree_across_steps():
    seen: dict[str, set[str]] = {}
    for step in PIPELINE.steps:
        for r in map(Requirement, step.requires):
            seen.setdefault(canonicalize_name(r.name), set()).add(str(r.specifier))
    assert {k: v for k, v in seen.items() if len(v) > 1} == {}
