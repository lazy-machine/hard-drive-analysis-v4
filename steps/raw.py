"""Build dataset/dataset.parquet from Backblaze's quarterly drive-stats zips. Run once.

The raw panel is about 250 GB of daily CSVs (3,652 of them for 2016-2025), which does not
fit on the SSD next to everything else, so these steps keep every large file on the
download disk (an HDD): the zips, the CSVs of the quarter being converted, DuckDB's spill
files and the per-quarter parquet files. Only the finished panel (about 23 GB) is written
to dataset/. The work is IO-bound and takes hours on that disk. That is accepted: both
steps are `once` steps, so the runner never reruns them while the panel is healthy, not
even with --force (see "Raw data" in README.md).

The conversion copies the CSVs as they are. Source quirks stay in the panel, because
later steps already handle them and v1 results were computed with them:

- 2017-01-30.csv has only a header; 2017-01-28, -29 and -31 have 46-61 rows.
- 2019-06-17.csv ends lines with a bare CR, writes dates as 6/17/19, and holds one row
  dated 6/15/19.
- 2022-12-10.csv holds the rows of 2022-12-11, so that day appears twice.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
import tomllib
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from common import Progress, connect, q, scalar

# Bump when the conversion changes, so per-quarter files staged by older code are redone.
CONVERTER_VERSION = 1
_DAY_CSV = re.compile(r"^\d{4}-\d{2}-\d{2}\.csv$")
_QUARTER = re.compile(r"^(\d{4})Q([1-4])$")
_CHUNK = 8 * 2**20


def quarters(first: str, last: str) -> list[str]:
    """Zip names from `first` to `last` inclusive, e.g. 2016Q1 -> data_Q1_2016.zip."""
    def parse(s: str) -> tuple[int, int]:
        m = _QUARTER.match(s)
        if not m:
            raise ValueError(f"quarter must look like 2016Q1, got {s!r}")
        return int(m[1]), int(m[2])

    (y, n), end = parse(first), parse(last)
    if (y, n) > end:
        raise ValueError(f"first quarter {first} is after last quarter {last}")
    names = []
    while (y, n) <= end:
        names.append(f"data_Q{n}_{y}.zip")
        y, n = (y + 1, 1) if n == 4 else (y, n + 1)
    return names


# -- download ----------------------------------------------------------------------------

def download(ctx):
    """Fetch the quarterly zips into the download folder, skipping complete ones.

    A zip is complete when its size matches the server's Content-Length, so an
    interrupted run resumes where it stopped. Each zip is written to `.part` and renamed
    only after its size and zip directory check out. manifest.json lists what panel_convert
    reads, so stray files in the folder are never converted.
    """
    log = Progress(ctx)
    out = ctx.outputs["zips"]
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("*.part"):
        stale.unlink()
    base = ctx.params["base_url"].rstrip("/")
    names = quarters(ctx.params["first"], ctx.params["last"])
    entries = []
    for i, name in enumerate(names, 1):
        size, fetched = _fetch(f"{base}/{name}", out / name, log)
        entries.append({"name": name, "bytes": size})
        log(f"[{i}/{len(names)}] {name} {size / 2**30:.2f} GiB "
            f"{'downloaded' if fetched else 'already complete'}")
    manifest = {"base_url": base, "zips": entries}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    total = sum(e["bytes"] for e in entries)
    return {"zips": len(entries), "gib": round(total / 2**30, 2)}


def _fetch(url: str, dest: Path, log, attempts: int = 4) -> tuple[int, bool]:
    """(size, whether it was downloaded) for one zip, retrying network errors."""
    request = urllib.request.Request(url, headers={"User-Agent": "hard-drive-analysis"})
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=60) as resp:
                length = int(resp.headers.get("Content-Length") or -1)
                if length >= 0 and dest.is_file() and dest.stat().st_size == length:
                    return length, False
                part = dest.with_name(dest.name + ".part")
                with part.open("wb") as fh:
                    shutil.copyfileobj(resp, fh, _CHUNK)
            got = part.stat().st_size
            if length >= 0 and got != length:
                raise OSError(f"{url}: got {got} bytes, expected {length}")
            zipfile.ZipFile(part).close()  # a truncated zip has no readable directory
            os.replace(part, dest)
            return got, True
        except (OSError, zipfile.BadZipFile) as exc:  # URLError and HTTPError are OSErrors
            if attempt == attempts:
                raise
            log(f"{url}: {exc}; retry {attempt}/{attempts - 1}")
            time.sleep(10 * attempt)
    raise AssertionError("unreachable")


# -- convert -----------------------------------------------------------------------------

def convert(ctx):
    """Every quarter's CSVs to one parquet file, typed by config/panel-schema.toml.

    One quarter at a time: unzip its CSVs to the scratch folder, convert them to a staged
    parquet file there, delete the CSVs. A staged quarter is reused when its zip, schema
    and CONVERTER_VERSION are unchanged, so a failed run resumes. Then the staged files
    are concatenated in date order into dataset/dataset.parquet, through a temporary file
    renamed into place: the rename replaces a symlink there rather than writing through it.
    """
    log = Progress(ctx)
    zips, out = ctx.inputs["zips"], ctx.outputs["panel"]
    schema_bytes = ctx.inputs["schema"].read_bytes()
    columns: dict[str, str] = tomllib.loads(schema_bytes.decode())["columns"]
    scratch = Path(ctx.params["scratch"])
    stage_dir, csv_root, spill = scratch / "stage", scratch / "csv", scratch / "duckdb"
    for d in (stage_dir, csv_root, spill):
        d.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    _require_free(scratch, ctx.params.get("scratch_free_gib", 0))
    _require_free(out.parent, ctx.params.get("panel_free_gib", 0))

    manifest = json.loads((zips / "manifest.json").read_text())
    # the spill folder is on the HDD: DuckDB asks for tens of GB of it, which the SSD can't spare
    con = connect(ctx.workdir, ctx.name, tmp=spill, ordered=True)
    schema_sha = hashlib.sha256(schema_bytes).hexdigest()
    staged, rows = [], 0
    for i, entry in enumerate(manifest["zips"], 1):
        path = zips / entry["name"]
        if path.stat().st_size != entry["bytes"]:
            raise ValueError(f"{path}: size differs from manifest.json; rerun panel_download")
        key = json.dumps({"zip": entry, "schema": schema_sha, "converter": CONVERTER_VERSION},
                         sort_keys=True)
        t0 = time.perf_counter()
        dest, n, reused = _stage_quarter(con, path, key, columns, stage_dir, csv_root)
        staged.append(dest)
        rows += n
        log(f"[{i}/{len(manifest['zips'])}] {path.stem}: {n:,} rows "
            f"{'(staged earlier)' if reused else f'in {time.perf_counter() - t0:.0f}s'}")

    part = out.with_name(f".{out.name}.part")
    part.unlink(missing_ok=True)
    log(f"writing {rows:,} rows to {out}")
    con.sql(f"""COPY (SELECT * FROM read_parquet([{', '.join(q(p) for p in staged)}]))
                TO {q(part)} (FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 122880)""")
    _check_panel(con, part, columns, rows)
    days = scalar(con, f"SELECT count(DISTINCT date) FROM read_parquet({q(part)})")
    con.close()
    os.replace(part, out)
    sha = _sha256(out)
    log(f"sha256 {sha}: pin it in pipeline.toml ([pipeline].fixed and drive_index)")
    for d in (stage_dir, csv_root, spill):
        shutil.rmtree(d, ignore_errors=True)
    return {"rows": rows, "days": days, "sha256": sha}


def _stage_quarter(con, path: Path, key: str, columns: dict[str, str], stage_dir: Path,
                   csv_root: Path) -> tuple[Path, int, bool]:
    """(staged parquet, rows, reused) for one quarter's zip."""
    stem = path.stem
    dest, key_file = stage_dir / f"{stem}.parquet", stage_dir / f"{stem}.key"
    if dest.is_file() and key_file.is_file() and key_file.read_text() == key:
        return dest, scalar(con, f"SELECT count(*) FROM read_parquet({q(dest)})"), True
    key_file.unlink(missing_ok=True)

    csv_dir = csv_root / stem
    shutil.rmtree(csv_dir, ignore_errors=True)
    csv_dir.mkdir(parents=True)
    files = _extract_days(path, csv_dir)
    src = (f"read_csv([{', '.join(q(f) for f in files)}], union_by_name = true, "
           "all_varchar = true, filename = true)")
    found = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {src}").fetchall()]
    unknown = [c for c in found if c not in columns and c != "filename"]
    if unknown:
        raise ValueError(f"{path.name}: columns not in config/panel-schema.toml: {unknown}")

    def expr(name: str, typ: str) -> str:
        if name == "filename":
            return f"'./raw-data/{stem}/' || parse_filename(filename) AS filename"
        if name not in found:
            return f'NULL::{typ} AS "{name}"'
        if name == "date":  # 2019-06-17.csv writes 6/17/19
            return ("coalesce(try_strptime(date, '%Y-%m-%d'), "
                    "try_strptime(date, '%m/%d/%y'))::DATE AS date")
        if typ == "VARCHAR":
            return f'"{name}"'
        return f'CAST("{name}" AS {typ}) AS "{name}"'  # strict: a bad value stops the run

    select = ", ".join(expr(n, t) for n, t in columns.items())
    part = dest.with_name(dest.name + ".part")
    con.sql(f"COPY (SELECT {select} FROM {src}) TO {q(part)} (FORMAT parquet, COMPRESSION zstd)")
    bad, n = con.sql(f"SELECT count(*) FILTER (WHERE date IS NULL), count(*) "
                     f"FROM read_parquet({q(part)})").fetchall()[0]
    if bad:
        raise ValueError(f"{path.name}: {bad} rows have a date in neither YYYY-MM-DD nor M/D/YY")
    os.replace(part, dest)
    key_file.write_text(key)
    shutil.rmtree(csv_dir)
    return dest, n, False


def _extract_days(path: Path, csv_dir: Path) -> list[Path]:
    """Unzip the daily CSVs flat into csv_dir, sorted by name; skip macOS metadata.

    Most zips carry an AppleDouble copy of every CSV (__MACOSX/._2019-04-01.csv), which is
    why the archives hold over 6,000 files for 3,652 days. Some zips nest the CSVs in a
    folder, so only the file name is kept.
    """
    days: dict[str, zipfile.ZipInfo] = {}
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            parts = PurePosixPath(info.filename).parts
            name = parts[-1] if parts else ""
            if info.is_dir() or "__MACOSX" in parts or name.startswith("."):
                continue
            if not name.endswith(".csv"):
                continue
            if not _DAY_CSV.match(name):
                raise ValueError(f"{path.name}: unexpected CSV {info.filename}")
            if name in days:
                raise ValueError(f"{path.name}: {name} appears twice")
            days[name] = info
        if not days:
            raise ValueError(f"{path.name}: no daily CSVs")
        for name in sorted(days):
            with zf.open(days[name]) as src, (csv_dir / name).open("wb") as dst:
                shutil.copyfileobj(src, dst, _CHUNK)  # zipfile checks each file's CRC
    return [csv_dir / name for name in sorted(days)]


def _check_panel(con, path: Path, columns: dict[str, str], rows: int) -> None:
    got = [(r[0], r[1]) for r in con.execute(
        f"DESCRIBE SELECT * FROM read_parquet({q(path)})").fetchall()]
    if got != list(columns.items()):
        raise ValueError(f"{path}: schema differs from config/panel-schema.toml")
    n = scalar(con, f"SELECT count(*) FROM read_parquet({q(path)})")
    if n != rows:
        raise ValueError(f"{path}: {n} rows, the staged quarters hold {rows}")


def _require_free(path: Path, gib: float) -> None:
    free = shutil.disk_usage(path).free / 2**30
    if free < gib:
        raise OSError(f"{path}: {free:.0f} GiB free, panel_convert needs {gib} GiB there")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()
