"""panel_download and panel_convert on tiny zips shaped like Backblaze's, quirks included."""

from __future__ import annotations

import json
import re
import tomllib
import zipfile
from pathlib import Path, PurePosixPath

import duckdb
import pytest

import raw
from data_pipeline.context import StepContext

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = ROOT / "config" / "panel-schema.toml"

HEAD_2016 = ("date,serial_number,model,capacity_bytes,failure,"
             "smart_1_normalized,smart_1_raw,smart_9_raw")
HEAD_LATER = HEAD_2016 + ",smart_7_raw,is_legacy_format"


def write_zip(path: Path, files: dict[str, str], header: str | None = None) -> None:
    """A zip with `files`, padded with header-only CSVs for the quarter's other days."""
    days = {PurePosixPath(n).name for n in files}
    pad = {d: header + "\n" for d in raw.quarter_days(path.stem) if d not in days} if header else {}
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, text in {**files, **pad}.items():
            zf.writestr(name, text)


@pytest.fixture
def source(tmp_path):
    """A fake bucket: two quarters, with AppleDouble junk, a nested folder, a CR-only file
    with M/D/YY dates, a header-only day and a value in scientific notation."""
    src = tmp_path / "bucket"
    src.mkdir()
    write_zip(src / "data_Q1_2016.zip", {
        "2016-01-02.csv": f"{HEAD_2016}\n2016-01-02,A,M1,4000787030016,0,100,0,10\n"
                          "2016-01-02,B,M2,-1,1,98,5,20\n",
        "2016-01-01.csv": f"{HEAD_2016}\n2016-01-01,A,M1,4000787030016,0,100,0,9\n",
        "__MACOSX/._2016-01-01.csv": "\x00\x05\x16\x07 mac metadata",
        ".DS_Store": "junk",
    }, HEAD_2016)
    write_zip(src / "data_Q2_2016.zip", {
        "data_Q2_2016/2016-04-01.csv": "\r".join([
            HEAD_LATER, "4/1/16,A,M1,4000787030016,0,100,0,99,2.81472E+14,False",
            "3/31/16,C,M3,4000787030016,1,90,7,50,,True"]) + "\r",
        "data_Q2_2016/2016-04-02.csv": HEAD_LATER + "\n",
        "__MACOSX/data_Q2_2016/._2016-04-01.csv": "junk",
    }, HEAD_LATER)
    return src


def ctx(tmp_path: Path, name: str, inputs=None, outputs=None, params=None) -> StepContext:
    return StepContext(name=name, inputs=inputs or {}, outputs=outputs or {},
                       params=params or {}, workdir=tmp_path)


def download(tmp_path, source):
    zips = tmp_path / "hdd" / "zips"
    summary = raw.download(ctx(tmp_path, "panel_download", outputs={"zips": zips},
                               params={"base_url": source.as_uri(), "first": "2016Q1",
                                       "last": "2016Q2"}))
    return zips, summary


def convert(tmp_path, zips, out=None):
    out = out or tmp_path / "dataset" / "dataset.parquet"
    summary = raw.convert(ctx(tmp_path, "panel_convert",
                              inputs={"zips": zips, "schema": SCHEMA}, outputs={"panel": out},
                              params={"scratch": str(tmp_path / "hdd" / "scratch")}))
    return out, summary


@pytest.fixture(autouse=True)
def few_threads(monkeypatch):
    monkeypatch.setenv("HDA_THREADS", "2")


def test_quarters():
    assert raw.quarters("2016Q3", "2017Q1") == ["data_Q3_2016.zip", "data_Q4_2016.zip",
                                                "data_Q1_2017.zip"]
    with pytest.raises(ValueError, match="after"):
        raw.quarters("2017Q1", "2016Q4")
    with pytest.raises(ValueError, match="2016Q1"):
        raw.quarters("2016-1", "2016Q4")


def test_download_writes_manifest_and_resumes(tmp_path, source):
    zips, summary = download(tmp_path, source)
    assert summary["zips"] == 2
    manifest = json.loads((zips / "manifest.json").read_text())
    assert [e["name"] for e in manifest["zips"]] == ["data_Q1_2016.zip", "data_Q2_2016.zip"]
    good = (zips / "data_Q1_2016.zip").read_bytes()
    (zips / "data_Q1_2016.zip").write_bytes(b"x" * len(good))  # right size, not a zip
    download(tmp_path, source)
    assert (zips / "data_Q1_2016.zip").read_bytes() == good
    mtime = (zips / "data_Q1_2016.zip").stat().st_mtime_ns
    (zips / "data_Q2_2016.zip").write_bytes(b"truncated")
    (zips / "data_Q2_2016.zip.part").write_bytes(b"left by a crash")
    download(tmp_path, source)
    assert (zips / "data_Q1_2016.zip").stat().st_mtime_ns == mtime  # complete: not fetched again
    assert zipfile.is_zipfile(zips / "data_Q2_2016.zip")
    assert not list(zips.glob("*.part"))
    log = (tmp_path / "build" / "logs" / "panel_download.log").read_text()
    assert "data_Q1_2016.zip 0.00 GiB already complete" in log


def test_convert_matches_schema_and_keeps_quirks(tmp_path, source):
    zips, _ = download(tmp_path, source)
    out, summary = convert(tmp_path, zips)
    columns = tomllib.loads(SCHEMA.read_text())["columns"]
    con = duckdb.connect()
    got = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{out}')").fetchall()
    assert [(r[0], r[1]) for r in got] == list(columns.items())
    rows = con.execute(f"""SELECT date::VARCHAR, serial_number, capacity_bytes, smart_9_raw,
                                  smart_7_raw, is_legacy_format, filename
                           FROM read_parquet('{out}')""").fetchall()
    q1, q2 = "./raw-data/data_Q1_2016/", "./raw-data/data_Q2_2016/"
    cap = 4000787030016.0
    assert rows == [  # file order, rows in file order; the CR file's 3/31/16 row kept as is
        ("2016-01-01", "A", cap, 9, None, None, q1 + "2016-01-01.csv"),
        ("2016-01-02", "A", cap, 10, None, None, q1 + "2016-01-02.csv"),
        ("2016-01-02", "B", -1.0, 20, None, None, q1 + "2016-01-02.csv"),
        ("2016-04-01", "A", cap, 99, 281472000000000.0, False, q2 + "2016-04-01.csv"),
        ("2016-03-31", "C", cap, 50, None, True, q2 + "2016-04-01.csv"),
    ]
    assert summary["rows"] == 5 and summary["days"] == 4
    assert re.fullmatch(r"[0-9a-f]{64}", summary["sha256"])
    scratch = tmp_path / "hdd" / "scratch"
    assert not any(scratch.iterdir())  # CSVs, spill and staged quarters cleaned up
    assert not list(out.parent.glob(".*.part"))


def test_rebuild_is_byte_identical(tmp_path, source):
    """The pin in pipeline.toml assumes the same zips always give the same panel."""
    zips, _ = download(tmp_path, source)
    _, first = convert(tmp_path, zips)
    _, second = convert(tmp_path, zips)
    assert first["sha256"] == second["sha256"]


def test_stale_part_does_not_count_against_free_space(tmp_path, source, monkeypatch):
    zips, _ = download(tmp_path, source)
    part = tmp_path / "dataset" / ".dataset.parquet.part"
    part.parent.mkdir()
    part.write_bytes(b"left by a crash")
    checked = []

    def require_free(path, gib):
        assert not part.exists(), "free space checked while a crashed run's part remains"
        checked.append(path)

    monkeypatch.setattr(raw, "_require_free", require_free)
    convert(tmp_path, zips)
    assert len(checked) == 2 and not part.exists()


def test_too_little_free_space_stops(tmp_path):
    with pytest.raises(OSError, match="panel_convert needs"):
        raw._require_free(tmp_path, 10**9)


def test_missing_day_stops(tmp_path, source):
    write_zip(source / "data_Q2_2016.zip", {f"2016-0{m}-0{d}.csv": HEAD_LATER + "\n"
                                            for m in (4, 5) for d in (1, 2)})
    zips, _ = download(tmp_path, source)
    with pytest.raises(ValueError, match="missing .'2016-04-03.csv'"):
        convert(tmp_path, zips)


def test_unknown_column_stops_and_staged_quarters_are_reused(tmp_path, source):
    good = (source / "data_Q2_2016.zip").read_bytes()
    write_zip(source / "data_Q2_2016.zip", {"2016-04-01.csv": HEAD_LATER + ",smart_999_raw\n"},
              HEAD_LATER)
    zips, _ = download(tmp_path, source)
    with pytest.raises(ValueError, match="smart_999_raw"):
        convert(tmp_path, zips)
    assert (tmp_path / "hdd" / "scratch" / "stage" / "data_Q1_2016.parquet").is_file()

    (source / "data_Q2_2016.zip").write_bytes(good)
    download(tmp_path, source)
    convert(tmp_path, zips)
    log = (tmp_path / "build" / "logs" / "panel_convert.log").read_text()
    assert "data_Q1_2016: 3 rows (staged earlier)" in log


def test_bad_value_in_a_typed_column_stops(tmp_path, source):
    write_zip(source / "data_Q2_2016.zip", {
        "2016-04-01.csv": f"{HEAD_2016}\n2016-04-01,A,M1,4000787030016,0,100,0,lots\n"},
        HEAD_2016)
    zips, _ = download(tmp_path, source)
    with pytest.raises(duckdb.ConversionException):
        convert(tmp_path, zips)


def test_fraction_in_an_integer_column_stops(tmp_path, source):
    write_zip(source / "data_Q2_2016.zip", {
        "2016-04-01.csv": f"{HEAD_2016}\n2016-04-01,A,M1,4000787030016,0,100,0,1.5\n"},
        HEAD_2016)
    zips, _ = download(tmp_path, source)
    with pytest.raises(duckdb.Error, match="smart_9_raw = 1.5 is not an integer"):
        convert(tmp_path, zips)


def test_unparseable_date_stops(tmp_path, source):
    write_zip(source / "data_Q2_2016.zip", {
        "2016-04-01.csv": f"{HEAD_2016}\n01.04.2016,A,M1,4000787030016,0,100,0,9\n"},
        HEAD_2016)
    zips, _ = download(tmp_path, source)
    with pytest.raises(ValueError, match="neither YYYY-MM-DD nor M/D/YY"):
        convert(tmp_path, zips)


def test_quarter_days():
    assert len(raw.quarter_days("data_Q1_2016")) == 91  # leap year
    assert raw.quarter_days("data_Q4_2025")[-1] == "2025-12-31.csv"
    assert sum(len(raw.quarter_days(z[:-4])) for z in raw.quarters("2016Q1", "2025Q4")) == 3653


def test_zip_layout_checks(tmp_path):
    for files, message in [({"notes.csv": "x"}, "unexpected CSV"),
                           ({"a/2016-01-01.csv": "x", "b/2016-01-01.csv": "x"}, "appears twice"),
                           ({"__MACOSX/._2016-01-01.csv": "x"}, "missing")]:
        z = tmp_path / "data_Q1_2016.zip"
        write_zip(z, files)
        (tmp_path / "out").mkdir(exist_ok=True)
        with pytest.raises(ValueError, match=message):
            raw._extract_days(z, tmp_path / "out")


def test_pipeline_pins_agree():
    """[pipeline].fixed and drive_index pin the same panel."""
    cfg = tomllib.loads((ROOT / "pipeline.toml").read_text())
    pins = {f["sha256"] for f in cfg["pipeline"]["fixed"]}
    drive_index = next(s for s in cfg["steps"] if s["name"] == "drive_index")
    assert pins == {drive_index["params"]["sha256"]}
    once = [s["name"] for s in cfg["steps"] if s.get("once")]
    assert once == ["panel_download", "panel_convert"]
    for s in cfg["steps"]:
        if s["name"] in once:
            paths = [*s.get("outputs", {}).values(),
                     *(v for k, v in s.get("params", {}).items() if k == "scratch")]
            for p in paths:  # big files stay on the download HDD, the panel goes to dataset/
                assert p.startswith("/srv/downloads/") or p == "dataset/dataset.parquet", p
