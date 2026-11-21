# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Split import gate tests

"""Exercise the split import gate on the real tree and on a small tracked fixture."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.sc_nc_import_gate import main

REPO = Path(__file__).resolve().parents[2]


def _fixture_repository(root: Path, split: dict[str, object], sources: dict[str, str]) -> Path:
    """Write a tracked fixture repository with a split map and source files.

    Parameters
    ----------
    root : Path
        An empty directory.
    split : dict[str, object]
        The split map to write.
    sources : dict[str, str]
        Repository-relative paths and their contents.

    Returns
    -------
    Path
        The path of the written split map.
    """
    for relative, text in sources.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    map_path = root / "split.json"
    map_path.write_text(json.dumps(split), encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    return map_path


SPLIT: dict[str, object] = {
    "order": ["SC-NC-A", "SC-NC-B"],
    "imports": {"SC-NC-A": [], "SC-NC-B": ["SC-NC-A"]},
    "facade_packages": ["face"],
    "facade_modules": ["__init__"],
    "circuit_side_packages": {"board": "CIRCUIT-X", "panel": "CIRCUIT-Y"},
    "packages": {"alpha": "SC-NC-A", "beta": "SC-NC-B", "omega": "SC-NC-B"},
    "modules": {"consts": "SC-NC-A", "late": "SC-NC-B"},
}

SOURCES = {
    "src/sc_neurocore/alpha/__init__.py": "",
    "src/sc_neurocore/alpha/up.py": "import sc_neurocore.beta.core\n",
    "src/sc_neurocore/alpha/facade_use.py": "from sc_neurocore import face\n",
    "src/sc_neurocore/alpha/board_use.py": "from sc_neurocore.board import pins\n",
    "src/sc_neurocore/alpha/own.py": "from . import up\nfrom sc_neurocore.alpha import own\n",
    "src/sc_neurocore/alpha/mod_use.py": "from sc_neurocore.late import y\n",
    "src/sc_neurocore/alpha/root_use.py": "import sc_neurocore\nfrom sc_neurocore import __version__\n",
    "src/sc_neurocore/beta/__init__.py": "",
    "src/sc_neurocore/beta/core.py": "from ..alpha import up\nimport os\n",
    "src/sc_neurocore/beta/deep/leaf.py": "from ...alpha.own import x\n",
    "src/sc_neurocore/face/__init__.py": "import sc_neurocore.beta\nimport sc_neurocore.alpha\n",
    "src/sc_neurocore/gamma/new.py": "x = 1\n",
    "src/sc_neurocore/version.py": "import sc_neurocore.beta\n",
    "src/sc_neurocore/__init__.py": "import sc_neurocore.beta\n",
    "src/sc_neurocore/consts.py": "from .late import y\n",
    "src/sc_neurocore/late.py": "from . import consts\n",
    "src/sc_neurocore/board/pins.py": (
        "import sc_neurocore.alpha\nfrom sc_neurocore import consts\nimport sc_neurocore.board\n"
    ),
    "src/sc_neurocore/panel/view.py": "import sc_neurocore.board\n",
    "src/sc_neurocore/panel/version_use.py": "from sc_neurocore import __version__\n",
    "src/sc_neurocore/alpha/data.txt": "import sc_neurocore.beta\n",
}


def test_the_real_tree_is_fully_mapped_and_reports_without_failing(tmp_path: Path) -> None:
    """Every tracked package of the real tree is placed, and report mode exits 0."""
    report_path = tmp_path / "report.json"
    completed = subprocess.run(
        [sys.executable, "tools/sc_nc_import_gate.py", "--output", str(report_path)],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema"] == "sc-neurocore.split-import-gate.v1"
    assert report["status"] == "report"
    assert report["unmapped_packages"] == []
    assert report["unmapped_modules"] == []
    assert report["stale_entries"] == []
    assert report["tracked_python_files"] > 1000
    split = json.loads((REPO / "tools/sc_nc_split_map.json").read_text(encoding="utf-8"))
    for finding in report["findings"]:
        if finding["source"].startswith("circuit side"):
            assert not finding["target"].startswith("circuit side")
            continue
        assert finding["source"] in split["order"]
        assert finding["target"] not in split["imports"][finding["source"]]
        assert finding["target"] != finding["source"]
    assert sum(report["findings_by_destination"].values()) + report["circuit_side_findings"] == len(
        report["findings"]
    )
    assert "status: report" in completed.stdout


def test_the_fixture_reports_each_crossing_and_nothing_allowed(tmp_path: Path) -> None:
    """Packages, modules, facade and circuit-side names are classified exactly."""
    map_path = _fixture_repository(tmp_path, SPLIT, SOURCES)
    report_path = tmp_path / "report.json"
    assert (
        main(["--root", str(tmp_path), "--map", str(map_path), "--output", str(report_path)]) == 0
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    found = {
        (item["file"], item["line"], item["source"], item["imports"], item["target"])
        for item in report["findings"]
    }
    board = "circuit side (CIRCUIT-X)"
    root, facade = "sc_neurocore.__init__", "SC-NeuroCore facade"
    assert found == {
        ("src/sc_neurocore/alpha/up.py", 1, "SC-NC-A", "sc_neurocore.beta", "SC-NC-B"),
        (
            "src/sc_neurocore/alpha/facade_use.py",
            1,
            "SC-NC-A",
            "sc_neurocore.face",
            "SC-NeuroCore facade",
        ),
        ("src/sc_neurocore/alpha/board_use.py", 1, "SC-NC-A", "sc_neurocore.board", board),
        ("src/sc_neurocore/alpha/mod_use.py", 1, "SC-NC-A", "sc_neurocore.late", "SC-NC-B"),
        ("src/sc_neurocore/consts.py", 1, "SC-NC-A", "sc_neurocore.late", "SC-NC-B"),
        ("src/sc_neurocore/board/pins.py", 1, board, "sc_neurocore.alpha", "SC-NC-A"),
        ("src/sc_neurocore/board/pins.py", 2, board, "sc_neurocore.consts", "SC-NC-A"),
        ("src/sc_neurocore/alpha/root_use.py", 1, "SC-NC-A", root, facade),
        ("src/sc_neurocore/alpha/root_use.py", 2, "SC-NC-A", root, facade),
        ("src/sc_neurocore/panel/version_use.py", 1, "circuit side (CIRCUIT-Y)", root, facade),
    }
    assert report["findings_by_destination"] == {"SC-NC-A": 7, "SC-NC-B": 0}
    assert report["circuit_side_findings"] == 3
    assert report["unmapped_packages"] == ["gamma"]
    assert report["unmapped_modules"] == ["version"]
    assert report["stale_entries"] == ["omega"]
    assert report["tracked_python_files"] == 19
    assert report["status"] == "report"


def test_blocking_fails_only_a_destination_with_findings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A blocking destination with findings exits 1; a clean one passes."""
    map_path = _fixture_repository(tmp_path, SPLIT, SOURCES)
    assert main(["--root", str(tmp_path), "--map", str(map_path), "--blocking", "SC-NC-B"]) == 0
    assert "status: pass" in capsys.readouterr().out
    assert main(["--root", str(tmp_path), "--map", str(map_path), "--blocking", "SC-NC-A"]) == 1
    out = capsys.readouterr().out
    assert "status: fail" in out
    assert "unmapped packages: gamma" in out
    assert "unmapped modules: version" in out
    assert "map entries without a tracked file: omega" in out
    assert "circuit side: 3 import(s) of SC-NeuroCore code" in out


def test_a_fully_mapped_tree_prints_no_unmapped_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without an unmapped package the summary names none."""
    sources = {path: text for path, text in SOURCES.items() if "/gamma/" not in path}
    map_path = _fixture_repository(tmp_path, SPLIT, sources)
    assert main(["--root", str(tmp_path), "--map", str(map_path)]) == 0
    out = capsys.readouterr().out
    assert "unmapped packages" not in out
    assert "SC-NC-A: 7 import(s) outside its column" in out


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"order": []}, "must list each destination once"),
        ({"order": ["SC-NC-A", "SC-NC-A"]}, "must list each destination once"),
        ({"imports": {"SC-NC-A": []}}, "SC-NC-B has no import column"),
        (
            {"imports": {"SC-NC-A": ["SC-NC-B"], "SC-NC-B": []}},
            "may only import earlier destinations",
        ),
        ({"imports": {"SC-NC-A": [], "SC-NC-B": [], "SC-NC-Z": []}}, "outside the order"),
        ({"packages": {"alpha": "SC-NC-Q"}}, "unknown destinations"),
        ({"facade_packages": ["alpha"]}, "placed more than once"),
        ({"circuit_side_packages": {"face": "CIRCUIT-X"}}, "placed more than once"),
        ({"modules": {"alpha": "SC-NC-A"}}, "placed more than once"),
        ({"facade_modules": ["consts"]}, "placed more than once"),
        ({"modules": {"consts": "SC-NC-Q"}}, "unknown destinations"),
    ],
)
def test_a_malformed_map_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], change: dict[str, object], message: str
) -> None:
    """Every map defect exits 2 and names itself."""
    map_path = _fixture_repository(tmp_path, {**SPLIT, **change}, SOURCES)
    assert main(["--root", str(tmp_path), "--map", str(map_path)]) == 2
    assert message in capsys.readouterr().err


def test_an_unreadable_map_and_an_unknown_blocking_name_are_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A missing or invalid map file and an unknown --blocking name exit 2."""
    map_path = _fixture_repository(tmp_path, SPLIT, SOURCES)
    assert main(["--root", str(tmp_path), "--map", str(tmp_path / "absent.json")]) == 2
    assert "cannot read the split map" in capsys.readouterr().err
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    assert main(["--root", str(tmp_path), "--map", str(tmp_path / "bad.json")]) == 2
    assert "cannot read the split map" in capsys.readouterr().err
    assert main(["--root", str(tmp_path), "--map", str(map_path), "--blocking", "SC-NC-Q"]) == 2
    assert "--blocking names unknown destinations" in capsys.readouterr().err


def test_an_empty_measurement_fails_closed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A root that is not a work tree, or tracks no source file, exits 2 and never passes."""
    # Git must not find a work tree above the temporary directory.
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    outside = tmp_path / "plain"
    outside.mkdir()
    map_path = outside / "split.json"
    map_path.write_text(json.dumps(SPLIT), encoding="utf-8")
    assert main(["--root", str(outside), "--map", str(map_path), "--blocking", "SC-NC-A"]) == 2
    assert "cannot list tracked files" in capsys.readouterr().err
    empty = tmp_path / "empty"
    empty.mkdir()
    untracked = _fixture_repository(empty, SPLIT, {"README.md": "no sources\n"})
    (empty / "src/sc_neurocore/alpha").mkdir(parents=True)
    (empty / "src/sc_neurocore/alpha/untracked.py").write_text("import sc_neurocore.beta\n")
    assert main(["--root", str(empty), "--map", str(untracked), "--blocking", "SC-NC-A"]) == 2
    assert "no tracked Python file under src/sc_neurocore" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("relative", "content"),
    [
        ("src/sc_neurocore/alpha/broken.py", b"def broken(:\n"),
        ("src/sc_neurocore/alpha/latin.py", b"name = '\xff'\n"),
        ("src/sc_neurocore/alpha/nul.py", b"x = 1\x00\n"),
    ],
)
def test_an_unparsable_tracked_file_is_refused_by_name(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], relative: str, content: bytes
) -> None:
    """A tracked file that does not decode or parse exits 2 and names itself."""
    map_path = _fixture_repository(tmp_path, SPLIT, SOURCES)
    (tmp_path / relative).write_bytes(content)
    subprocess.run(["git", "-C", str(tmp_path), "add", relative], check=True)
    assert main(["--root", str(tmp_path), "--map", str(map_path)]) == 2
    assert f"{relative}: cannot read or parse a tracked file" in capsys.readouterr().err


def test_a_tracked_file_missing_from_the_work_tree_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A file that is tracked but deleted exits 2 instead of being skipped."""
    map_path = _fixture_repository(tmp_path, SPLIT, SOURCES)
    (tmp_path / "src/sc_neurocore/beta/core.py").unlink()
    assert main(["--root", str(tmp_path), "--map", str(map_path)]) == 2
    assert "src/sc_neurocore/beta/core.py: cannot read or parse" in capsys.readouterr().err


def test_a_map_with_a_repeated_key_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """JSON keeps the last of repeated keys silently; the map refuses them."""
    map_path = _fixture_repository(tmp_path, SPLIT, SOURCES)
    text = json.dumps(SPLIT).replace('"alpha": "SC-NC-A"', '"alpha": "SC-NC-A", "alpha": "SC-NC-B"')
    map_path.write_text(text, encoding="utf-8")
    assert main(["--root", str(tmp_path), "--map", str(map_path)]) == 2
    assert "duplicate key(s) ['alpha']" in capsys.readouterr().err
