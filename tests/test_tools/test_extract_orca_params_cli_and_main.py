# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (cli_and_main) from former test_extract_orca_params.py

"""Command line and entry point of the ORCA parameter extractor."""

from __future__ import annotations

import json
import pytest
import subprocess
import sys
from extract_orca_params_support import (
    _full_output,
    _load_tool,
    _write_output,
    REPO,
    TOOL,
)
from pathlib import (
    Path,
)


def test_cli_writes_deterministic_json_file(tmp_path: Path) -> None:
    """The command line writes the same JSON file on two runs."""
    out = _write_output(tmp_path / "run.out", _full_output())
    target = tmp_path / "nested" / "params.json"

    result = subprocess.run(
        [sys.executable, str(TOOL), "--input", str(out), "--output", str(target)],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=REPO,
    )

    assert result.returncode == 0, result.stderr
    first = target.read_bytes()
    subprocess.run(
        [sys.executable, str(TOOL), "--input", str(out), "--output", str(target)],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=REPO,
    )
    assert target.read_bytes() == first
    payload = json.loads(first)
    assert payload["hyperfine"]["nucleus_count"] == 2


def test_cli_stdout_when_no_output(tmp_path: Path) -> None:
    """Without an output path the command line prints the JSON record."""
    out = _write_output(tmp_path / "run.out", _full_output())

    result = subprocess.run(
        [sys.executable, str(TOOL), "--input", str(out)],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=REPO,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["final_single_point_energy_eh"] == pytest.approx(-9953.726192774189)


def test_cli_fails_closed_without_writing(tmp_path: Path) -> None:
    """A run without the termination marker fails and writes no file."""
    out = _write_output(tmp_path / "run.out", _full_output(terminated=False))
    target = tmp_path / "params.json"

    result = subprocess.run(
        [sys.executable, str(TOOL), "--input", str(out), "--output", str(target)],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=REPO,
    )

    assert result.returncode == 1
    assert not target.exists()
    assert "normal-termination marker not found" in result.stderr


def test_main_writes_file_and_extra_source(tmp_path: Path) -> None:
    """The entry point writes the record and lists the extra provenance source."""
    tool = _load_tool()
    out = _write_output(tmp_path / "run.out", _full_output())
    extra = tmp_path / "REPRO.md"
    extra.write_text("log\n", encoding="utf-8")
    target = tmp_path / "out" / "params.json"

    code = tool.main(["--input", str(out), "--output", str(target), "--source", str(extra)])

    assert code == 0
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert [s["role"] for s in payload["provenance"]["sources"]] == [
        "orca_output",
        "provenance",
    ]


def test_main_writes_to_stdout(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Without an output path the entry point prints the JSON record."""
    tool = _load_tool()
    out = _write_output(tmp_path / "run.out", _full_output())

    code = tool.main(["--input", str(out)])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["g_tensor"]["g_isotropic"] == pytest.approx(2.0438125)


def test_main_custom_required_element_override(tmp_path: Path) -> None:
    """A required-element override lets a run without calcium pass."""
    tool = _load_tool()
    out = _write_output(tmp_path / "run.out", _full_output(ca_block=""))

    code = tool.main(["--input", str(out), "--require-element", "P"])

    assert code == 0


def test_main_returns_one_and_writes_nothing_on_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """On an extraction error the entry point returns one and writes nothing."""
    tool = _load_tool()
    out = _write_output(tmp_path / "run.out", _full_output(terminated=False))
    target = tmp_path / "params.json"

    code = tool.main(["--input", str(out), "--output", str(target)])

    assert code == 1
    assert not target.exists()
    assert "normal-termination marker not found" in capsys.readouterr().err
