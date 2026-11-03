# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native Go profile preservation and productive refusal contracts

"""Exercise public profile admission with real native inputs and damaged copies.

The compressed inputs retain complete historical native reports. They qualify
parser behavior, without claiming a new model run or current native coverage.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools.go_coverage_profiles import compare_profiles, diagnostic_statements, read_profile

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/go_mat_coverage"
_DIGESTS = {
    "before": "f413651868f79c8bf4d8a9bf528e49e004178f6222f318962faf40d8ed9e14c3",
    "complete": "2dc6b2ad8a6440154528984897409ea021995b62e4265b0b8c3d7c49625e9c4a",
}


@pytest.fixture
def profiles(tmp_path: Path) -> tuple[Path, Path]:
    """Restore byte-identical original native inputs in an owned test allocation."""
    paths = []
    for name, expected in _DIGESTS.items():
        data = gzip.decompress((_FIXTURES / (name + ".txt.gz")).read_bytes())
        assert hashlib.sha256(data).hexdigest() == expected
        path = tmp_path / (name + ".txt")
        path.write_bytes(data)
        paths.append(path)
    return paths[0], paths[1]


def test_complete_native_profiles_preserve_all_blocks(profiles: tuple[Path, Path]) -> None:
    """The actual reports retain every native block and four full statement totals."""
    result = compare_profiles(*profiles)
    assert result["passed"] is True
    assert result["native_blocks_retained"] == 4328
    assert result["target_blocks"] == 82
    assert result["branch_or_mcdc_qualification"] is False
    assert result["all_function_qualification"] is False
    sources = result["sources"]
    assert isinstance(sources, list)
    assert [source["total_statements"] for source in sources] == [50, 45, 26, 23]
    assert sum(source["zero_statement_blocks"] for source in sources) == 2


def test_native_before_work_is_a_real_uncovered_refusal(profiles: tuple[Path, Path]) -> None:
    """The actual zero-work profile cannot become a successful statement verdict."""
    with pytest.raises(ValueError, match="Uncovered native statements"):
        compare_profiles(profiles[0], profiles[0])


def test_before_work_cannot_be_replaced_by_executed_counters(profiles: tuple[Path, Path]) -> None:
    """Replacing both inputs with the actual completed report remains a refusal."""
    with pytest.raises(ValueError, match="Before-work subject was already executed"):
        compare_profiles(profiles[1], profiles[1])


@pytest.mark.parametrize("damage", ["empty", "header", "mode", "syntax", "duplicate"])
def test_damaged_native_text_refuses(
    profiles: tuple[Path, Path], tmp_path: Path, damage: str
) -> None:
    """Truncated, changed-mode, malformed and duplicate native records fail closed."""
    lines = profiles[1].read_text().splitlines()
    if damage == "empty":
        lines = []
    elif damage == "header":
        lines = lines[:1]
    elif damage == "mode":
        lines[0] = "mode: set"
    elif damage == "syntax":
        lines[1] += " unexpected"
    else:
        lines.append(lines[1])
    path = tmp_path / "damaged.txt"
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError):
        read_profile(path)


@pytest.mark.parametrize("damage", ["nul", "backslash", "parent", "zero", "reversed"])
def test_damaged_source_geometry_refuses(
    profiles: tuple[Path, Path], tmp_path: Path, damage: str
) -> None:
    """Faults in actual native source names or one-based coordinates are rejected."""
    lines = profiles[1].read_text().splitlines()
    filename, remainder = lines[1].split(":", 1)
    geometry, statements, count = remainder.split()
    if damage == "nul":
        filename += "\x00"
    elif damage == "backslash":
        filename = filename.replace("/", "\\", 1)
    elif damage == "parent":
        filename = filename.replace("/mat/", "/../mat/", 1)
    elif damage == "zero":
        geometry = "0.1,1.1"
    else:
        geometry = "2.1,1.1"
    lines[1] = f"{filename}:{geometry} {statements} {count}"
    path = tmp_path / "geometry.txt"
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="Invalid native source geometry"):
        read_profile(path)


@pytest.mark.parametrize("damage", ["missing-block", "changed-denominator", "zero-counter"])
def test_changed_or_uncovered_native_universe_refuses(
    profiles: tuple[Path, Path], damage: str
) -> None:
    """A genuine completed report cannot pass after dropping or damaging one block."""
    lines = profiles[1].read_text().splitlines()
    if damage == "missing-block":
        del lines[1]
    else:
        geometry, statements, count = lines[1].split()
        if damage == "changed-denominator":
            statements = str(int(statements) + 1)
        else:
            count = "0"
        lines[1] = f"{geometry} {statements} {count}"
    profiles[1].write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError):
        compare_profiles(*profiles)


def test_omitting_a_subject_from_both_reports_cannot_shrink_scope(
    profiles: tuple[Path, Path],
) -> None:
    """Equal before/after omissions do not silently remove an original source owner."""
    for path in profiles:
        lines = path.read_text().splitlines()
        path.write_text("\n".join(line for line in lines if "/services/mat.go:" not in line) + "\n")
    with pytest.raises(ValueError, match="Native subject absent or statement scope shrunk"):
        compare_profiles(*profiles)


def test_missing_native_report_is_not_an_empty_success(tmp_path: Path) -> None:
    """A missing artifact propagates its real file-read failure."""
    with pytest.raises(FileNotFoundError):
        read_profile(tmp_path / "missing.txt")


def test_historical_model_result_cannot_qualify_new_diagnostic_code(
    profiles: tuple[Path, Path],
) -> None:
    """The old complete model profile cannot pass current snapshot-source coverage."""
    with pytest.raises(ValueError, match="Missing or uncovered native diagnostic statements"):
        diagnostic_statements(profiles[1])


@pytest.mark.parametrize("scenario", ["complete", "before", "missing", "malformed", "collision"])
def test_public_profile_cli_retains_real_verdicts_and_output_custody(
    profiles: tuple[Path, Path], tmp_path: Path, scenario: str
) -> None:
    """The real public process exposes native refusals and never replaces a prior result."""
    before, after = profiles
    if scenario == "before":
        after = before
    elif scenario == "missing":
        after = tmp_path / "missing.txt"
    elif scenario == "malformed":
        after.write_text("mode: atomic\ninvalid native row\n")
    output = tmp_path / "result.json"
    if scenario == "collision":
        output.write_bytes(b"retained result\n")
    tool = Path(__file__).resolve().parents[2] / "tools/go_coverage_profiles.py"
    actual = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            str(tool),
            "--before",
            str(before),
            "--after",
            str(after),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert actual.returncode == (0 if scenario == "complete" else 1)
    if scenario == "collision":
        assert output.read_bytes() == b"retained result\n"
        assert "FileExistsError" in actual.stderr
        return
    result = json.loads(output.read_text())
    assert result["passed"] is (scenario == "complete")
    if scenario in {"missing", "malformed"}:
        assert result["observed_after"] is None
        assert "observation_error" in result
    else:
        observed = result["observed_after"]["sources"]
        assert [row["total_statements"] for row in observed] == [50, 45, 26, 23]
        assert sum(row["covered_statements"] for row in observed) == (
            144 if scenario == "complete" else 0
        )
