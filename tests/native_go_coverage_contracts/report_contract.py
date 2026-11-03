# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Real own-code coverage report conservation

"""Consume genuine collected reports after execution without synthetic successful counters."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import cast

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.go_coverage_trace import check_report_membership, expected_sources


@pytest.fixture
def actual_report(pytestconfig: pytest.Config) -> tuple[Path, Path]:
    """Require explicit real counter-tool output and its actual source repository."""
    report = pytestconfig.getoption("--native-go-python-report")
    repository = pytestconfig.getoption("--native-go-report-repository")
    if not isinstance(report, Path) or not isinstance(repository, Path):
        raise ValueError("Report contracts require actual post-execution JSON and source paths")
    return report.resolve(strict=True), repository.resolve(strict=True)


def test_actual_python_report_keeps_the_entire_own_source_cohort(
    actual_report: tuple[Path, Path],
) -> None:
    """Actual file identities stay complete without claiming a passing counter total."""
    report, repository = actual_report
    observed = check_report_membership(report, repository)
    assert set(cast(list[str], observed["actual_sources"])) == set(expected_sources(repository))
    assert observed["counter_qualification_claimed"] is False


@pytest.mark.parametrize(
    "fault", ["shape", "mapping", "missing", "duplicate", "foreign", "excluded"]
)
def test_changed_actual_report_cannot_drop_duplicate_or_relabel_source(
    actual_report: tuple[Path, Path],
    tmp_path: Path,
    fault: str,
) -> None:
    """Damaged real JSON copies cannot change the measured source membership."""
    report, repository = actual_report
    raw = json.loads(report.read_text())
    if fault == "shape":
        raw = []
    elif fault == "mapping":
        raw["files"] = []
    else:
        files = raw["files"]
        original = next(iter(files))
        if fault == "missing":
            files.pop(original)
        elif fault == "excluded":
            files[original]["excluded_lines"] = [files[original]["executed_lines"][0]]
        elif fault == "duplicate":
            source = Path(original)
            actual = source if source.is_absolute() else repository / source
            files[str(actual.parent) + "/./" + actual.name] = files[original]
        else:
            files[str(report)] = files.pop(original)
    copied = tmp_path / "changed-report.json"
    copied.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        check_report_membership(copied, repository)
