# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Actual native gate artifact and budget fixtures

"""Bind dedicated contracts to this gate's real native artifacts and deadline."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import cast

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.go_mat_coverage_build import NativeGo


def pytest_addoption(parser: pytest.Parser) -> None:
    """Accept explicit current job and source inputs for the dedicated native contracts."""
    parser.addoption("--native-go-job", type=Path)
    parser.addoption("--native-go-repository", type=Path)
    parser.addoption("--native-go-python-report", type=Path)
    parser.addoption("--native-go-report-repository", type=Path)


@pytest.fixture(scope="session")
def native_job(pytestconfig: pytest.Config) -> Path:
    """Require a real producer record and completed native statement verdict."""
    value = pytestconfig.getoption("--native-go-job")
    if not isinstance(value, Path):
        raise ValueError("Dedicated native contracts require their actual gate allocation")
    job = value.resolve(strict=True)
    producer = json.loads((job / "producer-inputs.json").read_text())
    verdict = json.loads((job / "artifacts/complete-statement-verdict.json").read_text())
    if producer["resource_admission_claimed"] is not False or verdict["passed"] is not True:
        raise ValueError("The actual native producer/verdict inputs differ")
    return job


@pytest.fixture(scope="session")
def native_repository(pytestconfig: pytest.Config, native_job: Path) -> Path:
    """Bind all contract imports to the same actual repository used by the producer."""
    value = pytestconfig.getoption("--native-go-repository")
    if not isinstance(value, Path):
        raise ValueError("Dedicated native contracts require the actual repository")
    root = value.resolve(strict=True)
    producer = json.loads((native_job / "producer-inputs.json").read_text())
    if str(root) != producer["repository"]:
        raise ValueError("Dedicated contracts came from a different source checkout")
    return root


@pytest.fixture
def native_builder(tmp_path: Path, native_job: Path, native_repository: Path) -> NativeGo:
    """Give each actual command test exclusive outputs inside the original admitted deadline."""
    producer = json.loads((native_job / "producer-inputs.json").read_text())
    job = tmp_path / "producer"
    job.mkdir()
    return NativeGo(
        job,
        native_repository,
        Path(producer["go"]),
        float(producer["deadline_monotonic"]),
        int(producer["jobs"]),
        cast(dict[str, str], producer["environment"]),
        float(producer["stop_seconds"]),
        Path(producer["python_coverage_config"]),
    )
