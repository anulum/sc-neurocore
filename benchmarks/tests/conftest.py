# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Explicit original benchmark evidence for consumer tests

"""Require actual producer outputs for the standalone benchmark consumer suite."""

from pathlib import Path

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register paths to genuine capture outputs and their source checkout."""
    group = parser.getgroup("Lapicque timing evidence")
    for name, help_text in (
        ("samples", "Original duration-retention JSON from the complete capture CLI"),
        ("report", "Original report JSON from the same complete five-language run"),
        ("checkout", "Checkout containing the exact original benchmark sources"),
    ):
        group.addoption("--lapicque-" + name, type=Path, help=help_text)


@pytest.fixture(scope="session")
def original_lapicque_evidence(pytestconfig: pytest.Config) -> tuple[Path, Path, Path]:
    """Read required real producer inputs without creating substitute measurements."""
    paths: list[Path] = []
    for name in ("samples", "report", "checkout"):
        value = pytestconfig.getoption("lapicque_" + name)
        if not isinstance(value, Path):
            raise pytest.UsageError("provide --lapicque-" + name + " from a real capture")
        paths.append(value.resolve(strict=True))
    return paths[0], paths[1], paths[2]
