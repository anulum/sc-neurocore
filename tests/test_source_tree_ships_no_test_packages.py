# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — The installed package carries no test package

"""Keep test suites in ``tests/``, where CI collects them, and out of the wheel.

Tests once lived in ``src/sc_neurocore/quantum_cognition/tests``. Pytest only
collects ``tests/``, so those 88 cases never ran in CI, and the package
discovery shipped them in every wheel.
"""

from __future__ import annotations

from pathlib import Path
import sys

from setuptools import find_packages

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
TEST_PACKAGE_NAMES = {"tests", "test"}


def test_package_discovery_finds_no_test_package() -> None:
    """The wheel's own package discovery, read from ``pyproject.toml``, yields no test package."""
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    find = config["tool"]["setuptools"]["packages"]["find"]
    packages = find_packages(where=str(ROOT / find["where"][0]), exclude=find["exclude"])
    assert "sc_neurocore" in packages
    shipped = sorted(name for name in packages if TEST_PACKAGE_NAMES & set(name.split(".")))
    assert shipped == [], f"Move these test packages to tests/: {shipped}"


def test_moved_quantum_cognition_tests_are_collected_from_tests() -> None:
    """The relocated suite sits under the collected ``tests`` root as a package."""
    suite = ROOT / "tests" / "test_quantum_cognition"
    assert (suite / "__init__.py").is_file()
    assert len(list(suite.glob("test_*.py"))) == 12
    assert not (ROOT / "src/sc_neurocore/quantum_cognition/tests").exists()
