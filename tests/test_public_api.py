# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Verify all __all__ exports are importable and no

"""Verify all __all__ exports are importable and no regressions occur."""

from pathlib import Path
import tomllib

import sc_neurocore


def test_all_symbols_importable():
    for name in sc_neurocore.__all__:
        assert hasattr(sc_neurocore, name), f"Missing export: {name}"


def test_version_string():
    from importlib.metadata import version

    assert sc_neurocore.__version__ == version("sc-neurocore")


def test_all_count():
    assert len(sc_neurocore.__all__) == 38, f"Public API count changed: {len(sc_neurocore.__all__)}"


def test_project_does_not_require_separate_engine_pypi_package():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    dependencies = data["project"]["dependencies"]
    assert all(not dep.startswith("sc-neurocore-engine") for dep in dependencies)
