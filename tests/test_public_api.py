# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Verify all __all__ exports are importable and no

"""Verify all __all__ exports are importable and no regressions occur."""

from pathlib import Path
import sys

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

import sc_neurocore


def _project_metadata() -> dict:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    return tomllib.loads(pyproject.read_text(encoding="utf-8"))


def test_all_symbols_importable():
    for name in sc_neurocore.__all__:
        assert hasattr(sc_neurocore, name), f"Missing export: {name}"


def test_version_string():
    from importlib.metadata import version

    assert sc_neurocore.__version__ == version("sc-neurocore")


def test_all_count():
    assert len(sc_neurocore.__all__) == 39, f"Public API count changed: {len(sc_neurocore.__all__)}"


def test_project_does_not_require_separate_engine_pypi_package():
    data = _project_metadata()
    dependencies = data["project"]["dependencies"]
    assert all(not dep.startswith("sc-neurocore-engine") for dep in dependencies)


def test_install_extras_cover_documented_workflow_groups():
    extras = _project_metadata()["project"]["optional-dependencies"]

    assert extras["hdl"] == ["pint>=0.23"]

    full = set(extras["full"])
    assert {
        "numba>=0.56",
        "matplotlib>=3.5",
        "networkx",
        "onnx",
        "torch>=2.0",
        "nir>=1.0",
        "fastapi>=0.100",
        "uvicorn[standard]>=0.20",
        "httpx>=0.27",
        "PyWavelets>=1.4",
        "zstandard>=0.22",
        "scikit-learn>=1.3",
        "pint>=0.23",
        "qiskit",
        "pennylane",
        "qiskit-aer",
    } <= full


def test_hdl_install_profile_packages_source_artefacts():
    package_data = _project_metadata()["tool"]["setuptools"]["package-data"]["sc_neurocore"]

    assert "hardware/*.v" in package_data
    assert "hdl_gen/safety/*.sv" in package_data
    assert "hdl_gen/openroad_flow/*.sh" in package_data
