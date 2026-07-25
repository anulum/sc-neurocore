# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Version, dependency, and install-profile contracts

"""Verify public distribution metadata and documented installation profiles."""

import runpy
import sys
from typing import Any, cast

import sc_neurocore

from tests.public_api_support import _project_metadata, _repo_root


def test_version_string() -> None:
    """The source package version matches pyproject metadata."""
    assert sc_neurocore.__version__ == _project_metadata()["project"]["version"]


def test_all_count() -> None:
    """The root public API keeps its audited export count."""
    assert len(sc_neurocore.__all__) == 45, f"Public API count changed: {len(sc_neurocore.__all__)}"


def test_project_does_not_require_separate_engine_pypi_package() -> None:
    """The base package does not require a separate engine package."""
    dependencies = _project_metadata()["project"]["dependencies"]
    assert all(not dependency.startswith("sc-neurocore-engine") for dependency in dependencies)


def test_install_extras_cover_documented_workflow_groups() -> None:
    """Optional dependency groups match the documented workflow boundaries."""
    metadata = _project_metadata()
    extras = metadata["project"]["optional-dependencies"]
    dependencies = metadata["project"]["dependencies"]

    assert extras["minimal"] == []
    assert extras["core"] == []
    assert extras["minimal"] == extras["core"]
    assert extras["hdl"] == ["pint>=0.23"]
    assert extras["license"] == ["httpx>=0.27"]
    assert extras["annealing"] == ["dwave-neal", "dimod"]
    assert all(
        dependency.split(">=", maxsplit=1)[0].split("==", maxsplit=1)[0]
        not in {
            "dimod",
            "dwave-neal",
            "fastapi",
            "httpx",
            "jax",
            "jaxlib",
            "pennylane",
            "qiskit",
            "torch",
            "uvicorn",
        }
        for dependency in dependencies
    )

    full = set(extras["full"])
    assert set(extras["hdl"]) <= full
    assert set(extras["annealing"]).isdisjoint(full)
    assert {
        "numba>=0.56",
        "matplotlib>=3.5",
        "networkx",
        "onnx",
        "torch>=2.0",
        "nir>=1.0,<1.0.9",
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


def test_minimal_install_profile_is_documented_and_dependency_light() -> None:
    """The minimal profile stays documented, empty, and free of opt-in stacks."""
    root = _repo_root()
    extras = _project_metadata()["project"]["optional-dependencies"]
    readme = (root / "README.md").read_text(encoding="utf-8")
    install_profiles = (root / "docs" / "guides" / "install_profiles.md").read_text(
        encoding="utf-8"
    )
    package_boundary = (root / "docs" / "architecture" / "package_boundary_decision.md").read_text(
        encoding="utf-8"
    )
    demo_source = (root / "examples" / "minimal_smoke_demo.py").read_text(encoding="utf-8")
    documented_surfaces = "\n".join((readme, install_profiles, package_boundary))
    banned_imports = (
        "import torch",
        "import jax",
        "import qiskit",
        "import pennylane",
        "import lava",
        "import gdsfactory",
    )

    assert extras["minimal"] == []
    assert 'pip install "sc-neurocore[minimal]"' in documented_surfaces
    assert "examples/minimal_smoke_demo.py" in documented_surfaces
    for banned_import in banned_imports:
        assert banned_import not in demo_source


def test_minimal_smoke_demo_runs_under_profile_budget() -> None:
    """The minimal smoke demo runs quickly without loading opt-in stacks."""
    heavy_modules = ("torch", "jax", "qiskit", "pennylane", "lava", "gdsfactory")
    loaded_before = {module for module in heavy_modules if module in sys.modules}
    namespace = runpy.run_path(str(_repo_root() / "examples" / "minimal_smoke_demo.py"))
    main = namespace["main"]

    assert callable(main)
    payload = cast(dict[str, Any], main())
    assert cast(float, payload["elapsed_seconds"]) < 30.0
    assert cast(int, payload["hdl_primitive_count"]) >= 1
    assert cast(int, payload["spike_count"]) >= 0
    assert set(cast(list[str], payload["heavy_modules_loaded"])) <= loaded_before
