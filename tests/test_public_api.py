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
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

import pytest
import sc_neurocore


def _project_metadata() -> dict[str, Any]:
    """Load the project metadata from the repository pyproject file."""
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    return tomllib.loads(pyproject.read_text(encoding="utf-8"))


def _package_root() -> Path:
    """Return the source-tree package root used by package-data tests."""
    return Path(__file__).resolve().parents[1] / "src" / "sc_neurocore"


def test_all_symbols_importable() -> None:
    """Every advertised root package symbol resolves through the lazy facade."""
    for name in sc_neurocore.__all__:
        assert hasattr(sc_neurocore, name), f"Missing export: {name}"


def test_unknown_lazy_symbol_raises_attribute_error() -> None:
    """Unknown root-package attributes fail with the standard attribute error."""
    missing_name = "not_a_public_symbol"
    with pytest.raises(AttributeError, match="has no attribute 'not_a_public_symbol'"):
        getattr(sc_neurocore, missing_name)


def test_dir_lists_lazy_public_api() -> None:
    """The package directory includes lazily exposed public API names."""
    public_names = dir(sc_neurocore)
    assert "StochasticLIFNeuron" in public_names
    assert "BitstreamEncoder" in public_names
    assert "not_a_public_symbol" not in public_names


def test_version_string() -> None:
    """The source package version matches pyproject metadata."""
    assert sc_neurocore.__version__ == _project_metadata()["project"]["version"]


def test_all_count() -> None:
    """The root public API keeps its audited export count."""
    assert len(sc_neurocore.__all__) == 44, f"Public API count changed: {len(sc_neurocore.__all__)}"


def test_project_does_not_require_separate_engine_pypi_package() -> None:
    """The base package does not require a separate engine package."""
    data = _project_metadata()
    dependencies = data["project"]["dependencies"]
    assert all(not dep.startswith("sc-neurocore-engine") for dep in dependencies)


def test_install_extras_cover_documented_workflow_groups() -> None:
    """Optional dependency groups match the documented workflow boundaries."""
    extras = _project_metadata()["project"]["optional-dependencies"]
    dependencies = _project_metadata()["project"]["dependencies"]

    assert extras["core"] == []
    assert extras["hdl"] == ["pint>=0.23"]
    assert extras["license"] == ["httpx>=0.27"]
    assert all(
        dep.split(">=", maxsplit=1)[0].split("==", maxsplit=1)[0]
        not in {"torch", "jax", "jaxlib", "qiskit", "pennylane", "fastapi", "uvicorn", "httpx"}
        for dep in dependencies
    )

    full = set(extras["full"])
    assert set(extras["hdl"]) <= full
    assert {
        "numba>=0.56",
        "matplotlib>=3.5",
        "networkx",
        "onnx",
        "torch>=2.0",
        "nir>=1.0,<2.0",
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


def test_hdl_install_profile_packages_source_artefacts() -> None:
    """The HDL install profile ships the baseline RTL source artefacts."""
    package_data = _project_metadata()["tool"]["setuptools"]["package-data"]["sc_neurocore"]

    assert "hardware/*.v" in package_data
    assert "hdl/primitives/*.v" in package_data
    assert "hdl_gen/safety/*.sv" in package_data
    assert "hdl_gen/openroad_flow/*.sh" in package_data

    package_root = _package_root()
    matched = {
        str(path.relative_to(package_root))
        for pattern in package_data
        for path in package_root.glob(pattern)
    }

    assert "hardware/microtubule_neuron.v" in matched
    assert "hdl/primitives/sc_bitstream_encoder.v" in matched
    assert "hdl/primitives/sc_bitstream_synapse.v" in matched
    assert "hdl/primitives/sc_dense_layer_core.v" in matched
    assert "hdl/primitives/sc_dotproduct_to_current.v" in matched
    assert "hdl/primitives/sc_firing_rate_bank.v" in matched
    assert "hdl/primitives/sc_lif_neuron.v" in matched
    assert "hdl_gen/safety/safety_monitor.sv" in matched
    assert "hdl_gen/openroad_flow/run_asic_flow.sh" in matched


def test_hdl_resource_helper_rejects_unknown_primitive_names() -> None:
    """The HDL resource helper rejects traversal and unknown primitive names."""
    from sc_neurocore.hdl.resources import (
        baseline_primitive_text,
        list_baseline_primitive_rtl,
    )

    names = list_baseline_primitive_rtl()

    assert names == (
        "sc_bitstream_encoder.v",
        "sc_bitstream_synapse.v",
        "sc_dense_layer_core.v",
        "sc_dotproduct_to_current.v",
        "sc_firing_rate_bank.v",
        "sc_lif_neuron.v",
    )
    assert "module sc_bitstream_encoder" in baseline_primitive_text("sc_bitstream_encoder.v")
    try:
        baseline_primitive_text("../sc_bitstream_encoder.v")
    except ValueError as exc:
        assert "Unknown baseline HDL primitive" in str(exc)
    else:  # pragma: no cover - assertion guard.
        raise AssertionError("path traversal primitive name was accepted")


def test_hdl_resource_helper_contract_and_missing_packaged_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The HDL resource helper reports missing packaged primitive files."""
    from sc_neurocore.hdl import resources as hdl_resources

    names = hdl_resources.list_baseline_primitive_rtl()
    assert isinstance(names, tuple)
    assert len(set(names)) == len(names)

    class _MissingResource:
        """Minimal resource object that simulates a missing packaged file."""

        class _File:
            """Minimal file-like resource with a negative file predicate."""

            def is_file(self) -> bool:
                """Report that the resource does not exist as a file."""
                return False

        def joinpath(self, _name: str) -> _File:
            """Return a missing child resource for any primitive name."""
            return self._File()

    monkeypatch.setattr(hdl_resources, "files", lambda _pkg: _MissingResource())
    with pytest.raises(FileNotFoundError, match="Missing packaged HDL primitive"):
        hdl_resources.baseline_primitive_path("sc_lif_neuron.v")


def test_base_wheel_does_not_package_polyglot_research_sources() -> None:
    """The base wheel omits research-only polyglot source trees."""
    package_data = _project_metadata()["tool"]["setuptools"]["package-data"]["sc_neurocore"]

    forbidden_prefixes = (
        "accel/julia/",
        "accel/go/",
        "accel/mojo/",
    )

    assert not any(item.startswith(forbidden_prefixes) for item in package_data)
