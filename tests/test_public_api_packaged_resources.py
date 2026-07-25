# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Packaged HDL and schema resource contracts

"""Verify the public wheel resource boundary and HDL resource helpers."""

import pytest

from tests.public_api_support import _package_root, _project_metadata


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


def test_dsl_schema_assets_are_packaged() -> None:
    """The wheel ships UniversalNeuron schema assets used by bare-name loading."""
    package_data = _project_metadata()["tool"]["setuptools"]["package-data"]["sc_neurocore"]

    assert "neurons/model_schemas/*.json" in package_data
    assert "neurons/model_schemas/*.toml" in package_data

    package_root = _package_root()
    matched = {
        str(path.relative_to(package_root))
        for pattern in package_data
        for path in package_root.glob(pattern)
    }

    assert "neurons/model_schemas/perfect_integrator.json" in matched
    assert "neurons/model_schemas/perfect_integrator.toml" in matched
    assert "neurons/model_schemas/lif.toml" in matched


def test_hdl_resource_helper_rejects_unknown_primitive_names() -> None:
    """The HDL resource helper rejects traversal and unknown primitive names."""
    from sc_neurocore.hdl.resources import baseline_primitive_text, list_baseline_primitive_rtl

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

    forbidden_prefixes = ("accel/julia/", "accel/go/", "accel/mojo/")
    assert not any(item.startswith(forbidden_prefixes) for item in package_data)
