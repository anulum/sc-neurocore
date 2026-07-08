# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Spintronic package public API tests

"""Tests for the public `sc_neurocore.spintronic` package facade."""

from __future__ import annotations

import sc_neurocore.spintronic as spintronic


def test_spintronic_package_exports_mapper_surface() -> None:
    """The spintronic package facade exposes the documented mapper surface."""
    expected_symbols = {
        "AgingModel",
        "DefectEntry",
        "DefectMap",
        "MLCConfig",
        "MappingResult",
        "MaterialParams",
        "MuMax3OutputParser",
        "MuMax3Result",
        "MuMax3ScriptGenerator",
        "RacetrackShiftRegister",
        "RadiationModel",
        "SkyrmionHallCorrector",
        "SpintronicArray",
        "SpintronicCell",
        "SpintronicDeviceConfig",
        "SpintronicMapper",
        "SpintronicTech",
        "SpintronicVerilogGenerator",
        "VariabilityModel",
        "WriteVerifyResult",
        "retention_failure_probability",
        "switching_current_vs_temperature",
        "switching_time_vs_temperature",
        "write_verify",
    }

    assert expected_symbols == set(spintronic.__all__)
    for name in expected_symbols:
        assert getattr(spintronic, name).__name__ == name
