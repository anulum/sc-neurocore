# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Module-level tests from former test_memristor_mapper.py

"""Module-level tests from former test_memristor_mapper.py."""

from __future__ import annotations

from memristor_mapper_support import *  # noqa: F403

def test_memristor_package_exports_mapper_surface() -> None:
    """The memristor package facade exposes the documented mapper surface."""
    expected_symbols = {
        "AgingReport",
        "AgingSimulator",
        "CompensationLUT",
        "CompensationStrategy",
        "ConductanceModel",
        "CrossbarArray",
        "CrossbarEstimator",
        "CrossbarMapping",
        "CrossbarPowerEstimate",
        "CrossbarTopology",
        "IRDropModel",
        "MappingResult",
        "MemristorMapper",
        "MemristorTechnology",
        "MonteCarloReport",
        "MonteCarloSimulator",
        "SCAbsorbEncoder",
        "SneakPathModel",
        "StuckFaultMap",
        "VariabilityInjector",
        "VerilogEmitter",
        "WriteVerifyProtocol",
        "WriteVerifyResult",
    }

    assert expected_symbols == set(memristor.__all__)
    for name in expected_symbols:
        assert getattr(memristor, name).__name__ == name
def test_signal_to_sneak_ratio_is_infinite_without_sneak_current() -> None:
    # A zero off-conductance gives zero worst-case sneak current, so the
    # signal-to-sneak ratio is infinite rather than dividing by zero.
    assert SneakPathModel.signal_to_sneak_ratio(g_on=1e-3, g_off=0.0, rows=8, cols=8) == float(
        "inf"
    )
