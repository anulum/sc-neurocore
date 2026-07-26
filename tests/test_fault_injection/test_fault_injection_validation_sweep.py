# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Resilience sweep validation tests

"""Input validation tests for resilience benchmark sweeps."""

from __future__ import annotations

from fault_injection_support import *  # noqa: F403


def test_sweep_ber_rejects_non_fault_model():  # type: ignore[no-untyped-def]  # Preserved legacy test AST
    bench = ResilienceBenchmark(seed=0)
    with pytest.raises(ValueError, match="fault_model must be a FaultModel"):
        bench.sweep_ber(fault_model="bit_flip", ber_range=[0.1])  # type: ignore[arg-type]


def test_sweep_ber_rejects_non_numeric_entry():  # type: ignore[no-untyped-def]  # Preserved legacy test AST
    bench = ResilienceBenchmark(seed=0)
    with pytest.raises(ValueError, match="ber_range entries must be"):
        bench.sweep_ber(fault_model=FaultModel.BIT_FLIP, ber_range=["x"])  # type: ignore[list-item]
