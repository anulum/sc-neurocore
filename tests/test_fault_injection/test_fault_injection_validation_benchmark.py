# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Resilience benchmark validation tests

"""Input validation tests for resilience benchmark runs."""

from __future__ import annotations

from fault_injection_support import *  # noqa: F403


def test_generate_bitstream_rejects_non_numeric_probability():  # type: ignore[no-untyped-def]  # Preserved legacy test AST
    bench = ResilienceBenchmark(seed=0)
    with pytest.raises(ValueError, match="probability must be"):
        bench._generate_bitstream(8, "x")  # type: ignore[arg-type]


def test_run_rejects_non_numeric_probability():  # type: ignore[no-untyped-def]  # Preserved legacy test AST
    bench = ResilienceBenchmark(seed=0)
    with pytest.raises(ValueError, match="probability must be"):
        bench.run(fault_model=FaultModel.BIT_FLIP, ber=0.1, probability="x")  # type: ignore[arg-type]


def test_run_rejects_non_numeric_ber():  # type: ignore[no-untyped-def]  # Preserved legacy test AST
    bench = ResilienceBenchmark(seed=0)
    with pytest.raises(ValueError, match="ber must be"):
        bench.run(fault_model=FaultModel.BIT_FLIP, ber="x")  # type: ignore[arg-type]
