# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Fault-injector validation tests

"""Input validation tests for ``FaultInjector``."""

from __future__ import annotations

from fault_injection_support import *  # noqa: F403


def test_inject_rejects_non_numeric_ber():  # type: ignore[no-untyped-def]  # Preserved legacy test AST
    inj = FaultInjector(seed=0)
    with pytest.raises(ValueError, match="ber must be"):
        inj.inject(np.array([0, 1], dtype=np.uint8), FaultModel.BIT_FLIP, "x")  # type: ignore[arg-type]


def test_inject_gaussian_requires_numeric_bitstream():  # type: ignore[no-untyped-def]  # Preserved legacy test AST
    inj = FaultInjector(seed=0)
    with pytest.raises(ValueError, match="gaussian_noise requires numeric"):
        inj.inject(np.array(["a", "b"]), FaultModel.GAUSSIAN_NOISE, 0.1)


def test_inject_unsupported_fault_model_raises():  # type: ignore[no-untyped-def]  # Preserved legacy test AST
    # A FaultModel-typed object that matches none of the handled members reaches
    # the exhaustiveness guard (defended for forward compatibility / typing).
    inj = FaultInjector(seed=0)
    bogus = MagicMock(spec=FaultModel)
    with pytest.raises(ValueError, match="unsupported fault model"):
        inj.inject(np.array([0, 1], dtype=np.uint8), bogus, 0.5)


def test_inject_at_positions_rejects_non_array():  # type: ignore[no-untyped-def]  # Preserved legacy test AST
    inj = FaultInjector(seed=0)
    with pytest.raises(ValueError, match="must be a numpy.ndarray"):
        inj.inject_at_positions([0, 1, 0], [1])  # type: ignore[arg-type]


def test_inject_at_positions_rejects_non_1d():  # type: ignore[no-untyped-def]  # Preserved legacy test AST
    inj = FaultInjector(seed=0)
    with pytest.raises(ValueError, match="must be a 1-D array"):
        inj.inject_at_positions(np.zeros((2, 2), dtype=np.uint8), [0])
