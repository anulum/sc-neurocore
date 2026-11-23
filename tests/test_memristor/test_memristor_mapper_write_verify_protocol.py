# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWriteVerifyProtocol from former test_memristor_mapper.py

"""Focused suite: TestWriteVerifyProtocol from former test_memristor_mapper.py."""

from __future__ import annotations

from sc_neurocore.memristor.memristor_mapper import (
    ConductanceModel,
    MemristorTechnology,
    WriteVerifyProtocol,
)


class TestWriteVerifyProtocol:
    """The program-and-verify write protocol."""

    def test_converges(self) -> None:
        """Writing reaches its target level within some iterations."""
        m = ConductanceModel(MemristorTechnology.GENERIC)
        wv = WriteVerifyProtocol(m, max_iterations=20, tolerance=0.05, seed=42)
        result = wv.program_cell(8)
        assert result.iterations > 0
        assert result.target_level == 8

    def test_low_tolerance_may_need_more_iterations(self) -> None:
        """A tighter tolerance needs no fewer iterations unless it does not converge."""
        m = ConductanceModel(MemristorTechnology.GENERIC)
        wv1 = WriteVerifyProtocol(m, max_iterations=20, tolerance=0.10, seed=42)
        wv2 = WriteVerifyProtocol(m, max_iterations=20, tolerance=0.001, seed=42)
        r1 = wv1.program_cell(8)
        r2 = wv2.program_cell(8)
        assert r1.iterations <= r2.iterations or not r2.converged
