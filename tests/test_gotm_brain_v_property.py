# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestVProperty from former test_gotm_brain.py

"""Focused suite: TestVProperty from former test_gotm_brain.py."""

from __future__ import annotations

from tests.gotm_brain_support import *  # noqa: F403


class TestVProperty:
    def test_v_reads_Vm(self) -> None:
        from sc_neurocore.quantum_cognition import SpinPoolMPS, HybridFisherPosnerLIF

        pool = SpinPoolMPS(n_sites=4)
        n = HybridFisherPosnerLIF(0, pool)
        assert n.v == n.Vm == -70.0

    def test_v_writes_Vm(self) -> None:
        from sc_neurocore.quantum_cognition import SpinPoolMPS, HybridFisherPosnerLIF

        pool = SpinPoolMPS(n_sites=4)
        n = HybridFisherPosnerLIF(0, pool)
        n.v = -55.0
        assert n.Vm == -55.0
        assert n.v == -55.0
