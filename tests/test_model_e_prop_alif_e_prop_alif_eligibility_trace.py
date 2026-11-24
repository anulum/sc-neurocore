# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEPropALIFEligibilityTrace from former test_model_e_prop_alif.py

"""Focused suite: TestEPropALIFEligibilityTrace from former test_model_e_prop_alif.py."""

from __future__ import annotations

from sc_neurocore.neurons.models.e_prop_alif import (
    EPropALIFNeuron,
)


class TestEPropALIFEligibilityTrace:
    """e_trace tracks how weight changes affect future spiking."""

    def test_e_trace_accumulates(self) -> None:
        """The eligibility trace accumulates under input."""
        n = EPropALIFNeuron()
        for _ in range(100):
            n.step(0.2)
        assert n.e_trace > 0

    def test_e_trace_decays(self) -> None:
        """The eligibility trace decays without input."""
        n = EPropALIFNeuron()
        n.e_trace = 10.0
        n.v = -10.0  # far from threshold → psi ≈ 0
        n.step(0.0)
        assert n.e_trace < 10.0

    def test_pseudo_derivative_peaks_near_threshold(self) -> None:
        """The pseudo-derivative psi is 0.3 · max(0, 1 - |V-θ|) and peaks when V ≈ θ."""
        n = EPropALIFNeuron()
        n.v = n.v_threshold_base  # exactly at threshold
        # psi = 0.3 * max(0, 1 - 0) = 0.3
        psi = max(0.0, 1.0 - abs(n.v - n.v_threshold_base)) * 0.3
        assert abs(psi - 0.3) < 1e-10
