# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTCLIFIsolation from former test_model_tc_lif.py

"""Focused suite: TestTCLIFIsolation from former test_model_tc_lif.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.neurons.models.sc_leaky_tc_lif import (
    SCLeakyTwoCompartmentLIFNeuron,
)


class TestTCLIFIsolation:
    """The SCLeakyTwoCompartmentLIFNeuron in isolation."""

    def test_construction_defaults(self) -> None:
        """Construction defaults."""
        n = SCLeakyTwoCompartmentLIFNeuron()
        assert n.v_s == 0.0
        assert n.v_d == 0.0
        assert n.tau_s == 2.0
        assert n.tau_d == 20.0
        assert n.kappa == 0.5
        assert n.theta == 1.0

    def test_step_returns_binary(self) -> None:
        """One step returns a binary spike."""
        assert SCLeakyTwoCompartmentLIFNeuron().step(0.0) in (0, 1)

    def test_dual_input_signature(self) -> None:
        """step(i_soma, i_dend) — two current inputs."""
        n = SCLeakyTwoCompartmentLIFNeuron()
        s = n.step(1.0, 0.5)
        assert s in (0, 1)

    def test_both_compartments_evolve(self) -> None:
        """Both compartments evolve."""
        n = SCLeakyTwoCompartmentLIFNeuron()
        for _ in range(100):
            n.step(0.5, 1.0)
        assert n.v_s != 0.0
        assert n.v_d != 0.0

    def test_state_finite(self) -> None:
        """The state stays finite under drive."""
        n = SCLeakyTwoCompartmentLIFNeuron()
        for _ in range(50000):
            n.step(2.0, 1.0)
        assert np.isfinite(n.v_s) and np.isfinite(n.v_d)

    def test_reset(self) -> None:
        """A reset after drive leaves a valid state."""
        n = SCLeakyTwoCompartmentLIFNeuron()
        for _ in range(100):
            n.step(2.0, 1.0)
        n.reset()
        assert n.v_s == n.v_rest and n.v_d == n.v_rest
