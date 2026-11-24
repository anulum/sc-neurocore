# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpiNNakerLIFIsolation from former test_model_spinnaker_lif.py

"""Focused suite: TestSpiNNakerLIFIsolation from former test_model_spinnaker_lif.py."""

from __future__ import annotations

import numpy as np
import pytest
from sc_neurocore.neurons.models.spinnaker_lif import (
    SpiNNakerLIFNeuron,
)


class TestSpiNNakerLIFIsolation:
    """The SpiNNakerLIFNeuron in isolation."""

    def test_defaults(self) -> None:
        """The defaults match the published model."""
        n = SpiNNakerLIFNeuron()
        assert n.v == -70.0 and n.tau_m == 20.0 and n.tau_refrac == 2.0

    def test_step_returns_binary(self) -> None:
        """One step returns a binary spike."""
        assert SpiNNakerLIFNeuron().step(0.0) in (0, 1)

    def test_state_finite(self) -> None:
        """The state stays finite under drive."""
        n = SpiNNakerLIFNeuron()
        for _ in range(50000):
            n.step(25.0)
        assert np.isfinite(n.v)

    def test_reset(self) -> None:
        """A reset after drive leaves a valid state."""
        n = SpiNNakerLIFNeuron()
        for _ in range(100):
            n.step(25.0)
        n.reset()
        assert n.v == n.v_rest and n.refrac_count == 0.0

    def test_rejects_invalid_parameters(self) -> None:
        """The model rejects invalid parameters."""
        with pytest.raises(ValueError, match="tau_m must be positive"):
            SpiNNakerLIFNeuron(tau_m=0.0)
        with pytest.raises(ValueError, match="dt must be positive"):
            SpiNNakerLIFNeuron(dt=0.0)
        with pytest.raises(ValueError, match="refrac_count must be non-negative"):
            SpiNNakerLIFNeuron(refrac_count=-1.0)

    def test_rejects_invalid_current_without_mutation(self) -> None:
        """The model rejects invalid current without mutation."""
        n = SpiNNakerLIFNeuron()
        v0 = n.v
        with pytest.raises(ValueError, match="current must be finite"):
            n.step(float("nan"))
        assert n.v == v0
