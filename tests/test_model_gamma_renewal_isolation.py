# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestIsolation from former test_model_gamma_renewal.py

"""Focused suite: TestIsolation from former test_model_gamma_renewal.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.neurons.models.gamma_renewal import (
    GammaRenewalNeuron,
)


class TestIsolation:
    """The GammaRenewalNeuron in isolation."""

    def test_step_returns_binary(self) -> None:
        """One step returns a binary spike."""
        assert GammaRenewalNeuron().step(0.0) in (0, 1)

    def test_state_finite(self) -> None:
        """The state stays finite under drive."""
        n = GammaRenewalNeuron()
        for _ in range(5000):
            n.step(100.0)
        assert np.isfinite(getattr(n, "v", 0.0))

    def test_reset(self) -> None:
        """A reset after drive leaves a valid state."""
        n = GammaRenewalNeuron()
        for _ in range(100):
            n.step(100.0)
        n.reset()
