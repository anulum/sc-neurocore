# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDeSchutterIsolation from former test_model_de_schutter_purkinje.py

"""Focused suite: TestDeSchutterIsolation from former test_model_de_schutter_purkinje.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.neurons.models.de_schutter_purkinje import (
    DeSchutterPurkinjeNeuron,
)


class TestDeSchutterIsolation:
    """The DeSchutterPurkinjeNeuron in isolation."""

    def test_step_returns_binary(self) -> None:
        """One step returns a binary spike."""
        assert DeSchutterPurkinjeNeuron().step(0.0) in (0, 1)

    def test_state_finite(self) -> None:
        """The state stays finite under drive."""
        n = DeSchutterPurkinjeNeuron()
        for _ in range(20000):
            n.step(10.0)
        assert np.isfinite(n.v)

    def test_reset(self) -> None:
        """A reset after drive leaves a valid state."""
        n = DeSchutterPurkinjeNeuron()
        for _ in range(100):
            n.step(10.0)
        n.reset()
        assert np.isfinite(n.v)
