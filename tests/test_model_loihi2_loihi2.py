# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLoihi2 from former test_model_loihi2.py

"""Focused suite: TestLoihi2 from former test_model_loihi2.py."""

from __future__ import annotations

from tests.model_loihi2_support import *  # noqa: F403


class TestLoihi2:
    def test_fires(self):
        from sc_neurocore.neurons.models.loihi2 import Loihi2Neuron

        n = Loihi2Neuron()
        assert sum(n.step(200) for _ in range(100)) > 0

    def test_integer_state(self):
        from sc_neurocore.neurons.models.loihi2 import Loihi2Neuron

        n = Loihi2Neuron()
        n.step(100)
        assert isinstance(n.s1, int)
