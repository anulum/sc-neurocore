# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikeServerStep from former test_serve_server.py

"""Focused suite: TestSpikeServerStep from former test_serve_server.py."""

from __future__ import annotations

import numpy as np
import pytest
from sc_neurocore.serve import (
    SpikeServer,
)
from tests.serve_server_support import (
    MockNetwork,
    MockPopNetwork,
)


class TestSpikeServerStep:
    """Stepping a spike server directly, without HTTP."""

    def test_step_with_sc_network(self) -> None:
        """A step on a network with a step method returns outputs and timestep 1."""
        net = MockNetwork()
        server = SpikeServer(net)
        result = server.step({"input": [1.0, 2.0, 3.0]})
        assert "outputs" in result
        assert "timestep" in result
        assert result["timestep"] == 1
        np.testing.assert_array_almost_equal(
            result["outputs"]["input"],
            [0.5, 1.0, 1.5],
        )

    def test_step_increments_timestep(self) -> None:
        """Each step advances the timestep by one."""
        net = MockNetwork()
        server = SpikeServer(net)
        server.step({"input": [1.0]})
        server.step({"input": [2.0]})
        result = server.step({"input": [3.0]})
        assert result["timestep"] == 3

    def test_step_with_population_network(self) -> None:
        """A step on a population network returns outputs per population."""
        net = MockPopNetwork()
        server = SpikeServer(net)
        result = server.step({"exc": [1.0, 0.0, 1.0]})
        assert "outputs" in result
        assert "exc" in result["outputs"]

    def test_step_unsupported_network(self) -> None:
        """Stepping a network without a step interface is refused."""
        server = SpikeServer(object())
        with pytest.raises(TypeError, match="Unsupported"):
            server.step({"input": [1.0]})
