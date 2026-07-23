# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikeServerStep from former test_serve_server.py

"""Focused suite: TestSpikeServerStep from former test_serve_server.py."""

from __future__ import annotations

from tests.serve_server_support import *  # noqa: F403

class TestSpikeServerStep:
    def test_step_with_sc_network(self):
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

    def test_step_increments_timestep(self):
        net = MockNetwork()
        server = SpikeServer(net)
        server.step({"input": [1.0]})
        server.step({"input": [2.0]})
        result = server.step({"input": [3.0]})
        assert result["timestep"] == 3

    def test_step_with_population_network(self):
        net = MockPopNetwork()
        server = SpikeServer(net)
        result = server.step({"exc": [1.0, 0.0, 1.0]})
        assert "outputs" in result
        assert "exc" in result["outputs"]

    def test_step_unsupported_network(self):
        server = SpikeServer(object())
        with pytest.raises(TypeError, match="Unsupported"):
            server.step({"input": [1.0]})
