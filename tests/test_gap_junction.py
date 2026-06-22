# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for gap junctions

"""Tests for gap junction electrical synapses."""

import numpy as np
import pytest

from sc_neurocore.synapses.gap_junction import GapJunction


class TestGapJunction:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("conductance", -0.1),
            ("conductance", float("nan")),
            ("rectification", -0.1),
            ("rectification", 1.1),
            ("rectification", float("inf")),
        ],
    )
    def test_invalid_parameters_fail_closed(self, field, value):
        kwargs = {"conductance": 0.1, "rectification": 0.0}
        kwargs[field] = value
        with pytest.raises(ValueError, match=field):
            GapJunction(**kwargs)

    def test_current_positive_dv(self):
        gj = GapJunction(conductance=0.1)
        i = gj.current(v_pre=-50.0, v_post=-65.0)
        assert i == 0.1 * 15.0  # 1.5

    def test_current_negative_dv(self):
        gj = GapJunction(conductance=0.1)
        i = gj.current(v_pre=-65.0, v_post=-50.0)
        assert i == 0.1 * -15.0  # -1.5

    def test_current_zero_dv(self):
        gj = GapJunction(conductance=0.5)
        assert gj.current(v_pre=-60.0, v_post=-60.0) == 0.0

    def test_bidirectional_symmetry(self):
        gj = GapJunction(conductance=0.2)
        i_forward = gj.current(v_pre=-50.0, v_post=-70.0)
        i_backward = gj.current(v_pre=-70.0, v_post=-50.0)
        assert i_forward == -i_backward

    def test_rectification(self):
        gj = GapJunction(conductance=0.1, rectification=1.0)
        i_pos = gj.current(v_pre=-50.0, v_post=-65.0)  # dv > 0 → full current
        i_neg = gj.current(v_pre=-65.0, v_post=-50.0)  # dv < 0 → blocked
        assert i_pos == 0.1 * 15.0
        assert i_neg == 0.0

    def test_partial_rectification(self):
        gj = GapJunction(conductance=1.0, rectification=0.5)
        i_neg = gj.current(v_pre=-70.0, v_post=-50.0)  # dv < 0 → reduced
        expected = 1.0 * (-20.0) * (1.0 - 0.5)  # = -10.0
        assert i_neg == expected

    @pytest.mark.parametrize(
        ("v_pre", "v_post"),
        [(float("nan"), -65.0), (-50.0, float("inf"))],
    )
    def test_invalid_current_voltage_fails_closed(self, v_pre, v_post):
        gj = GapJunction(conductance=0.1)
        with pytest.raises(ValueError, match="voltage"):
            gj.current(v_pre=v_pre, v_post=v_post)


class TestGapJunctionPopulation:
    def test_current_matrix_two_neurons(self):
        gj = GapJunction(conductance=0.1)
        voltages = np.array([-50.0, -70.0])
        adj = np.array([[0, 1], [1, 0]])
        currents = gj.current_matrix(voltages, adj)
        # Neuron 0: g*(V1-V0) = 0.1*(-70-(-50)) = -2.0
        # Neuron 1: g*(V0-V1) = 0.1*(-50-(-70)) = +2.0
        np.testing.assert_allclose(currents, [-2.0, 2.0])

    def test_current_matrix_conserves_current(self):
        gj = GapJunction(conductance=0.5)
        voltages = np.array([-50.0, -60.0, -70.0])
        adj = np.ones((3, 3)) - np.eye(3)
        currents = gj.current_matrix(voltages, adj)
        np.testing.assert_allclose(currents.sum(), 0.0, atol=1e-10)

    def test_weighted_reciprocal_current_matrix_conserves_current(self):
        gj = GapJunction(conductance=0.25)
        voltages = np.array([-50.0, -60.0, -70.0])
        adj = np.array(
            [
                [0.0, 0.5, 0.25],
                [0.5, 0.0, 0.75],
                [0.25, 0.75, 0.0],
            ]
        )
        currents = gj.current_matrix(voltages, adj)
        np.testing.assert_allclose(currents.sum(), 0.0, atol=1e-10)

    def test_no_connections_no_current(self):
        gj = GapJunction(conductance=1.0)
        voltages = np.array([-50.0, -70.0, -90.0])
        adj = np.zeros((3, 3))
        currents = gj.current_matrix(voltages, adj)
        np.testing.assert_allclose(currents, [0.0, 0.0, 0.0])

    @pytest.mark.parametrize(
        ("voltages", "adjacency"),
        [
            (np.array([-50.0, float("nan")]), np.ones((2, 2))),
            (np.array([[-50.0, -60.0]]), np.ones((2, 2))),
            (np.array([-50.0, -60.0]), np.ones((2, 3))),
            (np.array([-50.0, -60.0]), np.array([[0.0, -1.0], [-1.0, 0.0]])),
            (np.array([-50.0, -60.0]), np.array([[0.0, 1.0], [0.0, 0.0]])),
            (np.array([-50.0, -60.0]), np.ones(2)),
            (
                np.array([-50.0, -60.0]),
                np.array([[0.0, float("inf")], [float("inf"), 0.0]]),
            ),
        ],
    )
    def test_invalid_current_matrix_inputs_fail_closed(self, voltages, adjacency):
        gj = GapJunction(conductance=0.1)
        with pytest.raises(ValueError, match="current_matrix"):
            gj.current_matrix(voltages, adjacency)
