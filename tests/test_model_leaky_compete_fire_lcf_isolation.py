# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLCFIsolation from former test_model_leaky_compete_fire.py

"""Focused suite: TestLCFIsolation from former test_model_leaky_compete_fire.py."""

from __future__ import annotations

from tests.model_leaky_compete_fire_support import *  # noqa: F403


class TestLCFIsolation:
    def test_defaults(self):
        n = LeakyCompeteFireNeuron()
        assert n.n_units == 4 and len(n.v) == 4
        assert n.tau == 10.0 and n.v_threshold == 1.0
        assert n.w_inh == 0.5

    def test_step_returns_list(self):
        n = LeakyCompeteFireNeuron()
        result = n.step(5.0)
        assert isinstance(result, list) and len(result) == 4

    def test_each_element_binary(self):
        n = LeakyCompeteFireNeuron()
        result = n.step(5.0)
        for s in result:
            assert s in (0, 1)

    def test_reset_zeroes_all(self):
        n = LeakyCompeteFireNeuron()
        for _ in range(100):
            n.step(5.0)
        n.reset()
        assert all(v == 0.0 for v in n.v)

    def test_initial_vector_state_is_preserved_when_valid(self):
        n = LeakyCompeteFireNeuron(n_units=3, v=[0.1, 0.2, 0.3])
        assert n.v == [0.1, 0.2, 0.3]

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = LeakyCompeteFireNeuron()
            trace = [tuple(n.step(5.0)) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"n_units": 0},
            {"tau": 0.0},
            {"tau": np.inf},
            {"v_threshold": np.nan},
            {"w_inh": -0.1},
            {"w_inh": np.inf},
            {"dt": 0.0},
            {"dt": np.inf},
        ],
    )
    def test_invalid_configuration_raises(self, kwargs):
        with pytest.raises(ValueError):
            LeakyCompeteFireNeuron(**kwargs)

    def test_step_rejects_current_length_mismatch(self):
        n = LeakyCompeteFireNeuron(n_units=3)
        with pytest.raises(ValueError, match="length"):
            n.step([1.0, 2.0])

    def test_step_rejects_non_finite_current(self):
        n = LeakyCompeteFireNeuron(n_units=2)
        with pytest.raises(ValueError, match="finite"):
            n.step([1.0, np.nan])

    def test_rejects_initial_voltage_length_mismatch(self):
        with pytest.raises(ValueError, match="v must have length"):
            LeakyCompeteFireNeuron(n_units=3, v=[0.1, 0.2])

    def test_rejects_non_finite_initial_voltage(self):
        with pytest.raises(ValueError, match="v must contain"):
            LeakyCompeteFireNeuron(n_units=2, v=[0.1, np.nan])
