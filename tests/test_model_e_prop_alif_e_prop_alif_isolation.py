# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEPropALIFIsolation from former test_model_e_prop_alif.py

"""Focused suite: TestEPropALIFIsolation from former test_model_e_prop_alif.py."""

from __future__ import annotations

from tests.model_e_prop_alif_support import *  # noqa: F403

class TestEPropALIFIsolation:
    def test_defaults(self):
        n = EPropALIFNeuron()
        assert n.v == 0.0 and n.a == 0.0 and n.e_trace == 0.0
        assert n.tau_m == 20.0 and n.tau_a == 200.0 and n.beta == 0.07

    def test_alpha_precomputed(self):
        n = EPropALIFNeuron()
        assert abs(n.alpha_m - np.exp(-1.0 / 20.0)) < 1e-12
        assert abs(n.alpha_a - np.exp(-1.0 / 200.0)) < 1e-12

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"v": float("nan")},
            {"a": float("inf")},
            {"e_trace": float("nan")},
            {"tau_m": 0.0},
            {"tau_m": float("inf")},
            {"tau_a": 0.0},
            {"tau_a": float("nan")},
            {"v_threshold_base": float("inf")},
            {"v_threshold_base": -0.1},
            {"beta": -0.01},
            {"beta": float("nan")},
            {"v_reset": float("inf")},
            {"v_reset": 1.1},
            {"dt": 0.0},
            {"dt": float("nan")},
            {"dt": 21.0},
            {"dt": 201.0},
        ],
    )
    def test_rejects_non_physical_configuration(self, kwargs):
        with pytest.raises(ValueError):
            EPropALIFNeuron(**kwargs)

    def test_step_returns_binary(self):
        assert EPropALIFNeuron().step(0.0) in (0, 1)

    @pytest.mark.parametrize("current", [float("nan"), float("inf"), -float("inf")])
    def test_rejects_non_finite_current(self, current):
        n = EPropALIFNeuron()
        with pytest.raises(ValueError, match="current"):
            n.step(current)

    def test_state_finite(self):
        n = EPropALIFNeuron()
        for _ in range(50000):
            n.step(0.2)
        assert all(np.isfinite(v) for v in [n.v, n.a, n.e_trace])

    def test_reset(self):
        n = EPropALIFNeuron(v_reset=-0.25)
        for _ in range(100):
            n.step(0.5)
        n.reset()
        assert n.v == n.v_reset
        assert n.a == 0.0
        assert n.e_trace == 0.0
