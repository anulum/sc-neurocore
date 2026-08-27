# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLBParameters from former test_model_larter_breakspear.py

"""Focused suite: TestLBParameters from former test_model_larter_breakspear.py."""

from __future__ import annotations

from tests.model_larter_breakspear_support import *  # noqa: F403


class TestLBParameters:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("dt", 0.0),
            ("dt", float("nan")),
            ("tau_k", 0.0),
            ("phi", -0.1),
            ("b", -0.1),
            ("w", -0.01),
            ("w", 1.01),
            ("g_ca", -0.1),
            ("g_na", -0.1),
            ("g_k", -0.1),
            ("g_l", -0.1),
            ("delta_v", 0.0),
            ("delta_z", 0.0),
            ("coupling_balance", 1.01),
        ],
    )
    def test_rejects_nonphysical_parameters(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            LarterBreakspearNeuron(**{field: value})

    @pytest.mark.parametrize("integrator", ["rk4", "euler"])
    def test_runtime_parameter_corruption_fails_before_mutation(self, integrator: str):
        n = LarterBreakspearNeuron(integrator=integrator)
        n.tau_k = float("nan")
        before = (n.v, n.w, n.z)

        with pytest.raises(ValueError):
            n.step(0.0)

        assert (n.v, n.w, n.z) == before

    @pytest.mark.parametrize("integrator", ["rk4", "euler"])
    def test_potassium_gate_bounds_fail_before_mutation(self, integrator: str):
        n = LarterBreakspearNeuron(w=0.0, dt=100.0, integrator=integrator)
        before = (n.v, n.w, n.z)

        with pytest.raises(FloatingPointError, match="potassium gate"):
            n.step(-100.0)

        assert (n.v, n.w, n.z) == before

    def test_rejects_unknown_integrator(self):
        with pytest.raises(ValueError, match="integrator"):
            LarterBreakspearNeuron(integrator="verlet")

    @pytest.mark.parametrize("g_ca", [0.5, 1.1, 2.0])
    def test_g_ca_sweep(self, g_ca: float):
        n = LarterBreakspearNeuron(g_ca=g_ca)
        for _ in range(5000):
            n.step(0.0)
        assert np.isfinite(n.v)

    @pytest.mark.parametrize("i_ext", [0.0, 0.3, 1.0])
    def test_i_ext_sweep(self, i_ext: float):
        n = LarterBreakspearNeuron(i_ext=i_ext)
        for _ in range(5000):
            n.step(0.0)
        assert np.isfinite(n.v)

    @pytest.mark.parametrize("a_ee", [0.0, 0.4, 0.5])
    def test_a_ee_sweep(self, a_ee: float):
        n = LarterBreakspearNeuron(a_ee=a_ee)
        for _ in range(5000):
            n.step(0.0)
        assert np.isfinite(n.v)
