# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPLIFDynamics from former test_model_plif.py

"""Focused suite: TestPLIFDynamics from former test_model_plif.py."""

from __future__ import annotations

from tests.model_plif_support import *  # noqa: F403


class TestPLIFDynamics:
    def test_voltage_accumulation(self):
        """V(t) = sum_{k=0}^{t-1} alpha^k · I = I · (1 - alpha^t) / (1 - alpha).

        For alpha=0.5, I=0.5: V(1)=0.5, V(2)=0.75, V(3)=0.875, ...
        """
        n = ParametricLIFNeuron(a=0.0)  # alpha=0.5
        expected = [0.5, 0.75, 0.875]
        for t, exp_v in enumerate(expected):
            n.step(0.5)
            assert abs(n.v - exp_v) < 1e-12, f"t={t + 1}: v={n.v}, expected={exp_v}"

    def test_steady_state_voltage(self):
        """V_ss = I / (1 - alpha) when V_ss < threshold (no spikes)."""
        n = ParametricLIFNeuron(a=-2.0)  # alpha ≈ 0.119
        I = 0.3
        v_ss_analytical = I / (1.0 - n.alpha)
        # Run long enough to converge
        for _ in range(500):
            n.step(I)
        assert abs(n.v - v_ss_analytical) < 1e-6, (
            f"v={n.v:.6f}, expected V_ss={v_ss_analytical:.6f}"
        )

    def test_geometric_convergence_from_zero(self):
        """Voltage approaches V_ss geometrically: error halves each step at alpha=0.5."""
        n = ParametricLIFNeuron(a=0.0)  # alpha=0.5
        I = 0.3  # V_ss = 0.6, below threshold
        errors = []
        for _ in range(10):
            n.step(I)
            errors.append(abs(n.v - 0.6))
        # Each error ≈ alpha × previous error
        for i in range(1, len(errors)):
            if errors[i - 1] > 1e-12:
                ratio = errors[i] / errors[i - 1]
                assert abs(ratio - 0.5) < 0.01, f"Error ratio = {ratio:.4f}, expected ≈0.5"

    def test_no_leak_when_alpha_near_1(self):
        """With alpha ≈ 1, voltage barely decays — nearly a perfect integrator."""
        n = ParametricLIFNeuron(a=10.0)  # alpha ≈ 0.99995
        n.step(0.3)
        v_after = n.v
        n.step(0.0)  # zero input — should decay by factor alpha
        assert n.v > 0.99 * v_after, f"v decayed from {v_after:.6f} to {n.v:.6f}"

    def test_fast_decay_when_alpha_near_0(self):
        """With alpha ≈ 0, voltage decays almost instantly."""
        n = ParametricLIFNeuron(a=-10.0)  # alpha ≈ 0.00005
        n.step(0.3)
        n.step(0.0)  # zero input
        assert n.v < 0.001, f"v = {n.v} — expected near-zero decay"
