# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDecoherencePhysics from former test_ibm_verification_circuits.py

"""Focused suite: TestDecoherencePhysics from former test_ibm_verification_circuits.py."""

from __future__ import annotations

import pytest

pytest.importorskip("qiskit")

from tests.ibm_verification_circuits_support import *  # noqa: F403


class TestDecoherencePhysics:
    def test_zero_delay_preserves(self):
        qc = build_posner_decoherence_circuit(delay_dt=0, hf1=HF1, hf2=HF2)
        r = analyse_rpm_8q(_sv(qc))
        assert r["singlet_probability"] > 0.3

    def test_exact_sim_stable(self):
        r0 = analyse_rpm_8q(_sv(build_posner_decoherence_circuit(delay_dt=0, hf1=HF1, hf2=HF2)))
        r1 = analyse_rpm_8q(_sv(build_posner_decoherence_circuit(delay_dt=5000, hf1=HF1, hf2=HF2)))
        assert abs(r0["singlet_probability"] - r1["singlet_probability"]) < 0.02

    def test_xy4_dd_circuit(self):
        """XY-4 DD: electron-only, proper symmetric spacing."""
        qc = build_posner_decoherence_circuit(delay_dt=4000, dd_sequence="xy4", hf1=HF1, hf2=HF2)
        ops = qc.count_ops()
        assert "delay" in ops, f"DD must have delay, got {ops}"
        # XY-4 on electrons only (q0, q1): 2 Y-pulses per electron × 2 = 4 Y total
        assert ops.get("y", 0) == 4, f"Expected 4 Y gates (electron-only DD), got {ops.get('y', 0)}"
        # X-pulses: 2 from DD per electron (4 total) + 1 from singlet prep (q1) = 5
        assert ops.get("x", 0) == 5, f"Expected 5 X gates, got {ops.get('x', 0)}"

    def test_dd_vs_raw_same_on_simulator(self):
        """On exact simulator, DD and raw give same result (no noise)."""
        r_raw = analyse_rpm_8q(
            _sv(build_posner_decoherence_circuit(delay_dt=4000, hf1=HF1, hf2=HF2))
        )
        r_dd = analyse_rpm_8q(
            _sv(
                build_posner_decoherence_circuit(delay_dt=4000, dd_sequence="xy4", hf1=HF1, hf2=HF2)
            )
        )
        # DD adds extra X,Y gates that cancel in noiseless sim
        # Allow some tolerance due to the gates not being perfectly transparent on state
        assert abs(r_raw["singlet_probability"] - r_dd["singlet_probability"]) < 0.05
