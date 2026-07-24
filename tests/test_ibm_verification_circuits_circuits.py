# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCircuits from former test_ibm_verification_circuits.py

"""Focused suite: TestCircuits from former test_ibm_verification_circuits.py."""

from __future__ import annotations

from tests.ibm_verification_circuits_support import *  # noqa: F403


class TestCircuits:
    def test_posner_8q(self):
        qc = build_posner_circuit(hf1=HF1, hf2=HF2)
        assert qc.num_qubits == 8

    def test_decoherence_uses_delay(self):
        qc = build_posner_decoherence_circuit(delay_dt=1000, hf1=HF1, hf2=HF2)
        ops = qc.count_ops()
        assert "delay" in ops, f"Must use delay, got {ops}"

    def test_decoherence_8q(self):
        qc = build_posner_decoherence_circuit(delay_dt=0, hf1=HF1, hf2=HF2)
        assert qc.num_qubits == 8

    def test_chain_10q(self):
        qc = build_chain_circuit(n_qubits=10)
        assert qc.num_qubits == 10

    def test_nuclear_dipolar_in_circuit(self):
        """Circuit must contain RZZ gates for nuclear dipolar coupling."""
        qc = build_posner_circuit(n_trotter=1, hf1=HF1, hf2=HF2)
        ops = qc.count_ops()
        assert ops.get("rzz", 0) >= 16, f"Must have tensor dipolar RZZ, got {ops}"

    def test_dipolar_angles_match_hamiltonian_convention_8q(self):
        """RPP(theta)=exp(-i theta PP/2), so theta=Aab*t/2."""
        t = 1.25
        qi, qj = 2, 3
        tensor = {"Axx": -1e-8, "Ayy": -2e-8, "Azz": 3e-8, "Axy": 0.0, "Axz": 0.0, "Ayz": 0.0}
        zero_hf = [ZERO_TENSOR] * 3
        old_tensors = vih._DIPOLAR_TENSORS
        try:
            vih._DIPOLAR_TENSORS = [(qi, qj, tensor)]
            qc = build_posner_circuit(
                J=0.0,
                omega_0=0.0,
                t=t,
                n_trotter=1,
                hf1=zero_hf,
                hf2=zero_hf,
            )
        finally:
            vih._DIPOLAR_TENSORS = old_tensors
        assert math.isclose(_angle_sum(qc, "rzz", qi, qj), tensor["Azz"] * t / 2, rel_tol=1e-12)
        assert math.isclose(_angle_sum(qc, "rxx", qi, qj), tensor["Axx"] * t / 2, rel_tol=1e-12)
        assert math.isclose(_angle_sum(qc, "ryy", qi, qj), tensor["Ayy"] * t / 2, rel_tol=1e-12)
