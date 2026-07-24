# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTwoPosnerTransport from former test_ibm_verification_circuits.py

"""Focused suite: TestTwoPosnerTransport from former test_ibm_verification_circuits.py."""

from __future__ import annotations

from tests.ibm_verification_circuits_support import *  # noqa: F403


class TestTwoPosnerTransport:
    def test_circuit_16q(self):
        ext = pytest.importorskip("posner_extended")
        qc = ext.build_two_posner_transport_circuit(
            hf_site1=HF1,
            hf_site2=HF2,
            incorporation_tensors=INCORPORATION_TEST_TENSORS,
            transport_delay_dt=0,
        )
        assert qc.num_qubits == 16
        assert qc.count_ops().get("cz", 0) == 0

    def test_analysis(self):
        ext = pytest.importorskip("posner_extended")
        qc = ext.build_two_posner_transport_circuit(
            hf_site1=HF1,
            hf_site2=HF2,
            incorporation_tensors=INCORPORATION_TEST_TENSORS,
            transport_delay_dt=0,
        )
        counts = _sv(qc, 50_000)
        r = ext.analyse_two_posner(counts)
        assert "binding_probability" in r
        assert 0 <= r["binding_probability"] <= 1

    def test_zero_transport_has_bounded_binding_probability(self):
        """Zero transport delay uses explicit fixture tensors and stays bounded."""
        ext = pytest.importorskip("posner_extended")
        qc = ext.build_two_posner_transport_circuit(
            hf_site1=HF1,
            hf_site2=HF2,
            incorporation_tensors=INCORPORATION_TEST_TENSORS,
            transport_delay_dt=0,
        )
        counts = _sv(qc, 50_000)
        r = ext.analyse_two_posner(counts)
        assert 0.0 <= r["binding_probability"] <= 1.0

    def test_missing_hf_fails_closed(self):
        ext = pytest.importorskip("posner_extended")
        with pytest.raises(ValueError, match="hf_site1"):
            ext.build_two_posner_transport_circuit(
                incorporation_tensors=INCORPORATION_TEST_TENSORS,
                transport_delay_dt=0,
            )

    def test_missing_incorporation_tensors_fails_closed(self):
        ext = pytest.importorskip("posner_extended")
        with pytest.raises(ValueError, match="incorporation_tensors"):
            ext.build_two_posner_transport_circuit(
                hf_site1=HF1,
                hf_site2=HF2,
                transport_delay_dt=0,
            )

    def test_nonzero_transport_requires_explicit_noise_rate(self):
        ext = pytest.importorskip("posner_extended")
        pytest.importorskip("qiskit_aer")
        with pytest.raises(ValueError, match="transport_depolarizing_rate"):
            ext.build_two_posner_transport_circuit(
                hf_site1=HF1,
                hf_site2=HF2,
                incorporation_tensors=INCORPORATION_TEST_TENSORS,
                transport_delay_dt=1000,
            )

    def test_dipolar_angles_match_hamiltonian_convention_16q(self, monkeypatch):
        ext = pytest.importorskip("posner_extended")
        orca = pytest.importorskip("orca_posner_hf")
        t = 1.25
        qi, qj = 2, 3
        tensor = {"Axx": -1e-8, "Ayy": -2e-8, "Azz": 3e-8, "Axy": 0.0, "Axz": 0.0, "Ayz": 0.0}
        row = {"qubit_i": qi, "qubit_j": qj, **tensor}
        monkeypatch.setattr(orca, "compute_qubit_dipolar_tensor_table", lambda: [row])
        zero_hf = [ZERO_TENSOR] * 3
        zero_inc = {key: ZERO_TENSOR for key in INCORPORATION_TEST_TENSORS}
        qc = ext.build_two_posner_transport_circuit(
            J=0.0,
            omega_0=0.0,
            t_evolve=t,
            n_trotter=1,
            transport_delay_dt=0,
            dd_during_transport=False,
            hf_site1=zero_hf,
            hf_site2=zero_hf,
            incorporation_tensors=zero_inc,
        )
        assert math.isclose(_angle_sum(qc, "rzz", qi, qj), tensor["Azz"] * t / 2, rel_tol=1e-12)
        assert math.isclose(_angle_sum(qc, "rxx", qi, qj), tensor["Axx"] * t / 2, rel_tol=1e-12)
        assert math.isclose(_angle_sum(qc, "ryy", qi, qj), tensor["Ayy"] * t / 2, rel_tol=1e-12)
