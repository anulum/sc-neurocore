# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCa43Enriched from former test_ibm_verification_circuits.py

"""Focused suite: TestCa43Enriched from former test_ibm_verification_circuits.py."""

from __future__ import annotations

from tests.ibm_verification_circuits_support import *  # noqa: F403


class TestCa43Enriched:
    def test_circuit_35q(self):
        ext = pytest.importorskip("posner_extended")
        qc = ext.build_posner_43ca_circuit(
            n_trotter=1,
            p31_hf_site1=HF1,
            p31_hf_site2=HF2,
            ca43_hf_tensors=CA43_ZERO_TENSORS,
            ca_electron_map=CA_ELECTRON_MAP_TEST,
        )
        assert qc.num_qubits == 35

    def test_missing_ca43_inputs_fail_closed(self):
        ext = pytest.importorskip("posner_extended")
        with pytest.raises(ValueError, match="ca43_hf_tensors"):
            ext.build_posner_43ca_circuit(
                p31_hf_site1=HF1,
                p31_hf_site2=HF2,
                ca_electron_map=CA_ELECTRON_MAP_TEST,
            )

    def test_analysis_keys(self):
        """35q circuit analysis with minimal HF (MPS-tractable).

        Full I·S coupling (ca43_hf=0.01) creates too much entanglement
        for MPS simulation. CI test uses near-zero HF to verify the
        analysis pipeline. Full model is validated on IBM hardware.
        """
        ext = pytest.importorskip("posner_extended")
        qiskit_aer = pytest.importorskip("qiskit_aer")
        from qiskit import transpile

        sim = qiskit_aer.AerSimulator(
            method="matrix_product_state",
            matrix_product_state_max_bond_dimension=32,
        )
        # Near-zero HF to keep MPS tractable (validates analysis pipeline)
        _small_hf = [
            {"Axx": 0.01, "Ayy": 0.01, "Azz": 0.01, "Axy": 0.0, "Axz": 0.0, "Ayz": 0.0}
        ] * 3
        qc = ext.build_posner_43ca_circuit(
            n_trotter=1,
            p31_hf_site1=_small_hf,
            p31_hf_site2=_small_hf,
            ca43_hf_tensors=CA43_ZERO_TENSORS,
            ca_electron_map=CA_ELECTRON_MAP_TEST,
            J=0.01,
        )
        tqc = transpile(qc, sim)
        res = sim.run(tqc, shots=200).result()
        counts = res.get_counts()
        r = ext.analyse_43ca(counts)
        assert "singlet_probability" in r
        assert "ca_polarizations" in r
        assert len(r["ca_polarizations"]) == 9  # 9 calcium ions

    def test_dipolar_angles_match_hamiltonian_convention_35q(self, monkeypatch):
        ext = pytest.importorskip("posner_extended")
        orca = pytest.importorskip("orca_posner_hf")
        t = 1.25
        qi, qj = 2, 3
        tensor = {"Axx": -1e-8, "Ayy": -2e-8, "Azz": 3e-8, "Axy": 0.0, "Axz": 0.0, "Ayz": 0.0}
        row = {"qubit_i": qi, "qubit_j": qj, **tensor}
        monkeypatch.setattr(orca, "compute_qubit_dipolar_tensor_table", lambda: [row])
        zero_hf = [ZERO_TENSOR] * 3
        qc = ext.build_posner_43ca_circuit(
            J=0.0,
            omega_0=0.0,
            t=t,
            n_trotter=1,
            p31_hf_site1=zero_hf,
            p31_hf_site2=zero_hf,
            ca43_hf_tensors=CA43_ZERO_TENSORS,
            ca_electron_map=CA_ELECTRON_MAP_TEST,
        )
        assert math.isclose(_angle_sum(qc, "rzz", qi, qj), tensor["Azz"] * t / 2, rel_tol=1e-12)
        assert math.isclose(_angle_sum(qc, "rxx", qi, qj), tensor["Axx"] * t / 2, rel_tol=1e-12)
        assert math.isclose(_angle_sum(qc, "ryy", qi, qj), tensor["Ayy"] * t / 2, rel_tol=1e-12)
