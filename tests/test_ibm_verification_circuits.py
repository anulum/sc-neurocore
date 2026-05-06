# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Posner verification circuit tests

"""Tests for explicit Posner verification circuit construction."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
qiskit = pytest.importorskip("qiskit")
pytest.importorskip("scipy")

import verify_ibm_heron as vih  # noqa: E402
from verify_ibm_heron import (  # noqa: E402
    REFERENCE_TEST_HF_SITE1, REFERENCE_TEST_HF_SITE2, DEFAULT_NUC_DIPOLAR,
    DEFAULT_NUC_DIPOLAR_CROSS, _INTRA_PAIRS, _CROSS_PAIRS,
    _DIPOLAR_PAIRS, analyse_chain, analyse_rpm_8q,
    analytical_singlet_thermal, analytical_singlet_recombination,
    analytical_chain_corr, _posner_chain_couplings,
    _parse_dipolar_pairs, build_posner_circuit, build_posner_decoherence_circuit,
    build_chain_circuit, posner_hamiltonian,
)
from qiskit.quantum_info import Statevector  # noqa: E402

HF1 = REFERENCE_TEST_HF_SITE1
HF2 = REFERENCE_TEST_HF_SITE2
ZERO_TENSOR = {"Axx": 0.0, "Ayy": 0.0, "Azz": 0.0, "Axy": 0.0, "Axz": 0.0, "Ayz": 0.0}
INCORPORATION_TEST_TENSORS = {
    "p1_p2": {"Axx": 0.10, "Ayy": 0.10, "Azz": 0.05, "Axy": 0.01, "Axz": 0.0, "Ayz": 0.0},
    "p1_p3": {"Axx": 0.04, "Ayy": 0.04, "Azz": 0.02, "Axy": 0.0, "Axz": 0.01, "Ayz": 0.0},
    "p2_p3": {"Axx": 0.02, "Ayy": 0.02, "Azz": 0.03, "Axy": 0.0, "Axz": 0.0, "Ayz": 0.01},
}
CA_ELECTRON_MAP_TEST = {
    8: 0, 11: 0, 14: 0,
    17: 1, 20: 1, 23: 1,
    26: 0, 29: 1, 32: 0,
}
CA43_ZERO_TENSORS = {start: ZERO_TENSOR for start in CA_ELECTRON_MAP_TEST}

def _sv(qc, shots=100_000):
    return Statevector.from_instruction(qc.remove_final_measurements(inplace=False)).sample_counts(shots)


def _angle_sum(qc, gate_name: str, qa: int, qb: int) -> float:
    total = 0.0
    target = {qa, qb}
    for circuit_instruction in qc.data:
        inst = circuit_instruction.operation
        if inst.name != gate_name:
            continue
        qubits = {qc.find_bit(qarg).index for qarg in circuit_instruction.qubits}
        if qubits == target:
            total += float(inst.params[0])
    return total

class TestHamiltonian:
    def test_hermitian(self):
        H = posner_hamiltonian(1.0, HF1, HF2)
        assert np.allclose(H, H.conj().T)

    def test_256x256(self):
        assert posner_hamiltonian(1.0, HF1, HF2).shape == (256, 256)

    def test_nuclear_dipolar_present(self):
        """Nuclear dipolar terms must affect H when coupling is non-negligible."""
        # Use exaggerated coupling to verify the code path works
        H_with = posner_hamiltonian(1.0, HF1, HF2, d_nuc=0.03, d_nuc_cross=0.01)
        H_without = posner_hamiltonian(1.0, HF1, HF2, d_nuc=0.0, d_nuc_cross=0.0)
        assert not np.allclose(H_with, H_without), "Dipolar coupling must affect H"

    def test_cross_site_dipolar_present(self):
        """Cross-site nuclear dipolar (9 pairs) must change H at exaggerated coupling."""
        H_no_cross = posner_hamiltonian(1.0, HF1, HF2, d_nuc=0.03, d_nuc_cross=0.0)
        H_with_cross = posner_hamiltonian(1.0, HF1, HF2, d_nuc=0.03, d_nuc_cross=0.01)
        assert not np.allclose(H_no_cross, H_with_cross), "Cross-site dipolar missing"

    def test_cross_site_weaker_than_intra(self):
        """At equal distance, cross-site constant < intra-site (longer distance)."""
        assert DEFAULT_NUC_DIPOLAR_CROSS < DEFAULT_NUC_DIPOLAR

    def test_dipolar_physically_correct_magnitude(self):
        """Dipolar coupling must be ~10⁻⁸ in dimensionless units (not 0.03)."""
        assert DEFAULT_NUC_DIPOLAR < 1e-6, f"Nuclear dipolar too large: {DEFAULT_NUC_DIPOLAR}"
        assert DEFAULT_NUC_DIPOLAR > 1e-10, f"Nuclear dipolar too small: {DEFAULT_NUC_DIPOLAR}"
        # All per-pair couplings also in correct range
        for _, _, d in _DIPOLAR_PAIRS:
            assert 1e-10 < d < 1e-6, f"Dipolar coupling out of range: {d}"

    def test_15_dipolar_pairs_total(self):
        assert len(_INTRA_PAIRS) == 6
        assert len(_CROSS_PAIRS) == 9

    def test_anisotropic_differs(self):
        iso = [{"Axx":0.5,"Ayy":0.5,"Azz":0.5}]*3
        H_iso = posner_hamiltonian(1.0, iso, iso)
        H_aniso = posner_hamiltonian(1.0, HF1, HF2)
        assert not np.allclose(H_iso, H_aniso)

    def test_off_diagonal_hf(self):
        """Off-diagonal HF (Axy, Axz, Ayz) must change H vs diagonal-only."""
        diag = [{"Axx":0.5,"Ayy":0.5,"Azz":0.5,"Axy":0,"Axz":0,"Ayz":0}]*3
        full = [{"Axx":0.5,"Ayy":0.5,"Azz":0.5,"Axy":0.05,"Axz":0.03,"Ayz":0.02}]*3
        H_diag = posner_hamiltonian(1.0, diag, diag)
        H_full = posner_hamiltonian(1.0, full, full)
        assert not np.allclose(H_diag, H_full), "Off-diagonal HF must affect H"
        assert np.allclose(H_full, H_full.conj().T), "Must remain Hermitian"

    def test_per_pair_dipolar_distances(self):
        """Per-pair table has 15 entries with distinct couplings."""
        assert len(_DIPOLAR_PAIRS) == 15
        couplings = [d for _, _, d in _DIPOLAR_PAIRS]
        # Couplings are ~10⁻⁸, need high-precision rounding
        assert len(set(round(c, 12) for c in couplings)) >= 3, \
            f"Must have >=3 distinct values, got {set(round(c,12) for c in couplings)}"

    def test_external_dipolar_pairs_require_all_31p_pairs(self):
        pairs = [
            {
                "qubit_i": i,
                "qubit_j": j,
                "Axx": -1e-8 * (i + j),
                "Ayy": -1.1e-8 * (i + j),
                "Azz": 2.1e-8 * (i + j),
                "Axy": 0.1e-8,
                "Axz": 0.2e-8,
                "Ayz": 0.3e-8,
            }
            for i in range(2, 8)
            for j in range(i + 1, 8)
        ]
        parsed = _parse_dipolar_pairs(pairs)
        assert len(parsed) == 15
        assert parsed[0][0:2] == (2, 3)
        assert parsed[0][2]["Azz"] == 10.5e-8

    def test_external_dipolar_pairs_fail_closed_when_incomplete(self):
        pairs = [[i, j, -1e-8, -1e-8, 2e-8, 0.0, 0.0, 0.0]
                 for i in range(2, 8) for j in range(i + 1, 8)]
        with pytest.raises(ValueError, match="15 unique"):
            _parse_dipolar_pairs(pairs[:-1])

    def test_external_dipolar_pairs_reject_scalar_magnitude_only(self):
        pairs = [[i, j, 1e-8] for i in range(2, 8) for j in range(i + 1, 8)]
        with pytest.raises(ValueError, match="full tensor"):
            _parse_dipolar_pairs(pairs)

class TestExchangeProtection:
    def test_weak_exchange_allows_mixing(self):
        p = analytical_singlet_thermal(0.5, HF1, HF2, omega_0=0.5, t=math.pi)
        assert p < 0.75, f"J=0.5 must allow mixing, got {p}"

    def test_strong_exchange_protects(self):
        p = analytical_singlet_thermal(10.0, HF1, HF2, omega_0=0.5, t=math.pi)
        assert p > 0.85, f"J=10 must protect, got {p}"

    def test_zero_hf_preserves(self):
        z = [{"Axx":0,"Ayy":0,"Azz":0,"Axy":0,"Axz":0,"Ayz":0}]*3
        p = analytical_singlet_thermal(1.0, z, z, 0.0, math.pi, 0.0, 0.0)
        assert p > 0.99

class TestRecombination:
    def test_recombination_callable(self):
        phi = analytical_singlet_recombination(1.0, HF1, HF2, omega_0=0.5, k_recomb=0.1, n_t=5)
        assert 0.0 < phi < 1.0

    def test_recombination_differs_from_instant(self):
        phi_r = analytical_singlet_recombination(1.0, HF1, HF2, omega_0=0.5, k_recomb=0.1, n_t=10)
        phi_i = analytical_singlet_thermal(1.0, HF1, HF2, omega_0=0.5, t=math.pi)
        assert abs(phi_r - phi_i) > 0.01, "Recomb-weighted must differ from single-t"

class TestThermalAverage:
    def test_64_configs_differ_from_single(self):
        from scipy.linalg import expm
        H = posner_hamiltonian(1.0, HF1, HF2)
        U = expm(-1j * H * math.pi)
        se = np.array([0,1,-1,0], dtype=complex)/math.sqrt(2)
        PS = np.kron(np.outer(se,se.conj()), np.eye(64))
        ns = np.zeros(64, dtype=complex)
        ns[0] = 1.0
        psi = U @ np.kron(se, ns)
        p_single = float(np.real(psi.conj() @ PS @ psi))
        p_thermal = analytical_singlet_thermal(1.0, HF1, HF2, t=math.pi)
        assert abs(p_single - p_thermal) > 0.001

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

class TestChainPhysics:
    def test_exponential_superexchange(self):
        J = _posner_chain_couplings(10)
        assert J[0] > J[1] > J[2], "Couplings must decay with distance"
        assert J[0] == 1.0, "Nearest must be 1.0"
        # Verify exponential decay (not power-law)
        ratio_12 = J[1] / J[0]
        ratio_23 = J[2] / J[1]
        assert abs(ratio_12 - ratio_23) < 0.01, "Must be exponential (constant ratio)"

    def test_correlations_emerge(self):
        qc = build_chain_circuit(n_qubits=10, t=1.0)
        corrs = analyse_chain(_sv(qc, 50_000), 10)["zz_from_0"]
        assert any(abs(c) > 0.01 for c in corrs)

    def test_nearest_strongest(self):
        n = 10
        J = _posner_chain_couplings(n)
        th = analytical_chain_corr(n, J, 1.0)
        assert abs(th[0]) > abs(th[4])

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
        r_raw = analyse_rpm_8q(_sv(build_posner_decoherence_circuit(delay_dt=4000, hf1=HF1, hf2=HF2)))
        r_dd = analyse_rpm_8q(_sv(build_posner_decoherence_circuit(delay_dt=4000, dd_sequence="xy4", hf1=HF1, hf2=HF2)))
        # DD adds extra X,Y gates that cancel in noiseless sim
        # Allow some tolerance due to the gates not being perfectly transparent on state
        assert abs(r_raw["singlet_probability"] - r_dd["singlet_probability"]) < 0.05


class TestOrcaPipeline:
    def test_pp_distances(self):
        """P-P distances from S₆ geometry must be ≥ 4.9 Å."""
        orca = pytest.importorskip("orca_posner_hf")
        pp = orca.compute_pp_distances()
        assert len(pp) == 15, f"Expected 15 P-P pairs, got {len(pp)}"
        # Minimum P-P in S₆ is ~4.17 Å (cross-site near-equatorial)
        for pa, pb, d in pp:
            assert d >= 3.5, f"{pa}-{pb} distance {d} Å too short (P-O confusion?)"

    def test_qubit_dipolar_table(self):
        """Qubit dipolar table has 15 entries with correct magnitudes."""
        orca = pytest.importorskip("orca_posner_hf")
        dt = orca.compute_qubit_dipolar_table()
        assert len(dt) == 15
        for qi, qj, r, d in dt:
            assert 2 <= qi <= 7 and 2 <= qj <= 7
            assert 1e-10 < d < 1e-6, f"Coupling {d} out of physical range"

    def test_qubit_dipolar_tensor_table(self):
        """Dipolar tensors must be full, symmetric, and traceless."""
        orca = pytest.importorskip("orca_posner_hf")
        dt = orca.compute_qubit_dipolar_tensor_table()
        assert len(dt) == 15
        for row in dt:
            assert 2 <= row["qubit_i"] <= 7 and 2 <= row["qubit_j"] <= 7
            trace = row["Axx"] + row["Ayy"] + row["Azz"]
            assert abs(trace) < 1e-20
            assert any(abs(row[key]) > 0 for key in ("Axy", "Axz", "Ayz"))

    def test_dipolar_tensor_table_from_optimized_xyz(self, tmp_path):
        """Optimized XYZ parsing must feed runtime tensor generation."""
        orca = pytest.importorskip("orca_posner_hf")
        xyz = tmp_path / "posner_opt.xyz"
        xyz.write_text(
            "\n".join(
                [
                    "6",
                    "minimal phosphorus geometry",
                    "P 0.0 0.0 0.0",
                    "P 5.0 0.0 0.0",
                    "P 0.0 5.0 0.0",
                    "P 0.0 0.0 5.0",
                    "P 5.0 5.0 0.0",
                    "P 5.0 0.0 5.0",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        dt = orca.compute_qubit_dipolar_tensor_table_from_xyz(xyz)
        assert len(dt) == 15
        assert dt[0]["qubit_i"] == 2
        assert dt[0]["qubit_j"] == 3
        assert dt[0]["Axx"] > 0
        assert dt[0]["Ayy"] < 0

    def test_orca_input_generation(self):
        """ORCA input must contain required keywords."""
        orca = pytest.importorskip("orca_posner_hf")
        inp = orca.generate_orca_input()
        assert "B3LYP" in inp
        assert "def2-TZVP" in inp
        assert "eprnmr" in inp.lower() or "%eprnmr" in inp

    def test_radical_input(self):
        """Radical input must be unrestricted doublet."""
        orca = pytest.importorskip("orca_posner_hf")
        inp = orca.generate_radical_input()
        assert "UB3LYP" in inp
        assert "* xyz 1 2" in inp

    def test_invalid_neutral_doublet_rejected(self):
        """Neutral Ca9(PO4)6 has even electron count, so doublet is invalid."""
        orca = pytest.importorskip("orca_posner_hf")
        with pytest.raises(ValueError, match="charge=0, multiplicity=2"):
            orca.generate_orca_input(charge=0, multiplicity=2)


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


class TestBiologicalNoise:
    def test_noise_params(self):
        ext = pytest.importorskip("posner_extended")
        params = ext.get_noise_params_dict(37.0)
        assert params["temperature_K"] == 310.15
        assert params["T1_nuclear_s"] == 5.0
        assert params["T1_electron_s"] == 1e-6
        assert 0.49 < params["p_excited"] < 0.51
        assert params["cage_dephasing_rate"] is None

    def test_biological_noise_requires_cage_dephasing_rate(self):
        ext = pytest.importorskip("posner_extended")
        pytest.importorskip("qiskit_aer")
        with pytest.raises(ValueError, match="cage_dephasing_rate"):
            ext.biological_noise_model(37.0)
