# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestOrcaPipeline from former test_ibm_verification_circuits.py

"""Focused suite: TestOrcaPipeline from former test_ibm_verification_circuits.py."""

from __future__ import annotations

import pytest

pytest.importorskip("qiskit")

from tests.ibm_verification_circuits_support import *  # noqa: F403


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
