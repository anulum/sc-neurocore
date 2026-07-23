# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestThermalAverage from former test_ibm_verification_circuits.py

"""Focused suite: TestThermalAverage from former test_ibm_verification_circuits.py."""

from __future__ import annotations

from tests.ibm_verification_circuits_support import *  # noqa: F403

class TestThermalAverage:
    def test_64_configs_differ_from_single(self):
        from scipy.linalg import expm

        H = posner_hamiltonian(1.0, HF1, HF2)
        U = expm(-1j * H * math.pi)
        se = np.array([0, 1, -1, 0], dtype=complex) / math.sqrt(2)
        PS = np.kron(np.outer(se, se.conj()), np.eye(64))
        ns = np.zeros(64, dtype=complex)
        ns[0] = 1.0
        psi = U @ np.kron(se, ns)
        p_single = float(np.real(psi.conj() @ PS @ psi))
        p_thermal = analytical_singlet_thermal(1.0, HF1, HF2, t=math.pi)
        assert abs(p_single - p_thermal) > 0.001
