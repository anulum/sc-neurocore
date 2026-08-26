# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestChainPhysics from former test_ibm_verification_circuits.py

"""Focused suite: TestChainPhysics from former test_ibm_verification_circuits.py."""

from __future__ import annotations

import pytest

pytest.importorskip("qiskit")

from tests.ibm_verification_circuits_support import *  # noqa: F403


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
