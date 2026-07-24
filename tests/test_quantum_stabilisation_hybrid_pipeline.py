# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHybridPipeline from former test_quantum_stabilisation.py

"""Focused suite: TestHybridPipeline from former test_quantum_stabilisation.py."""

from __future__ import annotations

from tests.quantum_stabilisation_support import *  # noqa: F403


class TestHybridPipeline:
    def test_circuit_returns_scalar(self):
        from sc_neurocore.quantum.hybrid_pipeline import HybridQuantumClassicalPipeline

        pipe = HybridQuantumClassicalPipeline(n_qubits=2, n_layers=1)
        params = np.zeros(pipe.n_params)
        val = pipe.circuit(params)
        assert isinstance(val, float)
        assert -1.0 <= val <= 1.0

    def test_vqe_converges(self):
        from sc_neurocore.quantum.hybrid_pipeline import HybridQuantumClassicalPipeline

        pipe = HybridQuantumClassicalPipeline(n_qubits=2, n_layers=1)
        history, params = pipe.train(n_steps=30, lr=0.05)
        assert history[-1] <= history[0] + 0.5

    def test_evaluate(self):
        from sc_neurocore.quantum.hybrid_pipeline import HybridQuantumClassicalPipeline

        pipe = HybridQuantumClassicalPipeline(n_qubits=2, n_layers=1)
        _, params = pipe.train(n_steps=10, lr=0.05)
        val = pipe.evaluate(params)
        assert isinstance(val, float)
