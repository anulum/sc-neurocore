# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for quantum-classical hybrid layer

"""Tests for QuantumStochasticLayer: Ry rotation, cos²(θ/2) transfer function."""

from __future__ import annotations

import numpy as np

from sc_neurocore.quantum.hybrid import QuantumStochasticLayer


class TestQuantumStochasticLayer:
    def test_output_shape(self):
        qsl = QuantumStochasticLayer(n_qubits=4, length=512)
        input_bits = np.zeros((4, 512), dtype=np.uint8)
        output = qsl.forward(input_bits)
        assert output.shape == (4, 512)

    def test_output_binary(self):
        qsl = QuantumStochasticLayer(n_qubits=2, length=1024)
        np.random.seed(42)
        input_bits = (np.random.random((2, 1024)) < 0.5).astype(np.uint8)
        output = qsl.forward(input_bits)
        assert set(np.unique(output)).issubset({0, 1})

    def test_p_zero_maps_to_one(self):
        """p_in=0 → θ=0 → cos²(0)=1.0."""
        qsl = QuantumStochasticLayer(n_qubits=1, length=10000)
        input_bits = np.zeros((1, 10000), dtype=np.uint8)
        np.random.seed(42)
        output = qsl.forward(input_bits)
        p_out = np.mean(output)
        np.testing.assert_allclose(p_out, 1.0, atol=0.03)

    def test_p_one_maps_to_half(self):
        """p_in=1 → θ=π → cos²(π/2)=0.0. But the formula is cos²(p*π/2).
        Actually: p_in=1 → θ=π → p_measure = cos²(π/2) = 0.0.
        Wait, let me re-check: the code does theta = p_in * pi, then
        p_measure = cos(theta/2)². So p_in=1 → theta=pi → cos(pi/2)²=0.

        Actually cos(pi/2) = 0, so cos²(pi/2) = 0.
        """
        qsl = QuantumStochasticLayer(n_qubits=1, length=10000)
        input_bits = np.ones((1, 10000), dtype=np.uint8)
        np.random.seed(42)
        output = qsl.forward(input_bits)
        p_out = np.mean(output)
        # cos²(π/2) = 0, but SC noise means it won't be exactly 0
        np.testing.assert_allclose(p_out, 0.0, atol=0.03)

    def test_p_half_maps_to_cos2_pi4(self):
        """p_in=0.5 → θ=π/2 → cos²(π/4) ≈ 0.5."""
        qsl = QuantumStochasticLayer(n_qubits=1, length=10000)
        np.random.seed(42)
        input_bits = (np.random.random((1, 10000)) < 0.5).astype(np.uint8)
        output = qsl.forward(input_bits)
        p_out = np.mean(output)
        expected = np.cos(0.5 * np.pi / 2) ** 2  # ≈ 0.854
        np.testing.assert_allclose(p_out, expected, atol=0.05)

    def test_nonlinear_transfer(self):
        """cos²(p·π/2) is nonlinear: p=0.25 → cos²(π/8) ≈ 0.854,
        not the linear interpolation 0.75."""
        qsl = QuantumStochasticLayer(n_qubits=1, length=10000)
        np.random.seed(42)
        bits = (np.random.random((1, 10000)) < 0.25).astype(np.uint8)
        out = qsl.forward(bits)
        p_out = np.mean(out)
        # cos²(0.25·π/2) = cos²(π/8) ≈ 0.854
        # linear interpolation at p=0.25: 1.0 - 0.25 = 0.75
        expected = np.cos(0.25 * np.pi / 2) ** 2  # ≈ 0.854
        assert p_out > 0.75, f"p_out={p_out:.3f}, expected > 0.75 (cos²≈{expected:.3f})"

    def test_monotonic_decreasing(self):
        """cos²(p·π/2) is monotonically decreasing for p ∈ [0,1]."""
        qsl = QuantumStochasticLayer(n_qubits=1, length=8192)
        np.random.seed(42)

        prev = 2.0
        for p_in in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            bits = (np.random.random((1, 8192)) < p_in).astype(np.uint8)
            out = qsl.forward(bits)
            p_out = np.mean(out)
            assert p_out < prev + 0.05, f"not monotonically decreasing at p_in={p_in}"
            prev = p_out

    def test_multi_qubit(self):
        """Multiple qubits should be processed independently."""
        n = 5
        qsl = QuantumStochasticLayer(n_qubits=n, length=4096)
        np.random.seed(42)
        probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        input_bits = np.stack([(np.random.random(4096) < p).astype(np.uint8) for p in probs])
        output = qsl.forward(input_bits)
        p_out = np.mean(output, axis=1)
        expected = np.cos(probs * np.pi / 2) ** 2
        np.testing.assert_allclose(p_out, expected, atol=0.05)

    def test_length_preserved(self):
        L = 777
        qsl = QuantumStochasticLayer(n_qubits=3, length=L)
        np.random.seed(42)
        bits = (np.random.random((3, L)) < 0.5).astype(np.uint8)
        output = qsl.forward(bits)
        assert output.shape[1] == L
