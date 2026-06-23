# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Contract tests for SCPN L1 quantum layer

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from sc_neurocore.quantum.hybrid import QuantumStochasticLayer
from sc_neurocore.scpn.layers.l1_quantum import L1_QuantumLayer, L1_StochasticParameters


def test_l1_seed_scopes_input_and_core_output_bitstreams() -> None:
    params = L1_StochasticParameters(n_qubits=8, bitstream_length=64, rng_seed=123)
    layer_a = L1_QuantumLayer(params)
    layer_b = L1_QuantumLayer(params)

    out_a0 = layer_a.step(0.01)
    out_b0 = layer_b.step(0.01)
    out_a1 = layer_a.step(0.01)
    out_b1 = layer_b.step(0.01)

    np.testing.assert_array_equal(out_a0, out_b0)
    np.testing.assert_array_equal(out_a1, out_b1)
    assert not np.array_equal(out_a0, out_a1)


def test_quantum_stochastic_layer_seed_scopes_measurement() -> None:
    input_bits = np.ones((4, 32), dtype=np.uint8)
    core_a = QuantumStochasticLayer(n_qubits=4, length=32, rng_seed=456)
    core_b = QuantumStochasticLayer(n_qubits=4, length=32, rng_seed=456)

    np.testing.assert_array_equal(core_a.forward(input_bits), core_b.forward(input_bits))


def test_l1_external_field_is_validated_and_applied() -> None:
    layer = L1_QuantumLayer(
        L1_StochasticParameters(
            n_qubits=6,
            bitstream_length=32,
            coupling_strength=0.5,
            decoherence_rate=0.0,
            rng_seed=789,
        )
    )

    before = layer.coherence_probs.copy()
    layer.step(0.01, external_field=np.zeros(6))

    assert np.mean(layer.coherence_probs) < np.mean(before)


def test_l1_rejects_invalid_parameters_and_inputs() -> None:
    with pytest.raises(ValueError, match="n_qubits"):
        L1_QuantumLayer(L1_StochasticParameters(n_qubits=0))
    with pytest.raises(ValueError, match="bitstream_length"):
        L1_QuantumLayer(L1_StochasticParameters(bitstream_length=0))
    with pytest.raises(ValueError, match="F_non_Markov"):
        L1_QuantumLayer(L1_StochasticParameters(F_non_Markov=1.0))
    with pytest.raises(ValueError, match="temperature"):
        L1_QuantumLayer(L1_StochasticParameters(temperature=0.0))
    with pytest.raises(ValueError, match="coupling_strength"):
        L1_QuantumLayer(L1_StochasticParameters(coupling_strength=1.1))
    with pytest.raises(ValueError, match="decoherence_rate"):
        L1_QuantumLayer(L1_StochasticParameters(decoherence_rate=-0.1))
    with pytest.raises(ValueError, match="backend"):
        L1_QuantumLayer(L1_StochasticParameters(backend=""))
    with pytest.raises(ValueError, match="rng_seed"):
        L1_QuantumLayer(L1_StochasticParameters(rng_seed=cast(Any, 1.5)))

    layer = L1_QuantumLayer(L1_StochasticParameters(n_qubits=4, bitstream_length=16))
    with pytest.raises(ValueError, match="dt"):
        layer.step(0.0)
    with pytest.raises(ValueError, match="external_field"):
        layer.step(0.01, external_field=np.ones(3))
    with pytest.raises(ValueError, match="external_field"):
        layer.step(0.01, external_field=np.array([0.0, 0.0, 0.0, np.nan]))


def test_l1_non_simulated_backend_builds_hardware_core() -> None:
    layer = L1_QuantumLayer(
        L1_StochasticParameters(n_qubits=2, bitstream_length=16, backend="qiskit", rng_seed=1)
    )
    assert type(layer.quantum_core).__name__ == "QuantumHardwareLayer"


def test_l1_get_global_metric_returns_mean_coherence() -> None:
    layer = L1_QuantumLayer(L1_StochasticParameters(n_qubits=4, bitstream_length=16, rng_seed=2))
    metric = layer.get_global_metric()
    assert 0.0 <= metric <= 1.0


def test_l1_negative_seed_and_out_of_range_external_field_rejected() -> None:
    with pytest.raises(ValueError, match="rng_seed"):
        L1_QuantumLayer(L1_StochasticParameters(rng_seed=-1))
    layer = L1_QuantumLayer(L1_StochasticParameters(n_qubits=4, bitstream_length=16))
    with pytest.raises(ValueError, match="external_field must be within"):
        layer.step(0.01, external_field=np.array([0.0, 0.0, 0.0, 1.5]))
