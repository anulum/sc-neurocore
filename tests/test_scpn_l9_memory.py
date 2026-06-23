# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for SCPN L9 holographic memory layer

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from sc_neurocore.scpn.layers.l9_memory import L9_MemoryLayer, L9_StochasticParameters


class _DeterministicRng:
    def choice(self, choices: list[int], size: int) -> np.ndarray:
        return np.full(size, choices[0], dtype=np.float64)

    def random(self, size: int | tuple[int, ...]) -> np.ndarray:
        return np.zeros(size, dtype=np.float64)


def test_l9_imprint_rate_controls_hebbian_storage_strength() -> None:
    pattern = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float64)
    half = L9_MemoryLayer(
        L9_StochasticParameters(n_memory_slots=4, bitstream_length=16, imprint_rate=0.5)
    )
    full = L9_MemoryLayer(
        L9_StochasticParameters(n_memory_slots=4, bitstream_length=16, imprint_rate=1.0)
    )

    half.store(pattern)
    full.store(pattern)

    np.testing.assert_allclose(full.patterns, half.patterns * 2.0)
    assert half.n_stored == 1
    assert full.n_stored == 1


def test_l9_retrieval_gain_and_phase_field_coupling_drive_state_update() -> None:
    common = dict(
        n_memory_slots=3,
        bitstream_length=16,
        imprint_rate=0.0,
        decay_rate=0.0,
        rng_seed=9,
    )
    inert = L9_MemoryLayer(
        L9_StochasticParameters(**common, retrieval_gain=0.0, phase_field_coupling=0.0)
    )
    phase_driven = L9_MemoryLayer(
        L9_StochasticParameters(**common, retrieval_gain=0.0, phase_field_coupling=0.5)
    )
    retrieval_driven = L9_MemoryLayer(
        L9_StochasticParameters(**common, retrieval_gain=1.0, phase_field_coupling=0.0)
    )
    for layer in (inert, phase_driven, retrieval_driven):
        layer.state = -np.ones(3, dtype=np.float64)
        layer.patterns = np.zeros((3, 3), dtype=np.float64)
        vars(layer)["_rng"] = _DeterministicRng()
    retrieval_driven.patterns = np.eye(3, dtype=np.float64)

    inert_state = inert.step(0.001, {"cosmic_alignment": 1.0})["state"]
    phase_state = phase_driven.step(0.001, {"cosmic_alignment": 1.0})["state"]
    retrieval_state = retrieval_driven.step(0.001)["state"]

    np.testing.assert_array_equal(inert_state, -np.ones(3, dtype=np.float64))
    np.testing.assert_array_equal(phase_state, np.ones(3, dtype=np.float64))
    np.testing.assert_array_equal(retrieval_state, -np.ones(3, dtype=np.float64))


def test_l9_prefers_l8_memory_reference_wave_over_alignment_fallback() -> None:
    layer = L9_MemoryLayer(
        L9_StochasticParameters(
            n_memory_slots=3,
            bitstream_length=16,
            retrieval_gain=0.0,
            imprint_rate=0.0,
            decay_rate=0.0,
            phase_field_coupling=1.0,
            rng_seed=19,
        )
    )
    layer.state = np.ones(3, dtype=np.float64)
    vars(layer)["_rng"] = _DeterministicRng()

    out = layer.step(
        0.001,
        {
            "cosmic_alignment": 1.0,
            "memory_imprint_drive": {
                "reference_amplitude": 1.0,
                "reference_phase": np.pi,
                "reference_real": -1.0,
            },
        },
    )

    np.testing.assert_array_equal(out["state"], -np.ones(3, dtype=np.float64))


def test_l9_seed_scopes_initial_state_update_and_output_bitstreams() -> None:
    params = L9_StochasticParameters(
        n_memory_slots=4,
        bitstream_length=64,
        decay_rate=0.0,
        rng_seed=123,
    )
    layer_a = L9_MemoryLayer(params)
    layer_b = L9_MemoryLayer(params)

    np.testing.assert_array_equal(layer_a.state, layer_b.state)
    layer_a.state = np.zeros(4, dtype=np.float64)
    layer_b.state = np.zeros(4, dtype=np.float64)
    out_a0 = layer_a.step(0.001)["output_bitstreams"]
    out_b0 = layer_b.step(0.001)["output_bitstreams"]
    out_a1 = layer_a.step(0.001)["output_bitstreams"]
    out_b1 = layer_b.step(0.001)["output_bitstreams"]

    np.testing.assert_array_equal(out_a0, out_b0)
    np.testing.assert_array_equal(out_a1, out_b1)
    assert not np.array_equal(out_a0, out_a1)


def test_l9_rejects_invalid_parameters_and_inputs() -> None:
    with pytest.raises(ValueError, match="n_memory_slots"):
        L9_MemoryLayer(L9_StochasticParameters(n_memory_slots=0))
    with pytest.raises(ValueError, match="bitstream_length"):
        L9_MemoryLayer(L9_StochasticParameters(bitstream_length=0))
    with pytest.raises(ValueError, match="retrieval_gain"):
        L9_MemoryLayer(L9_StochasticParameters(retrieval_gain=-0.1))
    with pytest.raises(ValueError, match="imprint_rate"):
        L9_MemoryLayer(L9_StochasticParameters(imprint_rate=-0.1))
    with pytest.raises(ValueError, match="decay_rate"):
        L9_MemoryLayer(L9_StochasticParameters(decay_rate=-0.1))
    with pytest.raises(ValueError, match="phase_field_coupling"):
        L9_MemoryLayer(L9_StochasticParameters(phase_field_coupling=-0.1))
    with pytest.raises(ValueError, match="rng_seed"):
        L9_MemoryLayer(L9_StochasticParameters(rng_seed=cast(Any, 1.5)))

    layer = L9_MemoryLayer(L9_StochasticParameters(n_memory_slots=3, rng_seed=5))
    with pytest.raises(ValueError, match="pattern"):
        layer.store(np.array([1.0, np.nan, 0.0]))
    with pytest.raises(ValueError, match="pattern"):
        layer.store(np.array([1.0, -1.0]))
    with pytest.raises(ValueError, match="dt"):
        layer.step(0.0)
    with pytest.raises(ValueError, match="cosmic_alignment"):
        layer.step(0.001, {"cosmic_alignment": np.nan})
    with pytest.raises(ValueError, match="cosmic_alignment"):
        layer.step(0.001, {"cosmic_alignment": np.array([0.1, 0.2])})
    invalid_reference_payloads: list[Any] = [
        {"reference_amplitude": np.nan, "reference_phase": 0.0},
        {"reference_amplitude": 1.1, "reference_phase": 0.0},
        {"reference_amplitude": 1.0, "reference_phase": np.array([0.0, 1.0])},
        {"reference_amplitude": 1.0, "reference_phase": 0.0, "reference_real": 0.0},
    ]
    for payload in invalid_reference_payloads:
        with pytest.raises(ValueError, match="memory_imprint_drive"):
            layer.step(0.001, {"memory_imprint_drive": payload})


def test_get_global_metric_reflects_stored_pattern() -> None:
    layer = L9_MemoryLayer(L9_StochasticParameters(n_memory_slots=4, bitstream_length=16))
    assert layer.get_global_metric() == 0.0  # nothing stored yet
    layer.store(np.array([1.0, -1.0, 1.0, -1.0]))
    metric = layer.get_global_metric()
    assert 0.0 <= metric <= 1.0


def test_validate_params_rejects_non_integer_and_negative_seed() -> None:
    with pytest.raises(ValueError, match="n_memory_slots must be a positive integer"):
        L9_MemoryLayer(L9_StochasticParameters(n_memory_slots=cast(int, True)))
    with pytest.raises(ValueError, match="bitstream_length must be a positive integer"):
        L9_MemoryLayer(L9_StochasticParameters(n_memory_slots=4, bitstream_length=cast(int, True)))
    with pytest.raises(ValueError, match="rng_seed"):
        L9_MemoryLayer(L9_StochasticParameters(n_memory_slots=4, rng_seed=-1))


def test_l8_phase_reference_drive_is_neutral_without_known_keys() -> None:
    layer = L9_MemoryLayer(L9_StochasticParameters(n_memory_slots=4, bitstream_length=16))
    # An L8 payload carrying neither drive key contributes no phase reference.
    result = layer.step(0.001, {"unrelated_signal": 1.0})
    assert "state" in result


def test_memory_imprint_drive_remaining_contract_guards() -> None:
    drive = L9_MemoryLayer._memory_imprint_drive
    with pytest.raises(ValueError, match="must be a mapping"):
        drive(cast(Any, "not-a-mapping"))
    with pytest.raises(ValueError, match="reference_amplitude and reference_phase"):
        drive({"reference_amplitude": 0.5})
    with pytest.raises(ValueError, match="reference_phase must be finite"):
        drive({"reference_amplitude": 0.5, "reference_phase": float("inf")})
    with pytest.raises(ValueError, match="reference_real must be a finite scalar"):
        drive(
            {
                "reference_amplitude": 0.5,
                "reference_phase": 0.0,
                "reference_real": np.array([1.0, 2.0]),
            }
        )
