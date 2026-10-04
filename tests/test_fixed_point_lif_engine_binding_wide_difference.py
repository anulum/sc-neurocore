# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Installed fixed-point LIF wide arithmetic contracts

"""Compare real native scalar and batch entry points with the Python model."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.fixed_point_lif import FixedPointLIFNeuron
from tests.engine_requirement import require_engine

require_engine()
import sc_neurocore_engine as engine


@pytest.mark.parametrize(
    ("v_rest", "currents", "expected"),
    [
        (32767, [-32768, 0], [(0, -1), (0, 127)]),
        (-32768, [-2, 0], [(0, 32766), (0, 32510)]),
    ],
)
def test_wide_leak_difference_matches_python_and_observed_rtl(
    v_rest: int, currents: list[int], expected: list[tuple[int, int]]
) -> None:
    """Keep all subtraction bits before fractional scaling of the leak term."""
    parameters = {
        "data_width": 16,
        "fraction": 8,
        "v_rest": v_rest,
        "v_reset": 0,
        "v_threshold": 32767,
        "refractory_period": 0,
    }
    python = FixedPointLIFNeuron(**parameters)
    native = engine.FixedPointLif(**parameters)
    assert [python.step(1, 256, current, 0) for current in currents] == expected
    assert [native.step(1, 256, current, 0) for current in currents] == expected
    assert native.get_state() == python.get_state()
    spikes, voltages = engine.batch_lif_run_varying(
        1, 256, np.asarray(currents, dtype=np.int16), **parameters
    )
    np.testing.assert_array_equal(spikes, [spike for spike, _ in expected])
    np.testing.assert_array_equal(voltages, [voltage for _, voltage in expected])


def _python_trace(
    parameters: dict[str, int],
    leak: int,
    gain: int,
    currents: list[int],
    noises: list[int],
) -> list[tuple[int, int]]:
    """Run the public Python neuron with the exact supplied runtime sequence."""
    neuron = FixedPointLIFNeuron(**parameters)
    return [
        neuron.step(leak, gain, current, noise)
        for current, noise in zip(currents, noises, strict=True)
    ]


@pytest.mark.parametrize(
    ("width", "fraction"), [(1, 0), (2, 1), (8, 4), (12, 8), (16, 8), (16, 15)]
)
@pytest.mark.parametrize("rest_at_minimum", [False, True])
@pytest.mark.parametrize("leak_direction", [-1, 0, 1])
def test_scalar_constant_varying_and_parallel_match_public_python_arithmetic(
    width: int, fraction: int, rest_at_minimum: bool, leak_direction: int
) -> None:
    """Exercise signed boundaries, fractional shifts, refractory state and reset."""
    minimum = -(1 << (width - 1))
    maximum = (1 << (width - 1)) - 1
    parameters = {
        "data_width": width,
        "fraction": fraction,
        "v_rest": minimum if rest_at_minimum else maximum,
        "v_reset": minimum,
        "v_threshold": maximum,
        "refractory_period": 2,
    }
    leak = minimum if leak_direction < 0 else maximum if leak_direction > 0 else 0
    gain = maximum
    rng = np.random.default_rng(1761)
    currents = rng.integers(minimum, maximum + 1, size=64, dtype=np.int16)
    noises = rng.integers(minimum, maximum + 1, size=64, dtype=np.int16)
    currents[:4] = [minimum, maximum, 0, minimum]
    noises[:4] = [0, minimum, maximum, 0]
    currents.setflags(write=False)
    noises.setflags(write=False)
    current_values, noise_values = currents.tolist(), noises.tolist()
    expected = _python_trace(parameters, leak, gain, current_values, noise_values)
    native = engine.FixedPointLif(**parameters)
    python = FixedPointLIFNeuron(**parameters)
    for current, noise in zip(current_values, noise_values, strict=True):
        assert native.step(leak, gain, current, noise) == python.step(leak, gain, current, noise)
        assert native.get_state() == python.get_state()
    native.reset_state()
    python.reset_state()
    assert native.get_state() == python.get_state()
    assert native.step(leak, gain, minimum, 0) == python.step(leak, gain, minimum, 0)
    native.reset()
    python.reset()
    assert native.get_state() == python.get_state()

    spikes, voltages = engine.batch_lif_run_varying(leak, gain, currents, noises, **parameters)
    np.testing.assert_array_equal(spikes, [spike for spike, _ in expected])
    np.testing.assert_array_equal(voltages, [voltage for _, voltage in expected])
    assert spikes.dtype == np.int32
    assert voltages.dtype == np.int16
    np.testing.assert_array_equal(currents, current_values)
    np.testing.assert_array_equal(noises, noise_values)
    assert not currents.flags.writeable
    assert not noises.flags.writeable

    constant = _python_trace(parameters, leak, gain, [minimum] * 32, [0] * 32)
    spikes, voltages = engine.batch_lif_run(32, leak, gain, minimum, **parameters)
    np.testing.assert_array_equal(spikes, [spike for spike, _ in constant])
    np.testing.assert_array_equal(voltages, [voltage for _, voltage in constant])

    parallel_currents = np.asarray([minimum, 0, maximum], dtype=np.int16)
    parallel_currents.setflags(write=False)
    spikes, voltages = engine.batch_lif_run_multi(
        3, 32, leak, gain, parallel_currents, **parameters
    )
    assert spikes.shape == voltages.shape == (3, 32)
    assert spikes.dtype == np.int32
    assert voltages.dtype == np.int16
    for index, current in enumerate(parallel_currents.tolist()):
        trace = _python_trace(parameters, leak, gain, [current] * 32, [0] * 32)
        np.testing.assert_array_equal(spikes[index], [spike for spike, _ in trace])
        np.testing.assert_array_equal(voltages[index], [voltage for _, voltage in trace])
    np.testing.assert_array_equal(parallel_currents, [minimum, 0, maximum])
    assert not parallel_currents.flags.writeable
