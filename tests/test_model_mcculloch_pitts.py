# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source-faithful McCulloch-Pitts model contracts

"""Exact source rule, validation, batch and network tests for model 33."""

from __future__ import annotations

import dataclasses
from typing import cast

import numpy as np
import pytest

from sc_neurocore.analysis.spike_stats.basic import firing_rate, spike_count
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.network import Network
from sc_neurocore.network.population import Population
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.neurons.models.mcculloch_pitts import (
    McCullochPittsNeuron,
    encode_hardware_input,
)

_INT32_MAX = (1 << 31) - 1


def test_defaults_are_a_stateless_positive_count_contract() -> None:
    """The default is one excitatory afferent and no invented membrane state."""
    neuron = McCullochPittsNeuron()
    assert dataclasses.is_dataclass(neuron)
    assert neuron.theta == 1
    assert type(neuron.theta) is int
    assert not hasattr(neuron, "v")


@pytest.mark.parametrize(
    ("count", "expected"),
    ((0, 0), (1, 1), (2, 1)),
)
def test_or_gate_is_the_theta_one_truth_table(count: int, expected: int) -> None:
    """One or more active excitatory afferents implement inclusive OR."""
    assert McCullochPittsNeuron(theta=1).step(count) == expected


@pytest.mark.parametrize(
    ("count", "expected"),
    ((0, 0), (1, 0), (2, 1)),
)
def test_and_gate_is_the_theta_two_truth_table(count: int, expected: int) -> None:
    """Both of two excitatory afferents are required at theta two."""
    assert McCullochPittsNeuron(theta=2).step(count) == expected


def test_absolute_inhibition_vetoes_every_excitatory_count() -> None:
    """One inhibitory afferent dominates even an int32-maximum excitation."""
    neuron = McCullochPittsNeuron(theta=1)
    assert neuron.step(0, True) == 0
    assert neuron.step(1, True) == 0
    assert neuron.step(_INT32_MAX, True) == 0


def test_conjoined_negation_uses_the_source_inhibitory_wire() -> None:
    """The 1943 not-B conjunction is an absolute veto, not a negative weight."""
    neuron = McCullochPittsNeuron(theta=1)
    truth = {
        (0, False): 0,
        (1, False): 1,
        (0, True): 0,
        (1, True): 0,
    }
    assert {key: neuron.step(*key) for key in truth} == truth


def test_three_input_majority_uses_a_fixed_excitatory_count() -> None:
    """Theta two implements the majority of three binary excitatory afferents."""
    neuron = McCullochPittsNeuron(theta=2)
    assert [neuron.step(count) for count in range(4)] == [0, 0, 1, 1]


def test_calls_are_stateless_and_reset_is_a_validating_noop() -> None:
    """History cannot affect one-delay logical activity."""
    neuron = McCullochPittsNeuron(theta=2)
    assert [neuron.step(value) for value in (2, 0, 2, 0)] == [1, 0, 1, 0]
    neuron.reset()
    assert neuron.theta == 2


@pytest.mark.parametrize(
    "theta",
    (0, -1, 1.5, True, np.nan, np.inf, -np.inf, _INT32_MAX + 1, "1"),
)
def test_constructor_rejects_non_positive_or_lossy_thresholds(theta: object) -> None:
    """The fixed threshold is a positive signed-ABI-safe afferent count."""
    with pytest.raises(ValueError, match="theta"):
        McCullochPittsNeuron(theta=cast(int, theta))


@pytest.mark.parametrize(
    "count",
    (-1, 0.5, True, np.nan, np.inf, -np.inf, _INT32_MAX + 1, "1", object()),
)
def test_step_rejects_invalid_excitatory_counts(count: object) -> None:
    """Negative, fractional, Boolean and out-of-domain counts fail closed."""
    with pytest.raises(ValueError, match="excitatory_count"):
        McCullochPittsNeuron().step(count)


@pytest.mark.parametrize("flag", (0, 1, 0.0, 1.0, "false", None))
def test_step_requires_an_exact_inhibitory_boolean(flag: object) -> None:
    """Numeric truthiness cannot silently alter the absolute-veto wire."""
    with pytest.raises(ValueError, match="inhibitory_active"):
        McCullochPittsNeuron().step(1, flag)


def test_integer_valued_transport_floats_and_numpy_scalars_are_normalised() -> None:
    """Network Float64 transport remains usable without accepting fractions."""
    neuron = McCullochPittsNeuron(theta=cast(int, np.float64(2.0)))
    assert type(neuron.theta) is int
    assert neuron.step(np.float64(2.0), np.bool_(False)) == 1
    assert neuron.step(np.int32(1), np.bool_(False)) == 0


def test_runtime_threshold_corruption_fails_closed() -> None:
    """Public dataclass mutation is revalidated on step and reset."""
    neuron = McCullochPittsNeuron(theta=2)
    neuron.theta = cast(int, 1.25)
    with pytest.raises(ValueError, match="theta"):
        neuron.step(2)
    with pytest.raises(ValueError, match="theta"):
        neuron.reset()


@pytest.mark.parametrize(
    ("count", "inhibited", "encoded"),
    ((0, False, 0), (7, False, 7), (_INT32_MAX, False, _INT32_MAX), (7, True, -1)),
)
def test_signed_q320_encoding_is_bijective_over_valid_logical_inputs(
    count: int,
    inhibited: bool,
    encoded: int,
) -> None:
    """The RTL input uses -1 only for inhibition and non-negative values for counts."""
    assert encode_hardware_input(count, inhibited) == encoded


def test_python_batch_matches_scalar_truth_rows() -> None:
    """Varying excitation and inhibition return an exact contiguous binary trace."""
    counts = np.array([0, 1, 2, 3, _INT32_MAX], dtype=np.int64)
    flags = np.array([False, False, False, True, True], dtype=np.bool_)
    neuron = McCullochPittsNeuron(theta=2)
    events, event_count = neuron.simulate(counts, flags, backend="python")
    assert events.tolist() == [0, 0, 1, 0, 0]
    assert events.dtype == np.uint8
    assert events.flags.c_contiguous
    assert event_count == 1 == int(events.sum())
    assert [neuron.step(count, flag) for count, flag in zip(counts, flags, strict=True)] == (
        events.tolist()
    )


def test_batch_defaults_to_no_inhibition_and_accepts_empty_input() -> None:
    """Absent flags mean no veto; an empty stateless batch has zero events."""
    neuron = McCullochPittsNeuron(theta=2)
    events, count = neuron.simulate([0.0, 1.0, 2.0], backend="python")
    assert events.tolist() == [0, 0, 1]
    assert count == 1
    empty, empty_count = neuron.simulate([], [], backend="python")
    assert empty.shape == (0,)
    assert empty_count == 0


@pytest.mark.parametrize(
    ("counts", "flags", "message"),
    (
        (np.array(1), None, "one-dimensional"),
        (np.zeros((1, 1)), None, "one-dimensional"),
        ([0, 1], [False], "match"),
        ([0, 1], np.zeros((1, 2), dtype=np.bool_), "one-dimensional"),
        ([0, 1], [False, 1], "inhibitory_flags"),
        ([0, -1], None, r"excitatory_counts\[1\]"),
    ),
)
def test_batch_validation_fails_before_dispatch(
    counts: object,
    flags: object,
    message: str,
) -> None:
    """Malformed shapes and values cannot reach a native pointer boundary."""
    with pytest.raises(ValueError, match=message):
        McCullochPittsNeuron().simulate(
            cast(list[object], counts),
            cast(list[object] | None, flags),
            backend="python",
        )


def test_unknown_and_unavailable_backends_do_not_fall_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit backend selection is fail-closed."""
    neuron = McCullochPittsNeuron()
    with pytest.raises(ValueError, match="backend"):
        neuron.simulate([1], backend="cuda")

    from sc_neurocore.accel import mcculloch_pitts as backends

    monkeypatch.setattr(backends, "backend_available", lambda _backend: False)
    with pytest.raises(RuntimeError, match="Go"):
        neuron.simulate([1], backend="go")


def test_population_network_and_analysis_accept_integer_valued_transport() -> None:
    """The source contract remains usable through the generic Float64 network accumulator."""
    population = Population(McCullochPittsNeuron, n=8, label="mp")
    drive = PoissonInput(n=8, rate_hz=500.0, weight=2.0, dt=0.001, seed=42)
    monitor = SpikeMonitor(population)
    Network(population, drive, monitor).run(duration=0.05, dt=0.001, backend="python")
    assert monitor.count > 0

    train = np.asarray([McCullochPittsNeuron().step(1) for _ in range(100)], dtype=float)
    assert spike_count(train) == 100
    assert firing_rate(train, dt=0.001) == pytest.approx(1000.0)
