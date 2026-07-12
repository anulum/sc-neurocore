# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Ibarz-Tanaka 2007 source-map tests

"""Source, invariant and pipeline tests for the Ibarz-Tanaka map."""

from __future__ import annotations

import hashlib
import inspect
import math

import numpy as np
import pytest

from sc_neurocore.network.population import Population
from sc_neurocore.neurons.model_catalogue import load_descriptor_payload
from sc_neurocore.neurons.models.ibarz_tanaka_map import IbarzTanakaMapNeuron


def _reference_step(neuron: IbarzTanakaMapNeuron, current: float) -> tuple[float, float, int]:
    """Independently evaluate Ibarz et al. (2007), Eqs. 2-3."""
    lower = -1.0 - neuron.alpha / 2.0
    upper = 1.0 + current + neuron.u
    if neuron.v < lower:
        v_next = -(neuron.alpha**2) / 4.0 - neuron.alpha + current + neuron.u
    elif neuron.v <= 0.0:
        v_next = neuron.alpha * neuron.v + (neuron.v + 1.0) ** 2 + current + neuron.u
    elif neuron.v < upper:
        v_next = upper
    else:
        v_next = -1.0
    u_next = neuron.u - neuron.mu * (neuron.v + 1.0 - neuron.sigma)
    return v_next, u_next, int(neuron.v >= upper)


def test_defaults_match_the_published_figure_protocol() -> None:
    """Defaults use the paper's alpha, mu, sigma and Fig. 2 map placement."""
    neuron = IbarzTanakaMapNeuron()
    assert (neuron.v, neuron.u) == (-1.0, -0.1)
    assert (neuron.alpha, neuron.mu, neuron.sigma) == (1.0, 0.001, 0.1)


def test_descriptor_structure_matches_the_discrete_map() -> None:
    """Only source parameters are exposed and dt remains integration metadata."""
    payload = load_descriptor_payload("IbarzTanakaMapNeuron")
    assert payload is not None
    assert "dt" not in inspect.signature(IbarzTanakaMapNeuron).parameters
    assert set(payload["state"]) == {"v", "u"}
    assert set(payload["parameters"]) == {"alpha", "mu", "sigma"}
    assert payload["integration"] == {"dt": 1.0, "method": "map"}


@pytest.mark.parametrize("v", (-2.0, -1.0, 0.5, 1.5))
def test_all_four_fast_branches_match_eq_3(v: float) -> None:
    """Each source branch commits the independently evaluated candidate."""
    neuron = IbarzTanakaMapNeuron(v=v)
    expected = _reference_step(neuron, 0.2)
    assert neuron.step(0.2) == expected[2]
    assert (neuron.v, neuron.u) == expected[:2]


def test_slow_state_uses_the_pre_step_fast_state() -> None:
    """The Eq. 2 update is simultaneous with the Eq. 3 fast update."""
    neuron = IbarzTanakaMapNeuron(v=0.5, u=-0.1)
    expected_u = neuron.u - neuron.mu * (neuron.v + 1.0 - neuron.sigma)
    neuron.step(0.2)
    assert neuron.u == expected_u


def test_event_marks_the_source_reset_branch() -> None:
    """The plateau branch precedes a reset event on the next iteration."""
    neuron = IbarzTanakaMapNeuron(v=0.5, u=-0.1)
    assert neuron.step(0.2) == 0
    assert neuron.v == pytest.approx(1.1)
    assert neuron.step(0.2) == 1
    assert neuron.v == -1.0


@pytest.mark.parametrize("current, expected", ((0.0, 9), (0.2, 33), (1.0, 195)))
def test_source_protocol_event_counts(current: float, expected: int) -> None:
    """The published default parameter set has stable derived event counts."""
    _trace, events = IbarzTanakaMapNeuron().simulate(1000, current, backend="python")
    assert events == expected


def test_reproducibility_hash_is_stable() -> None:
    """The descriptor's driven fast-state trace digest is exact."""
    trace, events = IbarzTanakaMapNeuron().simulate(1000, 0.2, backend="python")
    digest = hashlib.sha256(trace.tobytes()).hexdigest()
    assert events == 33
    assert digest == "68000d6955ffcaedffa3a851f70e8f118156312ab224638defb408ae0b3002ed"


def test_batch_matches_repeated_source_steps() -> None:
    """The batch dispatcher and public step surface commit the same recurrence."""
    batch = IbarzTanakaMapNeuron()
    trace, events = batch.simulate(300, 0.2, backend="python")
    stepper = IbarzTanakaMapNeuron()
    expected_trace = []
    expected_events = 0
    for _step in range(300):
        expected_events += stepper.step(0.2)
        expected_trace.append(stepper.v)
    np.testing.assert_array_equal(trace, np.asarray(expected_trace, dtype=np.float64))
    assert events == expected_events
    assert (batch.v, batch.u) == (stepper.v, stepper.u)


@pytest.mark.parametrize("current", (-1.0, 0.0, 0.2, 1.0, 10.0))
def test_long_run_remains_finite(current: float) -> None:
    """The source operating envelope keeps both state variables finite."""
    neuron = IbarzTanakaMapNeuron()
    trace, _events = neuron.simulate(10_000, current, backend="python")
    assert np.isfinite(trace).all()
    assert math.isfinite(neuron.u)


@pytest.mark.parametrize("overrides", ({"alpha": 0.0}, {"mu": 0.0}, {"sigma": math.inf}))
def test_invalid_parameter_topology_is_rejected(overrides: dict[str, float]) -> None:
    """Invalid source parameters cannot enter the runtime."""
    with pytest.raises(ValueError):
        IbarzTanakaMapNeuron(**overrides)


def test_failed_step_preserves_state() -> None:
    """Non-finite input fails before state mutation."""
    neuron = IbarzTanakaMapNeuron()
    before = (neuron.v, neuron.u)
    with pytest.raises(ValueError, match="current must be finite"):
        neuron.step(float("nan"))
    assert (neuron.v, neuron.u) == before


def test_failed_batch_preserves_state() -> None:
    """A mutable parameter fault rejects the batch without state mutation."""
    neuron = IbarzTanakaMapNeuron()
    before = (neuron.v, neuron.u)
    neuron.alpha = float("inf")
    with pytest.raises(ValueError, match="parameters must be finite"):
        neuron.simulate(10, 0.2, backend="python")
    assert (neuron.v, neuron.u) == before


def test_request_validation() -> None:
    """Batch bounds and backend selection are explicit."""
    neuron = IbarzTanakaMapNeuron()
    with pytest.raises(ValueError, match="n_steps must be an integer"):
        neuron.simulate(True)
    with pytest.raises(ValueError, match="n_steps must be between"):
        neuron.simulate(-1)
    with pytest.raises(ValueError, match="unsupported backend"):
        neuron.simulate(1, backend="cuda")


def test_reset_restores_only_the_source_initial_state() -> None:
    """Reset preserves parameters while restoring the published state placement."""
    neuron = IbarzTanakaMapNeuron(alpha=0.95)
    neuron.simulate(100, 0.2, backend="python")
    neuron.reset()
    assert (neuron.v, neuron.u) == (-1.0, -0.1)
    assert neuron.alpha == 0.95


def test_population_path_observes_the_fast_state_and_events() -> None:
    """The standard population surface consumes the renamed v state correctly."""
    population = Population(IbarzTanakaMapNeuron, n=4, label="ibarz-tanaka")
    events = 0
    current = np.full(4, 0.2, dtype=np.float64)
    for _step in range(1000):
        events += int(population.step_all(current).sum())
    assert events == 4 * 33
    np.testing.assert_array_equal(
        population.voltages,
        np.asarray([neuron.v for neuron in population.neurons]),
    )
