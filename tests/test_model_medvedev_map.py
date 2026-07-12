# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Medvedev 2005 first-return model tests

"""Source, invariant and pipeline tests for the Medvedev first-return map."""

from __future__ import annotations

import hashlib
import inspect
import math

import numpy as np
import pytest

from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.network import Network
from sc_neurocore.network.population import Population
from sc_neurocore.neurons.model_catalogue import load_descriptor_payload
from sc_neurocore.neurons.models.medvedev_map import MedvedevMapNeuron


def _boundaries(neuron: MedvedevMapNeuron) -> tuple[float, float, float]:
    """Return independently derived source branch boundaries."""
    return (
        neuron.beta_0 / (neuron.delta - neuron.beta_0),
        neuron.beta_hc / (neuron.delta - neuron.beta_hc),
        neuron.beta_sn / (neuron.delta - neuron.beta_sn),
    )


def _inner_reference(neuron: MedvedevMapNeuron, state: float, current: float) -> float:
    """Independently evaluate the calibrated Eqs. 4.8 and 4.13 branch."""
    u_1 = (1.0 - neuron.alpha_t0) * state + neuron.alpha_t0 * neuron.f_0
    gap = neuron.beta_hc - neuron.delta * u_1 / (1.0 + u_1)
    inner = neuron.f_1
    if gap > 0.0:
        scale = math.exp(neuron.homoclinic_exponent * math.log(neuron.d * gap))
        inner = scale * (u_1 - neuron.f_1) + neuron.f_1
    return inner + neuron.input_gain * current


def test_defaults_are_the_disclosed_source_calibration() -> None:
    """The initial state is the Eq. 4.15 saddle-node return, not zero."""
    neuron = MedvedevMapNeuron()
    u_0, u_hc, u_sn = _boundaries(neuron)
    assert u_0 == pytest.approx(0.1764705882352941)
    assert u_hc == pytest.approx(0.25470514429109165)
    assert u_sn == pytest.approx(0.2514078836724436)
    assert neuron.u == u_sn
    assert neuron.d > 127.996  # Signed Q8.8 cannot encode the calibrated scale.


def test_descriptor_structure_matches_map_runtime() -> None:
    """The unit iteration belongs only to integration, never to parameters."""
    payload = load_descriptor_payload("MedvedevMapNeuron")
    assert payload is not None
    assert "dt" not in inspect.signature(MedvedevMapNeuron).parameters
    assert "dt" not in payload["parameters"]
    assert payload["integration"] == {"dt": 1.0, "method": "map"}
    assert set(payload["state"]) == {"u"}
    assert set(payload["parameters"]) == {
        "beta_0",
        "beta_hc",
        "beta_sn",
        "delta",
        "decay_t0",
        "alpha_t0",
        "f_0",
        "f_1",
        "homoclinic_exponent",
        "d",
        "input_gain",
    }


def test_left_branch_matches_eq_4_4_calibration() -> None:
    """The active left branch uses the calibrated exponential relaxation."""
    neuron = MedvedevMapNeuron(u=0.1)
    current = 2.0
    expected = (
        neuron.decay_t0 * neuron.u
        + (1.0 - neuron.decay_t0) * neuron.f_0
        + neuron.input_gain * current
    )
    assert neuron.step(current) == 1
    assert neuron.u == expected


def test_inner_branch_matches_eq_4_8_and_eq_4_13_calibration() -> None:
    """The middle branch composes the affine and homoclinic returns."""
    neuron = MedvedevMapNeuron(u=0.2)
    expected = _inner_reference(neuron, neuron.u, 2.0)
    assert neuron.step(2.0) == 1
    assert neuron.u == expected


def test_right_branch_is_exact_eq_4_15_return_without_input() -> None:
    """External current does not perturb the slow right return."""
    neuron = MedvedevMapNeuron(u=0.3)
    _u_0, _u_hc, u_sn = _boundaries(neuron)
    assert neuron.step(1000.0) == 0
    assert neuron.u == u_sn


def test_event_uses_pre_state_fast_return_region() -> None:
    """The event is an observation of the pre-step active branch."""
    neuron = MedvedevMapNeuron(u=0.3)
    assert neuron.step(0.0) == 0
    assert neuron.step(0.0) == 1


def test_zero_current_golden_cycle() -> None:
    """The source map reproduces its calibrated 100-step orbit."""
    trace, events = MedvedevMapNeuron().simulate(100, 0.0, backend="python")
    assert events == 100
    assert trace[-1] == pytest.approx(0.19448491761002404, abs=1e-15)
    assert float(np.mean(trace)) == pytest.approx(0.21623098362239998, abs=1e-15)
    assert np.unique(trace).size == 7


def test_driven_golden_cycle_and_event_vector() -> None:
    """The maintained I=2 protocol has a four-state, 75-event cycle."""
    trace, events = MedvedevMapNeuron().simulate(100, 2.0, backend="python")
    expected_cycle = np.array(
        [
            0.20201527871456648,
            0.23396543697847846,
            0.26318342915295445,
            0.2514078836724436,
        ]
    )
    assert events == 75
    np.testing.assert_array_equal(trace[:4], expected_cycle)
    np.testing.assert_array_equal(trace, np.tile(expected_cycle, 25))


def test_reproducibility_hash_is_stable() -> None:
    """The descriptor's 1000-step little-endian trace hash is exact."""
    trace, events = MedvedevMapNeuron().simulate(1000, 2.0, backend="python")
    digest = hashlib.sha256(trace.astype("<f8", copy=False).tobytes()).hexdigest()
    assert events == 750
    assert digest == "4e45193f652b8c4ab1fc860b179585a52c565cfbe1769b17e850ab770a232f2c"


def test_batch_matches_repeated_checked_steps() -> None:
    """The batch surface and single-step surface commit the same recurrence."""
    batch = MedvedevMapNeuron()
    trace, events = batch.simulate(300, 2.0, backend="python")
    stepper = MedvedevMapNeuron()
    manual = []
    manual_events = 0
    for _step in range(300):
        manual_events += stepper.step(2.0)
        manual.append(stepper.u)
    np.testing.assert_array_equal(trace, np.asarray(manual, dtype=np.float64))
    assert events == manual_events
    assert batch.u == stepper.u


@pytest.mark.parametrize("current", (0.0, 2.0, 16.0, 1024.0))
def test_long_run_remains_finite(current: float) -> None:
    """The enrolled operating envelope never commits non-finite state."""
    trace, _events = MedvedevMapNeuron().simulate(10_000, current, backend="python")
    assert np.isfinite(trace).all()


@pytest.mark.parametrize(
    "overrides",
    (
        {"beta_sn": 0.001},
        {"beta_hc": 0.02},
        {"decay_t0": 1.0},
        {"alpha_t0": 0.0},
        {"f_1": 2.0},
        {"homoclinic_exponent": 0.0},
        {"d": 0.0},
        {"input_gain": -1.0},
    ),
)
def test_invalid_parameter_topology_is_rejected(overrides: dict[str, float]) -> None:
    """Invalid source topology cannot enter the runtime."""
    with pytest.raises(ValueError):
        MedvedevMapNeuron(**overrides)


def test_failed_step_preserves_state() -> None:
    """Non-finite input fails before state mutation."""
    neuron = MedvedevMapNeuron()
    before = neuron.u
    with pytest.raises(ValueError, match="current must be finite"):
        neuron.step(float("nan"))
    assert neuron.u == before


def test_failed_batch_preserves_state() -> None:
    """A mutable parameter fault rejects the batch without state mutation."""
    neuron = MedvedevMapNeuron()
    before = neuron.u
    neuron.d = float("inf")
    with pytest.raises(ValueError, match="parameters must be finite"):
        neuron.simulate(10, 2.0, backend="python")
    assert neuron.u == before


def test_request_validation() -> None:
    """Batch bounds and backend selection are explicit."""
    neuron = MedvedevMapNeuron()
    with pytest.raises(ValueError, match="n_steps must be an integer"):
        neuron.simulate(True)
    with pytest.raises(ValueError, match="n_steps must be between"):
        neuron.simulate(-1)
    with pytest.raises(ValueError, match="backend must be"):
        neuron.simulate(1, backend="cuda")


def test_reset_restores_only_the_derived_return_state() -> None:
    """Reset preserves calibration and recomputes u_SN from mutable parameters."""
    neuron = MedvedevMapNeuron()
    neuron.beta_sn = 0.0019
    neuron.u = 0.3
    neuron.reset()
    assert neuron.u == neuron.beta_sn / (neuron.delta - neuron.beta_sn)
    assert neuron.beta_sn == 0.0019


def test_population_network_path_observes_events() -> None:
    """The standard network loop can drive and monitor the renamed u state."""
    population = Population(MedvedevMapNeuron, n=4, label="medvedev")
    monitor = SpikeMonitor(population)
    network = Network(population, monitor)
    network.run(duration=0.01, dt=0.001, backend="python")
    assert monitor.count > 0
