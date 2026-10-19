# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Network.to_torch(): the bridge steps as the models step, or refuses

"""``Network.to_torch()`` reproduces each supported model's own step, per neuron.

Reproducers of RIGOR review 271: the bridge stepped an Euler LIF with a strict
threshold and a subtractive reset where the models take an exact flow (SC
Lapicque) or forward Euler (StochasticLIF) with ``>=`` and a hard reset; it
substituted the SC profile for the 1907 source profile; it gave every neuron
the first neuron's voltage; it overwrote parallel CSR entries and ignored the
weight threshold; and float64 failed on the second step. Each spike train here
is compared with stepping the Python neurons themselves.
"""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.network.network import Network
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.neurons.models.lapicque import LapicqueNeuron

torch = pytest.importorskip("torch")


def _python_trains(population: Population, drive: np.ndarray) -> np.ndarray:
    """Step each neuron of ``population`` through ``drive[t, i]`` and return its spikes."""
    trains = np.zeros(drive.shape, dtype=np.float64)
    for t in range(drive.shape[0]):
        for i, neuron in enumerate(population.neurons):
            trains[t, i] = neuron.step(float(drive[t, i]))
    return trains


def _bridge_trains(population: Population, drive: np.ndarray, dtype: object = None) -> np.ndarray:
    """Run a one-population bridge on ``drive`` and return its spike traces."""
    bridge = Network(population).to_torch()
    inputs = torch.tensor(drive[:, None, :], dtype=dtype or torch.float32)
    if dtype is not None:
        bridge = bridge.to(dtype=dtype)
    _counts, traces = bridge(inputs, return_traces=True)
    return np.asarray(traces[population.label][:, 0, :].detach().numpy(), dtype=np.float64)


@pytest.mark.parametrize(
    ("model", "drive_value"),
    [("SCLapicqueLIFNeuron", 1.5), ("SCLapicqueLIFNeuron", 1.0), ("StochasticLIFNeuron", 1.0)],
)
def test_spike_trains_equal_the_models(model: str, drive_value: float) -> None:
    """Exact flow / Euler update, >= threshold and hard reset: bridge = Python, with spikes."""
    drive = np.full((30, 3), drive_value)
    drive[:, 1] *= 2.0
    expected = _python_trains(Population(model, 3, params={"dt": 1.0}), drive)
    assert np.array_equal(_bridge_trains(Population(model, 3, params={"dt": 1.0}), drive), expected)
    assert expected.sum() > 0


def test_hard_reset_after_a_large_input() -> None:
    """StochasticLIF input [3, 0, 0]: one spike, then the reset voltage, not a subtraction."""
    drive = np.array([[3.0], [0.0], [0.0]])
    expected = _python_trains(Population("StochasticLIFNeuron", 1), drive)
    assert np.array_equal(_bridge_trains(Population("StochasticLIFNeuron", 1), drive), expected)
    assert expected[:, 0].tolist() == [1.0, 0.0, 0.0]


def test_each_neuron_starts_from_its_own_voltage() -> None:
    """set_voltages([0, 0.99]) then one 0.3 step: the second neuron spikes in both."""
    drive = np.array([[0.3, 0.3]])
    reference = Population("StochasticLIFNeuron", 2)
    reference.set_voltages(np.array([0.0, 0.99]))
    expected = _python_trains(reference, drive)
    population = Population("StochasticLIFNeuron", 2)
    population.set_voltages(np.array([0.0, 0.99]))
    assert np.array_equal(_bridge_trains(population, drive), expected)
    assert expected[0].tolist() == [0.0, 1.0]


def test_the_1907_source_profile_is_refused() -> None:
    """The source profile latches one event and has no reset; the bridge does not substitute."""
    with pytest.raises(NotImplementedError, match="lapicque_1907"):
        Network(Population("LapicqueNeuron", 1, params={"dt": 1.0})).to_torch()


def test_parallel_connections_and_weight_threshold_are_refused() -> None:
    """Two CSR entries for one pair, or a pruning threshold, have no lossless dense form."""
    source = Population("StochasticLIFNeuron", 1)
    target = Population("StochasticLIFNeuron", 2)
    parallel = (np.array([0, 2]), np.array([1, 1]), np.array([0.4, 0.8]))
    with pytest.raises(NotImplementedError, match="parallel"):
        Network(source, target, Projection(source, target, 1.0, topology=parallel)).to_torch()
    single = (np.array([0, 1]), np.array([1]), np.array([1.2]))
    pruned = Projection(source, target, 1.0, topology=single, weight_threshold=1.3)
    with pytest.raises(NotImplementedError, match="weight_threshold"):
        Network(source, target, pruned).to_torch()


def test_untrained_sync_keeps_every_weight() -> None:
    """A round trip through the bridge without training leaves the CSR weights unchanged."""
    source = Population("StochasticLIFNeuron", 2)
    target = Population("StochasticLIFNeuron", 2)
    csr = (np.array([0, 2, 3]), np.array([0, 1, 1]), np.array([0.4, 0.8, 0.6]))
    projection = Projection(source, target, 1.0, topology=csr)
    before = projection.data.copy()
    Network(source, target, projection).to_torch().sync_to_network()
    assert np.array_equal(projection.data, before)


def _edge_events(weight: float, dtype: object) -> tuple[list[float], list[float]]:
    """One edge source→target, source input [2, 0]: target events in Python and in Torch."""
    python_source = Population("StochasticLIFNeuron", 1)
    python_target = Population("StochasticLIFNeuron", 1)
    expected, carried = [], 0.0
    for drive in (2.0, 0.0):
        expected.append(python_target.neurons[0].step(weight * carried))
        carried = python_source.neurons[0].step(drive)
    source = Population("StochasticLIFNeuron", 1)
    target = Population("StochasticLIFNeuron", 1)
    edge = (np.array([0, 1]), np.array([0]), np.array([weight]))
    bridge = Network(source, target, Projection(source, target, 1.0, topology=edge)).to_torch()
    bridge = bridge.to(dtype=dtype)
    _counts, traces = bridge(torch.tensor([[[2.0]], [[0.0]]], dtype=dtype), return_traces=True)
    return [float(e) for e in expected], traces[target.label][:, 0, 0].tolist()


def test_a_double_bridge_keeps_float64_weights() -> None:
    """Weight 0.99999999 stays below the threshold of 1 in float64, as in the network."""
    expected, events = _edge_events(0.99999999, torch.float64)
    assert expected == [0.0, 0.0] and events == expected
    # Control: an exactly representable weight that reaches the threshold fires in both.
    expected, events = _edge_events(1.0, torch.float64)
    assert expected == [0.0, 1.0] and events == expected


def test_untrained_sync_is_exact_for_a_weight_float32_cannot_hold() -> None:
    """Sync without training writes back 0.99999999 itself, not its float32 rounding 1.0."""
    source = Population("StochasticLIFNeuron", 1)
    target = Population("StochasticLIFNeuron", 1)
    edge = (np.array([0, 1]), np.array([0]), np.array([0.99999999]))
    projection = Projection(source, target, 1.0, topology=edge)
    bridge = Network(source, target, projection).to_torch()
    bridge(torch.zeros((2, 1, 1)))
    bridge.sync_to_network()
    assert projection.data.tolist() == [0.99999999]


@pytest.mark.parametrize(
    ("dtype", "first", "resistance", "events"),
    [
        (torch.float64, 1e307, 100.0, [1.0, 0.0, 1.0, 1.0]),
        (torch.float32, 1e38, 10.0, [1.0, 0.0, 1.0, 1.0]),
        (torch.float64, 1e150, 2.0, [1.0, 0.0, 0.0, 0.0]),
    ],
)
def test_an_overflowing_input_still_resets(
    dtype: object, first: float, resistance: float, events: list[float]
) -> None:
    """A finite input whose step overflows spikes, resets to v_reset and keeps a finite state.

    The 1e150 row, whose step does not overflow, is the well-conditioned control.
    """
    drive = np.array([[first], [0.0], [0.2], [0.2]])
    params = {"resistance": resistance}
    population = Population("StochasticLIFNeuron", 1, params=params)
    expected = _python_trains(population, drive)
    assert expected[:, 0].tolist() == events
    bridge = Network(Population("StochasticLIFNeuron", 1, params=params)).to_torch().to(dtype=dtype)
    voltages: list[float] = []
    cell = next(module for _name, module in bridge.named_modules() if hasattr(module, "v_reset"))
    cell.register_forward_hook(lambda _m, _i, out: voltages.append(float(out[1][0, 0].detach())))
    _counts, traces = bridge(torch.tensor(drive[:, None, :], dtype=dtype), return_traces=True)
    assert traces[population.label][:, 0, 0].tolist() == expected[:, 0].tolist()
    assert all(np.isfinite(voltages)) and voltages[0] == 0.0


def test_float64_runs_several_projected_steps() -> None:
    """A double bridge with double input keeps its dtype through the spikes of every step."""
    source = Population("StochasticLIFNeuron", 2)
    target = Population("StochasticLIFNeuron", 2)
    projection = Projection(source, target, 0.8, topology="all_to_all")
    bridge = Network(source, target, projection).to_torch().to(dtype=torch.float64)
    counts = bridge(torch.full((5, 1, 2), 2.0, dtype=torch.float64))
    assert counts.dtype == torch.float64 and counts.sum().item() > 0
    counts.sum().backward()


# RIGOR review 280.
def _mixed_profiles(source_first: bool) -> Population:
    """Return a Lapicque population whose neurons hold the SC and the 1907 source profile."""
    population = Population("SCLapicqueLIFNeuron", 2, params={"dt": 1.0})
    sc = population.neurons[0]
    source = LapicqueNeuron(
        profile="lapicque_1907",
        tau=sc.tau,
        dt=sc.dt,
        resistance=sc.resistance,
        v_rest=sc.v_rest,
        v_reset=sc.v_reset,
        v_threshold=sc.v_threshold,
    )
    population.neurons[0 if source_first else 1] = source
    return population


@pytest.mark.parametrize("source_first", [False, True])
def test_every_neuron_profile_is_checked(source_first: bool) -> None:
    """NEURO-TORCH-PROFILE-01: [SC, 1907] was accepted (Torch [8, 8], Python [8, 0])."""
    with pytest.raises(NotImplementedError, match="lapicque_1907"):
        Network(_mixed_profiles(source_first)).to_torch()


@pytest.mark.parametrize(
    ("dtype", "resistance"),
    [(torch.float64, 1e308), (torch.float32, 1e38), (torch.float64, 1e307)],
)
def test_the_gain_keeps_the_model_order(dtype: object, resistance: float) -> None:
    """NEURO-TORCH-GAIN-ORDER-01: R·dt overflowed where (R·I)·dt is zero at zero input.

    The model forms (R·I)·dt; the bridge formed R·dt = inf and inf·0 = NaN from the
    first step. Every voltage stays 0 with finite input gradients; 1e307 is the control.
    """
    params = {"resistance": resistance, "dt": 10.0}
    population = Population("StochasticLIFNeuron", 1, params=params)
    assert _python_trains(population, np.zeros((3, 1))).sum() == 0.0
    bridge = Network(Population("StochasticLIFNeuron", 1, params=params)).to_torch()
    bridge = bridge.to(dtype=dtype)
    voltages: list[float] = []
    cell = next(module for _name, module in bridge.named_modules() if hasattr(module, "v_reset"))
    cell.register_forward_hook(lambda _m, _i, out: voltages.append(float(out[1][0, 0].detach())))
    inputs = torch.zeros((3, 1, 1), dtype=dtype, requires_grad=True)
    counts = bridge(inputs)
    counts.sum().backward()
    assert voltages == [0.0, 0.0, 0.0]
    assert inputs.grad is not None and bool(torch.isfinite(inputs.grad).all())


def test_a_float64_run_reproduces_the_python_voltages() -> None:
    """The cell repeats the models' operations in their order: float64 voltages are bit-equal."""
    drive = np.array([[0.37], [0.0], [1.3], [0.21], [0.9]])
    for model in ("SCLapicqueLIFNeuron", "StochasticLIFNeuron"):
        reference = Population(model, 1, params={"dt": 1.0})
        expected = []
        for value in drive[:, 0]:
            reference.neurons[0].step(float(value))
            expected.append(float(reference.neurons[0].v))
        bridge = Network(Population(model, 1, params={"dt": 1.0})).to_torch().double()
        seen: list[float] = []
        cell = next(m for _n, m in bridge.named_modules() if hasattr(m, "v_reset"))
        cell.register_forward_hook(lambda _m, _i, out: seen.append(float(out[1][0, 0].detach())))
        bridge(torch.tensor(drive[:, None, :], dtype=torch.float64))
        assert seen == expected, model
