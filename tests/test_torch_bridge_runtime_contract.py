# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Network.to_torch() keeps the models' precision, refusals and live counters

"""The Torch bridge steps in float64 with the models' runtime refusals and live state.

RIGOR review 286 found three gaps, each reproduced here against the Python model:

* NEURO-TORCH-GAIN-ORDER-01 (residual): a float32 bridge with ``R = 1e40``, outside
  float32, gave NaN states for zero input and spurious events for a tiny one; the
  bridge now evaluates in float64, the models' precision, whatever the input dtype.
* NEURO-TORCH-NONFINITE-STATE-01: a step whose candidate overflows negatively left
  a non-finite state that the bridge kept using, where StochasticLIF refuses the
  next step and Lapicque refuses the step itself; both refusals are kept, and a
  float32 input that only float32 overflows now stays finite as in the model.
* NEURO-TORCH-LIVE-REFRACTORY-01: a live refractory counter (with period 0) was
  dropped; it now holds the neuron at ``v_rest`` for its remaining steps.

Each case compares events and voltages with the model step for step; gradients
are checked finite. Healthy controls run beside every repair.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from sc_neurocore.network.network import Network
from sc_neurocore.network.population import Population

torch = pytest.importorskip("torch")


def _python_run(population: Population, drive: list[float]) -> tuple[list[float], list[float]]:
    """Step the model's first neuron through ``drive``; return events and voltages."""
    neuron = population.neurons[0]
    events, voltages = [], []
    for value in drive:
        events.append(float(neuron.step(value)))
        voltages.append(float(neuron.v))
    return events, voltages


def _bridge_run(
    population: Population, drive: list[float], dtype: object
) -> tuple[list[float], list[float], Any]:
    """Run a one-population bridge; return events, voltages and the input gradient."""
    bridge = Network(population).to_torch().to(dtype=dtype)
    voltages: list[float] = []
    cell = next(module for _name, module in bridge.named_modules() if hasattr(module, "v_reset"))
    cell.register_forward_hook(lambda _m, _i, out: voltages.append(float(out[1][0, 0].detach())))
    inputs = torch.tensor([[[value]] for value in drive], dtype=dtype, requires_grad=True)
    counts, traces = bridge(inputs, return_traces=True)
    counts.sum().backward()
    assert counts.dtype == dtype
    assert inputs.grad is not None
    return traces[population.label][:, 0, 0].tolist(), voltages, inputs.grad


def _as_dtype(values: list[float], dtype: object) -> list[float]:
    """Return the values the model receives from a tensor of ``dtype``."""
    return [float(v) for v in torch.tensor(values, dtype=dtype).tolist()]


@pytest.mark.parametrize(
    ("resistance", "drive", "threshold"),
    [
        (1e40, [0.0, 0.0, 0.0], 1.0),  # R outside float32, zero input
        (1e40, [1e-40, 1e-40, 1e-40], 10.0),  # R outside float32, a tiny float32 input
        (1e30, [1e-40, 1e-40, 1e-40], 10.0),  # control: R inside float32
    ],
)
def test_a_float32_input_is_stepped_in_the_models_precision(
    resistance: float, drive: list[float], threshold: float
) -> None:
    """Events and voltages equal the model's for the input's float32 values.

    The input gradient is the float64 gradient rounded to float32: with R = 1e40
    it is about 1e39, beyond float32, so it rounds to infinity (never NaN).
    """
    params = {"resistance": resistance, "v_threshold": threshold, "dt": 1.0}
    values = _as_dtype(drive, torch.float32)
    events, voltages = _python_run(Population("StochasticLIFNeuron", 1, params=params), values)
    got_events, got_voltages, grad = _bridge_run(
        Population("StochasticLIFNeuron", 1, params=params), drive, torch.float32
    )
    assert got_events == events and got_voltages == voltages
    assert all(np.isfinite(voltages))
    _e, _v, grad64 = _bridge_run(
        Population("StochasticLIFNeuron", 1, params=params), values, torch.float64
    )
    assert torch.equal(grad, grad64.to(torch.float32))
    assert not bool(torch.isnan(grad).any())


def test_a_negative_overflow_is_refused_where_the_model_refuses() -> None:
    """StochasticLIF: -1e307·R100 leaves -inf; the model refuses the next step, so does the bridge."""
    params = {"resistance": 100.0, "dt": 1.0}
    neuron = Population("StochasticLIFNeuron", 1, params=params).neurons[0]
    assert neuron.step(-1e307) == 0 and neuron.v == -np.inf
    with pytest.raises(ValueError, match="v must be finite"):
        neuron.step(0.0)
    with pytest.raises(ValueError, match="v must be finite"):
        _bridge_run(
            Population("StochasticLIFNeuron", 1, params=params), [-1e307, 0.0], torch.float64
        )


def test_the_last_step_may_overflow_as_in_the_model() -> None:
    """With no step after it, the model does not refuse; neither does the bridge."""
    params = {"resistance": 100.0, "dt": 1.0}
    events, _voltages, _grad = _bridge_run(
        Population("StochasticLIFNeuron", 1, params=params), [0.0, -1e307], torch.float64
    )
    assert events == [0.0, 0.0]


@pytest.mark.parametrize(
    ("dtype", "first", "resistance"),
    [
        (torch.float32, -1e38, 10.0),  # overflows only in float32: the model stays finite
        (torch.float64, -1e150, 2.0),  # control
        (torch.float32, -1e30, 2.0),  # control
    ],
)
def test_finite_negative_drives_agree(dtype: object, first: float, resistance: float) -> None:
    """Events and voltages equal the model's, every state finite, gradients finite."""
    drive = [first, 0.0, 0.2, 0.2]
    params = {"resistance": resistance, "dt": 1.0}
    values = _as_dtype(drive, dtype)
    events, voltages = _python_run(Population("StochasticLIFNeuron", 1, params=params), values)
    got_events, got_voltages, grad = _bridge_run(
        Population("StochasticLIFNeuron", 1, params=params), drive, dtype
    )
    assert got_events == events and got_voltages == voltages
    assert all(np.isfinite(voltages)) and bool(torch.isfinite(grad).all())


def test_a_lapicque_candidate_that_overflows_is_refused_in_its_step() -> None:
    """SC Lapicque refuses a non-finite v_inf in the step that forms it; so does the bridge."""
    params = {"resistance": 1e300, "dt": 1.0}
    with pytest.raises(ValueError, match="candidate must remain finite"):
        Population("SCLapicqueLIFNeuron", 1, params=params).neurons[0].step(1e10)
    with pytest.raises(ValueError, match="candidate must remain finite"):
        _bridge_run(Population("SCLapicqueLIFNeuron", 1, params=params), [1e10], torch.float64)
    # Control: a finite step is accepted by both.
    events, _v = _python_run(Population("SCLapicqueLIFNeuron", 1, params=params), [1e-300])
    got, _gv, _g = _bridge_run(
        Population("SCLapicqueLIFNeuron", 1, params=params), [1e-300], torch.float64
    )
    assert got == events


def _refractory(counter: int) -> Population:
    """Return a StochasticLIF population (period 0) whose neuron holds a live counter."""
    population = Population("StochasticLIFNeuron", 1, params={"dt": 1.0})
    population.neurons[0].refractory_counter = counter
    assert population.neurons[0].get_state()["refractory"] == counter
    return population


@pytest.mark.parametrize("counter", [2, 0])
def test_a_live_refractory_counter_is_carried(counter: int) -> None:
    """Counter 2 holds two steps at v_rest ([0, 0, 1, 1] at drive 1); counter 0 is the control."""
    drive = [1.0, 1.0, 1.0, 1.0]
    events, voltages = _python_run(_refractory(counter), drive)
    assert events == ([0.0, 0.0, 1.0, 1.0] if counter else [1.0, 1.0, 1.0, 1.0])
    got_events, got_voltages, grad = _bridge_run(_refractory(counter), drive, torch.float64)
    assert got_events == events and got_voltages == voltages
    assert bool(torch.isfinite(grad).all())
