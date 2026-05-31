import math

import pytest

from sc_neurocore.neurons.models.chandelier_neuron import ChandelierNeuron


def snapshot(neuron: ChandelierNeuron) -> tuple[float, float, float, float, float]:
    return neuron.v, neuron.h, neuron.n, neuron.d, neuron.p


def test_default_step_preserves_finite_gate_probabilities() -> None:
    neuron = ChandelierNeuron()

    spike = neuron.step(0.0)

    assert spike in (0, 1)
    assert math.isfinite(neuron.v)
    assert 0.0 <= neuron.h <= 1.0
    assert 0.0 <= neuron.n <= 1.0
    assert 0.0 <= neuron.d <= 1.0
    assert 0.0 <= neuron.p <= 1.0


def test_kv_currents_hyperpolarize_relative_to_no_kv_currents() -> None:
    with_kv = ChandelierNeuron(g_na=0.0, g_k=0.0, g_kv1=3.0, g_kv3=4.0, g_l=0.0, d=1.0, p=1.0)
    without_kv = ChandelierNeuron(g_na=0.0, g_k=0.0, g_kv1=0.0, g_kv3=0.0, g_l=0.0, d=1.0, p=1.0)

    with_kv.step(10.0)
    without_kv.step(10.0)

    assert with_kv.v < without_kv.v


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dt": 0.0},
        {"c_m": 0.0},
        {"phi": 0.0},
        {"h": -0.1},
        {"n": 1.1},
        {"d": math.nan},
        {"p": math.inf},
        {"g_kv1": -1.0},
        {"g_kv3": -1.0},
    ],
)
def test_invalid_physical_configuration_is_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        ChandelierNeuron(**kwargs)


def test_non_finite_current_does_not_mutate_state() -> None:
    neuron = ChandelierNeuron()
    before = snapshot(neuron)

    with pytest.raises(ValueError):
        neuron.step(math.nan)

    assert snapshot(neuron) == before


def test_corrupted_runtime_gate_does_not_mutate_state() -> None:
    neuron = ChandelierNeuron()
    neuron.p = 1.5
    before = snapshot(neuron)

    with pytest.raises(ValueError):
        neuron.step(1.0)

    assert snapshot(neuron) == before


def test_candidate_outside_safety_bounds_does_not_mutate_state() -> None:
    neuron = ChandelierNeuron(g_na=0.0, g_k=0.0, g_kv1=0.0, g_kv3=0.0, g_l=0.0, c_m=1e-12)
    before = snapshot(neuron)

    with pytest.raises(ValueError):
        neuron.step(1.0)

    assert snapshot(neuron) == before
