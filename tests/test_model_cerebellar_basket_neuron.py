import math

import pytest

from sc_neurocore.neurons.models.cerebellar_basket_neuron import CerebellarBasketNeuron


def snapshot(neuron: CerebellarBasketNeuron) -> tuple[float, float, float, float, float, float]:
    return neuron.v, neuron.h, neuron.n, neuron.a, neuron.b, neuron.ca


def test_default_step_preserves_finite_gates_and_calcium() -> None:
    neuron = CerebellarBasketNeuron()

    spike = neuron.step(0.0)

    assert spike in (0, 1)
    assert math.isfinite(neuron.v)
    assert 0.0 <= neuron.h <= 1.0
    assert 0.0 <= neuron.n <= 1.0
    assert 0.0 <= neuron.a <= 1.0
    assert 0.0 <= neuron.b <= 1.0
    assert neuron.ca >= 0.0


def test_kca_current_hyperpolarizes_relative_to_no_kca_current() -> None:
    with_kca = CerebellarBasketNeuron(g_na=0.0, g_k=0.0, g_a=0.0, g_kca=2.0, g_l=0.0, ca=1.0)
    without_kca = CerebellarBasketNeuron(g_na=0.0, g_k=0.0, g_a=0.0, g_kca=0.0, g_l=0.0, ca=1.0)

    with_kca.step(10.0)
    without_kca.step(10.0)

    assert with_kca.v < without_kca.v


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dt": 0.0},
        {"c_m": 0.0},
        {"phi": 0.0},
        {"h": -0.1},
        {"n": 1.1},
        {"a": math.nan},
        {"b": math.inf},
        {"ca": -0.1},
        {"g_kca": -1.0},
    ],
)
def test_invalid_physical_configuration_is_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        CerebellarBasketNeuron(**kwargs)


def test_non_finite_current_does_not_mutate_state() -> None:
    neuron = CerebellarBasketNeuron()
    before = snapshot(neuron)

    with pytest.raises(ValueError):
        neuron.step(math.nan)

    assert snapshot(neuron) == before


def test_corrupted_runtime_gate_does_not_mutate_state() -> None:
    neuron = CerebellarBasketNeuron()
    neuron.b = -0.1
    before = snapshot(neuron)

    with pytest.raises(ValueError):
        neuron.step(1.0)

    assert snapshot(neuron) == before


def test_corrupted_runtime_calcium_does_not_mutate_state() -> None:
    neuron = CerebellarBasketNeuron()
    neuron.ca = math.inf
    before = snapshot(neuron)

    with pytest.raises(ValueError):
        neuron.step(1.0)

    assert snapshot(neuron) == before


def test_candidate_outside_safety_bounds_does_not_mutate_state() -> None:
    neuron = CerebellarBasketNeuron(g_na=0.0, g_k=0.0, g_a=0.0, g_kca=0.0, g_l=0.0, c_m=1e-12)
    before = snapshot(neuron)

    with pytest.raises(ValueError):
        neuron.step(1.0)

    assert snapshot(neuron) == before
