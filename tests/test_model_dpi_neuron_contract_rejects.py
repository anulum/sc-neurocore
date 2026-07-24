# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (contract_rejects) from former test_model_dpi_neuron.py

from __future__ import annotations

from tests.model_dpi_neuron_support import *  # noqa: F403


@pytest.mark.parametrize(
    "overrides",
    [
        {"i_mem": 0.0},
        {"i_mem": math.nan},
        {"i_ahp": -1.0},
        {"refractory_time": -0.1},
        {"i_threshold": math.inf},
        {"i_threshold": 0.0},
        {"i_reset": 1.0},
        {"i_rest": -0.1},
        {"i_tau": 0.0},
        {"i_g": 0.0},
        {"i_tau_ahp": 0.0},
        {"i_ga": 0.0},
        {"i_spike": 0.0},
        {"i_0": 0.0},
        {"kappa": 0.0},
        {"alpha": 0.0},
        {"tau": 0.0},
        {"tau_ahp": 0.0},
        {"refractory_period": 0.05},
        {"dt": 0.0},
    ],
)
def test_constructor_rejects_nonphysical_contract(overrides: dict[str, float]) -> None:
    """Reject every invalid maintained state/parameter family at construction."""
    with pytest.raises(ValueError):
        DPINeuron(**cast(Any, overrides))


@pytest.mark.parametrize(
    ("field", "value"),
    [("i_mem", math.nan), ("i_ahp", -1.0), ("i_ga", 0.0), ("dt", 0.0)],
)
def test_mutated_runtime_contract_is_revalidated(field: str, value: float) -> None:
    """Do not assume dataclass construction is the only mutation boundary."""
    neuron = DPINeuron()
    setattr(neuron, field, value)
    with pytest.raises(ValueError):
        neuron.step(0.0)


@pytest.mark.parametrize("current", [math.nan, math.inf, math.ulp(math.inf)])
def test_non_finite_input_fails_without_mutation(current: float) -> None:
    """Reject non-finite input before evaluating either coupled equation."""
    neuron = DPINeuron()
    before = (neuron.i_mem, neuron.i_ahp, neuron.refractory_time)
    with pytest.raises(ValueError, match="current"):
        neuron.step(current)
    assert (neuron.i_mem, neuron.i_ahp, neuron.refractory_time) == before


def test_negative_total_input_fails_without_mutation() -> None:
    """Keep the source current inside the physical non-negative domain."""
    neuron = DPINeuron()
    before = (neuron.i_mem, neuron.i_ahp, neuron.refractory_time)
    with pytest.raises(ValueError, match=r"i_rest \+ current"):
        neuron.step(-0.1000001)
    assert (neuron.i_mem, neuron.i_ahp, neuron.refractory_time) == before


def test_non_finite_membrane_candidate_fails_without_reset_masking() -> None:
    """Validate the Euler candidate before a threshold reset can hide overflow."""
    neuron = DPINeuron(tau=float.fromhex("0x1.0p-1022"))
    before = (neuron.i_mem, neuron.i_ahp, neuron.refractory_time)
    with pytest.raises(ValueError, match="candidate"):
        neuron.step(float.fromhex("0x1.fffffffffffffp+1023"))
    assert (neuron.i_mem, neuron.i_ahp, neuron.refractory_time) == before


def test_nonlinear_evaluation_failure_is_translated_without_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Contain a nonlinear-domain failure behind the public value contract."""
    neuron = DPINeuron()
    before = (neuron.i_mem, neuron.i_ahp, neuron.refractory_time)

    def fail_derivatives(
        _self: DPINeuron,
        _current: float,
        *,
        spike_active: bool,
    ) -> tuple[float, float]:
        assert not spike_active
        raise OverflowError("nonlinear circuit overflow")

    monkeypatch.setattr(DPINeuron, "_derivatives", fail_derivatives)
    with pytest.raises(ValueError, match="nonlinear current evaluation failed"):
        neuron.step(0.0)
    assert (neuron.i_mem, neuron.i_ahp, neuron.refractory_time) == before


def test_non_finite_adaptation_candidate_fails_without_mutation() -> None:
    """Reject arithmetic overflow across the simultaneous Euler candidate set."""
    neuron = DPINeuron(
        i_ahp=float.fromhex("0x1.fffffffffffffp+1023"),
        refractory_time=2.0,
        i_tau_ahp=1.0,
        tau_ahp=1.0,
        refractory_period=2.0,
        dt=2.0,
    )
    before = (neuron.i_mem, neuron.i_ahp, neuron.refractory_time)
    with pytest.raises(ValueError, match="Euler update must remain finite"):
        neuron.step(0.0)
    assert (neuron.i_mem, neuron.i_ahp, neuron.refractory_time) == before


def test_nonphysical_adaptation_candidate_fails_without_mutation() -> None:
    """Reject a negative post-Euler AHP current atomically."""
    neuron = DPINeuron(i_ahp=0.01, tau_ahp=0.01, dt=0.1)
    before = (neuron.i_mem, neuron.i_ahp, neuron.refractory_time)
    with pytest.raises(ValueError, match="physical current domain"):
        neuron.step(0.0)
    assert (neuron.i_mem, neuron.i_ahp, neuron.refractory_time) == before


@pytest.mark.parametrize("n_steps", [-1, 1.0, True])
def test_invalid_step_count_fails_before_mutation(n_steps: object) -> None:
    """Require a non-negative integer at the public simulation boundary."""
    neuron = DPINeuron()
    before = (neuron.i_mem, neuron.i_ahp, neuron.refractory_time)
    with pytest.raises(ValueError, match="n_steps"):
        neuron.simulate(cast(int, n_steps), 0.0)
    assert (neuron.i_mem, neuron.i_ahp, neuron.refractory_time) == before


def test_invalid_backend_and_total_current_fail_before_mutation() -> None:
    """Reject dispatch and current-domain errors without fallback mutation."""
    neuron = DPINeuron()
    before = (neuron.i_mem, neuron.i_ahp, neuron.refractory_time)
    with pytest.raises(ValueError, match="backend"):
        neuron.simulate(1, 0.0, backend="cuda")
    with pytest.raises(ValueError, match="current"):
        neuron.simulate(1, math.nan)
    with pytest.raises(ValueError, match="finite and non-negative"):
        neuron.simulate(1, -0.2)
    assert (neuron.i_mem, neuron.i_ahp, neuron.refractory_time) == before
