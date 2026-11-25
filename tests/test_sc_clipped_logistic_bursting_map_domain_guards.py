# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Retained clipped-logistic map domain and finiteness guards

"""Fail-closed domain, type, and overflow-rollback tests for the clipped-logistic map."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pytest

from sc_neurocore.neurons.models.sc_clipped_logistic_bursting_map import (
    SCClippedLogisticBurstingMapNeuron,
)

_ALL_FIELDS = ("x", "y", "a", "epsilon", "sigma", "x_threshold")
_NON_FINITE_VALUES = (float("nan"), float("inf"), float("-inf"))
_NON_REAL_VALUES: tuple[object, ...] = (True, False, "0.5", None, 1.0 + 2.0j)
_GOLDEN_SHA256 = "4264c3f9f94eef93e4f641afc91b0f97aff57f492838b2cfd32ce17335099967"


def _advance_default_neuron(steps: int = 5) -> SCClippedLogisticBurstingMapNeuron:
    """Return a default neuron advanced by ``steps`` valid updates."""
    neuron = SCClippedLogisticBurstingMapNeuron()
    for _ in range(steps):
        neuron.step(0.05)
    return neuron


def _state_snapshot(neuron: SCClippedLogisticBurstingMapNeuron) -> tuple[float, float]:
    """Return the current ``(x, y)`` state pair."""
    return (neuron.x, neuron.y)


class TestConstructionValidation:
    """Constructor validation for all six state and parameter fields."""

    @pytest.mark.parametrize("field", _ALL_FIELDS)
    @pytest.mark.parametrize("bad_value", _NON_FINITE_VALUES)
    def test_rejects_non_finite_fields(self, field: str, bad_value: float) -> None:
        """Non-finite state or parameter values raise ValueError at construction."""
        kwargs: dict[str, Any] = {field: bad_value}
        with pytest.raises(ValueError, match=field):
            SCClippedLogisticBurstingMapNeuron(**kwargs)

    @pytest.mark.parametrize("field", _ALL_FIELDS)
    @pytest.mark.parametrize("bad_value", _NON_REAL_VALUES)
    def test_rejects_bool_and_non_real_fields(self, field: str, bad_value: object) -> None:
        """Boolean and non-real state or parameter values raise TypeError at construction."""
        kwargs: dict[str, Any] = {field: bad_value}
        with pytest.raises(TypeError, match=field):
            SCClippedLogisticBurstingMapNeuron(**kwargs)

    def test_coerces_integer_and_numpy_scalars_to_float(self) -> None:
        """Integer and NumPy real scalars are normalised to Python floats."""
        kwargs: dict[str, Any] = {
            "x": 0,
            "y": np.int64(0),
            "a": np.float32(3.5),
            "epsilon": np.float64(0.02),
            "sigma": 1,
            "x_threshold": np.float32(0.75),
        }
        neuron = SCClippedLogisticBurstingMapNeuron(**kwargs)
        for field in _ALL_FIELDS:
            assert type(getattr(neuron, field)) is float
        assert neuron.x == 0.0
        assert neuron.y == 0.0
        assert neuron.a == 3.5
        assert neuron.epsilon == 0.02
        assert neuron.sigma == 1.0
        assert neuron.x_threshold == 0.75


class TestStepValidationAndOverflowRollback:
    """Runtime validation and failure-atomic overflow guards in ``step``."""

    @pytest.mark.parametrize("bad_current", _NON_FINITE_VALUES)
    def test_step_rejects_non_finite_current_without_mutating_state(
        self, bad_current: float
    ) -> None:
        """Non-finite drive raises ValueError and preserves ``(x, y)``."""
        neuron = _advance_default_neuron()
        before = _state_snapshot(neuron)
        with pytest.raises(ValueError, match="current"):
            neuron.step(bad_current)
        assert _state_snapshot(neuron) == before

    @pytest.mark.parametrize("bad_current", _NON_REAL_VALUES)
    def test_step_rejects_bool_and_non_real_current_without_mutating_state(
        self, bad_current: Any
    ) -> None:
        """Boolean or non-real drive raises TypeError and preserves ``(x, y)``."""
        neuron = _advance_default_neuron()
        before = _state_snapshot(neuron)
        with pytest.raises(TypeError, match="current"):
            neuron.step(bad_current)
        assert _state_snapshot(neuron) == before

    def test_step_accepts_integer_and_numpy_scalar_currents(self) -> None:
        """Integer and NumPy real drive values match exact Python float stepping."""
        drives: tuple[Any, ...] = (0, np.int64(1), np.float32(0.25), np.float64(-0.1))
        for drive in drives:
            actual = SCClippedLogisticBurstingMapNeuron()
            expected = SCClippedLogisticBurstingMapNeuron()
            assert actual.step(drive) == expected.step(float(drive))
            assert _state_snapshot(actual) == _state_snapshot(expected)

    @pytest.mark.parametrize("field", _ALL_FIELDS)
    @pytest.mark.parametrize("bad_value", _NON_FINITE_VALUES)
    def test_step_rejects_mutated_non_finite_attributes(self, field: str, bad_value: float) -> None:
        """Post-construction non-finite attribute mutation raises ValueError in ``step``."""
        neuron = _advance_default_neuron()
        setattr(neuron, field, bad_value)
        before_other = neuron.y if field == "x" else neuron.x
        with pytest.raises(ValueError, match=field):
            neuron.step(0.05)
        after_other = neuron.y if field == "x" else neuron.x
        assert after_other == before_other

    @pytest.mark.parametrize("field", _ALL_FIELDS)
    @pytest.mark.parametrize("bad_value", _NON_REAL_VALUES)
    def test_step_rejects_mutated_non_real_attributes(self, field: str, bad_value: object) -> None:
        """Post-construction non-real attribute mutation raises TypeError in ``step``."""
        neuron = _advance_default_neuron()
        setattr(neuron, field, bad_value)
        before_other = neuron.y if field == "x" else neuron.x
        with pytest.raises(TypeError, match=field):
            neuron.step(0.05)
        after_other = neuron.y if field == "x" else neuron.x
        assert after_other == before_other

    def test_step_handles_mutated_integer_parameter_without_rewriting_parameter(self) -> None:
        """Integer attribute mutation validates on the slow path and preserves exact math."""
        neuron = SCClippedLogisticBurstingMapNeuron()
        neuron.a = 3
        ref = SCClippedLogisticBurstingMapNeuron(a=3.0)
        assert neuron.step(0.05) == ref.step(0.05)
        assert _state_snapshot(neuron) == _state_snapshot(ref)

    def test_step_rejects_x_new_overflow_without_mutating_state(self) -> None:
        """Overflow in ``a * x * (1 - x) - y + current`` raises ValueError instead of clipping."""
        neuron = SCClippedLogisticBurstingMapNeuron(x=2.0, y=0.0, a=1e308)
        before = _state_snapshot(neuron)
        with pytest.raises(ValueError, match="non-finite"):
            neuron.step(0.0)
        assert _state_snapshot(neuron) == before

    def test_step_rejects_y_new_overflow_without_mutating_state(self) -> None:
        """Overflow in ``y + epsilon * (x - sigma)`` raises ValueError and preserves state."""
        neuron = SCClippedLogisticBurstingMapNeuron(x=2.0, y=1e308, epsilon=1e308, sigma=0.0)
        before = _state_snapshot(neuron)
        with pytest.raises(ValueError, match="non-finite"):
            neuron.step(0.0)
        assert _state_snapshot(neuron) == before

    def test_step_clips_large_finite_x_new_to_bounds(self) -> None:
        """Large finite ``x_new`` still clips cleanly to ``[-2.0, 2.0]``."""
        high = SCClippedLogisticBurstingMapNeuron()
        assert high.step(1e308) == 1
        assert high.x == 2.0
        assert high.y == pytest.approx(-0.004)

        low = SCClippedLogisticBurstingMapNeuron()
        assert low.step(-1e308) == 0
        assert low.x == -2.0
        assert low.y == pytest.approx(-0.004)


class TestSimulateValidationOverflowAndParity:
    """Runtime validation, overflow rollback, and parity in ``simulate``."""

    @pytest.mark.parametrize("bad_steps", (True, False, 1.5, "10", None))
    def test_simulate_rejects_bool_and_non_integer_n_steps(self, bad_steps: Any) -> None:
        """Boolean and non-integer ``n_steps`` raise TypeError and preserve state."""
        neuron = _advance_default_neuron()
        before = _state_snapshot(neuron)
        with pytest.raises(TypeError, match="n_steps"):
            neuron.simulate(bad_steps, 0.0)
        assert _state_snapshot(neuron) == before

    @pytest.mark.parametrize("bad_steps", (-1, -100))
    def test_simulate_rejects_negative_n_steps(self, bad_steps: int) -> None:
        """Negative ``n_steps`` raises ValueError and preserves state."""
        neuron = _advance_default_neuron()
        before = _state_snapshot(neuron)
        with pytest.raises(ValueError, match="n_steps must be non-negative"):
            neuron.simulate(bad_steps, 0.0)
        assert _state_snapshot(neuron) == before

    def test_simulate_rejects_n_steps_above_c_int_limit(self) -> None:
        """Step counts exceeding 32-bit signed int range raise ValueError."""
        neuron = _advance_default_neuron()
        before = _state_snapshot(neuron)
        for bad_steps in (1 << 64, 1 << 31):
            with pytest.raises(ValueError, match="n_steps must be at most"):
                neuron.simulate(bad_steps, 0.0)
            assert _state_snapshot(neuron) == before

    @pytest.mark.parametrize("bad_backend", ("cuda", "", 123, None))
    def test_simulate_rejects_invalid_backend(self, bad_backend: Any) -> None:
        """Unsupported or non-string backend names raise ValueError and preserve state."""
        neuron = _advance_default_neuron()
        before = _state_snapshot(neuron)
        with pytest.raises(ValueError, match="backend must be"):
            neuron.simulate(5, 0.0, backend=bad_backend)
        assert _state_snapshot(neuron) == before

    @pytest.mark.parametrize("backend", ("python", "auto"))
    @pytest.mark.parametrize("bad_current", _NON_FINITE_VALUES)
    def test_simulate_rejects_non_finite_current(self, backend: str, bad_current: float) -> None:
        """Non-finite drive raises ValueError before backend dispatch."""
        neuron = _advance_default_neuron()
        before = _state_snapshot(neuron)
        with pytest.raises(ValueError, match="current"):
            neuron.simulate(5, bad_current, backend=backend)
        assert _state_snapshot(neuron) == before

    @pytest.mark.parametrize("backend", ("python", "auto"))
    @pytest.mark.parametrize("bad_current", _NON_REAL_VALUES)
    def test_simulate_rejects_bool_and_non_real_current(
        self, backend: str, bad_current: Any
    ) -> None:
        """Boolean or non-real drive raises TypeError before backend dispatch."""
        neuron = _advance_default_neuron()
        before = _state_snapshot(neuron)
        with pytest.raises(TypeError, match="current"):
            neuron.simulate(5, bad_current, backend=backend)
        assert _state_snapshot(neuron) == before

    @pytest.mark.parametrize("backend", ("python", "auto"))
    @pytest.mark.parametrize("field", _ALL_FIELDS)
    def test_simulate_rejects_mutated_non_finite_and_non_real_attributes(
        self, backend: str, field: str
    ) -> None:
        """Post-construction invalid attributes raise before dispatch."""
        neuron_nan = _advance_default_neuron()
        setattr(neuron_nan, field, float("nan"))
        with pytest.raises(ValueError, match=field):
            neuron_nan.simulate(3, 0.05, backend=backend)

        neuron_bool = _advance_default_neuron()
        setattr(neuron_bool, field, True)
        with pytest.raises(TypeError, match=field):
            neuron_bool.simulate(3, 0.05, backend=backend)

    @pytest.mark.parametrize("backend", ("python", "auto"))
    def test_simulate_accepts_numpy_integer_steps_and_zero_steps(self, backend: str) -> None:
        """NumPy integer step counts and zero-step calls preserve exact state contracts."""
        neuron_zero = _advance_default_neuron()
        before = _state_snapshot(neuron_zero)
        empty_trace, zero_spikes = neuron_zero.simulate(0, 0.05, backend=backend)
        assert empty_trace.shape == (0,)
        assert empty_trace.dtype == np.float64
        assert zero_spikes == 0
        assert _state_snapshot(neuron_zero) == before

        neuron_np = SCClippedLogisticBurstingMapNeuron()
        neuron_ref = SCClippedLogisticBurstingMapNeuron()
        np_steps: Any = np.int64(25)
        np_current: Any = np.float32(0.05)
        trace_np, spikes_np = neuron_np.simulate(np_steps, np_current, backend=backend)
        trace_ref, spikes_ref = neuron_ref.simulate(25, float(np.float32(0.05)), backend="python")
        np.testing.assert_array_equal(trace_np, trace_ref)
        assert spikes_np == spikes_ref
        assert _state_snapshot(neuron_np) == _state_snapshot(neuron_ref)

    def test_simulate_python_rejects_x_new_overflow_without_mutating_state(self) -> None:
        """Python simulation rolls back state when ``x_new`` overflows to infinity."""
        neuron = SCClippedLogisticBurstingMapNeuron(x=2.0, y=0.0, a=1e308)
        before = _state_snapshot(neuron)
        with pytest.raises(ValueError, match="non-finite"):
            neuron.simulate(2, 0.0, backend="python")
        assert _state_snapshot(neuron) == before

    @pytest.mark.parametrize("backend", ("python", "auto"))
    def test_simulate_rejects_y_overflow_without_mutating_state(self, backend: str) -> None:
        """Immediate and delayed ``y`` overflow raise ValueError and leave state unchanged."""
        immediate = SCClippedLogisticBurstingMapNeuron(x=2.0, y=1e308, epsilon=1e308, sigma=0.0)
        before_imm = _state_snapshot(immediate)
        with pytest.raises(ValueError, match="non-finite"):
            immediate.simulate(2, 0.0, backend=backend)
        assert _state_snapshot(immediate) == before_imm

        delayed = SCClippedLogisticBurstingMapNeuron(x=2.0, y=1.7e308, epsilon=1e307, sigma=0.0)
        before_del = _state_snapshot(delayed)
        with pytest.raises(ValueError, match="non-finite"):
            delayed.simulate(20, 0.0, backend=backend)
        assert _state_snapshot(delayed) == before_del

    def test_reset_restores_default_state_and_preserves_parameters(self) -> None:
        """Resetting restores ``x = 0.1, y = 0.0`` without altering configured parameters."""
        neuron = SCClippedLogisticBurstingMapNeuron(a=3.5, epsilon=0.02, sigma=0.4, x_threshold=0.8)
        neuron.simulate(20, 0.1, backend="python")
        assert _state_snapshot(neuron) != (0.1, 0.0)
        neuron.reset()
        assert _state_snapshot(neuron) == (0.1, 0.0)
        assert (neuron.a, neuron.epsilon, neuron.sigma, neuron.x_threshold) == (
            3.5,
            0.02,
            0.4,
            0.8,
        )

    @pytest.mark.parametrize("backend", ("python", "auto"))
    def test_golden_trace_sha256_and_step_parity_preserved(self, backend: str) -> None:
        """Default 1000-step trace at ``current = 2.0`` preserves the descriptor golden digest."""
        neuron = SCClippedLogisticBurstingMapNeuron()
        trace, spikes = neuron.simulate(1000, 2.0, backend=backend)
        digest = hashlib.sha256(trace.tobytes()).hexdigest()
        assert digest == _GOLDEN_SHA256

        stepper = SCClippedLogisticBurstingMapNeuron()
        step_trace = np.empty(1000, dtype=np.float64)
        step_spikes = 0
        for index in range(1000):
            step_spikes += stepper.step(2.0)
            step_trace[index] = stepper.x
        np.testing.assert_array_equal(trace, step_trace)
        assert spikes == step_spikes
        assert _state_snapshot(neuron) == _state_snapshot(stepper)
