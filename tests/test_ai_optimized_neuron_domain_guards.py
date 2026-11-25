# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — AI-optimised neuron domain, finiteness and overflow guards

"""Domain, type, finiteness and overflow guards for the eight AI-optimised neuron models.

Covers ``MultiTimescaleNeuron``, ``AttentionGatedNeuron``, ``PredictiveCodingNeuron``,
``SelfReferentialNeuron``, ``CompositionalBindingNeuron``, ``DifferentiableSurrogateNeuron``,
``ContinuousAttractorNeuron`` and ``MetaPlasticNeuron``. Every invalid input, invalid
configuration or candidate-state overflow must raise without mutating state, while finite
extreme sigmoid or activation arguments evaluate without spurious ``OverflowError`` and
valid trajectories remain bit-identical to the original equations.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from sc_neurocore.neurons.models.ai_optimized import (
    AttentionGatedNeuron,
    CompositionalBindingNeuron,
    ContinuousAttractorNeuron,
    DifferentiableSurrogateNeuron,
    MetaPlasticNeuron,
    MultiTimescaleNeuron,
    PredictiveCodingNeuron,
    SelfReferentialNeuron,
)

_NON_FINITE: tuple[object, ...] = (
    math.nan,
    math.inf,
    -math.inf,
    np.float64("nan"),
    np.float64("inf"),
    np.float64("-inf"),
)

_INVALID_REAL_TYPES: tuple[object, ...] = (
    True,
    False,
    np.bool_(True),
    1 + 2j,
    "1.0",
    None,
)


def _raw(value: object) -> Any:
    """Return an untyped handle so strict tests can probe runtime type guards."""
    return value


class TestMultiTimescaleGuards:
    """Finiteness, domain and overflow guards for ``MultiTimescaleNeuron``."""

    @pytest.mark.parametrize("bad_input", _NON_FINITE)
    def test_rejects_non_finite_input_and_preserves_state(self, bad_input: object) -> None:
        """Non-finite inputs raise ``ValueError`` with all three compartments intact."""
        neuron = MultiTimescaleNeuron(v_fast=0.4, v_medium=0.2, v_slow=0.1)
        with pytest.raises(ValueError, match="finite"):
            neuron.step(_raw(bad_input))
        assert (neuron.v_fast, neuron.v_medium, neuron.v_slow) == (0.4, 0.2, 0.1)

    @pytest.mark.parametrize("bad_input", _INVALID_REAL_TYPES)
    def test_rejects_invalid_types(self, bad_input: object) -> None:
        """Booleans, complex values and non-numbers raise ``TypeError``."""
        neuron = MultiTimescaleNeuron(v_fast=0.4)
        with pytest.raises(TypeError):
            neuron.step(_raw(bad_input))
        assert neuron.v_fast == 0.4

    @pytest.mark.parametrize(
        ("kwargs", "exc_type"),
        [
            ({"tau_fast": 0.0}, ValueError),
            ({"tau_medium": -1.0}, ValueError),
            ({"tau_slow": 0.0}, ValueError),
            ({"dt": -0.5}, ValueError),
            ({"v_fast": math.nan}, ValueError),
            ({"gamma": math.inf}, ValueError),
            ({"dt": True}, TypeError),
            ({"alpha": 1j}, TypeError),
        ],
    )
    def test_constructor_validation(
        self, kwargs: dict[str, object], exc_type: type[Exception]
    ) -> None:
        """Invalid time constants, non-positive ``dt`` and non-finite fields fail closed."""
        with pytest.raises(exc_type):
            MultiTimescaleNeuron(**_raw(kwargs))

    def test_post_mutation_overflow_and_reset(self) -> None:
        """Mutated time constants and huge finite overflows leave state unchanged."""
        mutated = MultiTimescaleNeuron(v_fast=0.2, v_medium=0.1, v_slow=0.05)
        mutated.tau_fast = 0.0
        with pytest.raises(ValueError, match="tau_fast"):
            mutated.step(1.0)
        assert (mutated.v_fast, mutated.v_medium, mutated.v_slow) == (0.2, 0.1, 0.05)

        overflow = MultiTimescaleNeuron(v_fast=0.0, tau_fast=0.1, dt=1.0)
        with pytest.raises(FloatingPointError):
            overflow.step(1e308)
        assert overflow.v_fast == 0.0

        huge_int = MultiTimescaleNeuron()
        with pytest.raises(FloatingPointError):
            huge_int.step(_raw(10**400))
        assert huge_int.v_fast == 0.0

        valid = MultiTimescaleNeuron()
        assert valid.step(10.0) == 1
        assert valid.v_fast == 0.0
        assert valid.v_medium > 0.0
        valid.reset()
        assert (valid.v_fast, valid.v_medium, valid.v_slow) == (0.0, 0.0, 0.0)


class TestAttentionGatedGuards:
    """Overflow-safe sigmoid and domain guards for ``AttentionGatedNeuron``."""

    @pytest.mark.parametrize("bad_input", _NON_FINITE)
    def test_rejects_non_finite_input_and_preserves_state(self, bad_input: object) -> None:
        """Non-finite inputs raise ``ValueError`` without altering ``v``."""
        neuron = AttentionGatedNeuron(v=0.3)
        with pytest.raises(ValueError, match="finite"):
            neuron.step(_raw(bad_input))
        assert neuron.v == 0.3

    @pytest.mark.parametrize("bad_input", _INVALID_REAL_TYPES)
    def test_rejects_invalid_types(self, bad_input: object) -> None:
        """Booleans, complex values and non-numbers raise ``TypeError``."""
        neuron = AttentionGatedNeuron(v=0.3)
        with pytest.raises(TypeError):
            neuron.step(_raw(bad_input))
        assert neuron.v == 0.3

    @pytest.mark.parametrize(
        ("kwargs", "exc_type"),
        [
            ({"tau": 0.0}, ValueError),
            ({"tau": -5.0}, ValueError),
            ({"dt": 0.0}, ValueError),
            ({"v": math.nan}, ValueError),
            ({"w_key": math.inf}, ValueError),
            ({"theta": True}, TypeError),
        ],
    )
    def test_constructor_validation(
        self, kwargs: dict[str, object], exc_type: type[Exception]
    ) -> None:
        """Invalid ``tau``, ``dt`` or non-finite parameters fail at construction."""
        with pytest.raises(exc_type):
            AttentionGatedNeuron(**_raw(kwargs))

    def test_large_negative_finite_input_does_not_overflow_sigmoid(self) -> None:
        """Strong negative inputs saturate the attention gate to 0 without ``OverflowError``."""
        neuron = AttentionGatedNeuron()
        assert neuron.step(-1000.0) == 0
        assert neuron.v == 0.0
        assert neuron.step(-1e6) == 0
        assert neuron.v == 0.0

    def test_post_mutation_overflow_and_reset(self) -> None:
        """Mutated ``tau`` and huge positive-drive overflow leave ``v`` untouched."""
        mutated = AttentionGatedNeuron(v=0.25)
        mutated.tau = -1.0
        with pytest.raises(ValueError, match="tau"):
            mutated.step(1.0)
        assert mutated.v == 0.25

        overflow = AttentionGatedNeuron(v=0.0, w_key=1.0, tau=0.1, dt=1.0)
        with pytest.raises(FloatingPointError):
            overflow.step(1e308)
        assert overflow.v == 0.0

        neuron = AttentionGatedNeuron()
        assert neuron.step(20.0) == 1
        assert neuron.v == 0.0
        neuron.step(2.0)
        neuron.reset()
        assert neuron.v == 0.0


class TestPredictiveCodingGuards:
    """Finiteness, domain and overflow guards for ``PredictiveCodingNeuron``."""

    @pytest.mark.parametrize("bad_input", _NON_FINITE)
    def test_rejects_non_finite_input_and_preserves_state(self, bad_input: object) -> None:
        """Non-finite inputs raise ``ValueError`` with ``(v, pred)`` intact."""
        neuron = PredictiveCodingNeuron(v=0.2, pred=1.5)
        with pytest.raises(ValueError, match="finite"):
            neuron.step(_raw(bad_input))
        assert (neuron.v, neuron.pred) == (0.2, 1.5)

    @pytest.mark.parametrize(
        ("kwargs", "exc_type"),
        [
            ({"tau": 0.0}, ValueError),
            ({"tau_pred": -10.0}, ValueError),
            ({"dt": 0.0}, ValueError),
            ({"pred": math.nan}, ValueError),
            ({"theta": math.inf}, ValueError),
            ({"tau": False}, TypeError),
        ],
    )
    def test_constructor_validation(
        self, kwargs: dict[str, object], exc_type: type[Exception]
    ) -> None:
        """Invalid time constants, ``dt`` or non-finite parameters fail closed."""
        with pytest.raises(exc_type):
            PredictiveCodingNeuron(**_raw(kwargs))

    def test_post_mutation_overflow_and_reset(self) -> None:
        """Corrupted parameters and huge surprise overflow leave ``(v, pred)`` intact."""
        mutated = PredictiveCodingNeuron(v=0.1, pred=0.5)
        mutated.tau_pred = 0.0
        with pytest.raises(ValueError, match="tau_pred"):
            mutated.step(2.0)
        assert (mutated.v, mutated.pred) == (0.1, 0.5)

        overflow = PredictiveCodingNeuron(v=0.0, pred=-1e308)
        with pytest.raises(FloatingPointError):
            overflow.step(1e308)
        assert (overflow.v, overflow.pred) == (0.0, -1e308)

        neuron = PredictiveCodingNeuron()
        assert neuron.step(20.0) == 1
        neuron.reset()
        assert (neuron.v, neuron.pred) == (0.0, 0.0)


class TestSelfReferentialGuards:
    """Finiteness, domain and transactional guards for ``SelfReferentialNeuron``."""

    @pytest.mark.parametrize("bad_input", _NON_FINITE)
    def test_rejects_non_finite_input_and_preserves_history_and_step_count(
        self, bad_input: object
    ) -> None:
        """Non-finite inputs leave ``v``, ``_history`` and ``_step_count`` untouched."""
        neuron = SelfReferentialNeuron()
        neuron.step(2.0)
        v_before = neuron.v
        hist_before = list(neuron._history)
        count_before = neuron._step_count

        with pytest.raises(ValueError, match="finite"):
            neuron.step(_raw(bad_input))
        assert neuron.v == v_before
        assert list(neuron._history) == hist_before
        assert neuron._step_count == count_before

    @pytest.mark.parametrize(
        ("kwargs", "exc_type"),
        [
            ({"tau": 0.0}, ValueError),
            ({"target_rate": 0.0}, ValueError),
            ({"target_rate": -0.1}, ValueError),
            ({"dt": 0.0}, ValueError),
            ({"window": 0}, ValueError),
            ({"window": -5}, ValueError),
            ({"window": 1.5}, TypeError),
            ({"window": True}, TypeError),
            ({"window": math.nan}, ValueError),
            ({"_step_count": -1}, ValueError),
            ({"_step_count": True}, TypeError),
            ({"v": math.nan}, ValueError),
            ({"theta": math.inf}, ValueError),
        ],
    )
    def test_constructor_validation(
        self, kwargs: dict[str, object], exc_type: type[Exception]
    ) -> None:
        """Invalid parameters, window or step count fail at instantiation."""
        with pytest.raises(exc_type):
            SelfReferentialNeuron(**_raw(kwargs))

    def test_post_mutation_overflow_and_reset(self) -> None:
        """Mutated ``target_rate`` and candidate overflow leave history untouched."""
        mutated = SelfReferentialNeuron(v=0.3)
        mutated.target_rate = 0.0
        with pytest.raises(ValueError, match="target_rate"):
            mutated.step(1.0)
        assert (mutated.v, mutated._step_count, len(mutated._history)) == (0.3, 0, 0)

        overflow = SelfReferentialNeuron(v=0.0, tau=0.1, dt=1.0)
        with pytest.raises(FloatingPointError):
            overflow.step(1e308)
        assert (overflow.v, overflow._step_count, len(overflow._history)) == (0.0, 0, 0)

        neuron = SelfReferentialNeuron(window=_raw(np.int64(50)))
        assert neuron.step(20.0) == 1
        assert list(neuron._history) == [1]
        neuron.reset()
        assert (neuron.v, neuron._step_count, len(neuron._history)) == (0.0, 0, 0)


class TestCompositionalBindingGuards:
    """Finiteness, domain and overflow guards for ``CompositionalBindingNeuron``."""

    @pytest.mark.parametrize("bad_input", _NON_FINITE)
    def test_rejects_non_finite_input_and_preserves_state(self, bad_input: object) -> None:
        """Non-finite inputs leave ``(phi, amplitude)`` untouched."""
        neuron = CompositionalBindingNeuron(phi=0.5, amplitude=0.6)
        with pytest.raises(ValueError, match="finite"):
            neuron.step(_raw(bad_input))
        assert (neuron.phi, neuron.amplitude) == (0.5, 0.6)

    @pytest.mark.parametrize(
        ("kwargs", "exc_type"),
        [
            ({"tau": 0.0}, ValueError),
            ({"dt": -1.0}, ValueError),
            ({"phi": math.nan}, ValueError),
            ({"omega": math.inf}, ValueError),
            ({"coupling": math.nan}, ValueError),
            ({"tau": True}, TypeError),
        ],
    )
    def test_constructor_validation(
        self, kwargs: dict[str, object], exc_type: type[Exception]
    ) -> None:
        """Invalid ``tau``, ``dt`` or non-finite parameters fail at construction."""
        with pytest.raises(exc_type):
            CompositionalBindingNeuron(**_raw(kwargs))

    def test_post_mutation_overflow_and_reset(self) -> None:
        """Corrupted parameters and candidate overflow leave ``(phi, amplitude)`` intact."""
        mutated = CompositionalBindingNeuron(phi=0.2, amplitude=0.3)
        mutated.dt = 0.0
        with pytest.raises(ValueError, match="dt"):
            mutated.step(1.0)
        assert (mutated.phi, mutated.amplitude) == (0.2, 0.3)

        overflow = CompositionalBindingNeuron(amplitude=0.0, tau=0.1, dt=1.0)
        with pytest.raises(FloatingPointError):
            overflow.step(1e308)
        assert overflow.amplitude == 0.0

        neuron = CompositionalBindingNeuron()
        assert neuron.step(20.0) == 1
        neuron.reset()
        assert (neuron.phi, neuron.amplitude) == (0.0, 0.0)


class TestDifferentiableSurrogateGuards:
    """Finiteness, domain, surrogate-gradient and overflow guards."""

    @pytest.mark.parametrize("bad_input", _NON_FINITE)
    def test_rejects_non_finite_input_and_preserves_state(self, bad_input: object) -> None:
        """Non-finite inputs leave ``v`` untouched."""
        neuron = DifferentiableSurrogateNeuron(v=0.4)
        with pytest.raises(ValueError, match="finite"):
            neuron.step(_raw(bad_input))
        assert neuron.v == 0.4

    @pytest.mark.parametrize(
        ("kwargs", "exc_type"),
        [
            ({"beta": -0.1}, ValueError),
            ({"v": math.nan}, ValueError),
            ({"alpha": math.inf}, ValueError),
            ({"theta": math.nan}, ValueError),
            ({"beta": True}, TypeError),
        ],
    )
    def test_constructor_validation(
        self, kwargs: dict[str, object], exc_type: type[Exception]
    ) -> None:
        """Negative ``beta`` and non-finite or non-real parameters fail closed."""
        with pytest.raises(exc_type):
            DifferentiableSurrogateNeuron(**_raw(kwargs))

    def test_surrogate_grad_overflow_post_mutation_and_reset(self) -> None:
        """Extreme voltage underflows ``surrogate_grad`` to 0; overflows preserve ``v``."""
        far = DifferentiableSurrogateNeuron(v=1e200, beta=5.0)
        assert far.surrogate_grad() == 0.0

        bad_beta = DifferentiableSurrogateNeuron()
        bad_beta.beta = -1.0
        with pytest.raises(ValueError, match="beta"):
            bad_beta.surrogate_grad()

        overflow = DifferentiableSurrogateNeuron(v=5e307, alpha=2.0, theta=1e308)
        with pytest.raises(FloatingPointError):
            overflow.step(1e308)
        assert overflow.v == 5e307

        neuron = DifferentiableSurrogateNeuron()
        assert neuron.step(2.0) == 0
        assert neuron.step(0.0) == 1
        assert 0.0 < neuron.surrogate_grad() <= 1.0
        neuron.reset()
        assert neuron.v == 0.0


class TestContinuousAttractorGuards:
    """Finiteness, shape, overflow and parity guards for ``ContinuousAttractorNeuron``."""

    @pytest.mark.parametrize("bad_input", _NON_FINITE)
    def test_rejects_non_finite_input_and_preserves_ring(self, bad_input: object) -> None:
        """Non-finite inputs raise ``ValueError`` and leave ``u`` untouched."""
        ring = ContinuousAttractorNeuron(n_units=4)
        ring.step(1.0)
        before = list(ring.u)
        with pytest.raises(ValueError, match="finite"):
            ring.step(_raw(bad_input))
        assert ring.u == before

    @pytest.mark.parametrize(
        ("kwargs", "exc_type"),
        [
            ({"n_units": 0}, ValueError),
            ({"n_units": -4}, ValueError),
            ({"n_units": 4.5}, TypeError),
            ({"n_units": True}, TypeError),
            ({"tau": 0.0}, ValueError),
            ({"sigma_e": 0.0}, ValueError),
            ({"dt": -0.1}, ValueError),
            ({"excitation": math.nan}, ValueError),
            ({"inhibition": math.inf}, ValueError),
            ({"n_units": 4, "u": [0.0, 0.0]}, ValueError),
            ({"n_units": 3, "_weights": [[0.0, 0.0], [0.0, 0.0]]}, ValueError),
            ({"n_units": 2, "_weights": [[0.0, 0.0], [0.0]]}, ValueError),
        ],
    )
    def test_constructor_validation(
        self, kwargs: dict[str, object], exc_type: type[Exception]
    ) -> None:
        """Invalid ring size, time constants, ``sigma_e`` or mismatched shapes fail closed."""
        with pytest.raises(exc_type):
            ContinuousAttractorNeuron(**_raw(kwargs))

    def test_activation_overflow_and_step_guards(self) -> None:
        """Large activation saturates cleanly to 1.0; NaN state/weights fail closed on ``step``."""
        assert ContinuousAttractorNeuron._activation(-5.0) == 0.0
        assert ContinuousAttractorNeuron._activation(1e200) == 1.0

        nan_ring = ContinuousAttractorNeuron(n_units=4, u=[1.0, math.nan, 2.0, 0.0])
        assert nan_ring.bump_position() == 2
        with pytest.raises(ValueError, match="finite"):
            nan_ring.step(0.0)
        assert math.isnan(nan_ring.u[1])

        bad_weights = ContinuousAttractorNeuron(
            n_units=2,
            u=[0.0, 0.0],
            _weights=[[1.0, math.nan], [0.0, 1.0]],
        )
        with pytest.raises(ValueError, match="finite"):
            bad_weights.step(0.0)

        overflow = ContinuousAttractorNeuron(n_units=2, u=[1e308, 0.0], tau=0.1, dt=1.0)
        with pytest.raises(FloatingPointError):
            overflow.step(1e308)
        assert overflow.u == [1e308, 0.0]

        empty_bump = ContinuousAttractorNeuron(n_units=2)
        empty_bump.u = []
        with pytest.raises(ValueError, match="empty"):
            empty_bump.bump_position()
        with pytest.raises(ValueError, match="length"):
            empty_bump.step(0.0)

        ring = ContinuousAttractorNeuron(n_units=_raw(np.int64(4)))
        ring.step(2.0)
        ring.reset()
        assert ring.u == [0.0, 0.0, 0.0, 0.0]


class TestMetaPlasticGuards:
    """Overflow-safe meta-learning sigmoid and domain guards for ``MetaPlasticNeuron``."""

    @pytest.mark.parametrize("bad_input", _NON_FINITE)
    def test_rejects_non_finite_step_and_reward(self, bad_input: object) -> None:
        """Non-finite ``current`` or ``reward`` raises ``ValueError`` without state mutation."""
        neuron = MetaPlasticNeuron(v=0.3, error_trace=0.2, expected_reward=0.5)
        with pytest.raises(ValueError, match="finite"):
            neuron.step(_raw(bad_input))
        assert neuron.v == 0.3

        with pytest.raises(ValueError, match="finite"):
            neuron.update_meta(_raw(bad_input))
        assert (neuron.error_trace, neuron.expected_reward) == (0.2, 0.5)

    @pytest.mark.parametrize("bad_input", _INVALID_REAL_TYPES)
    def test_rejects_invalid_types(self, bad_input: object) -> None:
        """Booleans, complex values and non-numbers raise ``TypeError``."""
        neuron = MetaPlasticNeuron()
        with pytest.raises(TypeError):
            neuron.step(_raw(bad_input))
        with pytest.raises(TypeError):
            neuron.update_meta(_raw(bad_input))

    @pytest.mark.parametrize(
        ("kwargs", "exc_type"),
        [
            ({"tau": 0.0}, ValueError),
            ({"tau_meta": -1.0}, ValueError),
            ({"dt": 0.0}, ValueError),
            ({"lr0": -0.01}, ValueError),
            ({"v": math.nan}, ValueError),
            ({"kappa": math.inf}, ValueError),
            ({"lr0": True}, TypeError),
        ],
    )
    def test_constructor_validation(
        self, kwargs: dict[str, object], exc_type: type[Exception]
    ) -> None:
        """Invalid time constants, negative ``lr0`` or non-finite parameters fail closed."""
        with pytest.raises(exc_type):
            MetaPlasticNeuron(**_raw(kwargs))

    def test_extreme_sigmoid_overflow_and_reset(self) -> None:
        """Extreme ``meta_lr`` arguments saturate without ``OverflowError``; overflows roll back."""
        extreme = MetaPlasticNeuron(error_trace=-1000.0, kappa=5.0, target_error=0.3)
        assert extreme.meta_lr == 0.0
        extreme.update_meta(0.0)
        assert math.isfinite(extreme.expected_reward)

        corrupt_lr = MetaPlasticNeuron()
        corrupt_lr.lr0 = -1.0
        with pytest.raises(ValueError, match="lr0"):
            _ = corrupt_lr.meta_lr

        step_overflow = MetaPlasticNeuron(v=0.0, tau=0.1, dt=1.0)
        with pytest.raises(FloatingPointError):
            step_overflow.step(1e308)
        assert step_overflow.v == 0.0

        meta_overflow = MetaPlasticNeuron(expected_reward=-1e308)
        with pytest.raises(FloatingPointError):
            meta_overflow.update_meta(1e308)
        assert meta_overflow.expected_reward == -1e308

        neuron = MetaPlasticNeuron()
        assert neuron.step(20.0) == 1
        neuron.update_meta(1.0)
        assert 0.0 <= neuron.meta_lr <= neuron.lr0
        neuron.reset()
        assert (neuron.v, neuron.error_trace, neuron.expected_reward) == (0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    "cls",
    [
        MultiTimescaleNeuron,
        AttentionGatedNeuron,
        PredictiveCodingNeuron,
        SelfReferentialNeuron,
        CompositionalBindingNeuron,
        DifferentiableSurrogateNeuron,
        ContinuousAttractorNeuron,
        MetaPlasticNeuron,
    ],
)
def test_integer_and_numpy_scalar_drives_match_float_drive(cls: type[Any]) -> None:
    """Finite ``int`` and NumPy real drives coerce cleanly and match Python ``float``."""
    expected = cls().step(2.0)
    assert cls().step(2) == expected
    assert cls().step(np.float64(2.0)) == expected
