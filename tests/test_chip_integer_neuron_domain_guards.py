# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Neuromorphic chip integer neuron domain and finiteness guards

"""Domain, type, finiteness and overflow guards for neuromorphic-chip integer neuron models.

Covers the five Python reference implementations modelled on fixed-point or integer
neuromorphic hardware: ``SpiNNaker2Neuron``, ``Loihi2Neuron``, ``LoihiCUBANeuron``,
``TrueNorthNeuron`` and ``AkidaNeuron``. Every invalid input, invalid configuration or
candidate-state overflow must raise without mutating neuron state, while valid integer
trajectories remain bit-identical and finite floating-point network drives stay compatible
for the models that support them.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from sc_neurocore.neurons.models.akida_neuron import AkidaNeuron
from sc_neurocore.neurons.models.loihi2 import Loihi2Neuron
from sc_neurocore.neurons.models.loihi_cuba import LoihiCUBANeuron
from sc_neurocore.neurons.models.spinnaker2 import SpiNNaker2Neuron
from sc_neurocore.neurons.models.truenorth import TrueNorthNeuron

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
    "5",
    None,
)


def _raw(value: object) -> Any:
    """Return an untyped handle so strict tests can probe runtime type guards."""
    return value


class TestSpiNNaker2Guards:
    """Finiteness, integer-type and configuration guards for ``SpiNNaker2Neuron``."""

    @pytest.mark.parametrize("bad_input", _NON_FINITE)
    def test_rejects_non_finite_input_and_preserves_state(self, bad_input: object) -> None:
        """Non-finite inputs raise ``ValueError`` in active and refractory states."""
        active = SpiNNaker2Neuron(v=300, _refrac_count=0)
        with pytest.raises(ValueError, match="finite"):
            active.step(_raw(bad_input))
        assert (active.v, active._refrac_count) == (300, 0)
        assert type(active.v) is int

        refractory = SpiNNaker2Neuron(v=0, _refrac_count=2)
        with pytest.raises(ValueError, match="finite"):
            refractory.step(_raw(bad_input))
        assert (refractory.v, refractory._refrac_count) == (0, 2)

    @pytest.mark.parametrize(
        "bad_input",
        (0.0, 5.0, 500.0, np.float64(5.0), *_INVALID_REAL_TYPES),
    )
    def test_rejects_float_and_non_int_on_first_step_without_drift(self, bad_input: object) -> None:
        """Finite floats and non-integers raise ``TypeError`` on step 1 with no drift."""
        neuron = SpiNNaker2Neuron(v=128)
        with pytest.raises(TypeError):
            neuron.step(_raw(bad_input))
        assert neuron.v == 128
        assert type(neuron.v) is int
        assert neuron._refrac_count == 0

        reference = SpiNNaker2Neuron(v=128)
        assert neuron.step(500) == reference.step(500)
        assert (neuron.v, neuron._refrac_count) == (reference.v, reference._refrac_count)

    def test_accepts_numpy_integral_and_preserves_python_int(self) -> None:
        """NumPy integer scalars coerce to exact Python integers with identical trajectory."""
        numpy_neuron = SpiNNaker2Neuron()
        int_neuron = SpiNNaker2Neuron()
        for _ in range(12):
            assert numpy_neuron.step(_raw(np.int64(400))) == int_neuron.step(400)
            assert type(numpy_neuron.v) is int
            assert (numpy_neuron.v, numpy_neuron._refrac_count) == (
                int_neuron.v,
                int_neuron._refrac_count,
            )
        numpy_neuron.reset()
        assert (numpy_neuron.v, numpy_neuron._refrac_count) == (numpy_neuron.v_rest, 0)

    @pytest.mark.parametrize(
        ("kwargs", "exc_type"),
        [
            ({"decay_mult": -1}, ValueError),
            ({"decay_shift": -1}, ValueError),
            ({"refrac_steps": -1}, ValueError),
            ({"_refrac_count": -1}, ValueError),
            ({"v": math.nan}, ValueError),
            ({"v_threshold": math.inf}, ValueError),
            ({"v": 1.5}, TypeError),
            ({"decay_shift": True}, TypeError),
            ({"v_rest": 1 + 0j}, TypeError),
        ],
    )
    def test_constructor_domain_and_type_validation(
        self, kwargs: dict[str, object], exc_type: type[Exception]
    ) -> None:
        """Invalid constructor values fail closed at instantiation."""
        with pytest.raises(exc_type):
            SpiNNaker2Neuron(**_raw(kwargs))

    def test_post_construction_mutation_is_caught(self) -> None:
        """Direct attribute corruption is caught on ``step`` and ``reset``."""
        neuron = SpiNNaker2Neuron()
        neuron.decay_shift = -1
        with pytest.raises(ValueError, match="decay_shift"):
            neuron.step(10)

        corrupt_state = SpiNNaker2Neuron()
        corrupt_state.v = _raw(math.nan)
        with pytest.raises(ValueError, match="finite"):
            corrupt_state.step(10)

        corrupt_rest = SpiNNaker2Neuron()
        corrupt_rest.v_rest = _raw(math.inf)
        with pytest.raises(ValueError, match="finite"):
            corrupt_rest.reset()


class TestLoihi2Guards:
    """Finiteness, domain, overflow and parity guards for ``Loihi2Neuron``."""

    @pytest.mark.parametrize("bad_input", _NON_FINITE)
    def test_rejects_non_finite_input_and_preserves_state(self, bad_input: object) -> None:
        """Non-finite inputs raise ``ValueError`` and leave all three states untouched."""
        neuron = Loihi2Neuron(s1=15, s2=-7, s3=3)
        with pytest.raises(ValueError, match="finite"):
            neuron.step(_raw(bad_input))
        assert (neuron.s1, neuron.s2, neuron.s3) == (15, -7, 3)

    @pytest.mark.parametrize("bad_input", _INVALID_REAL_TYPES)
    def test_rejects_non_real_types_and_preserves_state(self, bad_input: object) -> None:
        """Booleans, complex values and non-numbers raise ``TypeError``."""
        neuron = Loihi2Neuron(s1=15, s2=-7, s3=3)
        with pytest.raises(TypeError):
            neuron.step(_raw(bad_input))
        assert (neuron.s1, neuron.s2, neuron.s3) == (15, -7, 3)

    @pytest.mark.parametrize(
        ("kwargs", "exc_type"),
        [
            ({"tau1": 0}, ValueError),
            ({"tau2": -5}, ValueError),
            ({"tau3": 0.5}, ValueError),
            ({"tau1": math.nan}, ValueError),
            ({"tau2": math.inf}, ValueError),
            ({"s1": math.nan}, ValueError),
            ({"s1_threshold": math.inf}, ValueError),
            ({"tau1": True}, TypeError),
            ({"w12": 1 + 0j}, TypeError),
        ],
    )
    def test_constructor_domain_and_type_validation(
        self, kwargs: dict[str, object], exc_type: type[Exception]
    ) -> None:
        """Invalid time constants or non-finite parameters fail at instantiation."""
        with pytest.raises(exc_type):
            Loihi2Neuron(**_raw(kwargs))

    def test_post_construction_mutation_is_caught(self) -> None:
        """Mutating a time constant or state to an invalid value raises on ``step``."""
        neuron = Loihi2Neuron(s1=10, s2=5, s3=2)
        neuron.tau1 = 0
        with pytest.raises(ValueError, match="tau1"):
            neuron.step(10)
        assert (neuron.s1, neuron.s2, neuron.s3) == (10, 5, 2)

        corrupt = Loihi2Neuron()
        corrupt.s2 = _raw(math.nan)
        with pytest.raises(ValueError, match="finite"):
            corrupt.step(10)

    def test_finite_float_overflow_leaves_state_unchanged(self) -> None:
        """Floating-point overflow during candidate update raises ``FloatingPointError``."""
        neuron = Loihi2Neuron(s1=_raw(0.0), s2=_raw(1e308), s3=_raw(0.0))
        with pytest.raises(FloatingPointError):
            neuron.step(_raw(1e308))
        assert (neuron.s1, neuron.s2, neuron.s3) == (0.0, 1e308, 0.0)

        spike_overflow = Loihi2Neuron(
            s1=_raw(1000.0),
            s2=_raw(0.0),
            s3=_raw(1e308),
            s3_incr=_raw(1e308),
            tau3=100,
        )
        with pytest.raises(FloatingPointError):
            spike_overflow.step(_raw(500.0))
        assert (spike_overflow.s1, spike_overflow.s2, spike_overflow.s3) == (
            1000.0,
            0.0,
            1e308,
        )

    def test_exact_integer_and_finite_float_compatibility(self) -> None:
        """Integer drives keep exact ``int`` state; finite float drives stay operational."""
        exact = Loihi2Neuron()
        np_exact = Loihi2Neuron()
        for value in (-7, -3, 5, 200, 200, 200):
            assert exact.step(value) == np_exact.step(_raw(np.int64(value)))
            assert (type(exact.s1), type(exact.s2), type(exact.s3)) == (int, int, int)
            assert (type(np_exact.s1), type(np_exact.s2), type(np_exact.s3)) == (int, int, int)
            assert (exact.s1, exact.s2, exact.s3) == (np_exact.s1, np_exact.s2, np_exact.s3)

        float_driven = Loihi2Neuron(tau1=_raw(np.int64(10)))
        assert float_driven.step(_raw(200.0)) == 0
        assert float_driven.step(_raw(1000.0)) == 1
        assert math.isfinite(float_driven.s1) and math.isfinite(float_driven.s2)
        float_driven.reset()
        assert (float_driven.s1, float_driven.s2, float_driven.s3) == (0, 0, 0)

        int_float_overflow = Loihi2Neuron(s2=10**400, tau2=_raw(10.0))
        with pytest.raises(FloatingPointError):
            int_float_overflow.step(0)
        assert int_float_overflow.s2 == 10**400


class TestLoihiCUBAGuards:
    """Finiteness, domain, overflow and parity guards for ``LoihiCUBANeuron``."""

    @pytest.mark.parametrize("bad_input", _NON_FINITE)
    def test_rejects_non_finite_input_and_preserves_state(self, bad_input: object) -> None:
        """Non-finite inputs raise ``ValueError`` and leave ``(v, u)`` untouched."""
        neuron = LoihiCUBANeuron(v=42, u=-9)
        with pytest.raises(ValueError, match="finite"):
            neuron.step(_raw(bad_input))
        assert (neuron.v, neuron.u) == (42, -9)

    @pytest.mark.parametrize("bad_input", _INVALID_REAL_TYPES)
    def test_rejects_non_real_types_and_preserves_state(self, bad_input: object) -> None:
        """Booleans, complex values and non-numbers raise ``TypeError``."""
        neuron = LoihiCUBANeuron(v=42, u=-9)
        with pytest.raises(TypeError):
            neuron.step(_raw(bad_input))
        assert (neuron.v, neuron.u) == (42, -9)

    @pytest.mark.parametrize(
        ("kwargs", "exc_type"),
        [
            ({"tau_v": 0}, ValueError),
            ({"tau_u": -3}, ValueError),
            ({"tau_v": 0.5}, ValueError),
            ({"tau_u": math.nan}, ValueError),
            ({"v": math.inf}, ValueError),
            ({"v_threshold": math.nan}, ValueError),
            ({"tau_u": False}, TypeError),
            ({"u": 2j}, TypeError),
        ],
    )
    def test_constructor_domain_and_type_validation(
        self, kwargs: dict[str, object], exc_type: type[Exception]
    ) -> None:
        """Invalid decay time constants or non-finite parameters fail at construction."""
        with pytest.raises(exc_type):
            LoihiCUBANeuron(**_raw(kwargs))

    def test_post_construction_mutation_and_overflow_are_caught(self) -> None:
        """Mutated time constants, corrupted states and float overflows leave state intact."""
        mutated = LoihiCUBANeuron(v=20, u=10)
        mutated.tau_u = 0
        with pytest.raises(ValueError, match="tau_u"):
            mutated.step(5)
        assert (mutated.v, mutated.u) == (20, 10)

        overflow = LoihiCUBANeuron(v=_raw(0.0), u=_raw(1e308))
        with pytest.raises(FloatingPointError):
            overflow.step(_raw(1e308))
        assert (overflow.v, overflow.u) == (0.0, 1e308)

        int_float_overflow = LoihiCUBANeuron(u=10**400, tau_u=_raw(5.0))
        with pytest.raises(FloatingPointError):
            int_float_overflow.step(0)
        assert int_float_overflow.u == 10**400

    def test_exact_integer_and_finite_float_compatibility(self) -> None:
        """Integer drives preserve exact truncation parity; finite floats stay operational."""
        neuron = LoihiCUBANeuron()
        np_neuron = LoihiCUBANeuron(tau_v=_raw(np.int64(10)))
        for value in (-7, -3, 5, -9, -1, -4):
            neuron.step(value)
            np_neuron.step(_raw(np.int64(value)))
        assert (neuron.v, neuron.u) == (-48, -13)
        assert (np_neuron.v, np_neuron.u) == (-48, -13)
        assert (type(np_neuron.v), type(np_neuron.u)) == (int, int)

        float_neuron = LoihiCUBANeuron()
        assert float_neuron.step(_raw(200.0)) == 0
        assert float_neuron.u == 200.0
        assert float_neuron.step(_raw(1000.0)) == 1
        float_neuron.reset()
        assert (float_neuron.v, float_neuron.u) == (0, 0)


class TestTrueNorthGuards:
    """Finiteness, domain, overflow and parity guards for ``TrueNorthNeuron``."""

    @pytest.mark.parametrize("bad_input", _NON_FINITE)
    def test_rejects_non_finite_input_and_preserves_state(self, bad_input: object) -> None:
        """Non-finite inputs raise ``ValueError`` without modifying ``v``."""
        neuron = TrueNorthNeuron(v=25)
        with pytest.raises(ValueError, match="finite"):
            neuron.step(_raw(bad_input))
        assert neuron.v == 25

    @pytest.mark.parametrize("bad_input", _INVALID_REAL_TYPES)
    def test_rejects_non_real_types_and_preserves_state(self, bad_input: object) -> None:
        """Booleans, complex values and non-numbers raise ``TypeError``."""
        neuron = TrueNorthNeuron(v=25)
        with pytest.raises(TypeError):
            neuron.step(_raw(bad_input))
        assert neuron.v == 25

    @pytest.mark.parametrize(
        ("kwargs", "exc_type"),
        [
            ({"threshold": 0}, ValueError),
            ({"threshold": -10}, ValueError),
            ({"threshold": math.nan}, ValueError),
            ({"v": math.inf}, ValueError),
            ({"leak": -math.inf}, ValueError),
            ({"v_reset": math.nan}, ValueError),
            ({"threshold": True}, TypeError),
            ({"leak": 1j}, TypeError),
        ],
    )
    def test_constructor_domain_and_type_validation(
        self, kwargs: dict[str, object], exc_type: type[Exception]
    ) -> None:
        """Non-positive thresholds and non-finite or non-real parameters fail closed."""
        with pytest.raises(exc_type):
            TrueNorthNeuron(**_raw(kwargs))

    def test_post_construction_mutation_and_overflow_are_caught(self) -> None:
        """Corrupted thresholds or floating-point overflow leave ``v`` untouched."""
        mutated = TrueNorthNeuron(v=30)
        mutated.threshold = 0
        with pytest.raises(ValueError, match="threshold"):
            mutated.step(10)
        assert mutated.v == 30

        overflow = TrueNorthNeuron(v=_raw(1e308))
        with pytest.raises(FloatingPointError):
            overflow.step(_raw(1e308))
        assert overflow.v == 1e308

        int_float_overflow = TrueNorthNeuron(v=10**400)
        with pytest.raises(FloatingPointError):
            int_float_overflow.step(_raw(1.0))
        assert int_float_overflow.v == 10**400

    def test_exact_integer_and_finite_float_compatibility(self) -> None:
        """Integer drives keep exact ``int`` state and negative leak remains valid."""
        neuron = TrueNorthNeuron(v=_raw(np.int64(0)), leak=-2, threshold=100)
        assert neuron.step(_raw(np.int64(8))) == 0
        assert neuron.v == 10
        assert type(neuron.v) is int

        float_neuron = TrueNorthNeuron(threshold=100)
        assert float_neuron.step(_raw(50.0)) == 0
        assert float_neuron.v == 50.0
        assert float_neuron.step(_raw(60.0)) == 1
        assert float_neuron.step(_raw(-200.0)) == 0
        assert float_neuron.v == 0
        float_neuron.step(20)
        float_neuron.reset()
        assert float_neuron.v == 0


class TestAkidaGuards:
    """Finiteness, domain, overflow and parity guards for ``AkidaNeuron``."""

    @pytest.mark.parametrize("bad_weight", _NON_FINITE)
    def test_rejects_non_finite_weight_with_value_error_and_preserves_state(
        self, bad_weight: object
    ) -> None:
        """NaN and signed infinities raise ``ValueError`` (never raw ``OverflowError``)."""
        neuron = AkidaNeuron(v=40, threshold=100, modulation=0.75)
        neuron.step(10)
        snapshot = (neuron.v, neuron._rank, neuron._spiked, neuron._current_modulation)
        with pytest.raises(ValueError, match="finite"):
            neuron.step(_raw(bad_weight))
        assert (neuron.v, neuron._rank, neuron._spiked, neuron._current_modulation) == snapshot

    @pytest.mark.parametrize("bad_weight", _INVALID_REAL_TYPES)
    def test_rejects_non_real_types_and_preserves_state(self, bad_weight: object) -> None:
        """Booleans, complex values and non-numbers raise ``TypeError``."""
        neuron = AkidaNeuron()
        snapshot = (neuron.v, neuron._rank, neuron._spiked, neuron._current_modulation)
        with pytest.raises(TypeError):
            neuron.step(_raw(bad_weight))
        assert (neuron.v, neuron._rank, neuron._spiked, neuron._current_modulation) == snapshot

    @pytest.mark.parametrize(
        ("kwargs", "exc_type"),
        [
            ({"modulation": -0.01}, ValueError),
            ({"modulation": 1.01}, ValueError),
            ({"modulation": math.nan}, ValueError),
            ({"modulation": math.inf}, ValueError),
            ({"_current_modulation": -0.1}, ValueError),
            ({"_current_modulation": 1.5}, ValueError),
            ({"_current_modulation": math.nan}, ValueError),
            ({"_rank": -1}, ValueError),
            ({"_rank": 1.5}, TypeError),
            ({"_rank": True}, TypeError),
            ({"_rank": math.nan}, ValueError),
            ({"_spiked": 1}, TypeError),
            ({"v": math.nan}, ValueError),
            ({"threshold": math.inf}, ValueError),
            ({"modulation": True}, TypeError),
            ({"modulation": 1 + 0j}, TypeError),
            ({"v": 1 + 0j}, TypeError),
        ],
    )
    def test_constructor_domain_and_type_validation(
        self, kwargs: dict[str, object], exc_type: type[Exception]
    ) -> None:
        """Invalid modulation, rank, spiked flag or non-finite fields fail at construction."""
        with pytest.raises(exc_type):
            AkidaNeuron(**_raw(kwargs))

    def test_post_construction_mutation_and_overflow_are_caught(self) -> None:
        """Post-construction corruption and huge finite-weight overflow leave state intact."""
        mutated = AkidaNeuron(v=10)
        mutated.modulation = -0.5
        with pytest.raises(ValueError, match="modulation"):
            mutated.step(5)
        assert (mutated.v, mutated._rank, mutated._spiked, mutated._current_modulation) == (
            10,
            0,
            False,
            1.0,
        )

        huge_int = AkidaNeuron(v=5)
        with pytest.raises(FloatingPointError):
            huge_int.step(10**400)
        assert (huge_int.v, huge_int._rank, huge_int._spiked, huge_int._current_modulation) == (
            5,
            0,
            False,
            1.0,
        )

    def test_exact_integer_and_finite_float_compatibility(self) -> None:
        """Zero weight is a no-op, NumPy integers stay ``int``, and finite floats work."""
        neuron = AkidaNeuron(v=_raw(np.int64(0)), _rank=_raw(np.int64(0)))
        assert neuron.step(0) == 0
        assert (neuron.v, neuron._rank, neuron._current_modulation) == (0, 0, 1.0)
        assert neuron.step(_raw(np.int64(60))) == 0
        assert neuron.step(_raw(60.0)) == 1
        assert type(neuron.v) is int
        assert neuron._rank == 2
        assert neuron._spiked is True
        neuron.reset()
        assert (neuron.v, neuron._rank, neuron._spiked, neuron._current_modulation) == (
            0,
            0,
            False,
            1.0,
        )
