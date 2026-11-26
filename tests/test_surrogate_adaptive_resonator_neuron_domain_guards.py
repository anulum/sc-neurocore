# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Surrogate, adaptive, and resonator neuron domain guard tests

"""Verify finiteness, domain, overflow-safety, and atomic rollback guards.

Covers ``KLIFNeuron``, ``SuperSpikeNeuron``, ``LiquidTimeConstantNeuron``,
``sustain_oscillation_boundary``, and ``BalancedResonateAndFireNeuron``.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import fields
from typing import Any

import numpy as np
import pytest
from sc_neurocore.neurons.models.balanced_resonate_and_fire import (
    BalancedResonateAndFireNeuron,
    sustain_oscillation_boundary,
)
from sc_neurocore.neurons.models.klif import KLIFNeuron
from sc_neurocore.neurons.models.ltc import LiquidTimeConstantNeuron
from sc_neurocore.neurons.models.superspike_neuron import SuperSpikeNeuron

NON_FINITE_FLOATS = [float("nan"), float("inf"), float("-inf")]
NON_REAL_VALUES: list[object] = [True, False, "1.0", None, 1 + 2j]


def _raw(value: object) -> Any:
    """Return an untyped reference so tests can pass invalid runtime types."""
    return value


def _snapshot(neuron: Any) -> dict[str, Any]:
    """Capture every dataclass field value of a neuron instance."""
    return {f.name: getattr(neuron, f.name) for f in fields(neuron)}


class TestKLIFDomainGuards:
    """Finiteness, domain, overflow, and coercion guards for ``KLIFNeuron``."""

    @pytest.mark.parametrize("bad", NON_FINITE_FLOATS)
    @pytest.mark.parametrize("field_name", ["v", "k", "tau", "v_threshold", "v_reset", "dt"])
    def test_constructor_rejects_non_finite_fields(self, field_name: str, bad: float) -> None:
        """Non-finite values in any constructor parameter raise ``ValueError``."""
        with pytest.raises(ValueError, match=field_name):
            KLIFNeuron(**{field_name: bad})

    @pytest.mark.parametrize("bad", [True, "1.0"])
    @pytest.mark.parametrize("field_name", ["v", "k", "tau", "v_threshold", "v_reset", "dt"])
    def test_constructor_rejects_bool_and_non_real_fields(
        self, field_name: str, bad: object
    ) -> None:
        """Booleans and non-real values in any constructor field raise ``TypeError``."""
        with pytest.raises(TypeError, match=field_name):
            KLIFNeuron(**{field_name: _raw(bad)})

    @pytest.mark.parametrize("field_name", ["tau", "dt"])
    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_constructor_requires_positive_tau_and_dt(self, field_name: str, bad: float) -> None:
        """Non-positive ``tau`` or ``dt`` raises ``ValueError``."""
        with pytest.raises(ValueError, match=field_name):
            KLIFNeuron(**{field_name: bad})

    def test_constructor_allows_zero_and_negative_k(self) -> None:
        """Trainable gain ``k`` allows zero and negative finite values."""
        n_zero = KLIFNeuron(k=0.0)
        assert n_zero.step(5.0) == 0
        assert n_zero.v == 0.0
        n_neg = KLIFNeuron(k=-2.0)
        assert n_neg.step(1.0) == 0
        assert n_neg.v == pytest.approx(-2.0)

    @pytest.mark.parametrize("bad", NON_FINITE_FLOATS)
    def test_step_rejects_non_finite_current_without_mutating_state(self, bad: float) -> None:
        """Non-finite drive raises ``ValueError`` and leaves state unchanged."""
        n = KLIFNeuron(v=0.4)
        before = _snapshot(n)
        with pytest.raises(ValueError, match="current"):
            n.step(bad)
        assert _snapshot(n) == before

    @pytest.mark.parametrize("bad", NON_REAL_VALUES)
    def test_step_rejects_non_real_current_without_mutating_state(self, bad: object) -> None:
        """Boolean or non-real drive raises ``TypeError`` and leaves state unchanged."""
        n = KLIFNeuron(v=0.4)
        before = _snapshot(n)
        with pytest.raises(TypeError, match="current"):
            n.step(_raw(bad))
        assert _snapshot(n) == before

    @pytest.mark.parametrize(
        ("attr", "bad", "exc_type"),
        [
            ("v", float("nan"), ValueError),
            ("alpha", float("inf"), ValueError),
            ("k", float("-inf"), ValueError),
            ("v_threshold", float("nan"), ValueError),
            ("v_reset", float("nan"), ValueError),
            ("tau", 0.0, ValueError),
            ("dt", -1.0, ValueError),
            ("v", True, TypeError),
            ("alpha", "bad", TypeError),
            ("tau", False, TypeError),
        ],
    )
    def test_step_rejects_mutated_invalid_attributes(
        self, attr: str, bad: object, exc_type: type[Exception]
    ) -> None:
        """Mutated non-finite, out-of-domain, or non-real attributes fail closed."""
        n = KLIFNeuron()
        setattr(n, attr, bad)
        with pytest.raises(exc_type, match=attr):
            n.step(0.5)

    def test_step_overflow_rolls_back_state(self) -> None:
        """Membrane overflow raises ``ValueError`` and preserves prior ``v``."""
        n = KLIFNeuron(v=1e308, k=1.0)
        with pytest.raises(ValueError, match="finite"):
            n.step(1e308)
        assert n.v == 1e308

    def test_integer_and_numpy_scalar_coercion_matches_float(self) -> None:
        """Integer and NumPy scalar inputs and parameters match float trajectories."""
        n_float = KLIFNeuron(v=0.0, k=1.0, tau=10.0, v_threshold=1.0, v_reset=0.0, dt=1.0)
        n_mixed = KLIFNeuron(
            v=0,
            k=1,
            tau=10,
            v_threshold=1,
            v_reset=0,
            dt=1,
        )
        assert n_mixed.step(1) == n_float.step(1.0)
        assert n_mixed.v == n_float.v
        assert n_mixed.step(_raw(np.float64(0.5))) == n_float.step(0.5)
        assert n_mixed.v == n_float.v
        assert n_mixed.step(_raw(np.int64(2))) == n_float.step(2.0)
        assert n_mixed.v == n_float.v
        assert set(n_float.__dict__) == {f.name for f in fields(KLIFNeuron)}
        n_float.reset()
        assert n_float.v == 0.0


class TestSuperSpikeDomainGuards:
    """Finiteness, domain, overflow, and coercion guards for ``SuperSpikeNeuron``."""

    @pytest.mark.parametrize("bad", NON_FINITE_FLOATS)
    @pytest.mark.parametrize(
        "field_name",
        ["v", "trace", "tau_m", "tau_e", "v_threshold", "v_reset", "beta_sg", "dt"],
    )
    def test_constructor_rejects_non_finite_fields(self, field_name: str, bad: float) -> None:
        """Non-finite values in any constructor parameter raise ``ValueError``."""
        with pytest.raises(ValueError, match=field_name):
            SuperSpikeNeuron(**{field_name: bad})

    @pytest.mark.parametrize("bad", [True, "1.0"])
    @pytest.mark.parametrize(
        "field_name",
        ["v", "trace", "tau_m", "tau_e", "v_threshold", "v_reset", "beta_sg", "dt"],
    )
    def test_constructor_rejects_bool_and_non_real_fields(
        self, field_name: str, bad: object
    ) -> None:
        """Booleans and non-real values in any constructor field raise ``TypeError``."""
        with pytest.raises(TypeError, match=field_name):
            SuperSpikeNeuron(**{field_name: _raw(bad)})

    @pytest.mark.parametrize(
        ("field_name", "bad"),
        [
            ("tau_m", 0.0),
            ("tau_m", -1.0),
            ("tau_e", 0.0),
            ("tau_e", -1.0),
            ("dt", 0.0),
            ("dt", -0.5),
            ("beta_sg", -0.1),
        ],
    )
    def test_constructor_domain_constraints(self, field_name: str, bad: float) -> None:
        """Out-of-domain time constants or negative sharpness raise ``ValueError``."""
        with pytest.raises(ValueError, match=field_name):
            SuperSpikeNeuron(**{field_name: bad})

    def test_surrogate_grad_overflow_safe_and_zero_beta_valid(self) -> None:
        """Extreme membrane distance underflows smoothly without ``OverflowError``."""
        n_huge = SuperSpikeNeuron(v=1e200, beta_sg=10.0)
        assert n_huge.surrogate_grad() == 0.0

        n_subnormal = SuperSpikeNeuron(v=1e155, v_threshold=0.0, beta_sg=1.0)
        sg_sub = n_subnormal.surrogate_grad()
        assert math.isfinite(sg_sub) and sg_sub >= 0.0

        n_inf_dist = SuperSpikeNeuron(v=1e308, v_threshold=-1e308, beta_sg=10.0)
        assert n_inf_dist.surrogate_grad() == 0.0

        n_flat = SuperSpikeNeuron(v=1e308, v_threshold=-1e308, beta_sg=0.0)
        assert n_flat.surrogate_grad() == 1.0

    @pytest.mark.parametrize(
        ("attr", "bad", "exc_type"),
        [
            ("v", float("nan"), ValueError),
            ("v_threshold", float("inf"), ValueError),
            ("beta_sg", -1.0, ValueError),
            ("beta_sg", True, TypeError),
        ],
    )
    def test_surrogate_grad_rejects_mutated_invalid_fields(
        self, attr: str, bad: object, exc_type: type[Exception]
    ) -> None:
        """``surrogate_grad`` rejects mutated invalid state or parameters."""
        n = SuperSpikeNeuron()
        setattr(n, attr, bad)
        with pytest.raises(exc_type, match=attr):
            n.surrogate_grad()

    @pytest.mark.parametrize("bad", NON_FINITE_FLOATS)
    def test_step_rejects_non_finite_current_without_mutating_state(self, bad: float) -> None:
        """Non-finite drive raises ``ValueError`` and leaves state unchanged."""
        n = SuperSpikeNeuron(v=0.3, trace=0.2)
        before = _snapshot(n)
        with pytest.raises(ValueError, match="current"):
            n.step(bad)
        assert _snapshot(n) == before

    @pytest.mark.parametrize("bad", NON_REAL_VALUES)
    def test_step_rejects_non_real_current_without_mutating_state(self, bad: object) -> None:
        """Boolean or non-real drive raises ``TypeError`` and leaves state unchanged."""
        n = SuperSpikeNeuron(v=0.3, trace=0.2)
        before = _snapshot(n)
        with pytest.raises(TypeError, match="current"):
            n.step(_raw(bad))
        assert _snapshot(n) == before

    @pytest.mark.parametrize(
        ("attr", "bad", "exc_type"),
        [
            ("v", float("nan"), ValueError),
            ("trace", float("inf"), ValueError),
            ("alpha_m", float("nan"), ValueError),
            ("alpha_e", float("-inf"), ValueError),
            ("v_threshold", float("nan"), ValueError),
            ("v_reset", float("nan"), ValueError),
            ("tau_m", 0.0, ValueError),
            ("tau_e", -1.0, ValueError),
            ("dt", 0.0, ValueError),
            ("beta_sg", -0.5, ValueError),
            ("trace", True, TypeError),
            ("alpha_m", "bad", TypeError),
        ],
    )
    def test_step_rejects_mutated_invalid_attributes(
        self, attr: str, bad: object, exc_type: type[Exception]
    ) -> None:
        """Mutated invalid attributes fail closed on ``step``."""
        n = SuperSpikeNeuron()
        setattr(n, attr, bad)
        with pytest.raises(exc_type, match=attr):
            n.step(0.2)

    def test_step_overflow_rolls_back_both_v_and_trace(self) -> None:
        """Overflow in either ``v`` or ``trace`` preserves both prior state variables."""
        n_v = SuperSpikeNeuron(v=1e308, trace=0.5)
        with pytest.raises(ValueError, match="finite"):
            n_v.step(1e308)
        assert n_v.v == 1e308
        assert n_v.trace == 0.5

        n_t = SuperSpikeNeuron(v=0.2, trace=1e308)
        n_t.alpha_e = 2.0
        with pytest.raises(ValueError, match="finite"):
            n_t.step(0.5)
        assert n_t.v == 0.2
        assert n_t.trace == 1e308

    def test_integer_and_numpy_scalar_coercion_matches_float(self) -> None:
        """Integer and NumPy scalar drives match float stepping bit-for-bit."""
        n_float = SuperSpikeNeuron()
        n_mixed = SuperSpikeNeuron(
            v=0,
            trace=0,
            tau_m=10,
            tau_e=10,
            v_threshold=1,
            v_reset=0,
            beta_sg=10,
            dt=1,
        )
        assert n_mixed.step(1) == n_float.step(1.0)
        assert n_mixed.v == n_float.v
        assert n_mixed.trace == n_float.trace
        assert n_mixed.step(_raw(np.float64(0.25))) == n_float.step(0.25)
        assert n_mixed.v == n_float.v
        assert n_mixed.trace == n_float.trace
        assert set(n_float.__dict__) == {f.name for f in fields(SuperSpikeNeuron)}
        n_float.reset()
        assert n_float.v == 0.0
        assert n_float.trace == 0.0


class TestLiquidTimeConstantDomainGuards:
    """Finiteness, domain, overflow, and coercion guards for ``LiquidTimeConstantNeuron``."""

    @pytest.mark.parametrize("bad", NON_FINITE_FLOATS)
    @pytest.mark.parametrize(
        "field_name",
        ["x", "tau_base", "w_tau", "w_x", "w_in", "bias", "v_threshold", "dt"],
    )
    def test_constructor_rejects_non_finite_fields(self, field_name: str, bad: float) -> None:
        """Non-finite values in any constructor parameter raise ``ValueError``."""
        with pytest.raises(ValueError, match=field_name):
            LiquidTimeConstantNeuron(**{field_name: bad})

    @pytest.mark.parametrize("bad", [True, "1.0"])
    @pytest.mark.parametrize(
        "field_name",
        ["x", "tau_base", "w_tau", "w_x", "w_in", "bias", "v_threshold", "dt"],
    )
    def test_constructor_rejects_bool_and_non_real_fields(
        self, field_name: str, bad: object
    ) -> None:
        """Booleans and non-real values in any constructor field raise ``TypeError``."""
        with pytest.raises(TypeError, match=field_name):
            LiquidTimeConstantNeuron(**{field_name: _raw(bad)})

    @pytest.mark.parametrize("field_name", ["tau_base", "dt"])
    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_constructor_requires_positive_tau_base_and_dt(
        self, field_name: str, bad: float
    ) -> None:
        """Non-positive ``tau_base`` or ``dt`` raises ``ValueError``."""
        with pytest.raises(ValueError, match=field_name):
            LiquidTimeConstantNeuron(**{field_name: bad})

    @pytest.mark.parametrize("bad", NON_FINITE_FLOATS)
    def test_step_rejects_non_finite_current_without_mutating_state(self, bad: float) -> None:
        """Non-finite drive raises ``ValueError`` and leaves state unchanged."""
        n = LiquidTimeConstantNeuron(x=0.25)
        before = _snapshot(n)
        with pytest.raises(ValueError, match="current"):
            n.step(bad)
        assert _snapshot(n) == before

    @pytest.mark.parametrize("bad", NON_REAL_VALUES)
    def test_step_rejects_non_real_current_without_mutating_state(self, bad: object) -> None:
        """Boolean or non-real drive raises ``TypeError`` and leaves state unchanged."""
        n = LiquidTimeConstantNeuron(x=0.25)
        before = _snapshot(n)
        with pytest.raises(TypeError, match="current"):
            n.step(_raw(bad))
        assert _snapshot(n) == before

    def test_extreme_finite_current_has_no_exp_overflow_warning(self) -> None:
        """Large finite currents saturate the sigmoid cleanly with zero warnings."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for drive in (2000.0, -2000.0, 1e308, -1e308):
                n = LiquidTimeConstantNeuron(v_threshold=100.0)
                spike = n.step(drive)
                assert spike in (0, 1)
                assert math.isfinite(n.x)

    def test_step_overflow_and_nan_cancellation_roll_back_state(self) -> None:
        """Numeric overflow or inf-inf cancellation in ``step`` leaves ``x`` unchanged."""
        n_over = LiquidTimeConstantNeuron(x=1e308, tau_base=0.1, dt=1.0)
        with pytest.raises(ValueError, match="finite"):
            n_over.step(0.0)
        assert n_over.x == 1e308

        n_cancel = LiquidTimeConstantNeuron(x=1e308, w_x=10.0, w_in=10.0)
        with pytest.raises(ValueError, match="finite"):
            n_cancel.step(-1e308)
        assert n_cancel.x == 1e308

    @pytest.mark.parametrize(
        ("attr", "bad", "exc_type"),
        [
            ("x", float("nan"), ValueError),
            ("tau_base", 0.0, ValueError),
            ("w_tau", float("inf"), ValueError),
            ("w_x", float("nan"), ValueError),
            ("w_in", float("-inf"), ValueError),
            ("bias", float("nan"), ValueError),
            ("v_threshold", float("nan"), ValueError),
            ("dt", -1.0, ValueError),
            ("x", True, TypeError),
            ("tau_base", "bad", TypeError),
        ],
    )
    def test_step_rejects_mutated_invalid_attributes(
        self, attr: str, bad: object, exc_type: type[Exception]
    ) -> None:
        """Mutated invalid attributes fail closed on ``step``."""
        n = LiquidTimeConstantNeuron()
        setattr(n, attr, bad)
        with pytest.raises(exc_type, match=attr):
            n.step(1.0)

    def test_integer_and_numpy_scalar_coercion_matches_float(self) -> None:
        """Integer and NumPy scalar drives match float stepping bit-for-bit."""
        n_float = LiquidTimeConstantNeuron()
        n_mixed = LiquidTimeConstantNeuron(
            x=0,
            tau_base=10,
            w_tau=-0.5,
            w_x=0.8,
            w_in=1,
            bias=0,
            v_threshold=1,
            dt=1,
        )
        assert n_mixed.step(5) == n_float.step(5.0)
        assert n_mixed.x == n_float.x
        assert n_mixed.step(_raw(np.float64(1.5))) == n_float.step(1.5)
        assert n_mixed.x == n_float.x
        assert set(n_float.__dict__) == {f.name for f in fields(LiquidTimeConstantNeuron)}
        n_float.reset()
        assert n_float.x == 0.0


class TestBalancedResonateAndFireDomainGuards:
    """Finiteness, domain, overflow, and parity guards for BRF."""

    @pytest.mark.parametrize("bad", NON_FINITE_FLOATS)
    def test_sustain_oscillation_boundary_rejects_non_finite_arguments(self, bad: float) -> None:
        """Non-finite ``omega`` or ``dt`` raises ``ValueError`` instead of returning a number."""
        with pytest.raises(ValueError, match="omega"):
            sustain_oscillation_boundary(bad, 0.01)
        with pytest.raises(ValueError, match="dt"):
            sustain_oscillation_boundary(10.0, bad)

    @pytest.mark.parametrize("bad", [True, False, "0.01", None])
    def test_sustain_oscillation_boundary_rejects_bool_and_non_real(self, bad: object) -> None:
        """Booleans and non-real arguments raise ``TypeError``."""
        with pytest.raises(TypeError, match="omega"):
            sustain_oscillation_boundary(_raw(bad), 0.01)
        with pytest.raises(TypeError, match="dt"):
            sustain_oscillation_boundary(10.0, _raw(bad))

    def test_sustain_oscillation_boundary_accepts_int_and_numpy_scalars(self) -> None:
        """Integer and NumPy scalar arguments match float calculation bit-for-bit."""
        ref = sustain_oscillation_boundary(10.0, 0.01)
        assert sustain_oscillation_boundary(10, _raw(np.float64(0.01))) == ref

    @pytest.mark.parametrize("bad", NON_FINITE_FLOATS)
    @pytest.mark.parametrize(
        "field_name",
        ["x", "y", "q", "omega", "b_offset", "threshold", "gamma", "dt"],
    )
    def test_constructor_rejects_non_finite_fields(self, field_name: str, bad: float) -> None:
        """Non-finite values in any constructor parameter raise ``ValueError``."""
        with pytest.raises(ValueError, match="finite"):
            BalancedResonateAndFireNeuron(**{field_name: bad})

    @pytest.mark.parametrize("bad", [True, "1.0"])
    @pytest.mark.parametrize(
        "field_name",
        ["x", "y", "q", "omega", "b_offset", "threshold", "gamma", "dt"],
    )
    def test_constructor_rejects_bool_and_non_real_fields(
        self, field_name: str, bad: object
    ) -> None:
        """Booleans and non-real values in any constructor field raise ``TypeError``."""
        with pytest.raises(TypeError, match=field_name):
            BalancedResonateAndFireNeuron(**{field_name: _raw(bad)})

    @pytest.mark.parametrize("bad", NON_FINITE_FLOATS)
    def test_step_rejects_non_finite_current_without_mutating_state(self, bad: float) -> None:
        """Non-finite drive raises ``ValueError`` and leaves state unchanged."""
        n = BalancedResonateAndFireNeuron(x=0.2, y=-0.1, q=0.3)
        before = _snapshot(n)
        with pytest.raises(ValueError, match="current"):
            n.step(bad)
        assert _snapshot(n) == before

    @pytest.mark.parametrize("bad", NON_REAL_VALUES)
    def test_step_rejects_non_real_current_without_mutating_state(self, bad: object) -> None:
        """Boolean or non-real drive raises ``TypeError`` and leaves state unchanged."""
        n = BalancedResonateAndFireNeuron(x=0.2, y=-0.1, q=0.3)
        before = _snapshot(n)
        with pytest.raises(TypeError, match="current"):
            n.step(_raw(bad))
        assert _snapshot(n) == before

    def test_step_overflow_rolls_back_all_state_variables(self) -> None:
        """Overflow during ``step`` leaves ``x``, ``y``, and ``q`` completely unchanged."""
        n_x = BalancedResonateAndFireNeuron(x=1e308, y=0.0, q=0.5, omega=10.0, dt=0.01)
        before_x = _snapshot(n_x)
        with pytest.raises(ValueError, match="finite"):
            n_x.step(1e308)
        assert _snapshot(n_x) == before_x

        n_y = BalancedResonateAndFireNeuron(x=1e308, y=1e308, q=0.5, omega=10.0, dt=0.01)
        before_y = _snapshot(n_y)
        with pytest.raises(ValueError, match="finite"):
            n_y.step(0.0)
        assert _snapshot(n_y) == before_y

        n_damp = BalancedResonateAndFireNeuron(x=0.1, y=0.1, q=1e308, b_offset=1e308)
        before_damp = _snapshot(n_damp)
        with pytest.raises(ValueError, match="finite"):
            n_damp.step(0.0)
        assert _snapshot(n_damp) == before_damp

        n_thresh = BalancedResonateAndFireNeuron(x=0.1, y=0.1, q=1e308, threshold=1e308)
        before_thresh = _snapshot(n_thresh)
        with pytest.raises(ValueError, match="finite"):
            n_thresh.step(0.0)
        assert _snapshot(n_thresh) == before_thresh

    def test_properties_and_state_reject_mutated_invalid_fields(self) -> None:
        """Properties ``damping``, ``dynamic_threshold``, and ``state`` fail closed."""
        n = BalancedResonateAndFireNeuron()
        n.q = float("nan")
        with pytest.raises(ValueError, match="q"):
            _ = n.damping
        with pytest.raises(ValueError, match="q"):
            _ = n.dynamic_threshold
        with pytest.raises(ValueError, match="q"):
            n.state()

        n2 = BalancedResonateAndFireNeuron(q=1e308, b_offset=1e308)
        with pytest.raises(ValueError, match="damping"):
            _ = n2.damping

        n3 = BalancedResonateAndFireNeuron(q=1e308, threshold=1e308)
        with pytest.raises(ValueError, match="dynamic_threshold"):
            _ = n3.dynamic_threshold

    def test_domain_boundaries_and_valid_state_snapshot(self) -> None:
        """Boundary rejections and normal ``state()`` snapshots behave consistently."""
        with pytest.raises(ValueError, match="dt"):
            sustain_oscillation_boundary(10.0, 0.0)
        with pytest.raises(ValueError, match="omega"):
            sustain_oscillation_boundary(0.0, 0.01)
        with pytest.raises(ValueError, match=r"dt \* omega"):
            sustain_oscillation_boundary(200.0, 0.01)
        with pytest.raises(ValueError, match=r"dt \* omega"):
            BalancedResonateAndFireNeuron(omega=200.0, dt=0.01)
        with pytest.raises(ValueError, match="gamma"):
            BalancedResonateAndFireNeuron(gamma=1.0)
        with pytest.raises(ValueError, match="gamma"):
            BalancedResonateAndFireNeuron(gamma=-0.1)

        valid = BalancedResonateAndFireNeuron()
        snap = valid.state()
        assert snap["damping"] == valid.damping
        assert snap["dynamic_threshold"] == valid.dynamic_threshold
        assert snap["p_omega"] == valid.p_omega
        assert valid.step(200.0) == 1
        assert valid.q == pytest.approx(1.0)

    def test_integer_coercion_and_200k_rust_reference_parity(self) -> None:
        """Integer drives match float bit-for-bit and 200k steps match the Rust reference."""
        n_float = BalancedResonateAndFireNeuron()
        n_int = BalancedResonateAndFireNeuron(
            x=0, y=0, q=0, omega=10, b_offset=1, threshold=1, gamma=0.9, dt=0.01
        )
        assert n_int.step(2) == n_float.step(2.0)
        assert n_int.x == n_float.x
        assert n_int.y == n_float.y
        assert n_int.q == n_float.q
        assert set(n_float.__dict__) == {f.name for f in fields(BalancedResonateAndFireNeuron)}

        n_ref = BalancedResonateAndFireNeuron(omega=10.0, b_offset=1.0, dt=0.01)
        spikes = 0
        for _ in range(200_000):
            spikes += n_ref.step(2.0)
        assert spikes == 0
        assert n_ref.x == 0.029363343430484898
        assert n_ref.y == 0.1955918095998993
        assert n_ref.q == 0.0
        n_ref.reset()
        assert (n_ref.x, n_ref.y, n_ref.q) == (0.0, 0.0, 0.0)
