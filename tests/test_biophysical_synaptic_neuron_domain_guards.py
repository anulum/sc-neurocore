# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Biophysical and synaptic neuron domain guard tests

"""Verify finiteness, domain, overflow-safety, and atomic rollback guards.

Covers ``DendrifyNeuron``, ``HuberBraunNeuron``, ``TsodyksMarkramNeuron``,
``PlantR15Neuron``, and ``DestexheThalamicNeuron``.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import fields
from typing import Any

import numpy as np
import pytest
from sc_neurocore.neurons.models.dendrify import DendrifyNeuron
from sc_neurocore.neurons.models.destexhe_thalamic import DestexheThalamicNeuron
from sc_neurocore.neurons.models.huber_braun import HuberBraunNeuron
from sc_neurocore.neurons.models.plant_r15 import PlantR15Neuron
from sc_neurocore.neurons.models.tsodyks_markram import TsodyksMarkramNeuron

NON_FINITE_FLOATS = [float("nan"), float("inf"), float("-inf")]
NON_REAL_VALUES: list[object] = [True, False, "1.0", None, 1 + 2j]


def _raw(value: object) -> Any:
    """Return an untyped reference so tests can pass invalid runtime types."""
    return value


def _snapshot(neuron: Any) -> dict[str, Any]:
    """Capture every dataclass field value of a neuron instance."""
    return {f.name: getattr(neuron, f.name) for f in fields(neuron)}


class TestDendrifyDomainGuards:
    """Finiteness, boolean-flag, domain, overflow, and coercion guards for ``DendrifyNeuron``."""

    @pytest.mark.parametrize("bad", NON_FINITE_FLOATS)
    @pytest.mark.parametrize(
        "field_name",
        [
            "v_s",
            "v_d",
            "tau_s",
            "tau_d",
            "g_c",
            "d_threshold",
            "d_amplitude",
            "d_duration",
            "d_timer",
            "v_rest",
            "v_threshold",
            "v_reset",
            "dt",
        ],
    )
    def test_constructor_rejects_non_finite_float_fields(self, field_name: str, bad: float) -> None:
        """Non-finite values in any float constructor parameter raise ``ValueError``."""
        with pytest.raises(ValueError, match=field_name):
            DendrifyNeuron(**{field_name: _raw(bad)})

    @pytest.mark.parametrize("bad", [True, "1.0"])
    @pytest.mark.parametrize(
        "field_name",
        [
            "v_s",
            "v_d",
            "tau_s",
            "tau_d",
            "g_c",
            "d_threshold",
            "d_amplitude",
            "d_duration",
            "d_timer",
            "v_rest",
            "v_threshold",
            "v_reset",
            "dt",
        ],
    )
    def test_constructor_rejects_bool_and_non_real_float_fields(
        self, field_name: str, bad: object
    ) -> None:
        """Booleans and non-real values in any float field raise ``TypeError``."""
        with pytest.raises(TypeError, match=field_name):
            DendrifyNeuron(**{field_name: _raw(bad)})

    @pytest.mark.parametrize("bad", [1, 0, "true", None, 1.0])
    def test_constructor_rejects_non_bool_d_active(self, bad: object) -> None:
        """Non-boolean ``d_active`` raises ``TypeError``."""
        with pytest.raises(TypeError, match="d_active"):
            DendrifyNeuron(d_active=_raw(bad))

    @pytest.mark.parametrize(
        ("field_name", "bad"),
        [
            ("tau_s", 0.0),
            ("tau_s", -1.0),
            ("tau_d", 0.0),
            ("tau_d", -1.0),
            ("dt", 0.0),
            ("dt", -0.1),
            ("g_c", -0.1),
            ("d_duration", -1.0),
        ],
    )
    def test_constructor_domain_constraints(self, field_name: str, bad: float) -> None:
        """Out-of-domain time constants, coupling, or duration raise ``ValueError``."""
        with pytest.raises(ValueError, match=field_name):
            DendrifyNeuron(**{field_name: _raw(bad)})

    @pytest.mark.parametrize("bad", NON_FINITE_FLOATS)
    def test_step_rejects_non_finite_current_without_mutating_state(self, bad: float) -> None:
        """Non-finite drive raises ``ValueError`` and leaves state unchanged."""
        n = DendrifyNeuron(v_s=-60.0, v_d=-55.0)
        before = _snapshot(n)
        with pytest.raises(ValueError, match="current"):
            n.step(bad)
        assert _snapshot(n) == before

    @pytest.mark.parametrize("bad", NON_REAL_VALUES)
    def test_step_rejects_non_real_current_without_mutating_state(self, bad: object) -> None:
        """Boolean or non-real drive raises ``TypeError`` and leaves state unchanged."""
        n = DendrifyNeuron(v_s=-60.0, v_d=-55.0)
        before = _snapshot(n)
        with pytest.raises(TypeError, match="current"):
            n.step(_raw(bad))
        assert _snapshot(n) == before

    @pytest.mark.parametrize(
        ("attr", "bad", "exc_type"),
        [
            ("v_s", float("nan"), ValueError),
            ("v_d", float("inf"), ValueError),
            ("tau_s", 0.0, ValueError),
            ("tau_d", -1.0, ValueError),
            ("g_c", -0.1, ValueError),
            ("d_duration", -1.0, ValueError),
            ("dt", 0.0, ValueError),
            ("d_active", 1, TypeError),
            ("v_s", True, TypeError),
        ],
    )
    def test_step_rejects_mutated_invalid_attributes(
        self, attr: str, bad: object, exc_type: type[Exception]
    ) -> None:
        """Mutated invalid attributes fail closed on ``step``."""
        n = DendrifyNeuron()
        setattr(n, attr, bad)
        with pytest.raises(exc_type, match=attr):
            n.step(10.0)

    def test_step_overflow_rolls_back_all_compartments_and_plateau_state(self) -> None:
        """Overflow in dendritic or somatic update leaves all four state fields intact."""
        n_d = DendrifyNeuron(v_d=1e308, v_s=-65.0, tau_d=0.1, dt=1.0)
        before_d = _snapshot(n_d)
        with pytest.raises(ValueError, match="finite"):
            n_d.step(1e308)
        assert _snapshot(n_d) == before_d

        n_s = DendrifyNeuron(v_s=-1e308, v_d=0.0, tau_s=0.1, dt=1.0, d_threshold=-10.0)
        before_s = _snapshot(n_s)
        with pytest.raises(ValueError, match="finite"):
            n_s.step(0.0)
        assert _snapshot(n_s) == before_s

    def test_plateau_spike_reset_and_scalar_coercion(self) -> None:
        """Plateau initiation/termination, somatic spiking, reset, and scalar coercion work."""
        n_float = DendrifyNeuron(d_duration=0.1, dt=0.1)
        n_mixed = DendrifyNeuron(
            v_s=-65,
            v_d=-65,
            d_active=_raw(np.bool_(False)),
            tau_s=10,
            tau_d=20,
            g_c=0.8,
            d_threshold=-35,
            d_amplitude=30,
            d_duration=0.1,
            d_timer=0,
            v_rest=-65,
            v_threshold=-50,
            v_reset=-65,
            dt=0.1,
        )
        assert n_mixed.step(10_000) == n_float.step(10_000.0)
        assert _snapshot(n_mixed) == _snapshot(n_float)
        assert n_float.d_active is False

        n_spike = DendrifyNeuron(v_s=-51.0, v_d=500.0, tau_s=1.0, dt=0.1)
        assert n_spike.step(0.0) == 1
        assert n_spike.v_s == n_spike.v_reset
        n_spike.reset()
        assert (n_spike.v_s, n_spike.v_d, n_spike.d_active, n_spike.d_timer) == (
            -65.0,
            -65.0,
            False,
            0.0,
        )
        assert set(n_float.__dict__) == {f.name for f in fields(DendrifyNeuron)}


class TestHuberBraunDomainGuards:
    """Finiteness, domain, overflow-safety, RNG preservation, and rollback for ``HuberBraunNeuron``."""

    @pytest.mark.parametrize("bad", NON_FINITE_FLOATS)
    @pytest.mark.parametrize(
        "field_name",
        [
            "v",
            "a_sd",
            "a_sr",
            "g_sd",
            "g_sr",
            "g_l",
            "e_sd",
            "e_sr",
            "e_l",
            "tau_sd",
            "tau_sr",
            "eta",
            "dt",
            "v_threshold",
        ],
    )
    def test_constructor_rejects_non_finite_fields(self, field_name: str, bad: float) -> None:
        """Non-finite values in any constructor parameter raise ``ValueError``."""
        with pytest.raises(ValueError, match=field_name):
            HuberBraunNeuron(**{field_name: bad})

    @pytest.mark.parametrize("bad", [True, "1.0"])
    @pytest.mark.parametrize("field_name", ["v", "a_sd", "g_sd", "tau_sd", "eta", "dt"])
    def test_constructor_rejects_bool_and_non_real_fields(
        self, field_name: str, bad: object
    ) -> None:
        """Booleans and non-real values in constructor fields raise ``TypeError``."""
        with pytest.raises(TypeError, match=field_name):
            HuberBraunNeuron(**{field_name: _raw(bad)})

    @pytest.mark.parametrize(
        ("field_name", "bad"),
        [
            ("tau_sd", 0.0),
            ("tau_sd", -1.0),
            ("tau_sr", 0.0),
            ("tau_sr", -1.0),
            ("dt", 0.0),
            ("dt", -0.1),
            ("g_sd", -0.1),
            ("g_sr", -0.1),
            ("g_l", -0.1),
            ("eta", -0.01),
        ],
    )
    def test_constructor_domain_constraints(self, field_name: str, bad: float) -> None:
        """Out-of-domain time constants, conductances, or noise raise ``ValueError``."""
        with pytest.raises(ValueError, match=field_name):
            HuberBraunNeuron(**{field_name: bad})

    @pytest.mark.parametrize("bad", NON_FINITE_FLOATS)
    def test_step_rejects_non_finite_current_without_mutating_state_or_rng(
        self, bad: float
    ) -> None:
        """Non-finite drive raises ``ValueError`` without mutating state or consuming RNG."""
        n = HuberBraunNeuron(v=-45.0, a_sd=0.2, a_sr=0.3)
        before = _snapshot(n)
        np.random.seed(12345)
        expected_next = float(np.random.randn())
        np.random.seed(12345)
        with pytest.raises(ValueError, match="current"):
            n.step(bad)
        assert _snapshot(n) == before
        assert float(np.random.randn()) == expected_next

    @pytest.mark.parametrize("bad", NON_REAL_VALUES)
    def test_step_rejects_non_real_current_without_mutating_state(self, bad: object) -> None:
        """Boolean or non-real drive raises ``TypeError`` and leaves state unchanged."""
        n = HuberBraunNeuron(v=-45.0, a_sd=0.2, a_sr=0.3)
        before = _snapshot(n)
        with pytest.raises(TypeError, match="current"):
            n.step(_raw(bad))
        assert _snapshot(n) == before

    def test_extreme_voltages_have_no_sigmoid_overflow_warning(self) -> None:
        """Extreme finite voltages evaluate complementary sigmoids without warnings."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for v_init in (-10_000.0, 10_000.0):
                n = HuberBraunNeuron(v=v_init, eta=0.0)
                spike = n.step(0.0)
                assert spike in (0, 1)
                assert math.isfinite(n.v) and math.isfinite(n.a_sd) and math.isfinite(n.a_sr)

    def test_step_overflow_and_mutated_invalid_fields_roll_back(self) -> None:
        """Candidate overflow and mutated invalid attributes leave state intact."""
        n_over = HuberBraunNeuron(v=1e308, a_sd=0.1, a_sr=0.2, eta=0.0, dt=10.0)
        before = _snapshot(n_over)
        with pytest.raises(ValueError, match="finite"):
            n_over.step(1e308)
        assert _snapshot(n_over) == before

        n_mut = HuberBraunNeuron()
        n_mut.tau_sd = 0.0
        with pytest.raises(ValueError, match="tau_sd"):
            n_mut.step(1.0)

    def test_scalar_coercion_spiking_and_reset(self) -> None:
        """Integer drives match float trajectories bit-for-bit, spike crosses threshold, reset works."""
        n_float = HuberBraunNeuron(eta=0.0)
        n_int = HuberBraunNeuron(v=-50, a_sd=0, a_sr=0, eta=0)
        assert n_int.step(500) == n_float.step(500.0) == 1
        assert _snapshot(n_int) == _snapshot(n_float)
        n_float.reset()
        assert (n_float.v, n_float.a_sd, n_float.a_sr) == (-50.0, 0.0, 0.0)
        assert set(n_float.__dict__) == {f.name for f in fields(HuberBraunNeuron)}


class TestTsodyksMarkramDomainGuards:
    """Finiteness, boolean-spike, domain, overflow, and coercion guards for ``TsodyksMarkramNeuron``."""

    @pytest.mark.parametrize("bad", NON_FINITE_FLOATS)
    @pytest.mark.parametrize(
        "field_name",
        [
            "v",
            "x",
            "u",
            "v_rest",
            "v_reset",
            "v_threshold",
            "tau_m",
            "tau_d",
            "tau_f",
            "u_se",
            "a_se",
            "r_m",
            "dt",
        ],
    )
    def test_constructor_rejects_non_finite_fields(self, field_name: str, bad: float) -> None:
        """Non-finite values in any constructor parameter raise ``ValueError``."""
        with pytest.raises(ValueError, match=field_name):
            TsodyksMarkramNeuron(**{field_name: bad})

    @pytest.mark.parametrize("bad", [True, "1.0"])
    @pytest.mark.parametrize("field_name", ["v", "x", "u", "tau_m", "u_se", "dt"])
    def test_constructor_rejects_bool_and_non_real_fields(
        self, field_name: str, bad: object
    ) -> None:
        """Booleans and non-real values in constructor fields raise ``TypeError``."""
        with pytest.raises(TypeError, match=field_name):
            TsodyksMarkramNeuron(**{field_name: _raw(bad)})

    @pytest.mark.parametrize(
        ("field_name", "bad"),
        [
            ("tau_m", 0.0),
            ("tau_m", -1.0),
            ("tau_d", 0.0),
            ("tau_d", -1.0),
            ("tau_f", 0.0),
            ("tau_f", -1.0),
            ("dt", 0.0),
            ("dt", -0.1),
            ("r_m", -0.1),
            ("u_se", -0.01),
            ("u_se", 1.01),
        ],
    )
    def test_constructor_domain_constraints(self, field_name: str, bad: float) -> None:
        """Out-of-domain time constants, resistance, or release probability raise ``ValueError``."""
        with pytest.raises(ValueError, match=field_name):
            TsodyksMarkramNeuron(**{field_name: bad})

    @pytest.mark.parametrize("bad", NON_FINITE_FLOATS)
    def test_step_rejects_non_finite_current_without_mutating_state(self, bad: float) -> None:
        """Non-finite drive raises ``ValueError`` and leaves state unchanged."""
        n = TsodyksMarkramNeuron(v=-60.0, x=0.8, u=0.3)
        before = _snapshot(n)
        with pytest.raises(ValueError, match="current"):
            n.step(bad, presynaptic_spike=True)
        assert _snapshot(n) == before

    @pytest.mark.parametrize("bad", NON_REAL_VALUES)
    def test_step_rejects_non_real_current_without_mutating_state(self, bad: object) -> None:
        """Boolean or non-real drive raises ``TypeError`` and leaves state unchanged."""
        n = TsodyksMarkramNeuron(v=-60.0, x=0.8, u=0.3)
        before = _snapshot(n)
        with pytest.raises(TypeError, match="current"):
            n.step(_raw(bad))
        assert _snapshot(n) == before

    @pytest.mark.parametrize("bad", [1, 0, 1.0, "true", None])
    def test_step_rejects_non_bool_presynaptic_spike(self, bad: object) -> None:
        """Non-boolean ``presynaptic_spike`` raises ``TypeError`` and preserves state."""
        n = TsodyksMarkramNeuron(v=-60.0, x=0.8, u=0.3)
        before = _snapshot(n)
        with pytest.raises(TypeError, match="presynaptic_spike"):
            n.step(1.0, presynaptic_spike=_raw(bad))
        assert _snapshot(n) == before

    def test_step_accepts_numpy_bool_and_rolls_back_overflow(self) -> None:
        """``np.bool_`` spikes work, overflow rolls back ``(v, x, u)``, and reset restores defaults."""
        n_py = TsodyksMarkramNeuron()
        n_np = TsodyksMarkramNeuron()
        assert n_np.step(10, presynaptic_spike=_raw(np.bool_(True))) == n_py.step(
            10.0, presynaptic_spike=True
        )
        assert _snapshot(n_np) == _snapshot(n_py)
        assert n_np.step(_raw(np.float64(5000.0)), presynaptic_spike=_raw(np.bool_(False))) == 1
        assert n_np.v == n_np.v_reset

        n_over = TsodyksMarkramNeuron(v=0.0, x=0.8, u=0.3, tau_m=0.1, dt=1.0)
        before = _snapshot(n_over)
        with pytest.raises(ValueError, match="finite"):
            n_over.step(1e308, presynaptic_spike=True)
        assert _snapshot(n_over) == before

        n_mut = TsodyksMarkramNeuron()
        n_mut.tau_d = 0.0
        with pytest.raises(ValueError, match="tau_d"):
            n_mut.step(0.0)

        n_np.reset()
        assert (n_np.v, n_np.x, n_np.u) == (n_np.v_rest, 1.0, n_np.u_se)
        assert set(n_py.__dict__) == {f.name for f in fields(TsodyksMarkramNeuron)}


class TestPlantR15DomainGuards:
    """Finiteness, domain, overflow-safety, and atomic rollback guards for ``PlantR15Neuron``."""

    @pytest.mark.parametrize("bad", NON_FINITE_FLOATS)
    @pytest.mark.parametrize(
        "field_name",
        [
            "v",
            "m",
            "h",
            "n",
            "ca",
            "g_na",
            "g_k",
            "g_ca",
            "g_l",
            "g_kca",
            "e_na",
            "e_k",
            "e_ca",
            "e_l",
            "c_m",
            "k_ca",
            "tau_ca",
            "dt",
            "v_threshold",
        ],
    )
    def test_constructor_rejects_non_finite_fields(self, field_name: str, bad: float) -> None:
        """Non-finite values in any constructor parameter raise ``ValueError``."""
        with pytest.raises(ValueError, match=field_name):
            PlantR15Neuron(**{field_name: bad})

    @pytest.mark.parametrize("bad", [True, "1.0"])
    @pytest.mark.parametrize("field_name", ["v", "ca", "g_na", "c_m", "tau_ca", "dt"])
    def test_constructor_rejects_bool_and_non_real_fields(
        self, field_name: str, bad: object
    ) -> None:
        """Booleans and non-real values in constructor fields raise ``TypeError``."""
        with pytest.raises(TypeError, match=field_name):
            PlantR15Neuron(**{field_name: _raw(bad)})

    @pytest.mark.parametrize(
        ("field_name", "bad"),
        [
            ("c_m", 0.0),
            ("c_m", -1.0),
            ("tau_ca", 0.0),
            ("tau_ca", -1.0),
            ("dt", 0.0),
            ("dt", -0.05),
            ("g_na", -0.1),
            ("g_k", -0.1),
            ("g_ca", -0.1),
            ("g_l", -0.1),
            ("g_kca", -0.1),
            ("k_ca", -0.01),
            ("ca", -0.1),
        ],
    )
    def test_constructor_domain_constraints(self, field_name: str, bad: float) -> None:
        """Out-of-domain capacitance, time constants, conductances, or calcium raise ``ValueError``."""
        with pytest.raises(ValueError, match=field_name):
            PlantR15Neuron(**{field_name: bad})

    @pytest.mark.parametrize("bad", NON_FINITE_FLOATS)
    def test_step_rejects_non_finite_current_without_mutating_state(self, bad: float) -> None:
        """Non-finite drive raises ``ValueError`` and leaves all five states intact."""
        n = PlantR15Neuron()
        n.step(1.0)
        before = _snapshot(n)
        with pytest.raises(ValueError, match="current"):
            n.step(bad)
        assert _snapshot(n) == before

    @pytest.mark.parametrize("bad", NON_REAL_VALUES)
    def test_step_rejects_non_real_current_without_mutating_state(self, bad: object) -> None:
        """Boolean or non-real drive raises ``TypeError`` and leaves all five states intact."""
        n = PlantR15Neuron()
        before = _snapshot(n)
        with pytest.raises(TypeError, match="current"):
            n.step(_raw(bad))
        assert _snapshot(n) == before

    def test_step_overflow_and_rate_exp_overflow_roll_back_all_substeps(self) -> None:
        """Overflow during any of the five substeps leaves ``(v, m, h, n, ca)`` unchanged."""
        n_exp = PlantR15Neuron(v=-20_000.0)
        before_exp = _snapshot(n_exp)
        with pytest.raises(ValueError, match="finite"):
            n_exp.step(0.0)
        assert _snapshot(n_exp) == before_exp

        n_pow = PlantR15Neuron(m=1e150)
        before_pow = _snapshot(n_pow)
        with pytest.raises(ValueError, match="overflowed"):
            n_pow.step(0.0)
        assert _snapshot(n_pow) == before_pow

        n_inf = PlantR15Neuron(v=-50.0, g_na=0.0, g_k=0.0, g_ca=0.0, g_l=0.0, g_kca=0.0, c_m=1e-300)
        before_inf = _snapshot(n_inf)
        with pytest.raises(ValueError, match="finite"):
            n_inf.step(1e100)
        assert _snapshot(n_inf) == before_inf

        n_ca_clamp = PlantR15Neuron(v=200.0, ca=0.0, g_na=0.0, g_k=0.0, g_l=0.0, g_kca=0.0)
        assert n_ca_clamp.step(0.0) == 0
        assert n_ca_clamp.ca == 0.0

        n_mut = PlantR15Neuron()
        n_mut.c_m = 0.0
        with pytest.raises(ValueError, match="c_m"):
            n_mut.step(0.0)

    def test_scalar_coercion_spiking_and_reset(self) -> None:
        """Integer drives match float stepping bit-for-bit and reset restores rest state."""
        n_float = PlantR15Neuron()
        n_int = PlantR15Neuron(v=-50, c_m=1, tau_ca=500)
        assert n_int.step(10) == n_float.step(10.0)
        assert _snapshot(n_int) == _snapshot(n_float)

        n_spike = PlantR15Neuron(v=-15.0)
        assert n_spike.step(100.0) == 1
        n_spike.reset()
        assert (n_spike.v, n_spike.m, n_spike.h, n_spike.n, n_spike.ca) == (
            -50.0,
            0.05,
            0.6,
            0.3,
            0.1,
        )
        assert set(n_float.__dict__) == {f.name for f in fields(PlantR15Neuron)}


class TestDestexheThalamicDomainGuards:
    """Finiteness, domain, overflow-safety, and atomic rollback guards for ``DestexheThalamicNeuron``."""

    @pytest.mark.parametrize("bad", NON_FINITE_FLOATS)
    @pytest.mark.parametrize(
        "field_name",
        [
            "v",
            "h_na",
            "n_k",
            "m_t",
            "h_t",
            "g_na",
            "g_k",
            "g_t",
            "g_l",
            "e_na",
            "e_k",
            "e_ca",
            "e_l",
            "dt",
            "v_threshold",
        ],
    )
    def test_constructor_rejects_non_finite_fields(self, field_name: str, bad: float) -> None:
        """Non-finite values in any constructor parameter raise ``ValueError``."""
        with pytest.raises(ValueError, match=field_name):
            DestexheThalamicNeuron(**{field_name: bad})

    @pytest.mark.parametrize("bad", [True, "1.0"])
    @pytest.mark.parametrize("field_name", ["v", "h_na", "g_na", "g_t", "dt"])
    def test_constructor_rejects_bool_and_non_real_fields(
        self, field_name: str, bad: object
    ) -> None:
        """Booleans and non-real values in constructor fields raise ``TypeError``."""
        with pytest.raises(TypeError, match=field_name):
            DestexheThalamicNeuron(**{field_name: _raw(bad)})

    @pytest.mark.parametrize(
        ("field_name", "bad"),
        [
            ("dt", 0.0),
            ("dt", -0.02),
            ("g_na", -0.1),
            ("g_k", -0.1),
            ("g_t", -0.1),
            ("g_l", -0.1),
        ],
    )
    def test_constructor_domain_constraints(self, field_name: str, bad: float) -> None:
        """Non-positive ``dt`` or negative conductances raise ``ValueError``."""
        with pytest.raises(ValueError, match=field_name):
            DestexheThalamicNeuron(**{field_name: bad})

    @pytest.mark.parametrize("bad", NON_FINITE_FLOATS)
    def test_step_rejects_non_finite_current_without_mutating_state(self, bad: float) -> None:
        """Non-finite drive raises ``ValueError`` and leaves all five states intact."""
        n = DestexheThalamicNeuron()
        n.step(5.0)
        before = _snapshot(n)
        with pytest.raises(ValueError, match="current"):
            n.step(bad)
        assert _snapshot(n) == before

    @pytest.mark.parametrize("bad", NON_REAL_VALUES)
    def test_step_rejects_non_real_current_without_mutating_state(self, bad: object) -> None:
        """Boolean or non-real drive raises ``TypeError`` and leaves all five states intact."""
        n = DestexheThalamicNeuron()
        before = _snapshot(n)
        with pytest.raises(TypeError, match="current"):
            n.step(_raw(bad))
        assert _snapshot(n) == before

    def test_extreme_voltages_sigmoids_tau_and_overflow_rollback(self) -> None:
        """Extreme finite voltages evaluate safely without warnings; overflow rolls back."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            n_hyper = DestexheThalamicNeuron(v=-30_000.0, g_na=0.0, g_k=0.0, g_t=0.0, g_l=0.0)
            assert n_hyper.step(0.0) == 0
            assert math.isfinite(n_hyper.v)

            n_depol = DestexheThalamicNeuron(v=10_000.0, g_na=0.0, g_k=0.0, g_t=0.0, g_l=0.0)
            assert n_depol.step(0.0) == 0
            assert math.isfinite(n_depol.v)

        n_over = DestexheThalamicNeuron(v=1e308, g_na=100.0)
        before = _snapshot(n_over)
        with pytest.raises(ValueError, match="finite"):
            n_over.step(1e308)
        assert _snapshot(n_over) == before

        n_mut = DestexheThalamicNeuron()
        n_mut.dt = 0.0
        with pytest.raises(ValueError, match="dt"):
            n_mut.step(0.0)

    def test_scalar_coercion_spiking_and_reset(self) -> None:
        """Integer drives match float stepping bit-for-bit and reset restores rest state."""
        n_float = DestexheThalamicNeuron()
        n_int = DestexheThalamicNeuron(v=-65, g_na=100, g_k=10, g_t=2)
        assert n_int.step(5) == n_float.step(5.0)
        assert _snapshot(n_int) == _snapshot(n_float)

        n_spike = DestexheThalamicNeuron(v=-25.0)
        assert n_spike.step(1000.0) == 1
        n_spike.reset()
        assert (n_spike.v, n_spike.h_na, n_spike.n_k, n_spike.m_t, n_spike.h_t) == (
            -65.0,
            0.6,
            0.3,
            0.0,
            1.0,
        )
        assert set(n_float.__dict__) == {f.name for f in fields(DestexheThalamicNeuron)}


class TestPlantAndDestexheBaseIdentityWitness:
    """200,000-step base bit-identity witness for Plant R15 and Destexhe."""

    def test_plant_r15_and_destexhe_200k_step_base_bit_identity(self) -> None:
        """200,000-step trajectories remain bit-identical to the unguarded base ``np.exp`` ODEs."""
        plant = PlantR15Neuron()
        plant_spikes = sum(plant.step(0.0) for _ in range(200_000))
        assert plant_spikes == 1
        assert plant.v == -23.811914558998463
        assert plant.m == 0.9238532128110429
        assert plant.h == 0.04449483941587448
        assert plant.n == 0.8137215185231551
        assert plant.ca == 0.8705912612921731

        dest = DestexheThalamicNeuron()
        dest_spikes = sum(dest.step(5.0) for _ in range(200_000))
        assert dest_spikes == 1
        assert dest.v == -23.008485128272973
        assert dest.h_na == 0.011010016895934286
        assert dest.n_k == 0.5413949264908458
        assert dest.m_t == 0.994672141630726
        assert dest.h_t == 5.054184063168711e-07
