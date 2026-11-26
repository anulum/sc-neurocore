# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Fail-loud domain guard tests for biophysical and population models

"""Fail-loud domain and type guard tests for five neuron models.

Covers ``GIFPopulationNeuron``, ``AvRonCardiacNeuron``, ``GolgiCell``,
``MotorUnit``, and ``RenshawCell`` across constructor validation, runtime
input rejection, mutated-attribute refusal, candidate-overflow rollback,
and scalar numeric coercion.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

import numpy as np
import pytest
from sc_neurocore.neurons.models.av_ron_cardiac import AvRonCardiacNeuron
from sc_neurocore.neurons.models.gif_population import GIFPopulationNeuron
from sc_neurocore.neurons.models.golgi_cell import GolgiCell
from sc_neurocore.neurons.models.motor_unit import MotorUnit
from sc_neurocore.neurons.models.renshaw_cell import RenshawCell

NON_FINITE_FLOATS: tuple[float, ...] = (math.nan, math.inf, -math.inf)
INVALID_REAL_TYPES: tuple[object, ...] = (True, False, np.bool_(True), "1.0", None, 1 + 2j)
INVALID_INT_TYPES: tuple[object, ...] = (True, False, np.bool_(True), 1.5, "1", None, 1 + 2j)


def _raw(value: object) -> Any:
    """Return an untyped value so runtime refusal tests satisfy strict static typing."""
    return value


def _gif_snapshot(neuron: GIFPopulationNeuron) -> tuple[float, float, int]:
    """Capture dynamic membrane, adaptation, and PRNG state for a GIF neuron."""
    return (neuron.v, neuron.eta, neuron._rng.state)


# ---------------------------------------------------------------------------
# GIFPopulationNeuron
# ---------------------------------------------------------------------------

GIF_FLOAT_FIELDS: tuple[str, ...] = (
    "v",
    "theta",
    "eta",
    "tau_m",
    "tau_eta",
    "delta_v",
    "lambda_0",
    "eta_increment",
    "v_rest",
    "v_reset",
    "dt",
)
GIF_POSITIVE_FIELDS: tuple[str, ...] = ("tau_m", "tau_eta", "delta_v", "dt")


@pytest.mark.parametrize("field_name", GIF_FLOAT_FIELDS)
@pytest.mark.parametrize("bad", NON_FINITE_FLOATS)
def test_gif_constructor_rejects_non_finite_floats(field_name: str, bad: float) -> None:
    """Non-finite float arguments fail at GIFPopulationNeuron construction."""
    with pytest.raises(ValueError, match=field_name):
        GIFPopulationNeuron(**{field_name: _raw(bad)})


@pytest.mark.parametrize("field_name", GIF_FLOAT_FIELDS)
@pytest.mark.parametrize("bad", INVALID_REAL_TYPES)
def test_gif_constructor_rejects_invalid_float_types(field_name: str, bad: object) -> None:
    """Boolean and non-real float fields fail with TypeError at construction."""
    with pytest.raises(TypeError, match=field_name):
        GIFPopulationNeuron(**{field_name: _raw(bad)})


@pytest.mark.parametrize("field_name", GIF_POSITIVE_FIELDS)
@pytest.mark.parametrize("bad", (0.0, -1.0))
def test_gif_constructor_rejects_non_positive_fields(field_name: str, bad: float) -> None:
    """Strictly positive GIFPopulationNeuron time and scale constants reject <= 0."""
    with pytest.raises(ValueError, match=field_name):
        GIFPopulationNeuron(**{field_name: _raw(bad)})


def test_gif_constructor_rejects_negative_lambda_0() -> None:
    """Negative baseline escape rate fails at construction."""
    with pytest.raises(ValueError, match="lambda_0"):
        GIFPopulationNeuron(lambda_0=-0.01)


@pytest.mark.parametrize("bad", INVALID_INT_TYPES)
def test_gif_constructor_rejects_invalid_seed_types(bad: object) -> None:
    """Boolean and non-integer seeds fail with TypeError at construction."""
    with pytest.raises(TypeError, match="seed"):
        GIFPopulationNeuron(seed=_raw(bad))


def test_gif_constructor_accepts_negative_and_numpy_integer_seeds() -> None:
    """Negative integer seeds stay permitted for contract-level seed-domain checks."""
    neg = GIFPopulationNeuron(seed=-1)
    assert neg.seed == -1
    assert neg.step(1.0) in (0, 1)

    np_seeded = GIFPopulationNeuron(seed=_raw(np.int64(42)))
    assert type(np_seeded.seed) is int
    assert np_seeded.seed == 42


@pytest.mark.parametrize("bad", INVALID_REAL_TYPES)
def test_gif_step_rejects_invalid_current_type_and_preserves_state(bad: object) -> None:
    """Boolean and non-real currents raise TypeError without mutating state or PRNG."""
    neuron = GIFPopulationNeuron(v=-62.0, eta=0.5, seed=7)
    before = _gif_snapshot(neuron)

    with pytest.raises(TypeError, match="current"):
        neuron.step(_raw(bad))

    assert _gif_snapshot(neuron) == before


@pytest.mark.parametrize("bad", NON_FINITE_FLOATS)
def test_gif_step_rejects_non_finite_current_and_preserves_state(bad: float) -> None:
    """Non-finite currents raise ValueError without mutating state or PRNG."""
    neuron = GIFPopulationNeuron(v=-62.0, eta=0.5, seed=7)
    before = _gif_snapshot(neuron)

    with pytest.raises(ValueError, match="current"):
        neuron.step(bad)

    assert _gif_snapshot(neuron) == before


@pytest.mark.parametrize(
    ("field_name", "bad", "exc_type"),
    [
        ("v", math.nan, ValueError),
        ("v", True, TypeError),
        ("tau_eta", 0.0, ValueError),
        ("delta_v", -1.0, ValueError),
        ("lambda_0", -0.1, ValueError),
        ("dt", 0.0, ValueError),
    ],
)
def test_gif_step_rejects_mutated_invalid_attributes_and_preserves_state(
    field_name: str,
    bad: object,
    exc_type: type[Exception],
) -> None:
    """Mutated invalid attributes raise loudly on step without advancing PRNG."""
    neuron = GIFPopulationNeuron(v=-62.0, eta=0.5, seed=7)
    setattr(neuron, field_name, bad)
    rng_before = neuron._rng.state

    with pytest.raises(exc_type, match=field_name):
        neuron.step(1.0)

    assert neuron._rng.state == rng_before


def test_gif_post_spike_adaptation_overflow_rolls_back_state_and_rng() -> None:
    """Overflow in post-spike adaptation leaves membrane, adaptation, and PRNG intact."""
    neuron = GIFPopulationNeuron(
        v=-50.0,
        eta=-1.0e308,
        eta_increment=-1.0e308,
        theta=-50.0,
        lambda_0=1.0e9,
        seed=11,
    )
    before = _gif_snapshot(neuron)

    with pytest.raises(ValueError, match="finite"):
        neuron.step(0.0)

    assert _gif_snapshot(neuron) == before


def test_gif_step_accepts_int_and_numpy_scalar_and_validates_reset() -> None:
    """Numeric scalars coerce cleanly and reset rejects mutated invalid v_rest or seed."""
    ref = GIFPopulationNeuron(seed=9)
    int_n = GIFPopulationNeuron(seed=9)
    np_n = GIFPopulationNeuron(seed=9)

    assert ref.step(2.0) == int_n.step(2) == np_n.step(_raw(np.float64(2.0)))
    assert ref.v == int_n.v == np_n.v
    assert ref.eta == int_n.eta == np_n.eta

    ref.v_rest = math.nan
    with pytest.raises(ValueError, match="v_rest"):
        ref.reset()

    ok = GIFPopulationNeuron(seed=9)
    ok.seed = _raw(True)
    with pytest.raises(TypeError, match="seed"):
        ok.reset()


# ---------------------------------------------------------------------------
# AvRonCardiacNeuron
# ---------------------------------------------------------------------------

AVRON_FLOAT_FIELDS: tuple[str, ...] = (
    "v",
    "h",
    "n",
    "s",
    "g_na",
    "g_k",
    "g_s",
    "g_l",
    "e_na",
    "e_k",
    "e_s",
    "e_l",
    "dt",
    "v_threshold",
)
AVRON_GATES: tuple[str, ...] = ("h", "n", "s")
AVRON_NONNEG_FIELDS: tuple[str, ...] = ("g_na", "g_k", "g_s", "g_l")


@pytest.mark.parametrize("field_name", AVRON_FLOAT_FIELDS)
@pytest.mark.parametrize("bad", NON_FINITE_FLOATS)
def test_avron_constructor_rejects_non_finite_floats(field_name: str, bad: float) -> None:
    """Non-finite float arguments fail at AvRonCardiacNeuron construction."""
    with pytest.raises(ValueError, match=field_name):
        AvRonCardiacNeuron(**{field_name: _raw(bad)})


@pytest.mark.parametrize("field_name", AVRON_FLOAT_FIELDS)
@pytest.mark.parametrize("bad", INVALID_REAL_TYPES)
def test_avron_constructor_rejects_invalid_types(field_name: str, bad: object) -> None:
    """Boolean and non-real arguments fail with TypeError at construction."""
    with pytest.raises(TypeError, match=field_name):
        AvRonCardiacNeuron(**{field_name: _raw(bad)})


@pytest.mark.parametrize("field_name", AVRON_GATES)
@pytest.mark.parametrize("bad", (-0.01, 1.01))
def test_avron_constructor_rejects_out_of_range_gates(field_name: str, bad: float) -> None:
    """Gating variables outside [0, 1] fail at construction."""
    with pytest.raises(ValueError, match=field_name):
        AvRonCardiacNeuron(**{field_name: _raw(bad)})


@pytest.mark.parametrize("field_name", AVRON_NONNEG_FIELDS)
def test_avron_constructor_rejects_negative_conductances(field_name: str) -> None:
    """Negative conductances fail at construction."""
    with pytest.raises(ValueError, match=field_name):
        AvRonCardiacNeuron(**{field_name: _raw(-0.1)})


@pytest.mark.parametrize("bad", (0.0, -0.02))
def test_avron_constructor_rejects_non_positive_dt(bad: float) -> None:
    """Non-positive dt fails at construction."""
    with pytest.raises(ValueError, match="dt"):
        AvRonCardiacNeuron(dt=bad)


@pytest.mark.parametrize("bad", INVALID_REAL_TYPES)
def test_avron_step_rejects_invalid_current_type_and_preserves_state(bad: object) -> None:
    """Boolean and non-real currents raise TypeError without mutating state."""
    neuron = AvRonCardiacNeuron(v=-55.0, h=0.5, n=0.4, s=0.6)
    before = asdict(neuron)

    with pytest.raises(TypeError, match="current"):
        neuron.step(_raw(bad))

    assert asdict(neuron) == before


@pytest.mark.parametrize("bad", NON_FINITE_FLOATS)
def test_avron_step_rejects_non_finite_current_and_preserves_state(bad: float) -> None:
    """Non-finite currents raise ValueError without mutating state."""
    neuron = AvRonCardiacNeuron(v=-55.0, h=0.5, n=0.4, s=0.6)
    before = asdict(neuron)

    with pytest.raises(ValueError, match="current"):
        neuron.step(bad)

    assert asdict(neuron) == before


def test_avron_step_coerces_scalars_and_refuses_mutated_type_error() -> None:
    """Int and NumPy float scalars match float step; mutated bool raises TypeError."""
    ref = AvRonCardiacNeuron()
    int_n = AvRonCardiacNeuron()
    np_n = AvRonCardiacNeuron()

    assert ref.step(2.0) == int_n.step(2) == np_n.step(_raw(np.float64(2.0)))
    assert (ref.v, ref.h, ref.n, ref.s) == (int_n.v, int_n.h, int_n.n, int_n.s)
    assert (ref.v, ref.h, ref.n, ref.s) == (np_n.v, np_n.h, np_n.n, np_n.s)

    ref.g_na = _raw(True)
    with pytest.raises(TypeError, match="g_na"):
        ref.step(1.0)


# ---------------------------------------------------------------------------
# GolgiCell
# ---------------------------------------------------------------------------

GOLGI_GATES: tuple[str, ...] = ("m", "h", "p_na", "n", "a", "b", "w", "m_t", "s", "c_n", "r")
GOLGI_CONDUCTANCES: tuple[str, ...] = (
    "g_na_t",
    "g_na_p",
    "g_kdr",
    "g_ka",
    "g_km",
    "g_cat",
    "g_can",
    "g_bk",
    "g_sk",
    "g_h",
    "g_l",
)
GOLGI_REVERSALS: tuple[str, ...] = ("e_na", "e_k", "e_ca", "e_h", "e_l")
GOLGI_POSITIVE_FIELDS: tuple[str, ...] = ("c_m", "tau_ca", "kd_bk", "kd_sk", "dt")
GOLGI_FLOAT_FIELDS: tuple[str, ...] = (
    "v",
    *GOLGI_GATES,
    "ca",
    *GOLGI_CONDUCTANCES,
    *GOLGI_REVERSALS,
    *GOLGI_POSITIVE_FIELDS,
    "gain",
)


@pytest.mark.parametrize("field_name", GOLGI_FLOAT_FIELDS)
@pytest.mark.parametrize("bad", NON_FINITE_FLOATS)
def test_golgi_constructor_rejects_non_finite_floats(field_name: str, bad: float) -> None:
    """Non-finite float arguments fail at GolgiCell construction."""
    with pytest.raises(ValueError, match=field_name):
        GolgiCell(**{field_name: _raw(bad)})


@pytest.mark.parametrize("field_name", GOLGI_FLOAT_FIELDS)
@pytest.mark.parametrize("bad", INVALID_REAL_TYPES)
def test_golgi_constructor_rejects_invalid_float_types(field_name: str, bad: object) -> None:
    """Boolean and non-real float fields fail with TypeError at construction."""
    with pytest.raises(TypeError, match=field_name):
        GolgiCell(**{field_name: _raw(bad)})


@pytest.mark.parametrize("bad", (-100.1, 60.1))
def test_golgi_constructor_rejects_out_of_range_voltage(bad: float) -> None:
    """Membrane voltage outside [-100, 60] fails at construction."""
    with pytest.raises(ValueError, match="v"):
        GolgiCell(v=bad)


@pytest.mark.parametrize("field_name", GOLGI_GATES)
@pytest.mark.parametrize("bad", (-0.01, 1.01))
def test_golgi_constructor_rejects_out_of_range_gates(field_name: str, bad: float) -> None:
    """Gating variables outside [0, 1] fail at construction."""
    with pytest.raises(ValueError, match=field_name):
        GolgiCell(**{field_name: _raw(bad)})


@pytest.mark.parametrize("field_name", (*GOLGI_CONDUCTANCES, "ca", "gain"))
def test_golgi_constructor_rejects_negative_nonneg_fields(field_name: str) -> None:
    """Non-negative GolgiCell parameters reject negative values at construction."""
    with pytest.raises(ValueError, match=field_name):
        GolgiCell(**{field_name: _raw(-0.1)})


@pytest.mark.parametrize("field_name", GOLGI_POSITIVE_FIELDS)
@pytest.mark.parametrize("bad", (0.0, -1.0))
def test_golgi_constructor_rejects_non_positive_fields(field_name: str, bad: float) -> None:
    """Strictly positive GolgiCell parameters reject <= 0 at construction."""
    with pytest.raises(ValueError, match=field_name):
        GolgiCell(**{field_name: _raw(bad)})


@pytest.mark.parametrize("bad", INVALID_INT_TYPES)
def test_golgi_constructor_rejects_invalid_sub_steps_types(bad: object) -> None:
    """Boolean and non-integer sub_steps fail with TypeError at construction."""
    with pytest.raises(TypeError, match="sub_steps"):
        GolgiCell(sub_steps=_raw(bad))


@pytest.mark.parametrize("bad", (0, -3))
def test_golgi_constructor_rejects_non_positive_sub_steps(bad: int) -> None:
    """Non-positive sub_steps fail with ValueError at construction."""
    with pytest.raises(ValueError, match="sub_steps"):
        GolgiCell(sub_steps=bad)


@pytest.mark.parametrize("bad", INVALID_REAL_TYPES)
def test_golgi_step_rejects_invalid_current_type_and_preserves_state(bad: object) -> None:
    """Boolean and non-real currents raise TypeError without mutating state."""
    cell = GolgiCell()
    before = asdict(cell)

    with pytest.raises(TypeError, match="current"):
        cell.step(_raw(bad))

    assert asdict(cell) == before


def test_golgi_step_rejects_zero_and_non_finite_conductance_and_calcium_overflow() -> None:
    """Zero conductance, non-finite conductance, and calcium overflow raise and roll back."""
    zero_g = GolgiCell(**{name: _raw(0.0) for name in GOLGI_CONDUCTANCES})
    before_zero = asdict(zero_g)
    with pytest.raises(ValueError, match="conductance"):
        zero_g.step(0.0)
    assert asdict(zero_g) == before_zero

    huge_g = GolgiCell(g_l=1.0e308, g_km=1.0e308, w=1.0)
    before_huge = asdict(huge_g)
    with pytest.raises(ValueError, match="conductance"):
        huge_g.step(0.0)
    assert asdict(huge_g) == before_huge

    ca_overflow = GolgiCell(g_cat=1.0e308, e_ca=1.0e308, m_t=0.5, s=0.5)
    before_ca = asdict(ca_overflow)
    with pytest.raises(ValueError, match="calcium"):
        ca_overflow.step(0.0)
    assert asdict(ca_overflow) == before_ca

    neg_excess = GolgiCell()
    before_neg = asdict(neg_excess)
    with pytest.raises(ValueError, match="voltage"):
        neg_excess.step(-1.0e8)
    assert asdict(neg_excess) == before_neg


def test_golgi_step_singularity_reversal_mutated_attrs_scalars_and_reset() -> None:
    """Covers rate singularities, non-inward Ca, scalar coercion, mutated attrs, and reset."""
    for v_sing in (-35.0, -34.0):
        cell = GolgiCell(v=v_sing, sub_steps=1)
        assert cell.step(0.0) in (0, 1)

    outward_ca = GolgiCell(v=20.0, e_ca=10.0, sub_steps=_raw(np.int64(1)))
    assert type(outward_ca.sub_steps) is int
    assert outward_ca.step(0.0) in (0, 1)

    ref = GolgiCell()
    int_c = GolgiCell()
    np_c = GolgiCell()
    assert ref.step(5.0) == int_c.step(5) == np_c.step(_raw(np.float64(5.0)))
    assert asdict(ref) == asdict(int_c) == asdict(np_c)

    ref.sub_steps = 0
    with pytest.raises(ValueError, match="sub_steps"):
        ref.step(1.0)

    ref.sub_steps = 10
    ref.reset()
    assert asdict(ref) == asdict(GolgiCell())


# ---------------------------------------------------------------------------
# MotorUnit
# ---------------------------------------------------------------------------

MOTOR_VOLTAGE_FIELDS: tuple[str, ...] = ("v", "v_rest", "v_reset", "v_threshold")
MOTOR_POSITIVE_FIELDS: tuple[str, ...] = ("tau_m", "tau_adapt", "tau_twitch", "dt")
MOTOR_NONNEG_FIELDS: tuple[str, ...] = ("gain", "twitch_amp")
MOTOR_FLOAT_FIELDS: tuple[str, ...] = (
    *MOTOR_VOLTAGE_FIELDS,
    *MOTOR_POSITIVE_FIELDS,
    "adapt",
    "a_adapt",
    *MOTOR_NONNEG_FIELDS,
    "force",
    "force_decay",
)


@pytest.mark.parametrize("field_name", MOTOR_FLOAT_FIELDS)
@pytest.mark.parametrize("bad", NON_FINITE_FLOATS)
def test_motor_unit_constructor_rejects_non_finite_floats(field_name: str, bad: float) -> None:
    """Non-finite float arguments fail at MotorUnit construction."""
    with pytest.raises(ValueError, match=field_name):
        MotorUnit(**{field_name: _raw(bad)})


@pytest.mark.parametrize("field_name", MOTOR_FLOAT_FIELDS)
@pytest.mark.parametrize("bad", INVALID_REAL_TYPES)
def test_motor_unit_constructor_rejects_invalid_types(field_name: str, bad: object) -> None:
    """Boolean and non-real arguments fail with TypeError at construction."""
    with pytest.raises(TypeError, match=field_name):
        MotorUnit(**{field_name: _raw(bad)})


@pytest.mark.parametrize("field_name", MOTOR_VOLTAGE_FIELDS)
@pytest.mark.parametrize("bad", (-150.1, 100.1))
def test_motor_unit_constructor_rejects_out_of_range_voltages(field_name: str, bad: float) -> None:
    """Voltage fields outside [-150, 100] fail at construction."""
    with pytest.raises(ValueError, match=field_name):
        MotorUnit(**{field_name: _raw(bad)})


@pytest.mark.parametrize("bad", (-0.01, 1.01))
def test_motor_unit_constructor_rejects_out_of_range_force(bad: float) -> None:
    """Force outside [0, 1] fails at construction."""
    with pytest.raises(ValueError, match="force"):
        MotorUnit(force=bad)


@pytest.mark.parametrize("field_name", MOTOR_POSITIVE_FIELDS)
@pytest.mark.parametrize("bad", (0.0, -1.0))
def test_motor_unit_constructor_rejects_non_positive_fields(field_name: str, bad: float) -> None:
    """Strictly positive MotorUnit time constants reject <= 0 at construction."""
    with pytest.raises(ValueError, match=field_name):
        MotorUnit(**{field_name: _raw(bad)})


@pytest.mark.parametrize("field_name", MOTOR_NONNEG_FIELDS)
def test_motor_unit_constructor_rejects_negative_nonneg_fields(field_name: str) -> None:
    """Negative gain or twitch_amp fails at construction."""
    with pytest.raises(ValueError, match=field_name):
        MotorUnit(**{field_name: _raw(-0.1)})


@pytest.mark.parametrize("v_reset", (-50.0, -45.0))
def test_motor_unit_constructor_rejects_reset_at_or_above_threshold(v_reset: float) -> None:
    """v_reset >= v_threshold fails at construction."""
    with pytest.raises(ValueError, match="v_reset"):
        MotorUnit(v_reset=v_reset, v_threshold=-50.0)


@pytest.mark.parametrize("bad", INVALID_REAL_TYPES)
def test_motor_unit_step_rejects_invalid_drive_type_and_preserves_state(bad: object) -> None:
    """Boolean and non-real drive raises TypeError without mutating state."""
    unit = MotorUnit.slow()
    before = asdict(unit)

    with pytest.raises(TypeError, match="drive"):
        unit.step(_raw(bad))

    assert asdict(unit) == before


def test_motor_unit_adaptation_overflow_mutated_attrs_scalars_and_reset() -> None:
    """Covers adaptation overflow rollback, negative drive, scalars, mutated attrs, and reset."""
    overflow_unit = MotorUnit(a_adapt=1.0e308)
    before = asdict(overflow_unit)
    with pytest.raises(ValueError, match="adapt"):
        overflow_unit.step(100.0)
    assert asdict(overflow_unit) == before

    ref = MotorUnit.slow()
    int_u = MotorUnit.slow()
    np_u = MotorUnit.slow()
    assert ref.step(20.0) == int_u.step(20) == np_u.step(_raw(np.float64(20.0)))
    assert asdict(ref) == asdict(int_u) == asdict(np_u)

    ref.tau_m = 0.0
    with pytest.raises(ValueError, match="tau_m"):
        ref.step(1.0)

    ref.tau_m = 10.0
    ref.reset()
    assert asdict(ref) == asdict(MotorUnit.slow())

    ref.v_rest = math.nan
    with pytest.raises(ValueError, match="v_rest"):
        ref.reset()


# ---------------------------------------------------------------------------
# RenshawCell
# ---------------------------------------------------------------------------

RENSHAW_GATES: tuple[str, ...] = ("h", "n", "adapt")
RENSHAW_CONDUCTANCES: tuple[str, ...] = ("g_na", "g_k", "g_adapt", "g_l")
RENSHAW_REVERSALS: tuple[str, ...] = ("e_na", "e_k", "e_l")
RENSHAW_POSITIVE_FIELDS: tuple[str, ...] = ("c_m", "phi", "tau_adapt", "dt")
RENSHAW_FLOAT_FIELDS: tuple[str, ...] = (
    "v",
    *RENSHAW_GATES,
    *RENSHAW_CONDUCTANCES,
    *RENSHAW_REVERSALS,
    *RENSHAW_POSITIVE_FIELDS,
    "v_threshold",
)


@pytest.mark.parametrize("field_name", RENSHAW_FLOAT_FIELDS)
@pytest.mark.parametrize("bad", NON_FINITE_FLOATS)
def test_renshaw_constructor_rejects_non_finite_floats(field_name: str, bad: float) -> None:
    """Non-finite float arguments fail at RenshawCell construction."""
    with pytest.raises(ValueError, match=field_name):
        RenshawCell(**{field_name: _raw(bad)})


@pytest.mark.parametrize("field_name", RENSHAW_FLOAT_FIELDS)
@pytest.mark.parametrize("bad", INVALID_REAL_TYPES)
def test_renshaw_constructor_rejects_invalid_types(field_name: str, bad: object) -> None:
    """Boolean and non-real arguments fail with TypeError at construction."""
    with pytest.raises(TypeError, match=field_name):
        RenshawCell(**{field_name: _raw(bad)})


@pytest.mark.parametrize("bad", (-150.1, 100.1))
def test_renshaw_constructor_rejects_out_of_range_voltage(bad: float) -> None:
    """Membrane voltage outside [-150, 100] fails at construction."""
    with pytest.raises(ValueError, match="v"):
        RenshawCell(v=bad)


@pytest.mark.parametrize("field_name", RENSHAW_GATES)
@pytest.mark.parametrize("bad", (-0.01, 1.01))
def test_renshaw_constructor_rejects_out_of_range_gates(field_name: str, bad: float) -> None:
    """Gating variables outside [0, 1] fail at construction."""
    with pytest.raises(ValueError, match=field_name):
        RenshawCell(**{field_name: _raw(bad)})


@pytest.mark.parametrize("field_name", RENSHAW_CONDUCTANCES)
def test_renshaw_constructor_rejects_negative_conductances(field_name: str) -> None:
    """Negative conductances fail at construction."""
    with pytest.raises(ValueError, match=field_name):
        RenshawCell(**{field_name: _raw(-0.1)})


@pytest.mark.parametrize("field_name", RENSHAW_POSITIVE_FIELDS)
@pytest.mark.parametrize("bad", (0.0, -1.0))
def test_renshaw_constructor_rejects_non_positive_fields(field_name: str, bad: float) -> None:
    """Strictly positive RenshawCell parameters reject <= 0 at construction."""
    with pytest.raises(ValueError, match=field_name):
        RenshawCell(**{field_name: _raw(bad)})


@pytest.mark.parametrize("bad", INVALID_REAL_TYPES)
def test_renshaw_step_rejects_invalid_current_type_and_preserves_state(bad: object) -> None:
    """Boolean and non-real currents raise TypeError without mutating state."""
    cell = RenshawCell()
    before = asdict(cell)

    with pytest.raises(TypeError, match="current"):
        cell.step(_raw(bad))

    assert asdict(cell) == before


def test_renshaw_gate_rate_overflow_zero_and_non_finite_conductance() -> None:
    """Overflowing gate rate, zero conductance, and non-finite conductance raise and roll back."""
    gate_overflow = RenshawCell(v=-150.0, phi=1.0e308)
    before_gate = asdict(gate_overflow)
    with pytest.raises(ValueError, match="gate"):
        gate_overflow.step(0.0)
    assert asdict(gate_overflow) == before_gate

    zero_g = RenshawCell(g_na=0.0, g_k=0.0, g_adapt=0.0, g_l=0.0)
    before_zero = asdict(zero_g)
    with pytest.raises(ValueError, match="conductance"):
        zero_g.step(0.0)
    assert asdict(zero_g) == before_zero

    huge_g = RenshawCell(g_l=1.0e308, g_adapt=1.0e308, adapt=1.0)
    before_huge = asdict(huge_g)
    with pytest.raises(ValueError, match="conductance"):
        huge_g.step(0.0)
    assert asdict(huge_g) == before_huge

    neg_excess = RenshawCell()
    before_neg = asdict(neg_excess)
    with pytest.raises(ValueError, match="voltage"):
        neg_excess.step(-1.0e8)
    assert asdict(neg_excess) == before_neg


def test_renshaw_singularity_mutated_attrs_scalars_and_reset() -> None:
    """Covers rate singularities, scalar coercion, mutated attrs, and reset."""
    for v_sing in (-35.0, -34.0):
        cell = RenshawCell(v=v_sing, dt=0.5)
        assert cell.step(0.0) in (0, 1)

    ref = RenshawCell()
    int_c = RenshawCell()
    np_c = RenshawCell()
    assert ref.step(4.0) == int_c.step(4) == np_c.step(_raw(np.float64(4.0)))
    assert asdict(ref) == asdict(int_c) == asdict(np_c)

    ref.phi = 0.0
    with pytest.raises(ValueError, match="phi"):
        ref.step(1.0)

    ref.phi = 5.0
    ref.reset()
    assert asdict(ref) == asdict(RenshawCell())


_BASE_20K_BIT_IDENTITY: dict[str, dict[float, tuple[int, dict[str, str]]]] = {
    "AvRonCardiacNeuron": {
        0.5: (
            135,
            {
                "v": "-0x1.a6a908a5f9990p+5",
                "h": "0x1.b0e1404141ea8p-3",
                "n": "0x1.5ef3b09a23753p-1",
                "s": "0x1.ba7444c8981e5p-2",
                "g_na": "0x1.4000000000000p+6",
                "g_k": "0x1.4000000000000p+5",
                "g_s": "0x1.4000000000000p+4",
                "g_l": "0x1.999999999999ap-4",
                "e_na": "0x1.4000000000000p+5",
                "e_k": "-0x1.4000000000000p+6",
                "e_s": "-0x1.9000000000000p+4",
                "e_l": "-0x1.e000000000000p+5",
                "dt": "0x1.47ae147ae147bp-6",
                "v_threshold": "-0x1.4000000000000p+4",
            },
        ),
        2.0: (
            135,
            {
                "v": "-0x1.5b29a9c92879bp+5",
                "h": "0x1.3165939a6d700p-2",
                "n": "0x1.438a3ac8c4456p-1",
                "s": "0x1.bafcb063c0836p-2",
                "g_na": "0x1.4000000000000p+6",
                "g_k": "0x1.4000000000000p+5",
                "g_s": "0x1.4000000000000p+4",
                "g_l": "0x1.999999999999ap-4",
                "e_na": "0x1.4000000000000p+5",
                "e_k": "-0x1.4000000000000p+6",
                "e_s": "-0x1.9000000000000p+4",
                "e_l": "-0x1.e000000000000p+5",
                "dt": "0x1.47ae147ae147bp-6",
                "v_threshold": "-0x1.4000000000000p+4",
            },
        ),
        10.0: (
            138,
            {
                "v": "-0x1.b15077bbf71a2p+5",
                "h": "0x1.763f473534a25p-3",
                "n": "0x1.67b3ba8dceb2bp-1",
                "s": "0x1.b86de763a8326p-2",
                "g_na": "0x1.4000000000000p+6",
                "g_k": "0x1.4000000000000p+5",
                "g_s": "0x1.4000000000000p+4",
                "g_l": "0x1.999999999999ap-4",
                "e_na": "0x1.4000000000000p+5",
                "e_k": "-0x1.4000000000000p+6",
                "e_s": "-0x1.9000000000000p+4",
                "e_l": "-0x1.e000000000000p+5",
                "dt": "0x1.47ae147ae147bp-6",
                "v_threshold": "-0x1.4000000000000p+4",
            },
        ),
    },
    "GIFPopulationNeuron": {
        0.5: (
            1,
            {
                "v": "-0x1.0307661de4162p+6",
                "theta": "-0x1.9000000000000p+5",
                "eta": "0x1.1ff8b5ee1ceb6p-90",
                "tau_m": "0x1.4000000000000p+4",
                "tau_eta": "0x1.9000000000000p+6",
                "delta_v": "0x1.0000000000000p+1",
                "lambda_0": "0x1.0624dd2f1a9fcp-10",
                "eta_increment": "0x1.4000000000000p+2",
                "v_rest": "-0x1.0400000000000p+6",
                "v_reset": "-0x1.0400000000000p+6",
                "dt": "0x1.0000000000000p-1",
            },
        ),
        2.0: (
            1,
            {
                "v": "-0x1.001d987790588p+6",
                "theta": "-0x1.9000000000000p+5",
                "eta": "0x1.1ff8b5ee1ceb6p-90",
                "tau_m": "0x1.4000000000000p+4",
                "tau_eta": "0x1.9000000000000p+6",
                "delta_v": "0x1.0000000000000p+1",
                "lambda_0": "0x1.0624dd2f1a9fcp-10",
                "eta_increment": "0x1.4000000000000p+2",
                "v_rest": "-0x1.0400000000000p+6",
                "v_reset": "-0x1.0400000000000p+6",
                "dt": "0x1.0000000000000p-1",
            },
        ),
        10.0: (
            1,
            {
                "v": "-0x1.e127f4aba3760p+5",
                "theta": "-0x1.9000000000000p+5",
                "eta": "0x1.1ff8b5ee1ceb6p-90",
                "tau_m": "0x1.4000000000000p+4",
                "tau_eta": "0x1.9000000000000p+6",
                "delta_v": "0x1.0000000000000p+1",
                "lambda_0": "0x1.0624dd2f1a9fcp-10",
                "eta_increment": "0x1.4000000000000p+2",
                "v_rest": "-0x1.0400000000000p+6",
                "v_reset": "-0x1.0400000000000p+6",
                "dt": "0x1.0000000000000p-1",
            },
        ),
    },
    "GolgiCell": {
        0.5: (
            51,
            {
                "v": "-0x1.252597b4072adp+6",
                "m": "0x1.4928c1b5818adp-7",
                "h": "0x1.de9e2b2bb2711p-1",
                "p_na": "0x1.caa2567754ea1p-8",
                "n": "0x1.53e2e1673bc7bp-5",
                "a": "0x1.a9cdf98f11414p-5",
                "b": "0x1.f582dce602041p-3",
                "w": "0x1.067b3b4952598p-5",
                "m_t": "0x1.bf99d48307dc0p-7",
                "s": "0x1.92aa3db3162b5p-1",
                "c_n": "0x1.7bf2a75f080a7p-16",
                "r": "0x1.ec5823e101c38p-3",
                "ca": "0x1.3a46fdf1e8311p-3",
                "g_na_t": "0x1.8000000000000p+5",
                "g_na_p": "0x1.999999999999ap-3",
                "g_kdr": "0x1.0000000000000p+4",
                "g_ka": "0x1.0000000000000p+3",
                "g_km": "0x1.0000000000000p+0",
                "g_cat": "0x1.0000000000000p-1",
                "g_can": "0x1.0000000000000p+0",
                "g_bk": "0x1.8000000000000p+1",
                "g_sk": "0x1.0000000000000p+0",
                "g_h": "0x1.999999999999ap-4",
                "g_l": "0x1.999999999999ap-5",
                "e_na": "0x1.b800000000000p+5",
                "e_k": "-0x1.6800000000000p+6",
                "e_ca": "0x1.e000000000000p+6",
                "e_h": "-0x1.4000000000000p+5",
                "e_l": "-0x1.b800000000000p+5",
                "c_m": "0x1.0000000000000p+0",
                "tau_ca": "0x1.9000000000000p+7",
                "kd_bk": "0x1.0000000000000p+0",
                "kd_sk": "0x1.0000000000000p-1",
                "dt": "0x1.0000000000000p-1",
                "gain": "0x1.0000000000000p+0",
            },
        ),
        2.0: (
            70,
            {
                "v": "-0x1.089f619bf86a2p+6",
                "m": "0x1.986bbebb21931p-6",
                "h": "0x1.a7086bf21d05bp-1",
                "p_na": "0x1.a4825aac41954p-6",
                "n": "0x1.36dd5809bf9c4p-4",
                "a": "0x1.4acac54a3b472p-4",
                "b": "0x1.adb4357b4da74p-4",
                "w": "0x1.43aea06108428p-5",
                "m_t": "0x1.cd26698b4bd48p-5",
                "s": "0x1.8bd12cbb1dbd0p-1",
                "c_n": "0x1.b41ee3791b0b4p-14",
                "r": "0x1.df8e38dbdcd94p-3",
                "ca": "0x1.2d5f3f88eeb80p-3",
                "g_na_t": "0x1.8000000000000p+5",
                "g_na_p": "0x1.999999999999ap-3",
                "g_kdr": "0x1.0000000000000p+4",
                "g_ka": "0x1.0000000000000p+3",
                "g_km": "0x1.0000000000000p+0",
                "g_cat": "0x1.0000000000000p-1",
                "g_can": "0x1.0000000000000p+0",
                "g_bk": "0x1.8000000000000p+1",
                "g_sk": "0x1.0000000000000p+0",
                "g_h": "0x1.999999999999ap-4",
                "g_l": "0x1.999999999999ap-5",
                "e_na": "0x1.b800000000000p+5",
                "e_k": "-0x1.6800000000000p+6",
                "e_ca": "0x1.e000000000000p+6",
                "e_h": "-0x1.4000000000000p+5",
                "e_l": "-0x1.b800000000000p+5",
                "c_m": "0x1.0000000000000p+0",
                "tau_ca": "0x1.9000000000000p+7",
                "kd_bk": "0x1.0000000000000p+0",
                "kd_sk": "0x1.0000000000000p-1",
                "dt": "0x1.0000000000000p-1",
                "gain": "0x1.0000000000000p+0",
            },
        ),
        10.0: (
            106,
            {
                "v": "-0x1.21f05e343d367p+6",
                "m": "0x1.6591f7971540ep-7",
                "h": "0x1.d95f8bc98e1cbp-1",
                "p_na": "0x1.60780c3f57819p-6",
                "n": "0x1.6b62980d87e73p-5",
                "a": "0x1.d6fac8ac67d4ap-5",
                "b": "0x1.183f32dee3be2p-3",
                "w": "0x1.16317390d0629p-4",
                "m_t": "0x1.067b4ef2f69eap-6",
                "s": "0x1.5da8c147fe29cp-1",
                "c_n": "0x1.39c15a4e98b50p-15",
                "r": "0x1.8608d60a3ca8dp-3",
                "ca": "0x1.2273e3ac00237p-2",
                "g_na_t": "0x1.8000000000000p+5",
                "g_na_p": "0x1.999999999999ap-3",
                "g_kdr": "0x1.0000000000000p+4",
                "g_ka": "0x1.0000000000000p+3",
                "g_km": "0x1.0000000000000p+0",
                "g_cat": "0x1.0000000000000p-1",
                "g_can": "0x1.0000000000000p+0",
                "g_bk": "0x1.8000000000000p+1",
                "g_sk": "0x1.0000000000000p+0",
                "g_h": "0x1.999999999999ap-4",
                "g_l": "0x1.999999999999ap-5",
                "e_na": "0x1.b800000000000p+5",
                "e_k": "-0x1.6800000000000p+6",
                "e_ca": "0x1.e000000000000p+6",
                "e_h": "-0x1.4000000000000p+5",
                "e_l": "-0x1.b800000000000p+5",
                "c_m": "0x1.0000000000000p+0",
                "tau_ca": "0x1.9000000000000p+7",
                "kd_bk": "0x1.0000000000000p+0",
                "kd_sk": "0x1.0000000000000p-1",
                "dt": "0x1.0000000000000p-1",
                "gain": "0x1.0000000000000p+0",
            },
        ),
    },
    "MotorUnit": {
        0.5: (
            0,
            {
                "v": "-0x1.03447cbc4528cp+6",
                "v_rest": "-0x1.0400000000000p+6",
                "v_reset": "-0x1.1800000000000p+6",
                "v_threshold": "-0x1.9000000000000p+5",
                "tau_m": "0x1.4000000000000p+3",
                "adapt": "0x1.5e451e80393a4p-5",
                "tau_adapt": "0x1.9000000000000p+6",
                "a_adapt": "0x1.999999999999ap-3",
                "gain": "0x1.0000000000000p+0",
                "force": "0x0.0p+0",
                "twitch_amp": "0x1.999999999999ap-5",
                "tau_twitch": "0x1.6800000000000p+6",
                "force_decay": "0x0.0p+0",
                "dt": "0x1.0000000000000p-1",
            },
        ),
        2.0: (
            0,
            {
                "v": "-0x1.0111f2f114a2fp+6",
                "v_rest": "-0x1.0400000000000p+6",
                "v_reset": "-0x1.1800000000000p+6",
                "v_threshold": "-0x1.9000000000000p+5",
                "tau_m": "0x1.4000000000000p+3",
                "adapt": "0x1.5e451e8039435p-3",
                "tau_adapt": "0x1.9000000000000p+6",
                "a_adapt": "0x1.999999999999ap-3",
                "gain": "0x1.0000000000000p+0",
                "force": "0x0.0p+0",
                "twitch_amp": "0x1.999999999999ap-5",
                "tau_twitch": "0x1.6800000000000p+6",
                "force_decay": "0x0.0p+0",
                "dt": "0x1.0000000000000p-1",
            },
        ),
        10.0: (
            0,
            {
                "v": "-0x1.eab37d6ace5c1p+5",
                "v_rest": "-0x1.0400000000000p+6",
                "v_reset": "-0x1.1800000000000p+6",
                "v_threshold": "-0x1.9000000000000p+5",
                "tau_m": "0x1.4000000000000p+3",
                "adapt": "0x1.b5d666204794ep-1",
                "tau_adapt": "0x1.9000000000000p+6",
                "a_adapt": "0x1.999999999999ap-3",
                "gain": "0x1.0000000000000p+0",
                "force": "0x0.0p+0",
                "twitch_amp": "0x1.999999999999ap-5",
                "tau_twitch": "0x1.6800000000000p+6",
                "force_decay": "0x0.0p+0",
                "dt": "0x1.0000000000000p-1",
            },
        ),
    },
    "RenshawCell": {
        0.5: (
            0,
            {
                "v": "-0x1.fc77f6f3c78b5p+5",
                "h": "0x1.880fe9c5691c5p-1",
                "n": "0x1.7c0aec10428f3p-4",
                "adapt": "0x1.538861e9f05f6p-10",
                "g_na": "0x1.1800000000000p+5",
                "g_k": "0x1.2000000000000p+3",
                "g_adapt": "0x1.4000000000000p+2",
                "g_l": "0x1.eb851eb851eb8p-4",
                "e_na": "0x1.b800000000000p+5",
                "e_k": "-0x1.6800000000000p+6",
                "e_l": "-0x1.0400000000000p+6",
                "c_m": "0x1.0000000000000p+0",
                "phi": "0x1.4000000000000p+2",
                "tau_adapt": "0x1.9000000000000p+5",
                "dt": "0x1.47ae147ae147bp-7",
                "v_threshold": "-0x1.4000000000000p+4",
            },
        ),
        2.0: (
            81,
            {
                "v": "-0x1.130e7ec341308p+6",
                "h": "0x1.c0d1ca9361a2ap-1",
                "n": "0x1.f527ca89cf664p-5",
                "adapt": "0x1.640f1b5e55546p-7",
                "g_na": "0x1.1800000000000p+5",
                "g_k": "0x1.2000000000000p+3",
                "g_adapt": "0x1.4000000000000p+2",
                "g_l": "0x1.eb851eb851eb8p-4",
                "e_na": "0x1.b800000000000p+5",
                "e_k": "-0x1.6800000000000p+6",
                "e_l": "-0x1.0400000000000p+6",
                "c_m": "0x1.0000000000000p+0",
                "phi": "0x1.4000000000000p+2",
                "tau_adapt": "0x1.9000000000000p+5",
                "dt": "0x1.47ae147ae147bp-7",
                "v_threshold": "-0x1.4000000000000p+4",
            },
        ),
        10.0: (
            384,
            {
                "v": "-0x1.de5eaadf6d69cp+5",
                "h": "0x1.30f36f8950291p-1",
                "n": "0x1.0e425380884c4p-3",
                "adapt": "0x1.a006161631e14p-6",
                "g_na": "0x1.1800000000000p+5",
                "g_k": "0x1.2000000000000p+3",
                "g_adapt": "0x1.4000000000000p+2",
                "g_l": "0x1.eb851eb851eb8p-4",
                "e_na": "0x1.b800000000000p+5",
                "e_k": "-0x1.6800000000000p+6",
                "e_l": "-0x1.0400000000000p+6",
                "c_m": "0x1.0000000000000p+0",
                "phi": "0x1.4000000000000p+2",
                "tau_adapt": "0x1.9000000000000p+5",
                "dt": "0x1.47ae147ae147bp-7",
                "v_threshold": "-0x1.4000000000000p+4",
            },
        ),
    },
}


_DYNAMIC_STATE_FIELDS: dict[str, frozenset[str]] = {
    "AvRonCardiacNeuron": frozenset({"v", "h", "n", "s"}),
    "GIFPopulationNeuron": frozenset({"v", "theta", "eta"}),
    "GolgiCell": frozenset(
        {"v", "m", "h", "p_na", "n", "a", "b", "w", "m_t", "s", "c_n", "r", "ca"}
    ),
    "MotorUnit": frozenset({"v", "adapt", "force", "force_decay"}),
    "RenshawCell": frozenset({"v", "h", "n", "adapt"}),
}


@pytest.mark.parametrize(
    ("model_name", "model_cls"),
    [
        ("AvRonCardiacNeuron", AvRonCardiacNeuron),
        ("GIFPopulationNeuron", GIFPopulationNeuron),
        ("GolgiCell", GolgiCell),
        ("MotorUnit", MotorUnit),
        ("RenshawCell", RenshawCell),
    ],
)
@pytest.mark.parametrize("amplitude", [0.5, 2.0, 10.0])
def test_all_five_models_20k_step_trajectory_and_spike_regression(
    model_name: str, model_cls: type[Any], amplitude: float
) -> None:
    """Verify 20,000-step spike counts and trajectories across FMA and non-FMA libm hosts.

    Unchanging parameter fields match bit-for-bit; dynamic ODE states use a
    tight cross-``libm`` tolerance (``rel_tol=1e-6``, ``abs_tol=1e-12``) so
    1-ULP transcendental differences between FMA and non-FMA ``libm``
    implementations over 20,000 spiking steps do not cause spurious host
    failures while spike counts remain exact.
    """
    drives = np.random.default_rng(7).uniform(0.0, amplitude, size=20_000)
    cell = model_cls()
    spikes = 0
    for drive in drives:
        spikes += cell.step(float(drive))
    expected_spikes, expected_hex = _BASE_20K_BIT_IDENTITY[model_name][amplitude]
    actual = {k: float(v) for k, v in asdict(cell).items() if isinstance(v, float)}
    dynamic_fields = _DYNAMIC_STATE_FIELDS[model_name]
    assert spikes == expected_spikes
    assert set(actual) == set(expected_hex)
    for field_name, exp_hex in expected_hex.items():
        expected_val = float.fromhex(exp_hex)
        actual_val = actual[field_name]
        assert math.isfinite(actual_val)
        if field_name in dynamic_fields:
            assert math.isclose(actual_val, expected_val, rel_tol=1e-6, abs_tol=1e-12), (
                f"{model_name} amp={amplitude} field={field_name}: "
                f"{actual_val!r} vs {expected_val!r}"
            )
        else:
            assert actual_val.hex() == exp_hex


def _golgi_bk_sk_reference_step(cell: GolgiCell) -> float:
    """Compute a single BK/SK-only Golgi substep using the exact ``** 2`` operator."""
    dt_sub = cell.dt / cell.sub_steps
    decay = math.exp(-dt_sub / cell.tau_ca)
    ca_next = max(0.0, cell.ca * decay)
    ca2 = ca_next**2
    kd2 = cell.kd_bk**2
    v_half = 100.0 - 120.0 * ca2 / (ca2 + kd2)
    x = (cell.v - v_half) / 15.0
    if x >= 0.0:
        bk_v = 1.0 / (1.0 + math.exp(-x))
    else:
        ex = math.exp(x)
        bk_v = ex / (1.0 + ex)
    sk_inf = ca2 / (ca2 + cell.kd_sk**2)
    g_k = cell.g_bk * bk_v + cell.g_sk * sk_inf
    g_total = g_k + cell.g_l
    steady_v = (g_k * cell.e_k + cell.g_l * cell.e_l) / g_total
    return steady_v + (cell.v - steady_v) * math.exp(-(g_total / cell.c_m) * dt_sub)


@pytest.mark.parametrize(
    ("label", "kwargs"),
    [
        (
            "ca_next_pow2",
            {
                "v": 0.0,
                "ca": 13.033,
                "kd_bk": 1.0,
                "kd_sk": 13.033,
                "g_bk": 0.0,
                "g_sk": 4.0,
                "g_l": 1.0,
            },
        ),
        (
            "kd_bk_pow2",
            {
                "v": -60.0,
                "ca": 15.0,
                "kd_bk": 4.536,
                "kd_sk": 1.0,
                "g_bk": 12.0,
                "g_sk": 0.0,
                "g_l": 0.05,
            },
        ),
        (
            "kd_sk_pow2",
            {
                "v": 0.0,
                "ca": 7.964,
                "kd_bk": 1.0,
                "kd_sk": 7.964,
                "g_bk": 0.0,
                "g_sk": 4.0,
                "g_l": 1.0,
            },
        ),
    ],
)
def test_golgi_cell_preserves_power_operator_bit_identity(
    label: str, kwargs: dict[str, float]
) -> None:
    """Pin 0-ULP same-host ``float.hex()`` identity for all three ``** 2`` sites in ``GolgiCell``."""
    cell = GolgiCell(
        sub_steps=1,
        g_na_t=0.0,
        g_na_p=0.0,
        g_kdr=0.0,
        g_ka=0.0,
        g_km=0.0,
        g_cat=0.0,
        g_can=0.0,
        g_h=0.0,
        **kwargs,
    )
    expected_v = _golgi_bk_sk_reference_step(cell)
    cell.step(0.0)
    assert cell.v.hex() == expected_v.hex(), label
