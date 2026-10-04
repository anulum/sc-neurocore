# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — installed Brunel-Wang PyO3 boundary contracts

"""Exercise configured construction, full-gate stepping, and reset semantics."""

from __future__ import annotations

import copy
import pickle
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.accel.brunel_wang import PARITY_ATOL
from sc_neurocore.neurons.models.brunel_wang import BrunelWangNeuron as ReferenceNeuron
from tests.engine_requirement import require_engine

extension = require_engine()
import sc_neurocore_engine

_PARAMETERS: tuple[tuple[str, float, float], ...] = (
    ("v", -70.0, -63.0),
    ("v_rest", -70.0, -68.0),
    ("v_reset", -55.0, -56.0),
    ("v_threshold", -50.0, -49.0),
    ("tau_m", 20.0, 18.0),
    ("tau_ref", 2.0, 1.5),
    ("g_ampa_ext", 2.08, 1.8),
    ("g_ampa_rec", 0.104, 0.13),
    ("g_nmda", 0.327, 0.4),
    ("g_gaba", 1.25, 1.1),
    ("v_ampa", 0.0, 2.0),
    ("v_nmda", 0.0, 1.0),
    ("v_gaba", -70.0, -72.0),
    ("c_m", 0.5, 0.6),
    ("mg_conc", 1.0, 0.7),
    ("dt", 0.1, 0.05),
    ("ref_remaining", 0.0, 0.35),
)
_GATE_NAMES = ("i_ampa_ext", "s_ampa_rec", "s_nmda_rec", "s_gaba")
_CONFIGURATIONS = (
    pytest.param({}, id="defaults"),
    *(pytest.param({name: value}, id=name) for name, _, value in _PARAMETERS),
    pytest.param({name: value for name, _, value in _PARAMETERS}, id="all-configured"),
    pytest.param({"v": -49.0, "ref_remaining": 0.0}, id="initial-threshold-crossing"),
    pytest.param({"ref_remaining": 0.05}, id="partial-refractory-step"),
)


class NativeNeuron(Protocol):
    """Describe the public state and scalar conversion boundary under test."""

    def step(
        self,
        i_ampa_ext: object = 0.0,
        s_ampa_rec: object = 0.0,
        s_nmda_rec: object = 0.0,
        s_gaba: object = 0.0,
    ) -> int:
        """Advance one four-gate step or refuse before mutating state."""
        ...

    def reset(self) -> None:
        """Restore configured resting voltage and clear refractory time."""
        ...

    def get_state(self) -> tuple[float, float]:
        """Return voltage and refractory time as an immutable tuple."""
        ...


class Constructor(Protocol):
    """Describe the class's positional and keyword conversion surface."""

    def __call__(self, *args: object, **kwargs: object) -> NativeNeuron:
        """Construct a native cell from the actual exposed parameters."""
        ...


_CONSTRUCTOR = cast(Constructor, sc_neurocore_engine.BrunelWangNeuron)


def _reference(parameters: dict[str, float]) -> ReferenceNeuron:
    """Adapt native c_m spelling and initial refractory state to Python."""
    arguments = dict(parameters)
    refractory = arguments.pop("ref_remaining", 0.0)
    if "c_m" in arguments:
        arguments["C_m"] = arguments.pop("c_m")
    neuron = ReferenceNeuron(**arguments)
    neuron.ref_remaining = refractory
    return neuron


def _gates(index: int) -> tuple[float, float, float, float]:
    """Drive all four channels through silent, firing and mixed intervals."""
    return (
        (0.0, 1.0, 0.2, 0.8)[index % 4],
        0.1 * (index % 3),
        0.2 * (index % 5),
        0.05 * (index % 7),
    )


@pytest.mark.parametrize("parameters", _CONFIGURATIONS)
def test_all_exposed_configuration_fields_match_python_trace_and_reset(
    parameters: dict[str, float],
) -> None:
    """Compare configured native state/events with the actual Python model."""
    expected = _reference(parameters)
    native = _CONSTRUCTOR(**parameters)
    positional = _CONSTRUCTOR(*(parameters.get(name, default) for name, default, _ in _PARAMETERS))
    assert native.get_state() == (expected.v, expected.ref_remaining)
    assert positional.get_state() == native.get_state()
    events = 0
    for index in range(256):
        gates = _gates(index)
        event = native.step(*gates)
        events += event
        assert type(event) is int
        assert event == positional.step(*gates) == expected.step(*gates)
        assert positional.get_state() == native.get_state()
        assert native.get_state() == pytest.approx(
            (expected.v, expected.ref_remaining), rel=0.0, abs=2.0e-12
        )
    assert events > 0
    expected.reset()
    native.reset()
    assert native.get_state() == (expected.v, expected.ref_remaining)
    for index in range(32):
        assert native.step(*_gates(index)) == expected.step(*_gates(index))
        assert native.get_state() == pytest.approx(
            (expected.v, expected.ref_remaining), rel=0.0, abs=2.0e-12
        )


@pytest.mark.parametrize("parameters", _CONFIGURATIONS)
@pytest.mark.parametrize("backend", ["python", "rust", "julia", "go", "mojo"])
def test_configured_native_trace_matches_every_public_backend(
    parameters: dict[str, float], backend: str
) -> None:
    """Compare each configured PyO3 trace through all five actual public routes."""
    gates = np.asarray([_gates(index) for index in range(256)], dtype=np.float64)
    native = _CONSTRUCTOR(**parameters)
    events: list[int] = []
    states: list[tuple[float, float]] = []
    for gate_values in gates:
        events.append(native.step(*gate_values))
        states.append(native.get_state())
    reference = _reference(parameters)
    result = reference.simulate(*(gates[:, index] for index in range(4)), backend=backend)
    actual_events = result["events"]
    voltages = result["voltages"]
    refractory = result["refractory"]
    assert isinstance(actual_events, np.ndarray)
    assert isinstance(voltages, np.ndarray)
    assert isinstance(refractory, np.ndarray)
    assert actual_events.dtype == np.int64
    assert voltages.dtype == refractory.dtype == np.float64
    assert actual_events.shape == voltages.shape == refractory.shape == (256,)
    tolerance = max(PARITY_ATOL["rust"], PARITY_ATOL[backend])
    np.testing.assert_array_equal(actual_events, events)
    np.testing.assert_allclose(
        np.column_stack(
            (
                cast(npt.NDArray[np.float64], voltages),
                cast(npt.NDArray[np.float64], refractory),
            )
        ),
        states,
        rtol=0.0,
        atol=tolerance,
    )
    assert (result["v_final"], result["ref_final"]) == pytest.approx(
        native.get_state(), rel=0.0, abs=tolerance
    )
    assert (reference.v, reference.ref_remaining) == (result["v_final"], result["ref_final"])


@pytest.mark.parametrize("field", [name for name, _, _ in _PARAMETERS])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_constructor_fields_refuse_with_exact_native_error(
    field: str, value: float
) -> None:
    """Refuse every non-finite state/configuration field during construction."""
    with pytest.raises(ValueError) as refusal:
        _CONSTRUCTOR(**{field: value})
    assert str(refusal.value) == "invalid Brunel-Wang configuration or aggregate gate"


@pytest.mark.parametrize("field", ["tau_m", "tau_ref", "c_m", "dt"])
@pytest.mark.parametrize("value", [0.0, -1.0])
def test_nonpositive_constructor_scales_refuse(field: str, value: float) -> None:
    """Keep all exposed integration and capacitance scales strictly positive."""
    with pytest.raises(ValueError) as refusal:
        _CONSTRUCTOR(**{field: value})
    assert str(refusal.value) == "invalid Brunel-Wang configuration or aggregate gate"


@pytest.mark.parametrize(
    "field", ["g_ampa_ext", "g_ampa_rec", "g_nmda", "g_gaba", "mg_conc", "ref_remaining"]
)
def test_negative_constructor_conductances_or_refractory_time_refuse(field: str) -> None:
    """Refuse negative gates, magnesium concentration and initial refractory time."""
    with pytest.raises(ValueError) as refusal:
        _CONSTRUCTOR(**{field: -1.0})
    assert str(refusal.value) == "invalid Brunel-Wang configuration or aggregate gate"


@pytest.mark.parametrize("field", [name for name, _, _ in _PARAMETERS])
def test_constructor_scalar_type_refusal_names_the_python_type(field: str) -> None:
    """Exercise every constructor argument's real f64 conversion refusal."""
    with pytest.raises(TypeError) as refusal:
        _CONSTRUCTOR(**{field: None})
    assert str(refusal.value) == "must be real number, not NoneType"


def test_unknown_or_extra_constructor_arguments_refuse() -> None:
    """Preserve the measured keyword and positional constructor signature."""
    with pytest.raises(TypeError) as refusal:
        _CONSTRUCTOR(unknown=1.0)
    assert str(refusal.value) == (
        "BrunelWangNeuron.__new__() got an unexpected keyword argument 'unknown'"
    )
    with pytest.raises(TypeError) as refusal:
        _CONSTRUCTOR(*([0.0] * 18))
    assert str(refusal.value) == (
        "BrunelWangNeuron.__new__() takes from 0 to 17 positional arguments but 18 were given"
    )


@pytest.mark.parametrize("field,value", [(name, value) for name, _, value in _PARAMETERS])
def test_numpy_constructor_scalars_and_readonly_zero_rank_conversion(
    field: str, value: float
) -> None:
    """Accept real NumPy scalar extraction without changing caller-owned storage."""
    scalar = np.array(value, dtype=np.float64)
    scalar.setflags(write=False)
    before = scalar.tobytes()
    native = _CONSTRUCTOR(**{field: scalar})
    control = _CONSTRUCTOR(**{field: np.float64(value)})
    assert native.get_state() == control.get_state()
    for index in range(16):
        assert native.step(*_gates(index)) == control.step(*_gates(index))
        assert native.get_state() == control.get_state()
    assert scalar.shape == ()
    assert scalar.tobytes() == before
    assert not scalar.flags.writeable


@pytest.mark.parametrize("field", _GATE_NAMES)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), -1.0])
def test_each_invalid_aggregate_gate_is_atomic_and_recovers(field: str, value: float) -> None:
    """Validate all four gates before touching membrane or refractory state."""
    for refractory in (0.0, 0.35):
        native = _CONSTRUCTOR(v=-63.0, ref_remaining=refractory)
        control = _CONSTRUCTOR(v=-63.0, ref_remaining=refractory)
        before = native.get_state()
        with pytest.raises(ValueError) as refusal:
            native.step(**{field: value})
        assert str(refusal.value) == "invalid Brunel-Wang configuration or aggregate gate"
        assert native.get_state() == before
        assert native.step(*_gates(1)) == control.step(*_gates(1))
        assert native.get_state() == control.get_state()


@pytest.mark.parametrize("field", _GATE_NAMES)
def test_each_gate_scalar_extraction_refuses_without_advancing_state(field: str) -> None:
    """Preserve the next configured trace after a Python scalar conversion error."""
    native = _CONSTRUCTOR(v=-63.0)
    control = _CONSTRUCTOR(v=-63.0)
    before = native.get_state()
    with pytest.raises(TypeError) as refusal:
        native.step(**{field: None})
    assert str(refusal.value) == "must be real number, not NoneType"
    assert native.get_state() == before
    assert native.step(*_gates(1)) == control.step(*_gates(1))
    assert native.get_state() == control.get_state()


def test_nonfinite_candidate_refusal_and_refractory_branch_preserve_recovery() -> None:
    """Distinguish RK2 overflow from valid finite gates during refractory time."""
    native = _CONSTRUCTOR()
    before = native.get_state()
    with pytest.raises(ValueError) as refusal:
        native.step(1.0e308)
    assert str(refusal.value) == "non-finite Brunel-Wang RK2 candidate"
    assert native.get_state() == before
    control = _CONSTRUCTOR()
    assert native.step(*_gates(1)) == control.step(*_gates(1))
    assert native.get_state() == control.get_state()
    refractory = _CONSTRUCTOR(ref_remaining=0.35, v_reset=-56.0)
    assert refractory.step(1.0e308) == 0
    assert refractory.get_state() == (-56.0, 0.35 - 0.1)


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_configured_instance_pickle_refusal_preserves_identity_and_following_trace(
    protocol: int,
) -> None:
    """Round-trip the native global and refuse unsupported configured persistence."""
    cls = sc_neurocore_engine.BrunelWangNeuron
    assert cls is extension.BrunelWangNeuron
    assert pickle.loads(pickle.dumps(cls, protocol=protocol)) is cls
    native = _CONSTRUCTOR(v=-63.0, g_nmda=0.4, dt=0.05)
    control = _CONSTRUCTOR(v=-63.0, g_nmda=0.4, dt=0.05)
    before = native.get_state()
    with pytest.raises(TypeError) as refusal:
        pickle.dumps(native, protocol=protocol)
    expected = (
        "cannot pickle 'BrunelWangNeuron' object"
        if protocol < 2
        else "cannot pickle 'sc_neurocore_engine.sc_neurocore_engine.BrunelWangNeuron' object"
    )
    assert str(refusal.value) == expected
    assert native.get_state() == before
    assert native.step(*_gates(1)) == control.step(*_gates(1))
    assert native.get_state() == control.get_state()


def test_configured_copy_refusals_preserve_immutable_returned_state() -> None:
    """Refuse copy/deepcopy while preserving the independent tuple snapshot."""
    native = _CONSTRUCTOR(v=-63.0)
    control = _CONSTRUCTOR(v=-63.0)
    before = native.get_state()
    assert type(before) is tuple
    for copier in (copy.copy, copy.deepcopy):
        with pytest.raises(TypeError):
            copier(native)
        assert native.get_state() == before
    assert native.step(*_gates(1)) == control.step(*_gates(1))
    assert native.get_state() == control.get_state()


def test_configured_engine_boundary_and_reset() -> None:
    """Preserve non-default configuration while resetting only dynamic state."""
    neuron = sc_neurocore_engine.BrunelWangNeuron(g_nmda=0.4, dt=0.05)
    assert neuron.step(0.2, 0.1, 0.3, 0.0) in (0, 1)
    neuron.reset()
    assert neuron.get_state() == (-70.0, 0.0)


def test_engine_failure_is_atomic() -> None:
    """Reject invalid aggregate input without mutating dynamic state."""
    neuron = sc_neurocore_engine.BrunelWangNeuron(v=-63.0)
    before = neuron.get_state()
    with pytest.raises(ValueError):
        neuron.step(float("nan"), 0.0, 0.0, 0.0)
    assert neuron.get_state() == before
