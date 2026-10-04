# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — MAT family C ABI admission and caller-buffer contracts

"""Exercise both actual MAT Go/Mojo exports, including crash-fenced pointers."""

from __future__ import annotations

import ctypes
from collections.abc import Callable
from dataclasses import fields
import json
from pathlib import Path
import subprocess
import sys
from typing import TypeAlias, cast

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.neurons.models.mat import MATNeuron
from sc_neurocore.neurons.models.sc_resetting_mat import SCResettingMATNeuron

FloatArray: TypeAlias = npt.NDArray[np.float64]
CallResult: TypeAlias = tuple[
    int, FloatArray, list[FloatArray], npt.NDArray[np.int64], list[FloatArray]
]
_REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULTS = {
    "mat": {field.name: float(getattr(MATNeuron(), field.name)) for field in fields(MATNeuron)},
    "sc_resetting_mat": {
        field.name: float(getattr(SCResettingMATNeuron(), field.name))
        for field in fields(SCResettingMATNeuron)
    },
}
_COMMON_INVALID = [
    {field: value} for field in ("theta1", "theta2") for value in (-1.0, 1e9 + 1.0)
] + [
    {field: value}
    for field in ("tau_m", "tau_1", "tau_2", "resistance", "dt")
    for value in (0.0, -1.0)
]
_DOMAIN_INVALID = {
    "mat": [
        *_COMMON_INVALID,
        *[{"v": value} for value in (-201.0, 201.0)],
        *[{"omega": value} for value in (-1e9 - 1.0, 1e9 + 1.0)],
        *[{"refractory_remaining": value} for value in (-1.0, 3.0)],
        {"refractory_period": -1.0},
        *[{field: value} for field in ("alpha_1", "alpha_2") for value in (-1.0, 1e9 + 1.0)],
    ],
    "sc_resetting_mat": [
        *_COMMON_INVALID,
        *[{field: value} for field in ("v", "v_reset") for value in (-201.0, 101.0)],
        *[{field: value} for field in ("h1", "h2") for value in (-1.0, 1e9 + 1.0)],
    ],
}
INVALID_CONFIGURATIONS = [
    (model, parameters)
    for model in DEFAULTS
    for parameters in [
        *_DOMAIN_INVALID[model],
        *[
            {field: value}
            for field in DEFAULTS[model]
            for value in (float("nan"), float("inf"), -float("inf"))
        ],
    ]
]


def _call(
    model: str,
    backend: str,
    parameters: dict[str, float],
    count: int,
    current: float | FloatArray = 0.0,
    null_buffer: int | None = None,
    library_path: Path | None = None,
) -> CallResult:
    """Call a real exported ABI with typed live buffers and all thirteen fields."""
    path = library_path or (_REPOSITORY / f"src/sc_neurocore/accel/{backend}/{model}/lib{model}.so")
    library = ctypes.CDLL(str(path))
    native = getattr(library, f"{model}_simulate_c")
    native.restype = ctypes.c_int if backend == "go" else ctypes.c_ssize_t
    function = cast(Callable[..., int], native)
    configuration = DEFAULTS[model] | parameters
    size = max(count, 1)
    drive = np.full(size, current, dtype=np.float64)
    trace_count = 4 if model == "mat" else 3
    traces = [np.full(size, -777.0) for _ in range(trace_count)]
    events = np.full(size, -777, dtype=np.int64)
    finals = [np.full(1, -777.0) for _ in range(trace_count)]
    addresses = [array.ctypes.data for array in [drive, *traces, events, *finals]]
    if null_buffer is not None:
        addresses[null_buffer] = 0
    pointers = (
        [ctypes.c_void_p(address) for address in addresses]
        if backend == "go"
        else [ctypes.c_ssize_t(address) for address in addresses]
    )
    steps = ctypes.c_int(count) if backend == "go" else ctypes.c_ssize_t(count)
    status = function(steps, *(ctypes.c_double(v) for v in configuration.values()), *pointers)
    return status, drive, traces, events, finals


def _assert_refused_outputs(result: CallResult, status: int) -> None:
    """Require refusal to preserve every caller-owned trace and final buffer."""
    code, _, traces, events, finals = result
    assert code == status
    for array in [*traces, *finals]:
        np.testing.assert_array_equal(array, np.full(array.size, -777.0))
    np.testing.assert_array_equal(events, np.full(events.size, -777))


@pytest.mark.parametrize(("model", "parameters"), INVALID_CONFIGURATIONS)
@pytest.mark.parametrize("count", [0, 1])
@pytest.mark.parametrize("backend", ["go", "mojo"])
def test_complete_configuration_refuses_before_output(
    model: str, parameters: dict[str, float], count: int, backend: str
) -> None:
    """Invalid complete profiles cannot pass zero-step admission or poison recovery."""
    result = _call(model, backend, parameters, count)
    _assert_refused_outputs(result, 2)
    np.testing.assert_array_equal(result[1], np.zeros(max(count, 1)))
    assert _call(model, backend, {}, 1)[0] == 0


@pytest.mark.parametrize("model", DEFAULTS)
@pytest.mark.parametrize("backend", ["go", "mojo"])
def test_empty_batch_writes_only_exact_initial_finals(model: str, backend: str) -> None:
    """A valid empty call retains trace sentinels and copies all initial dynamics."""
    status, _, traces, events, finals = _call(model, backend, {}, 0)
    assert status == 0
    initial = list(DEFAULTS[model].values())[: len(finals)]
    assert [float(array[0]) for array in finals] == initial
    for array in traces:
        assert array[0] == -777.0
    assert events[0] == -777


_POINTER_CASES = [
    (model, pointer) for model in DEFAULTS for pointer in range(10 if model == "mat" else 8)
]


@pytest.mark.parametrize(("model", "pointer"), _POINTER_CASES)
@pytest.mark.parametrize("count", [0, 1])
@pytest.mark.parametrize("backend", ["go", "mojo"])
def test_null_pointer_refusal_is_crash_fenced(
    model: str, pointer: int, count: int, backend: str, tmp_path: Path
) -> None:
    """Every absent pointer refuses in a finite child with preserved raw evidence."""
    code = (
        "import json,resource,sys; from pathlib import Path; "
        "resource.setrlimit(resource.RLIMIT_CORE,(0,0)); "
        "repo=Path(sys.argv[1]); sys.path[:0]=json.loads(sys.argv[6]); "
        "from tests.test_mat_family_c_abi import _call,_assert_refused_outputs; "
        "_assert_refused_outputs(_call(sys.argv[2],sys.argv[3],{},int(sys.argv[4]),"
        "null_buffer=int(sys.argv[5])),1); print('null pointer refused without writes')"
    )
    with (
        (tmp_path / "stdout.log").open("wb") as stdout,
        (tmp_path / "stderr.log").open("wb") as stderr,
    ):
        child = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                code,
                str(_REPOSITORY),
                model,
                backend,
                str(count),
                str(pointer),
                json.dumps(sys.path),
            ],
            stdout=stdout,
            stderr=stderr,
            check=False,
            timeout=20,
        )
    assert child.returncode == 0, (tmp_path / "stderr.log").read_text()


@pytest.mark.parametrize("model", DEFAULTS)
@pytest.mark.parametrize("backend", ["go", "mojo"])
def test_negative_length_refuses_all_writes(model: str, backend: str) -> None:
    """A negative count is rejected before creating any buffer view."""
    _assert_refused_outputs(_call(model, backend, {}, -1), 1)


@pytest.mark.parametrize("model", DEFAULTS)
@pytest.mark.parametrize("backend", ["go", "mojo"])
@pytest.mark.parametrize("current", [float("nan"), float("inf"), -float("inf"), 1e308])
def test_first_transition_refusal_preserves_all_outputs(
    model: str, backend: str, current: float
) -> None:
    """Nonfinite and finite-overflow first samples leave every output untouched."""
    _assert_refused_outputs(_call(model, backend, {}, 1, current), 2)


@pytest.mark.parametrize("model", DEFAULTS)
@pytest.mark.parametrize("backend", ["go", "mojo"])
@pytest.mark.parametrize("current", [float("nan"), float("inf"), -float("inf"), 1e308])
def test_late_refusal_retains_only_the_valid_prefix(
    model: str, backend: str, current: float
) -> None:
    """Raw C callers receive the valid prefix and no uncommitted suffix or finals."""
    drive = np.array([0.0, current, 0.0])
    status, actual_drive, traces, events, finals = _call(model, backend, {}, 3, drive)
    reference = MATNeuron() if model == "mat" else SCResettingMATNeuron()
    event = reference.step(0.0)
    assert status == 2 and events[0] == event
    names = list(DEFAULTS[model])[: len(traces)]
    np.testing.assert_allclose(
        [trace[0] for trace in traces],
        [getattr(reference, name) for name in names],
        rtol=0,
        atol=2e-12,
    )
    for trace in traces:
        np.testing.assert_array_equal(trace[1:], [-777.0, -777.0])
    np.testing.assert_array_equal(events[1:], [-777, -777])
    for final in finals:
        assert final[0] == -777.0
    np.testing.assert_array_equal(actual_drive, drive)


@pytest.mark.parametrize("backend", ["go", "mojo"])
@pytest.mark.parametrize(
    ("model", "parameters", "current"),
    [
        ("mat", {"theta1": 1e9, "tau_1": 1e308, "omega": -1e9}, 0.0),
        ("mat", {"theta2": 1e9, "tau_2": 1e308, "omega": -1e9}, 0.0),
        ("mat", {"v": -200.0, "dt": 10.0}, 1.0),
        ("sc_resetting_mat", {"theta1": 1e9, "tau_1": 1e308, "v_threshold_base": -1e308}, 0.0),
        ("sc_resetting_mat", {"theta2": 1e9, "tau_2": 1e308, "v_threshold_base": -1e308}, 0.0),
        ("sc_resetting_mat", {"theta1": 1.0, "tau_1": 0.001}, 0.0),
        ("sc_resetting_mat", {"theta2": 1.0, "tau_2": 0.001}, 0.0),
        ("sc_resetting_mat", {"v": 100.0, "v_rest": 5000.0}, 0.0),
    ],
)
def test_event_and_candidate_envelopes_refuse_before_first_write(
    model: str, parameters: dict[str, float], current: float, backend: str
) -> None:
    """Admitted profiles cannot commit invalid RK4, voltage or post-event candidates."""
    _assert_refused_outputs(_call(model, backend, parameters, 1, current), 2)
    assert _call(model, backend, {}, 1)[0] == 0


@pytest.mark.parametrize("backend", ["go", "mojo"])
def test_raw_finite_cancellation_preserves_all_states_and_events(backend: str) -> None:
    """The complete raw ABI retains the source derivative's finite cancellation."""
    status, drive, traces, events, finals = _call(
        "sc_resetting_mat", backend, {"v_rest": 1e308, "resistance": 1e308}, 4, -1.0
    )
    assert status == 0
    np.testing.assert_array_equal(drive, np.full(4, -1.0))
    np.testing.assert_array_equal(traces[0], np.full(4, -70.0))
    for trace in traces[1:]:
        np.testing.assert_array_equal(trace, np.zeros(4))
    np.testing.assert_array_equal(events, np.zeros(4, dtype=np.int64))
    assert [float(final[0]) for final in finals] == [-70.0, 0.0, 0.0]
