# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Executed Wilson-Cowan five-runtime parity

"""Exercise configured traces, public dispatch, and failure atomicity."""

from __future__ import annotations

import ctypes
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pytest

from sc_neurocore.accel import wilson_cowan as backends
from sc_neurocore.neurons.models.wilson_cowan import WilsonCowanUnit

_REPOSITORY = Path(__file__).resolve().parents[1]
_COMPILED_BACKENDS = ("rust", "julia", "go", "mojo")
_CONFIG = {
    "e": 0.24,
    "i": 0.11,
    "w_ee": 10.0,
    "w_ei": 6.0,
    "w_ie": 10.0,
    "w_ii": 1.0,
    "tau_e": 1.0,
    "tau_i": 2.0,
    "a": 1.2,
    "theta": 4.0,
    "dt": 0.1,
}


@pytest.mark.parametrize("backend", ("python", *_COMPILED_BACKENDS))
def test_complete_configured_contract_matches_python(backend: str) -> None:
    tolerance = backends.PARITY_ATOL[backend]
    reference_unit = WilsonCowanUnit(**_CONFIG)
    reference_e, reference_i = reference_unit.simulate(128, 3.0, backend="python")
    actual_unit = WilsonCowanUnit(**_CONFIG)
    actual_e, actual_i = actual_unit.simulate(128, 3.0, backend=backend)
    np.testing.assert_allclose(actual_e, reference_e, rtol=0.0, atol=tolerance)
    np.testing.assert_allclose(actual_i, reference_i, rtol=0.0, atol=tolerance)
    assert actual_unit.e == pytest.approx(reference_unit.e, abs=tolerance)
    assert actual_unit.i == pytest.approx(reference_unit.i, abs=tolerance)


@pytest.mark.parametrize("backend", ("python", *_COMPILED_BACKENDS))
def test_empty_batch_preserves_both_rates(backend: str) -> None:
    unit = WilsonCowanUnit(**_CONFIG)
    e_trace, i_trace = unit.simulate(0, 3.0, backend=backend)
    np.testing.assert_array_equal(e_trace, np.empty(0))
    np.testing.assert_array_equal(i_trace, np.empty(0))
    assert (unit.e, unit.i) == (_CONFIG["e"], _CONFIG["i"])


@pytest.mark.parametrize("backend", ("python", *_COMPILED_BACKENDS))
def test_saturated_initial_boundary_matches_python(backend: str) -> None:
    tolerance = backends.PARITY_ATOL[backend]
    reference = WilsonCowanUnit(e=1.0, i=0.5)
    reference_e, reference_i = reference.simulate(200, 2.0, backend="python")
    actual = WilsonCowanUnit(e=1.0, i=0.5)
    actual_e, actual_i = actual.simulate(200, 2.0, backend=backend)

    np.testing.assert_allclose(actual_e, reference_e, rtol=0.0, atol=tolerance)
    np.testing.assert_allclose(actual_i, reference_i, rtol=0.0, atol=tolerance)


def test_auto_executes_a_maintained_lane() -> None:
    unit = WilsonCowanUnit(**_CONFIG)
    e_trace, i_trace = unit.simulate(4, 3.0)
    assert e_trace.shape == i_trace.shape == (4,)
    assert (unit.e, unit.i) == (e_trace[-1], i_trace[-1])


@pytest.mark.parametrize("backend", _COMPILED_BACKENDS)
def test_requested_backend_reports_unavailable(
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(backends, "backend_available", lambda name: name != backend)
    unit = WilsonCowanUnit(**_CONFIG)
    before = (unit.e, unit.i)
    with pytest.raises(RuntimeError, match=backend.title()):
        unit.simulate(1, 3.0, backend=backend)
    assert (unit.e, unit.i) == before


@pytest.mark.parametrize(
    ("e_trace", "i_trace", "e_final", "i_final", "message"),
    (
        ([[0.2]], [0.1], 0.2, 0.1, "malformed E trace"),
        ([0.2], [[0.1]], 0.2, 0.1, "malformed I trace"),
        ([float("nan")], [0.1], 0.2, 0.1, "non-finite rates"),
        ([2.0], [0.1], 2.0, 0.1, "out-of-range E"),
        ([0.2], [0.1], 0.5, 0.1, "final E rate disagrees"),
        ([0.2], [0.1], 0.2, 0.5, "final I rate disagrees"),
    ),
)
def test_normaliser_rejects_malformed_results(
    e_trace: object,
    i_trace: object,
    e_final: object,
    i_final: object,
    message: str,
) -> None:
    with pytest.raises(FloatingPointError, match=message):
        backends.normalise_result(
            e_trace,
            i_trace,
            e_final,
            i_final,
            n_steps=1,
            initial_e=0.24,
            initial_i=0.11,
            a=1.2,
            theta=4.0,
        )


def test_malformed_backend_result_preserves_public_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unit = WilsonCowanUnit(**_CONFIG)
    before = (unit.e, unit.i)
    malformed = (np.array([0.2]), np.array([0.1]), 0.6, 0.1)
    monkeypatch.setattr(backends, "backend_available", lambda _backend: True)
    monkeypatch.setattr(backends, "simulate_go", lambda *_args: malformed)
    with pytest.raises(FloatingPointError, match="final E rate disagrees"):
        unit.simulate(1, 3.0, backend="go")
    assert (unit.e, unit.i) == before


@pytest.mark.parametrize("backend", ("go", "mojo"))
def test_c_abi_invalid_contract_writes_nothing(backend: str) -> None:
    module = __import__(
        f"sc_neurocore.accel.{backend}.wilson_cowan",
        fromlist=["wilson_cowan"],
    )
    assert bool(getattr(module, f"_HAS_{backend.upper()}_WILSON_COWAN"))
    library: Any = module._lib
    ext = np.full(2, 3.0, dtype=np.float64)
    e_out = np.full(2, -999.0, dtype=np.float64)
    i_out = np.full(2, -999.0, dtype=np.float64)
    e_final = np.full(1, -999.0, dtype=np.float64)
    i_final = np.full(1, -999.0, dtype=np.float64)
    if backend == "go":
        final_e_arg: Any = e_final.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        final_i_arg: Any = i_final.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    else:
        final_e_arg = e_final.ctypes.data
        final_i_arg = i_final.ctypes.data
    status = library.wilson_cowan_simulate_c(
        2,
        0.24,
        0.11,
        10.0,
        6.0,
        10.0,
        1.0,
        1.0,
        2.0,
        1.2,
        4.0,
        -0.1,
        ext.ctypes.data,
        e_out.ctypes.data,
        i_out.ctypes.data,
        final_e_arg,
        final_i_arg,
    )
    assert status == -1
    np.testing.assert_array_equal(e_out, np.full(2, -999.0))
    np.testing.assert_array_equal(i_out, np.full(2, -999.0))
    np.testing.assert_array_equal(e_final, np.full(1, -999.0))
    np.testing.assert_array_equal(i_final, np.full(1, -999.0))


@pytest.mark.parametrize("backend", ("go", "mojo"))
def test_c_abi_empty_batch_accepts_null_trace_buffers(backend: str) -> None:
    module = __import__(
        f"sc_neurocore.accel.{backend}.wilson_cowan",
        fromlist=["wilson_cowan"],
    )
    assert bool(getattr(module, f"_HAS_{backend.upper()}_WILSON_COWAN"))
    library: Any = module._lib
    e_final = np.full(1, -999.0, dtype=np.float64)
    i_final = np.full(1, -999.0, dtype=np.float64)
    if backend == "go":
        final_e_arg: Any = e_final.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        final_i_arg: Any = i_final.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    else:
        final_e_arg = e_final.ctypes.data
        final_i_arg = i_final.ctypes.data
    status = library.wilson_cowan_simulate_c(
        0,
        0.24,
        0.11,
        10.0,
        6.0,
        10.0,
        1.0,
        1.0,
        2.0,
        1.2,
        4.0,
        0.1,
        None,
        None,
        None,
        final_e_arg,
        final_i_arg,
    )
    assert status == 0
    np.testing.assert_array_equal(e_final, np.array([0.24]))
    np.testing.assert_array_equal(i_final, np.array([0.11]))


def test_standalone_rust_safety_matches_python_trace(tmp_path: Path) -> None:
    source = _REPOSITORY / "src/sc_neurocore/accel/rust/safety/wilson_cowan.rs"
    program = tmp_path / "wilson_cowan_trace.rs"
    binary = tmp_path / "wilson_cowan_trace"
    program.write_text(
        f'''include!(r#"{source}"#);

fn main() {{
    let mut state = WilsonCowanUnit::new();
    for _ in 0..6 {{
        state.step(3.0).expect("valid Wilson-Cowan input");
        println!("{{:.17}} {{:.17}}", state.e, state.i);
    }}
}}
''',
        encoding="utf-8",
    )
    subprocess.run(
        ["rustc", "--edition", "2021", "-O", str(program), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    completed = subprocess.run(
        [str(binary)], check=True, capture_output=True, text=True, timeout=30
    )
    actual = np.asarray(
        [[float(value) for value in line.split()] for line in completed.stdout.splitlines()]
    )
    reference = WilsonCowanUnit()
    reference_e, reference_i = reference.simulate(6, 3.0, backend="python")
    np.testing.assert_allclose(actual[:, 0], reference_e, rtol=0.0, atol=1.0e-15)
    np.testing.assert_allclose(actual[:, 1], reference_i, rtol=0.0, atol=1.0e-15)
