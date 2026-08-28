# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Izhikevich-2007 dispatch and validation tests

"""Dispatch, input validation, and algorithm selection contracts."""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from sc_neurocore.neurons.models.izhikevich2007 import Izhikevich2007Neuron
from sc_neurocore.neurons.models import izhikevich2007 as izh
from tests.izhikevich2007_backends_support import _go, _julia, _mojo, _run

_ROOT = Path(__file__).resolve().parents[1]


def test_auto_matches_python_bit_exact() -> None:
    ref, ref_spikes, _rv, _ru = _run("python")
    got, spikes, _vf, _uf = _run("auto")
    np.testing.assert_array_equal(got, ref)
    assert spikes == ref_spikes


def test_invalid_backend_raises() -> None:
    with pytest.raises(ValueError, match="backend must be"):
        Izhikevich2007Neuron().simulate(10, 0.0, backend="cuda")


def test_negative_n_steps_raises() -> None:
    with pytest.raises(ValueError, match="n_steps must be between"):
        Izhikevich2007Neuron().simulate(-1, 0.0)


@pytest.mark.parametrize("n_steps", (True, 1.5, 1 << 31))
def test_invalid_step_count_raises(n_steps: object) -> None:
    with pytest.raises(ValueError, match="n_steps"):
        Izhikevich2007Neuron().simulate(n_steps, 0.0)  # type: ignore[arg-type]


def test_non_finite_current_raises() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        Izhikevich2007Neuron().simulate(10, np.inf)


def test_non_rk4_integrator_rejected() -> None:
    with pytest.raises(ValueError, match="RK4 integrator only"):
        Izhikevich2007Neuron(integrator="euler").simulate(10, 300.0)


def test_mutated_configuration_and_overflow_are_atomic() -> None:
    neuron = Izhikevich2007Neuron()
    neuron.C = 0.0
    before = (neuron.v, neuron.u)
    with pytest.raises(ValueError, match="C must be positive"):
        neuron.simulate(1, 300.0, backend="python")
    assert (neuron.v, neuron.u) == before

    overflow = Izhikevich2007Neuron()
    overflow.v = 1.0e200
    before = (overflow.v, overflow.u)
    with pytest.raises(FloatingPointError, match="candidate state"):
        overflow.step(0.0)
    assert (overflow.v, overflow.u) == before


def test_non_finite_reset_candidate_preserves_state() -> None:
    neuron = Izhikevich2007Neuron()
    neuron.v, neuron.u = -55.0, 7.0
    before = (neuron.v, neuron.u)
    neuron.v0 = 1.0e308
    neuron.vr = -1.0e308
    neuron.b = 1.0e308

    with pytest.raises(FloatingPointError, match="reset state became non-finite"):
        neuron.reset_state()

    assert (neuron.v, neuron.u) == before


def test_go_and_mojo_abis_reject_invalid_input_without_writing() -> None:
    for backend, available, loader in (("go", _go, "_go_lib"), ("mojo", _mojo, "_mojo_lib")):
        if not available():
            continue
        trace = np.full(3, 123.0, dtype=np.float64)
        args = (-60.0, 0.0, 100.0, 0.7, -60.0, -40.0, 35.0, 0.03, -2.0, -50.0, 100.0, 0.1)
        library = getattr(izh, loader)
        if backend == "go":
            result = library.izhikevich2007_simulate_c(
                *(ctypes.c_double(value) for value in args),
                ctypes.c_int(1),
                ctypes.c_double(math.nan),
                trace.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            )
        else:
            result = library.izhikevich2007_simulate_c(*args, 1, math.nan, int(trace.ctypes.data))
        assert result == -1
        np.testing.assert_array_equal(trace, np.full(3, 123.0))


def test_julia_native_api_rejects_invalid_input() -> None:
    if not _julia():
        pytest.skip("Julia Izhikevich 2007 backend unavailable")
    juliacall = pytest.importorskip("juliacall")
    with pytest.raises(juliacall.JuliaError, match="inputs must be finite"):
        izh._julia_module.simulate_trace(
            -60.0,
            0.0,
            100.0,
            0.7,
            -60.0,
            -40.0,
            35.0,
            0.03,
            -2.0,
            -50.0,
            100.0,
            0.1,
            1,
            math.nan,
        )


def test_committed_benchmark_has_complete_backend_and_source_custody() -> None:
    report = json.loads(
        (_ROOT / "benchmarks/results/bench_izhikevich2007_simulate.json").read_text()
    )
    rows = report["backend_summary"]
    assert set(rows) == {"python", "rust", "julia", "go", "mojo"}
    assert all(not row.get("skipped", False) for row in rows.values())

    reference = rows["python"]
    for backend in ("rust", "julia", "go"):
        row = rows[backend]
        assert row["spikes"] == reference["spikes"]
        assert row["v_final"] == reference["v_final"]
        assert row["u_final"] == reference["u_final"]
        assert row["trace_sha256"] == reference["trace_sha256"]
    assert rows["mojo"]["spikes"] == reference["spikes"]
    assert rows["mojo"]["parity_max_abs_diff"] <= 1.0e-6

    for relative in (
        "benchmarks/bench_izhikevich2007_simulate.py",
        "src/sc_neurocore/neurons/models/izhikevich2007.py",
        "engine/src/rk4_neurons.rs",
        "engine/src/bindings/izhikevich2007.rs",
        "src/sc_neurocore/accel/rust/safety/izhikevich2007.rs",
        "src/sc_neurocore/accel/go/neurons/izhikevich2007/izhikevich2007.go",
        "src/sc_neurocore/accel/julia/neurons/izhikevich2007.jl",
        "src/sc_neurocore/accel/mojo/neurons/izhikevich2007.mojo",
        "src/sc_neurocore/neurons/model_schemas/izhikevich2007.toml",
        "src/sc_neurocore/neurons/model_schemas/izhikevich2007.json",
        "src/sc_neurocore/neurons/reference_receipts/izhikevich_2007.json",
        "src/sc_neurocore/neurons/reference_receipts/izhikevich_2007_rk4.json",
    ):
        expected = hashlib.sha256((_ROOT / relative).read_bytes()).hexdigest()
        assert report["source_hashes"][relative] == expected
