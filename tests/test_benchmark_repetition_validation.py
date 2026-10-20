# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Retained benchmark repetition contracts

"""Exercise repetition retention through the maintained Python MAT dispatcher."""

from collections.abc import Callable, Mapping

import json
import os
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from benchmarks import (
    bench_model_mat,
    bench_model_non_resetting_lif,
    bench_model_sc_non_resetting_adaptive_lif,
    bench_model_sc_resetting_mat,
)
from benchmarks._repeated_measurements import measure_repetitions, summarize_repetitions
from sc_neurocore.accel.mat import simulate_mat


def test_all_repetitions_preserve_complete_mat_trajectories() -> None:
    """Keep all five full trajectories and events beside their measured samples."""
    currents = np.full(128, 0.7, dtype=np.float64)
    original = currents.copy()
    samples, results = measure_repetitions(simulate_mat, currents, backend="python", repeats=5)
    control = simulate_mat(currents, backend="python")
    assert len(samples) == len(results) == 5
    assert all(sample > 0 for sample in samples)
    for result in results:
        for key in ("voltages", "theta1", "theta2", "refractory", "events"):
            np.testing.assert_array_equal(result[key], control[key])
        for key in ("v_final", "theta1_final", "theta2_final", "refractory_final"):
            assert result[key] == control[key]
    np.testing.assert_array_equal(currents, original)


def test_runtime_owned_arrays_cannot_rewrite_retained_repetitions() -> None:
    """Preserve measured trajectories after the dispatcher owner changes its arrays."""
    returned: list[Mapping[str, object]] = []

    def recording_dispatcher(
        currents: npt.NDArray[np.float64], *, backend: str
    ) -> Mapping[str, object]:
        """Retain the actual MAT result so its owner can later change its arrays."""
        result = simulate_mat(currents, backend=backend)
        returned.append(result)
        return result

    currents = np.full(64, 0.7, dtype=np.float64)
    samples, results = measure_repetitions(
        recording_dispatcher, currents, backend="python", repeats=5
    )
    assert len(returned) == 6
    assert len(samples) == len(results) == 5
    control = simulate_mat(currents, backend="python")
    for raw in returned:
        for value in raw.values():
            if isinstance(value, np.ndarray):
                value.fill(0)
    for result in results:
        assert all(result is not raw for raw in returned)
        for key in ("voltages", "theta1", "theta2", "refractory", "events"):
            np.testing.assert_array_equal(result[key], control[key])
            for raw in returned:
                assert not np.shares_memory(result[key], raw[key])


@pytest.mark.parametrize("repeats", [0, -1])
def test_nonpositive_repetitions_refuse_before_dispatch(repeats: int) -> None:
    """Refuse an invalid count before an independently invalid current can run."""
    with pytest.raises(ValueError, match="^repeats must be positive$"):
        measure_repetitions(simulate_mat, np.array([np.nan]), backend="python", repeats=repeats)


def test_dispatcher_error_is_not_replaced_by_success() -> None:
    """Propagate a real dispatcher refusal rather than returning timing evidence."""
    with pytest.raises(ValueError, match="unknown MAT backend"):
        measure_repetitions(
            simulate_mat, np.full(64, 0.7), backend="unavailable-runtime", repeats=5
        )


@pytest.mark.parametrize(
    "run",
    [
        bench_model_mat.main,
        bench_model_sc_resetting_mat.main,
        bench_model_non_resetting_lif.main,
        bench_model_sc_non_resetting_adaptive_lif.main,
    ],
)
def test_model_cli_retains_every_runtime_repetition(
    run: Callable[[list[str] | None], int], tmp_path: Path
) -> None:
    """Run real dispatchers and bind each sample to a complete parity verdict."""
    output = tmp_path / "measurement.json"
    status = run(["--steps", "64", "--repeats", "3", "--output", str(output)])
    if status == 2:
        assert not output.exists()
        assert os.environ.get("SC_NEUROCORE_REQUIRE_ALL_BENCHMARK_RUNTIMES") != "1"
        return
    assert status == 0
    payload = json.loads(output.read_text())
    assert payload["steps"] == 64
    assert payload["repeats"] == 3
    assert payload["production_speed_claim"] is False
    assert payload["evidence_class"] == "local_regression_non_isolated"
    assert set(payload["backend_summary"]) == {"python", "rust", "julia", "go", "mojo"}
    for row in payload["backend_summary"].values():
        assert len(row["repetitions"]) == len(row["samples_ns"]) == 3
        assert row["trace_matches_python"]
        assert row["event_vector_matches_python"]
        assert row["parity_max_abs_diff"] == max(
            check["parity_max_abs_diff"] for check in row["repetitions"]
        )
        for sample, check in zip(row["samples_ns"], row["repetitions"], strict=True):
            assert sample > 0
            assert check["samples_ns"] == [sample]
            assert check["median_ns_per_step"] == sample / 64
            assert check["trace_matches_python"]
            assert check["event_vector_matches_python"]
            assert len(check["trace_sha256"]) == 64
            assert np.isfinite(check["final_state"]).all()
        assert row["trace_sha256"] == row["repetitions"][-1]["trace_sha256"]
        assert row["final_state"] == row["repetitions"][-1]["final_state"]
    versions = payload["tool_versions"]
    for tool in ("rustc", "go", "go_cli", "julia", "julia_cli", "mojo"):
        assert versions[tool] and versions[tool] != "unavailable"
        assert not versions[tool].startswith("exit ")
    assert versions["go"].startswith("go1.")
    assert versions["julia"].startswith("julia version ")
    context = payload["measurement_context"]
    assert context["host"]
    assert context["architecture"]
    assert context["cpu_model"]
    assert context["logical_cpu_count"] == os.cpu_count()
    assert context["python_version"]
    assert context["numpy_version"] == np.__version__
    if hasattr(os, "sysconf"):
        assert context["ram_bytes"] == os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    assert "load_average_before" in context
    assert "load_average_after" in context
    for path in ("benchmarks/_repeated_measurements.py", "benchmarks/_benchmark_context.py"):
        assert len(payload["source_hashes"][path]) == 64


@pytest.mark.parametrize("fault", ["state", "event", "nan", "shape"])
def test_model_cli_refuses_a_bad_middle_repetition(
    fault: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject an actual MAT run with a corrupted intermediate runtime result."""
    import sc_neurocore.accel.mat as mat_backend

    original = mat_backend.simulate_mat
    counts: dict[str, int] = {}
    observed: list[Mapping[str, object]] = []

    def observed_dispatcher(
        currents: npt.NDArray[np.float64], *, backend: str
    ) -> Mapping[str, object]:
        """Run the real backend and alter only its second measured trajectory."""
        result = original(currents, backend=backend)
        counts[backend] = counts.get(backend, 0) + 1
        if backend == "rust":
            observed.append(result)
            if counts[backend] == 3:
                if fault == "event":
                    events = np.asarray(result["events"])
                    events[0] = 1 - events[0]
                elif fault == "shape":
                    result["theta1"] = np.asarray(result["theta1"])[:1]
                else:
                    np.asarray(result["theta1"])[0] = np.nan if fault == "nan" else 1e12
        return result

    monkeypatch.setattr(mat_backend, "simulate_mat", observed_dispatcher)
    output = tmp_path / "invalid-measurement.json"
    status = bench_model_mat.main(["--steps", "64", "--repeats", "3", "--output", str(output)])
    assert not output.exists()
    if status == 2:
        assert os.environ.get("SC_NEUROCORE_REQUIRE_ALL_BENCHMARK_RUNTIMES") != "1"
        return
    assert status == 3
    assert counts == dict.fromkeys(("python", "rust", "julia", "go", "mojo"), 4)
    assert len(observed) == 4
    healthy = original(np.full(64, 0.7), backend="rust")
    for key in ("voltages", "theta1", "theta2", "refractory", "events"):
        np.testing.assert_array_equal(observed[-1][key], healthy[key])


@pytest.mark.parametrize("remove_sample", [False, True])
def test_summary_refuses_missing_trajectory_evidence(remove_sample: bool) -> None:
    """Refuse empty or unequal evidence produced from real retained MAT runs."""
    samples, results = measure_repetitions(
        simulate_mat, np.full(64, 0.7), backend="python", repeats=3
    )
    if remove_sample:
        samples.pop()
    else:
        samples.clear()
        results.clear()
    with pytest.raises(ValueError, match="each timed repetition requires one retained trajectory"):
        summarize_repetitions(
            samples,
            results,
            lambda _samples, _result: pytest.fail("invalid evidence reached model comparison"),
        )
