# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

"""Public dispatch, failure, and consolidated custody tests for the SC network."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, cast

import pytest

from sc_neurocore.network import (
    SCCompteWMBackend,
    SCCompteWMBackendUnavailable,
    SCCompteWMNetworkSpec,
    SCCompteWMStimulus,
    run_sc_compte_wm_network,
    sc_compte_wm_backend_status,
)

REPOSITORY = Path(__file__).resolve().parents[1]
BACKENDS: tuple[SCCompteWMBackend, ...] = ("python", "rust", "julia", "go", "mojo")
DISPATCHER = REPOSITORY / "src/sc_neurocore/network/sc_compte_wm_backends.py"
RUST_RUNNER = REPOSITORY / "engine/examples/sc_compte_wm_network_run.rs"
RUST_SOURCE = REPOSITORY / "engine/src/sc_compte_wm_network.rs"
JULIA_RUNNER = (
    REPOSITORY / "src/sc_neurocore/accel/julia/sc_compte_wm_network/run_sc_compte_wm_network.jl"
)
GO_RUNNER = REPOSITORY / "src/sc_neurocore/accel/go/cmd/run_sc_compte_wm_network/main.go"
BENCHMARK = REPOSITORY / "benchmarks/bench_sc_compte_wm_network_all_runtimes.py"
RESULT = REPOSITORY / "benchmarks/results/bench_sc_compte_wm_network_all_runtimes.json"
EXPECTED_INPUT = "9e0a75fc79cbf1d985d3d58664116c0aa2fa0adda85a317a625a54176051cfd6"
EXPECTED_SPIKES = "59ec91dcb7dc65b5f928091cb0e25c26729a0a4453ebe7d8244fc1ceae7d9712"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_public_status_reports_every_runtime_active_without_auto_fallback() -> None:
    statuses = sc_compte_wm_backend_status()
    assert tuple(status.backend for status in statuses) == BACKENDS
    assert all(status.available and status.reason is None for status in statuses)
    assert {status.execution_mode for status in statuses} == {
        "in-process",
        "repository-native-command",
        "in-process-shared-library",
    }


@pytest.mark.parametrize("backend", BACKENDS)
def test_explicit_backends_execute_exact_five_step_custody(backend: SCCompteWMBackend) -> None:
    result = run_sc_compte_wm_network(
        0.1, backend=backend, statistics_window_ms=0.1, timeout_s=600.0
    )
    assert result.backend == backend
    assert result.execution_ns > 0
    assert result.receipt.steps == 5
    assert result.receipt.input_sha256 == EXPECTED_INPUT
    assert result.receipt.spike_sha256 == EXPECTED_SPIKES
    assert result.receipt.excitatory_spikes == result.receipt.inhibitory_spikes == 0
    assert len(result.receipt.windows) == 1


@pytest.mark.parametrize("backend", BACKENDS)
def test_explicit_backends_execute_full_population_stimulus(
    backend: SCCompteWMBackend,
) -> None:
    stimulus = SCCompteWMStimulus(0.0, 0.02, 600_000.0, kind="global_current", center_deg=None)
    result = run_sc_compte_wm_network(
        0.02,
        backend=backend,
        stimuli=(stimulus,),
        statistics_window_ms=0.02,
        timeout_s=600.0,
    )
    assert result.receipt.excitatory_spikes == 2048
    assert result.receipt.inhibitory_spikes == 0
    assert result.receipt.windows[0].statistics is not None


def test_native_dispatch_rejects_unrepresented_spec_and_never_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="fix these specification fields"):
        run_sc_compte_wm_network(
            0.02,
            backend="rust",
            spec=SCCompteWMNetworkSpec(dt_ms=0.01),
            statistics_window_ms=0.02,
        )

    def fail(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], 17, stdout="", stderr="selected runtime failed")

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(SCCompteWMBackendUnavailable, match="exited 17"):
        run_sc_compte_wm_network(0.02, backend="go", statistics_window_ms=0.02, timeout_s=1.0)
    with pytest.raises(ValueError, match="unknown"):
        run_sc_compte_wm_network(
            0.02,
            backend=cast(SCCompteWMBackend, "auto"),
            statistics_window_ms=0.02,
        )


def test_native_adapters_and_public_dispatch_are_documented() -> None:
    assert "without fallback" in DISPATCHER.read_text(encoding="utf-8")
    assert "JSON command adapter" in RUST_RUNNER.read_text(encoding="utf-8")
    assert "JSON command adapter" in JULIA_RUNNER.read_text(encoding="utf-8")
    assert "JSON adapter for public Go dispatch" in GO_RUNNER.read_text(encoding="utf-8")
    rust = RUST_SOURCE.read_text(encoding="utf-8")
    for symbol in (
        "pub enum SCCompteWMStimulusKind",
        "pub struct SCCompteWMStimulus",
        "pub struct SCCompteWMActivityStatistics",
        "pub struct SCCompteWMWindowReceipt",
        "pub struct SCCompteWMRunReceipt",
        "pub fn run(",
        "pub fn state_sha256(",
    ):
        offset = rust.index(symbol)
        assert "///" in rust[max(0, offset - 500) : offset]


def test_consolidated_benchmark_is_source_bound_and_five_runtime_exact() -> None:
    payload = json.loads(RESULT.read_text(encoding="utf-8"))
    assert payload["model"] == "SC-COMPTE-WM-NETWORK"
    assert payload["configuration"]["cells"] == 2560
    assert payload["configuration"]["steps"] == 1000
    assert payload["configuration"]["repeats"] == 3
    assert tuple(payload["backends"]) == BACKENDS
    assert payload["all_backends_available"] is True
    assert payload["all_runtime_input_spike_count_exact"] is True
    assert payload["all_repeat_receipts_exact"] is True
    assert payload["passed"] is True
    assert payload["persistent_bump_claimed"] is False
    assert payload["distractor_resistance_claimed"] is False
    for backend in BACKENDS:
        assert payload["backends"][backend]["repeat_receipts_exact"] is True
    for path in (
        DISPATCHER,
        RUST_RUNNER,
        JULIA_RUNNER,
        GO_RUNNER,
        BENCHMARK,
        REPOSITORY / "src/sc_neurocore/accel/mojo/sc_compte_wm_network/libsc_compte_wm_network.so",
    ):
        relative = path.relative_to(REPOSITORY).as_posix()
        assert payload["source_sha256"][relative] == _sha256(path)
