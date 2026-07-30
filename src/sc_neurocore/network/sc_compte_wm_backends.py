# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — explicit public backend dispatch for the SC Compte network

"""Select complete ``SC-COMPTE-WM-NETWORK`` runtimes without fallback.

Python and Mojo execute in-process. Rust, Julia, and Go execute documented
repository-native JSON adapters with every dependency/cache rooted in
``.venv``. A requested unavailable or failing backend raises; it is never
replaced by another recurrence. Short dispatch receipts are execution custody,
not persistent-working-memory evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Literal, cast

from .sc_compte_wm import SCCompteWMActivityStatistics, SCCompteWMNetworkSpec
from .sc_compte_wm_network import (
    SCCompteWMNetwork,
    SCCompteWMRunReceipt,
    SCCompteWMStimulus,
    SCCompteWMWindowReceipt,
)

SCCompteWMBackend = Literal["python", "rust", "julia", "go", "mojo"]
_BACKENDS: tuple[SCCompteWMBackend, ...] = ("python", "rust", "julia", "go", "mojo")
_REPOSITORY = Path(__file__).resolve().parents[3]
_VENV = _REPOSITORY / ".venv"
_RUST_MANIFEST = _REPOSITORY / "engine/Cargo.toml"
_RUST_RUNNER = _REPOSITORY / "engine/examples/sc_compte_wm_network_run.rs"
_JULIA_ROOT = _REPOSITORY / "src/sc_neurocore/accel/julia/sc_compte_wm_network"
_JULIA_RUNNER = _JULIA_ROOT / "run_sc_compte_wm_network.jl"
_GO_ROOT = _REPOSITORY / "src/sc_neurocore/accel/go"
_GO_RUNNER = _GO_ROOT / "cmd/run_sc_compte_wm_network/main.go"
_NATIVE_FIXED_FIELDS = (
    "n_excitatory",
    "n_inhibitory",
    "dt_ms",
    "external_rate_hz",
    "external_exc_conductance_ns",
    "external_inh_conductance_ns",
    "recurrent_ee_conductance_ns",
    "recurrent_ei_conductance_ns",
    "recurrent_ie_conductance_ns",
    "recurrent_ii_conductance_ns",
    "ee_j_plus",
    "ee_sigma_deg",
    "ei_j_plus",
    "ei_sigma_deg",
    "tau_ampa_ms",
    "tau_nmda_ms",
    "tau_nmda_rise_ms",
    "alpha_nmda_per_ms",
    "tau_gabaa_ms",
    "magnesium_mm",
    "excitatory",
    "inhibitory",
)


class SCCompteWMBackendUnavailable(RuntimeError):
    """Raised when an explicitly selected native runtime cannot execute."""


@dataclass(frozen=True, slots=True)
class SCCompteWMBackendStatus:
    """Availability of one explicitly named full-network runtime."""

    backend: SCCompteWMBackend
    available: bool
    execution_mode: str
    reason: str | None


@dataclass(frozen=True, slots=True)
class SCCompteWMBackendRun:
    """One selected-backend receipt with native execution timing."""

    backend: SCCompteWMBackend
    execution_ns: int
    receipt: SCCompteWMRunReceipt


def _executable(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def sc_compte_wm_backend_status() -> tuple[SCCompteWMBackendStatus, ...]:
    """Return deterministic availability details for all five runtimes."""
    from sc_neurocore.accel.mojo.sc_compte_wm_network import (
        _HAS_MOJO_SC_COMPTE_WM_NETWORK,
    )

    checks: dict[SCCompteWMBackend, tuple[bool, str, str | None]] = {
        "python": (True, "in-process", None),
        "rust": (
            _executable(_VENV / "bin/cargo") and _RUST_RUNNER.is_file(),
            "repository-native-command",
            None,
        ),
        "julia": (
            _executable(_VENV / "bin/julia") and _JULIA_RUNNER.is_file(),
            "repository-native-command",
            None,
        ),
        "go": (
            _executable(_VENV / "bin/go") and _GO_RUNNER.is_file(),
            "repository-native-command",
            None,
        ),
        "mojo": (
            _HAS_MOJO_SC_COMPTE_WM_NETWORK,
            "in-process-shared-library",
            None,
        ),
    }
    statuses = []
    for backend in _BACKENDS:
        available, mode, _ = checks[backend]
        reason = None if available else f"repository-local {backend} runtime is unavailable"
        statuses.append(SCCompteWMBackendStatus(backend, available, mode, reason))
    return tuple(statuses)


def _validate_native_spec(spec: SCCompteWMNetworkSpec) -> None:
    baseline = SCCompteWMNetworkSpec()
    changed = [
        name for name in _NATIVE_FIXED_FIELDS if getattr(spec, name) != getattr(baseline, name)
    ]
    if changed:
        raise ValueError("native v1 backends fix these specification fields: " + ", ".join(changed))


def _native_args(
    spec: SCCompteWMNetworkSpec,
    duration_ms: float,
    statistics_window_ms: float,
    stimuli: tuple[SCCompteWMStimulus, ...],
) -> list[str]:
    arguments = [
        "--duration-ms",
        repr(duration_ms),
        "--statistics-window-ms",
        repr(statistics_window_ms),
        "--seed",
        str(spec.seed),
    ]
    if spec.structured_ei:
        arguments.append("--structured-ei")
    if spec.modulated:
        arguments.append("--modulated")
    if spec.allow_recurrent_autapses:
        arguments.append("--allow-recurrent-autapses")
    for stimulus in stimuli:
        center = "none" if stimulus.center_deg is None else repr(stimulus.center_deg)
        arguments.extend(
            [
                "--stimulus",
                ",".join(
                    (
                        repr(stimulus.start_ms),
                        repr(stimulus.duration_ms),
                        repr(stimulus.current_pa),
                        stimulus.kind,
                        center,
                    )
                ),
            ]
        )
    return arguments


def _environment(backend: SCCompteWMBackend) -> dict[str, str]:
    environment = os.environ.copy()
    if backend == "rust":
        environment.update(
            {
                "CARGO_HOME": str(_VENV / "cargo-home"),
                "CARGO_TARGET_DIR": str(_VENV / "cargo-target"),
            }
        )
    elif backend == "julia":
        environment["JULIA_DEPOT_PATH"] = str(_VENV / "julia_depot")
    elif backend == "go":
        go_home = _VENV / "go"
        environment.update(
            {
                "GOPATH": str(go_home),
                "GOMODCACHE": str(go_home / "pkg/mod"),
                "GOCACHE": str(go_home / "cache"),
                "GOTOOLCHAIN": "auto",
            }
        )
    return environment


def _command(backend: SCCompteWMBackend, arguments: list[str]) -> tuple[list[str], Path]:
    if backend == "rust":
        return (
            [
                str(_VENV / "bin/cargo"),
                "run",
                "--quiet",
                "--release",
                "--manifest-path",
                str(_RUST_MANIFEST),
                "--example",
                "sc_compte_wm_network_run",
                "--no-default-features",
                "--",
                *arguments,
            ],
            _REPOSITORY,
        )
    if backend == "julia":
        return (
            [
                str(_VENV / "bin/julia"),
                f"--project={_JULIA_ROOT}",
                str(_JULIA_RUNNER),
                *arguments,
            ],
            _REPOSITORY,
        )
    if backend == "go":
        return (
            [str(_VENV / "bin/go"), "run", "./cmd/run_sc_compte_wm_network", *arguments],
            _GO_ROOT,
        )
    raise ValueError(f"{backend} does not use native command dispatch")


def _statistics(payload: dict[str, Any] | None) -> SCCompteWMActivityStatistics | None:
    if payload is None:
        return None
    return SCCompteWMActivityStatistics(
        excitatory_rate_hz=float(payload["excitatory_rate_hz"]),
        inhibitory_rate_hz=float(payload["inhibitory_rate_hz"]),
        bump_angle_deg=float(payload["bump_angle_deg"]),
        resultant_length=float(payload["resultant_length"]),
        circular_width_deg=(
            None if payload["circular_width_deg"] is None else float(payload["circular_width_deg"])
        ),
    )


def _receipt(payload: dict[str, Any]) -> SCCompteWMRunReceipt:
    windows = tuple(
        SCCompteWMWindowReceipt(
            start_ms=float(window["start_ms"]),
            end_ms=float(window["end_ms"]),
            excitatory_spikes=int(window["excitatory_spikes"]),
            inhibitory_spikes=int(window["inhibitory_spikes"]),
            statistics=_statistics(window["statistics"]),
        )
        for window in payload["windows"]
    )
    return SCCompteWMRunReceipt(
        specification_version=str(payload["specification_version"]),
        seed=int(payload["seed"]),
        duration_ms=float(payload["duration_ms"]),
        steps=int(payload["steps"]),
        excitatory_spikes=int(payload["excitatory_spikes"]),
        inhibitory_spikes=int(payload["inhibitory_spikes"]),
        windows=windows,
        input_sha256=str(payload["input_sha256"]),
        spike_sha256=str(payload["spike_sha256"]),
        final_state_sha256=str(payload["final_state_sha256"]),
    )


def _run_native_command(
    backend: SCCompteWMBackend,
    arguments: list[str],
    timeout_s: float | None,
) -> SCCompteWMBackendRun:
    command, cwd = _command(backend, arguments)
    try:
        executed = subprocess.run(
            command,
            cwd=cwd,
            env=_environment(backend),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise SCCompteWMBackendUnavailable(f"{backend} execution failed: {error}") from error
    if executed.returncode != 0:
        detail = executed.stderr.strip()[-2000:]
        raise SCCompteWMBackendUnavailable(
            f"{backend} execution exited {executed.returncode}: {detail}"
        )
    try:
        payload = cast(dict[str, Any], json.loads(executed.stdout))
        if payload.get("runtime") != backend:
            raise ValueError("runtime identity mismatch")
        execution_ns = int(payload["execution_ns"])
        if execution_ns < 0:
            raise ValueError("negative execution time")
        return SCCompteWMBackendRun(backend, execution_ns, _receipt(payload))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise SCCompteWMBackendUnavailable(f"{backend} emitted an invalid run receipt") from error


def run_sc_compte_wm_network(
    duration_ms: float,
    *,
    backend: SCCompteWMBackend,
    spec: SCCompteWMNetworkSpec | None = None,
    stimuli: tuple[SCCompteWMStimulus, ...] = (),
    statistics_window_ms: float | None = None,
    timeout_s: float | None = None,
) -> SCCompteWMBackendRun:
    """Execute one explicitly selected complete runtime without fallback.

    ``timeout_s=None`` permits long scientific runs. A finite timeout must be
    positive. Native v1 backends accept the fixed public constants plus the
    seed and three documented mode flags; incompatible parameter changes fail
    before process launch.
    """
    if backend not in _BACKENDS:
        raise ValueError(f"unknown SC Compte backend: {backend}")
    selected_spec = SCCompteWMNetworkSpec() if spec is None else spec
    window_ms = (
        selected_spec.protocol.statistics_window_ms
        if statistics_window_ms is None
        else statistics_window_ms
    )
    if timeout_s is not None and (not math.isfinite(timeout_s) or timeout_s <= 0.0):
        raise ValueError("timeout_s must be finite and positive")
    status = {item.backend: item for item in sc_compte_wm_backend_status()}[backend]
    if not status.available:
        raise SCCompteWMBackendUnavailable(status.reason or f"{backend} is unavailable")
    if backend == "python":
        python_runtime = SCCompteWMNetwork(selected_spec)
        started = time.perf_counter_ns()
        receipt = python_runtime.run(duration_ms, stimuli=stimuli, statistics_window_ms=window_ms)
        return SCCompteWMBackendRun(backend, time.perf_counter_ns() - started, receipt)
    _validate_native_spec(selected_spec)
    if backend == "mojo":
        from sc_neurocore.accel.mojo.sc_compte_wm_network import SCCompteWMMojoNetwork

        mojo_runtime = SCCompteWMMojoNetwork(selected_spec)
        started = time.perf_counter_ns()
        receipt = mojo_runtime.run(duration_ms, stimuli=stimuli, statistics_window_ms=window_ms)
        return SCCompteWMBackendRun(backend, time.perf_counter_ns() - started, receipt)
    arguments = _native_args(selected_spec, duration_ms, window_ms, stimuli)
    return _run_native_command(backend, arguments, timeout_s)


__all__ = [
    "SCCompteWMBackend",
    "SCCompteWMBackendRun",
    "SCCompteWMBackendStatus",
    "SCCompteWMBackendUnavailable",
    "run_sc_compte_wm_network",
    "sc_compte_wm_backend_status",
]
