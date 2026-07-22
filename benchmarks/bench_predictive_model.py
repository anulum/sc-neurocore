# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source-bound LGSSM backend benchmark

"""Measure the maintained LGSSM implementations with provenance and parity checks.

The forward-filter workload executes Python, Rust, Julia, Go, and Mojo on the
same controlled sequence. RTS smoothing and EM are explicitly recorded as
Python-only surfaces. Results are local-regression evidence: CPU affinity and
host load are disclosed, but no exclusive-core or cross-host performance claim
is inferred from the measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, TypeVar, cast

import numpy as np

from sc_neurocore.world_model import _lgssm_backends as backend_runtime
from sc_neurocore.world_model._lgssm_backends import ExplicitBackendName
from sc_neurocore.world_model._lgssm_types import FilterResult, FloatArray
from sc_neurocore.world_model.predictive_model import (
    EMLearner,
    KalmanFilter,
    LinearGaussianSSM,
    RTSSmoother,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
ALL_BACKENDS: tuple[ExplicitBackendName, ...] = (
    "python",
    "rust",
    "julia",
    "go",
    "mojo",
)
SOURCE_PATHS = (
    "src/sc_neurocore/world_model/predictive_model.py",
    "src/sc_neurocore/world_model/_lgssm_types.py",
    "src/sc_neurocore/world_model/_lgssm_backends.py",
    "src/sc_neurocore/world_model/_lgssm_filter.py",
    "src/sc_neurocore/world_model/_lgssm_smoothing.py",
    "src/sc_neurocore/world_model/_lgssm_em.py",
    "src/sc_neurocore/world_model/_predictive_world_model.py",
    "engine/src/lgssm.rs",
    "bridge/sc_neurocore_engine/__init__.py",
    "bridge/sc_neurocore_engine/world_model.py",
    "src/sc_neurocore/accel/rust/safety/predictive_model.rs",
    "src/sc_neurocore/accel/julia/world_model/predictive_model.jl",
    "src/sc_neurocore/accel/go/lgssm/lgssm.go",
    "src/sc_neurocore/accel/mojo/world_model/lgssm.mojo",
    "benchmarks/bench_predictive_model.py",
)
ARRAY_PARITY_ATOL = 1e-9
LIKELIHOOD_PARITY_ATOL = 1e-7
DISPATCH_MATERIALITY_RATIO = 1.10

T = TypeVar("T")


@dataclass(frozen=True)
class BenchmarkConfig:
    """Validated command-line benchmark configuration."""

    json_path: Path | None
    repeats: int
    steps: int
    em_iterations: int
    backends: tuple[ExplicitBackendName, ...]
    other_heavy_jobs_running: str
    other_heavy_jobs_note: str
    isolation_note: str
    argv: tuple[str, ...]


def _positive_int(raw_value: str) -> int:
    """Parse a strictly positive integer for argparse."""
    value = int(raw_value)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _parse_args(argv: list[str]) -> BenchmarkConfig:
    """Parse the reproducibility and workload controls."""
    parser = argparse.ArgumentParser(
        description="Source-bound LGSSM five-backend local benchmark.",
    )
    parser.add_argument("--json", type=Path)
    parser.add_argument("--repeats", type=_positive_int, default=25)
    parser.add_argument("--steps", type=_positive_int, default=200)
    parser.add_argument("--em-iterations", type=_positive_int, default=10)
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=ALL_BACKENDS,
        default=list(ALL_BACKENDS),
    )
    parser.add_argument(
        "--other-heavy-jobs-running",
        choices=("yes", "no", "unknown"),
        default="unknown",
    )
    parser.add_argument("--other-heavy-jobs-note", default="not disclosed")
    parser.add_argument(
        "--isolation-note",
        default="ordinary loaded-host run; no exclusive-core reservation",
    )
    parsed = parser.parse_args(argv)
    selected = tuple(cast(ExplicitBackendName, name) for name in cast(list[str], parsed.backends))
    if len(set(selected)) != len(selected):
        parser.error("--backends entries must be unique")
    return BenchmarkConfig(
        json_path=cast(Path | None, parsed.json),
        repeats=cast(int, parsed.repeats),
        steps=cast(int, parsed.steps),
        em_iterations=cast(int, parsed.em_iterations),
        backends=selected,
        other_heavy_jobs_running=cast(str, parsed.other_heavy_jobs_running),
        other_heavy_jobs_note=cast(str, parsed.other_heavy_jobs_note),
        isolation_note=cast(str, parsed.isolation_note),
        argv=tuple(argv),
    )


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one regular file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hashes() -> dict[str, str]:
    """Bind evidence to every maintained implementation and dispatcher source."""
    hashes: dict[str, str] = {}
    for relative_path in SOURCE_PATHS:
        path = ROOT / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"benchmark source is missing: {relative_path}")
        hashes[relative_path] = _sha256(path)
    return hashes


def _command_output(command: list[str]) -> str:
    """Capture a short best-effort tool version without invoking a shell."""
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "unavailable"
    output = completed.stdout.strip() or completed.stderr.strip()
    return output.splitlines()[0] if output else f"exit {completed.returncode}"


def _rust_extension_path() -> Path:
    """Resolve the installed PyO3 extension used by the Rust backend."""
    extension = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")
    module_path = getattr(extension, "__file__", None)
    if not isinstance(module_path, str):
        raise RuntimeError("installed Rust extension does not expose __file__")
    path = Path(module_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"installed Rust extension is missing: {path}")
    return path


def _display_path(path: Path) -> str:
    """Render repository-local paths without leaking a host mount prefix."""
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _binary_hashes(backends: tuple[ExplicitBackendName, ...]) -> dict[str, object]:
    """Hash the native binaries actually selected for this run."""
    paths: dict[str, Path] = {}
    if "rust" in backends:
        paths["rust"] = _rust_extension_path()
    if "go" in backends:
        paths["go"] = ROOT / "src/sc_neurocore/accel/go/lgssm/liblgssm.so"
    if "mojo" in backends:
        paths["mojo"] = ROOT / "src/sc_neurocore/accel/mojo/world_model/liblgssm.so"

    payload: dict[str, object] = {}
    for backend, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{backend} runtime binary is missing: {path}")
        payload[backend] = {
            "path": _display_path(path),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
    return payload


def _runtime_versions(backends: tuple[ExplicitBackendName, ...]) -> dict[str, str]:
    """Record the language runtimes participating in this run."""
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    if "rust" in backends:
        versions["rust_engine_package"] = importlib.metadata.version(
            "sc_neurocore_engine",
        )
        versions["rust_extension"] = _display_path(_rust_extension_path())
        versions["rustc"] = _command_output(["rustc", "--version"])
    if "julia" in backends:
        juliacall = importlib.import_module("juliacall")
        versions["juliacall"] = importlib.metadata.version("juliacall")
        versions["julia"] = str(juliacall.Main.seval("string(VERSION)"))
    if "go" in backends:
        versions["go"] = _command_output(["go", "version"])
    if "mojo" in backends:
        mojo = shutil.which("mojo")
        versions["mojo"] = _command_output([mojo, "--version"]) if mojo else "unavailable"
    return versions


def _read_first_line(path: Path) -> str:
    """Read one sysfs value without failing the benchmark on absent metadata."""
    try:
        return path.read_text(encoding="utf-8").strip().splitlines()[0]
    except (FileNotFoundError, IndexError, OSError):
        return "unavailable"


def _cpu_model() -> str:
    """Return the first Linux CPU model description when available."""
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _affinity() -> list[int]:
    """Return the process CPU-affinity mask as sorted logical CPUs."""
    try:
        return sorted(os.sched_getaffinity(0))
    except AttributeError:
        return []


def _host_metadata(affinity: list[int]) -> dict[str, object]:
    """Capture the loaded-host context needed to interpret local timings."""
    cpu = affinity[0] if affinity else 0
    frequency_root = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq")
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": _cpu_model(),
        "logical_cpu_count": os.cpu_count(),
        "process_affinity": affinity,
        "taskset_pinned_to_one_cpu": len(affinity) == 1,
        "exclusive_core_reserved": False,
        "scaling_governor": _read_first_line(frequency_root / "scaling_governor"),
        "scaling_current_khz_before": _read_first_line(
            frequency_root / "scaling_cur_freq",
        ),
    }


def _build_workload(
    *,
    steps: int,
    seed: int = 20260714,
) -> tuple[LinearGaussianSSM, FloatArray, FloatArray]:
    """Build one deterministic controlled LGSSM sequence for every backend."""
    model = LinearGaussianSSM.random(
        state_dim=4,
        obs_dim=3,
        control_dim=2,
        seed=seed,
    )
    rng = np.random.default_rng(seed + 1)
    controls = rng.normal(size=(steps, model.control_dim))
    observations = np.zeros((steps, model.obs_dim), dtype=np.float64)
    state = rng.multivariate_normal(model.mu_0, model.Sigma_0)
    for time_index in range(steps):
        observations[time_index] = (
            model.C @ state
            + model.D @ controls[time_index]
            + rng.multivariate_normal(np.zeros(model.obs_dim), model.R)
        )
        state = (
            model.A @ state
            + model.B @ controls[time_index]
            + rng.multivariate_normal(np.zeros(model.state_dim), model.Q)
        )
    return model, observations, controls


def _measure(operation: Callable[[], T], *, repeats: int) -> tuple[list[float], T]:
    """Warm an operation once, then return every timed sample and final value."""
    operation()
    samples: list[float] = []
    result: T | None = None
    for _ in range(repeats):
        started = time.perf_counter_ns()
        result = operation()
        samples.append((time.perf_counter_ns() - started) / 1_000_000.0)
    if result is None:
        raise RuntimeError("benchmark repeats unexpectedly produced no result")
    return samples, result


def _timing_payload(samples: list[float]) -> dict[str, object]:
    """Summarise timings while retaining all raw samples."""
    return {
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def _parity_payload(candidate: FilterResult, reference: FilterResult) -> dict[str, float]:
    """Compute explicit value deltas against the Python reference."""
    return {
        "means_max_abs": float(np.max(np.abs(candidate.means - reference.means))),
        "covariances_max_abs": float(
            np.max(np.abs(candidate.covariances - reference.covariances)),
        ),
        "pred_means_max_abs": float(
            np.max(np.abs(candidate.pred_means - reference.pred_means)),
        ),
        "pred_covariances_max_abs": float(
            np.max(np.abs(candidate.pred_covariances - reference.pred_covariances)),
        ),
        "log_likelihood_abs": abs(candidate.log_likelihood - reference.log_likelihood),
    }


def _assert_parity(backend: str, parity: dict[str, float]) -> None:
    """Fail evidence generation when a maintained backend diverges materially."""
    array_fields = (
        "means_max_abs",
        "covariances_max_abs",
        "pred_means_max_abs",
        "pred_covariances_max_abs",
    )
    if any(parity[field] > ARRAY_PARITY_ATOL for field in array_fields):
        raise RuntimeError(f"{backend} array parity failed: {parity}")
    if parity["log_likelihood_abs"] > LIKELIHOOD_PARITY_ATOL:
        raise RuntimeError(f"{backend} likelihood parity failed: {parity}")


def _forward_filter_evidence(
    config: BenchmarkConfig,
    model: LinearGaussianSSM,
    observations: FloatArray,
    controls: FloatArray,
) -> dict[str, object]:
    """Measure interleaved forward filters and validate each against Python."""
    reference = KalmanFilter(model).filter(
        observations,
        controls,
        backend="python",
    )
    operations: dict[ExplicitBackendName, Callable[[], FilterResult]] = {}
    samples: dict[ExplicitBackendName, list[float]] = {}
    results: dict[ExplicitBackendName, FilterResult] = {}
    for backend in config.backends:
        available, reason = backend_runtime.probe_backend(backend)
        if not available:
            raise RuntimeError(reason)

        def filter_once(
            backend_name: ExplicitBackendName = backend,
        ) -> FilterResult:
            return KalmanFilter(model).filter(
                observations,
                controls,
                backend=backend_name,
            )

        operations[backend] = filter_once
        samples[backend] = []
        results[backend] = filter_once()

    backend_count = len(config.backends)
    for repeat_index in range(config.repeats):
        offset = repeat_index % backend_count
        round_order = config.backends[offset:] + config.backends[:offset]
        for backend in round_order:
            started = time.perf_counter_ns()
            results[backend] = operations[backend]()
            samples[backend].append(
                (time.perf_counter_ns() - started) / 1_000_000.0,
            )

    evidence: dict[str, object] = {}
    python_median: float | None = None
    for backend in config.backends:
        result = results[backend]
        parity = _parity_payload(result, reference)
        _assert_parity(backend, parity)
        timing = _timing_payload(samples[backend])
        if backend == "python":
            python_median = cast(float, timing["median_ms"])
        evidence[backend] = {
            **timing,
            "log_likelihood": result.log_likelihood,
            "parity_vs_python": parity,
        }

    if python_median is not None:
        for backend_payload in evidence.values():
            typed_payload = cast(dict[str, object], backend_payload)
            median = cast(float, typed_payload["median_ms"])
            typed_payload["python_median_ratio"] = python_median / median
    return evidence


def _python_only_evidence(
    config: BenchmarkConfig,
    model: LinearGaussianSSM,
    observations: FloatArray,
    controls: FloatArray,
) -> dict[str, object]:
    """Measure the explicitly Python-only RTS and controlled-EM surfaces."""
    filtered = KalmanFilter(model).filter(
        observations,
        controls,
        backend="python",
    )
    smoother_samples, _ = _measure(
        lambda: RTSSmoother(model).smooth(filtered),
        repeats=config.repeats,
    )

    def fit_once() -> LinearGaussianSSM:
        learner = EMLearner(max_iter=config.em_iterations, tol=0.0)
        return learner.fit(
            observations,
            model,
            controls,
            backend="python",
        )

    em_samples, _ = _measure(fit_once, repeats=config.repeats)
    return {
        "rts_smoother": {
            "implementation_scope": "python_only",
            **_timing_payload(smoother_samples),
        },
        "em_learner": {
            "implementation_scope": "python_only",
            "iterations_per_sample": config.em_iterations,
            **_timing_payload(em_samples),
        },
    }


def _git_metadata() -> dict[str, object]:
    """Record the parent checkout while source hashes bind dirty-tree content."""
    head = _command_output(["git", "rev-parse", "HEAD"])
    dirty = _command_output(["git", "status", "--porcelain", "--untracked-files=no"])
    return {
        "head": head,
        "tracked_worktree_dirty": bool(dirty and dirty != "unavailable"),
        "source_binding": "sha256_per_file",
    }


def _render_summary(payload: dict[str, object]) -> None:
    """Print a compact human-readable view of the measured filter rows."""
    forward = cast(dict[str, object], payload["forward_filter"])
    print("backend      median_ms      min_ms    ll_abs_delta")
    for backend, raw_row in forward.items():
        row = cast(dict[str, object], raw_row)
        parity = cast(dict[str, float], row["parity_vs_python"])
        print(
            f"{backend:<10}  {cast(float, row['median_ms']):>10.3f}  "
            f"{cast(float, row['min_ms']):>10.3f}  "
            f"{parity['log_likelihood_abs']:.3e}",
        )


def run(config: BenchmarkConfig) -> dict[str, object]:
    """Execute the benchmark and return a self-contained evidence payload."""
    affinity = _affinity()
    load_before = list(os.getloadavg())
    host = _host_metadata(affinity)
    measured_at = datetime.now(timezone.utc).isoformat()
    benchmark_started = time.perf_counter()
    availability: dict[str, dict[str, object]] = {}
    for backend in config.backends:
        probe_started = time.perf_counter_ns()
        available, reason = backend_runtime.probe_backend(backend)
        availability[backend] = {
            "available": available,
            "reason": reason,
            "post_import_probe_ms": (time.perf_counter_ns() - probe_started) / 1_000_000.0,
        }
    unavailable = {
        backend: data["reason"]
        for backend, data in availability.items()
        if not cast(bool, data["available"])
    }
    if unavailable:
        raise RuntimeError(f"requested LGSSM backends unavailable: {unavailable}")

    model, observations, controls = _build_workload(steps=config.steps)
    forward = _forward_filter_evidence(config, model, observations, controls)
    measured_order = sorted(
        config.backends,
        key=lambda backend: cast(
            float,
            cast(dict[str, object], forward[backend])["median_ms"],
        ),
    )
    python_only = _python_only_evidence(
        config,
        model,
        observations,
        controls,
    )
    host["load_average_before"] = load_before
    host["load_average_after"] = list(os.getloadavg())
    host["benchmark_elapsed_seconds"] = time.perf_counter() - benchmark_started
    cpu = affinity[0] if affinity else 0
    host["scaling_current_khz_after"] = _read_first_line(
        Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq"),
    )

    executable = _display_path(Path(sys.executable).resolve())
    command = shlex.join(
        [
            executable,
            str(SCRIPT_PATH.relative_to(ROOT)),
            *config.argv,
        ],
    )
    return {
        "schema_version": 2,
        "evidence_class": "local_regression",
        "measured_at_utc": measured_at,
        "command": command,
        "working_directory": "repository_root",
        "git": _git_metadata(),
        "workload": {
            "seed": 20260714,
            "time_steps": config.steps,
            "state_dim": model.state_dim,
            "observation_dim": model.obs_dim,
            "control_dim": model.control_dim,
            "repeats": config.repeats,
            "warmups_per_cell": 1,
            "forward_sampling": "round_robin_rotating_start",
        },
        "parity_tolerances": {
            "array_max_abs": ARRAY_PARITY_ATOL,
            "log_likelihood_abs": LIKELIHOOD_PARITY_ATOL,
        },
        "dispatch_order": list(backend_runtime.AUTO_BACKEND_ORDER),
        "measured_order": measured_order,
        "dispatch_policy": {
            "basis": "post_import_probe_cost_then_interleaved_median_ms",
            "material_inversion_ratio": DISPATCH_MATERIALITY_RATIO,
            "tie_policy": "preserve_stable_order",
            "warm_order_exceptions": [
                "rust_before_julia_avoids_lazy_julia_initialisation",
            ],
        },
        "requested_backends": list(config.backends),
        "backend_availability": availability,
        "forward_filter": forward,
        "python_only_workloads": python_only,
        "source_sha256": _source_hashes(),
        "binary_evidence": _binary_hashes(config.backends),
        "runtime_versions": _runtime_versions(config.backends),
        "host": host,
        "isolation": {
            "exclusive_core_reserved": False,
            "classification": "loaded_host",
            "note": config.isolation_note,
            "other_heavy_jobs_running": config.other_heavy_jobs_running,
            "other_heavy_jobs_note": config.other_heavy_jobs_note,
        },
        "interpretation": (
            "Timings are suitable for regression comparison on this disclosed host only; "
            "they are not promotion-grade cross-host or exclusive-core evidence."
        ),
    }


def main(argv: list[str]) -> int:
    """Run the CLI, print its summary, and optionally write canonical JSON."""
    config = _parse_args(argv)
    payload = run(config)
    _render_summary(payload)
    if config.json_path is not None:
        config.json_path.parent.mkdir(parents=True, exist_ok=True)
        config.json_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {config.json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
