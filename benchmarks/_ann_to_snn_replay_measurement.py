# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Full-response IF latency measurement

"""Measure owned public-call latency after verifying every response against NumPy."""

import importlib.metadata
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from _ann_to_snn_replay_profiles import replay_profiles, response_digest

from sc_neurocore.conversion.if_benchmark_identity import (
    configured_artifacts,
)
from sc_neurocore.conversion.if_benchmark_identity import (
    source_digests as runtime_source_digests,
)
from sc_neurocore.conversion.if_dispatch import ReplayBackend

KERNEL = "dense-if-f64-sequential-v1"
__all__ = [
    "BACKENDS",
    "KERNEL",
    "REPOSITORY",
    "configured_artifacts",
    "measure",
    "metadata",
    "source_digests",
]

BACKENDS: tuple[ReplayBackend, ...] = ("numpy", "rust", "go", "mojo", "julia")
REPOSITORY = Path(__file__).resolve().parents[1]


def source_digests() -> dict[str, str]:
    """Bind the imported package and these executing scripts as public dispatch does.

    Returns
    -------
    dict of str to str
        Repository-relative owning source paths and their SHA-256 digests.
    """
    return runtime_source_digests(Path(__file__).resolve().parent)


def metadata(cpu: int, samples: int, warmup: int) -> dict[str, object]:
    """Describe call boundaries, dependency versions, affinity and concurrent host load.

    Parameters
    ----------
    cpu : int
        Common allowed CPU assigned to each fresh runtime process.
    samples, warmup : int
        Warm sample and preceding exact-workload repetition counts.

    Returns
    -------
    dict
        Actual Linux host/dependency context and precise measurement boundaries.
    """
    cpu_model = next(
        line.split(":", 1)[1].strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines()
        if line.startswith("model name")
    )
    return {
        "cpu": cpu_model,
        "cpu_affinity": [cpu],
        "kernel": platform.release(),
        "python": platform.python_version(),
        "numpy": importlib.metadata.version("numpy"),
        "juliacall": importlib.metadata.version("juliacall"),
        "samples": samples,
        "warmup": warmup,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "load_average": list(os.getloadavg()),
        "cpu_isolation": "one CPU affinity; host load is not isolated",
        "coverage_process_start_requested": bool(os.environ.get("COVERAGE_PROCESS_START")),
        "latency_boundary": (
            "public call and collection of all owned vectors; encoding, Python admission and "
            "native transport included; final release, interpreter startup and builds excluded"
        ),
        "first_call": (
            "first selected-provider call after the NumPy reference; first workload includes "
            "lazy native library/runtime initialization; later workloads may reuse that runtime"
        ),
        "warmup_definition": "exact workload repetitions before recorded warm samples",
        "energy": "not measured; no instrument, calibration or operation-count estimate",
        "model_provenance": (
            "seeded untrained runtime workload, not trained source-to-target loss evidence"
        ),
    }


def measure(backend: ReplayBackend, samples: int, warmup: int) -> dict[str, object]:
    """Measure actual cold/warm calls and refuse any complete-response bit mismatch.

    Parameters
    ----------
    backend : str
        Explicit actual native runtime or NumPy reference.
    samples, warmup : int
        Positive warm sample and preceding warmup repetition counts.

    Returns
    -------
    dict
        Every workload's raw call times, input/response digests and aggregate latency.

    Raises
    ------
    RuntimeError
        Requested provider absent, native refusal or full response mismatch.
    """
    cases: list[dict[str, object]] = []
    medians = []
    for profile in replay_profiles():
        expected = response_digest(profile.execute("numpy"))
        first_call = 0
        timings = []
        for repeat in range(1 + warmup + samples):
            started = time.perf_counter_ns()
            actual = profile.execute(backend)
            elapsed = time.perf_counter_ns() - started
            if response_digest(actual) != expected:
                raise RuntimeError(f"{backend} complete response mismatch: {profile.name}")
            del actual
            if repeat == 0:
                first_call = elapsed
            elif repeat > warmup:
                timings.append(elapsed)
        median = statistics.median(timings) / 1e6
        medians.append(median)
        cases.append(
            {
                "name": profile.name,
                "input_sha256": profile.input_digest(),
                "response_sha256": expected,
                "first_call_ns": first_call,
                "samples_ns": timings,
                "median_call_ms": median,
            }
        )
    return {
        "backend": backend,
        "available": True,
        "used": True,
        "full_bit_parity": True,
        "median_call_ms": statistics.geometric_mean(medians),
        "aggregation": "equal-weight geometric mean of twenty per-workload warm medians",
        "cases": cases,
    }
