# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Controlled local photonic crosstalk benchmark

"""Benchmark the executable Python, Rust, Go, Julia, and Mojo pair kernels."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
from typing import Any

import numpy as np

import sc_neurocore.optics.photonic_emitter as photonic
from _photonic_crosstalk_native import (
    GO_ROOT,
    JULIA_SOURCE,
    MOJO_SOURCE,
    RUST_SOURCE,
    go_benchmark,
    julia_benchmark,
    measure,
    mojo_benchmark,
    rust_benchmark,
    tool_version,
)

REPOSITORY = Path(__file__).resolve().parents[1]
SOURCES = (
    Path(__file__).resolve(),
    Path(__file__).with_name("_photonic_crosstalk_native.py"),
    REPOSITORY / "src/sc_neurocore/optics/_photonic_crosstalk.py",
    RUST_SOURCE,
    GO_ROOT / "services/photonic_emitter/photonic_emitter.go",
    GO_ROOT / "services/photonic_emitter/photonic_emitter_test.go",
    JULIA_SOURCE,
    REPOSITORY / "src/sc_neurocore/accel/julia/photonic_emitter_parity_test.jl",
    MOJO_SOURCE,
    REPOSITORY / "engine/src/photonic.rs",
)


def _source_hashes() -> dict[str, str]:
    """Hash every source that can affect a reported runtime."""
    return {
        str(path.relative_to(REPOSITORY)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in SOURCES
    }


def _python_benchmark(
    pair_indices: list[tuple[int, int]], gaps: list[float], lengths: list[float], iterations: int
) -> dict[str, object]:
    """Measure the public validated Python fallback."""
    model = photonic.CrosstalkModel()
    previous = photonic._HAS_RUST_PH
    photonic._HAS_RUST_PH = False
    try:
        result = model.analyze_pairs(pair_indices, gaps, lengths)
        timings = measure(
            lambda: model.analyze_pairs(pair_indices, gaps, lengths), iterations=iterations
        )
    finally:
        photonic._HAS_RUST_PH = previous
    return {
        **timings,
        "mode": "public validated Python fallback",
        "first_pair": [
            result["coupling_coefficient_per_um"][0],
            result["coupling_ratio"][0],
            result["isolation_db"][0],
        ],
    }


def _host_context(cpu: int) -> dict[str, object]:
    """Capture the local-only scheduling and promotion context."""
    isolated_path = Path("/sys/devices/system/cpu/isolated")
    governor_path = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor")
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "selected_cpu": cpu,
        "isolated_cpus": isolated_path.read_text(encoding="utf-8").strip()
        if isolated_path.exists()
        else "",
        "governor": governor_path.read_text(encoding="utf-8").strip()
        if governor_path.exists()
        else "unknown",
        "load_average_start": list(os.getloadavg()),
    }


def main() -> None:
    """Run the controlled local benchmark and emit source-hashed JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--native-iterations", type=int, default=250)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.pairs < 1 or arguments.iterations < 1 or arguments.native_iterations < 1:
        raise ValueError("pairs and iteration counts must be positive")

    available = sorted(os.sched_getaffinity(0))
    selected_cpu = available[0]
    os.sched_setaffinity(0, {selected_cpu})
    host = _host_context(selected_cpu)
    pair_indices = [(index, index + 1) for index in range(arguments.pairs)]
    gaps_array = np.asarray([180.0 + index % 64 for index in range(arguments.pairs)])
    lengths_array = np.asarray([8.0 + index % 17 for index in range(arguments.pairs)])
    gaps = gaps_array.tolist()
    lengths = lengths_array.tolist()

    with tempfile.TemporaryDirectory(prefix="photonic-crosstalk-bench-") as temporary:
        temp = Path(temporary)
        runtimes = {
            "python": _python_benchmark(pair_indices, gaps, lengths, arguments.iterations),
            "rust": rust_benchmark(temp, arguments.pairs, arguments.native_iterations),
            "go": go_benchmark(),
            "julia": julia_benchmark(arguments.pairs, arguments.native_iterations),
            "mojo": mojo_benchmark(temp, gaps_array, lengths_array, arguments.native_iterations),
        }

    reference = np.asarray(runtimes["python"]["first_pair"], dtype=np.float64)
    parity: dict[str, float] = {}
    for runtime in ("rust", "go", "julia", "mojo"):
        parity[runtime] = float(
            np.max(
                np.abs(np.asarray(runtimes[runtime]["first_pair"], dtype=np.float64) - reference)
            )
        )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "benchmark": "photonic_crosstalk_pairs",
        "evidence_class": "local_regression_non_isolated",
        "promotion_eligible": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "workload": {
            "pair_count": arguments.pairs,
            "gap_nm": "180 + index mod 64",
            "coupling_length_um": "8 + index mod 17",
            "wavelength_nm": 1550.0,
            "core_index": 3.48,
            "cladding_index": 1.45,
        },
        "host": host,
        "versions": {
            "python": platform.python_version(),
            "rust": tool_version(["rustc", "--version"]),
            "go": tool_version(["go", "version"]),
            "julia": tool_version(["julia", "--version"]),
            "mojo": tool_version(["mojo", "--version"]),
        },
        "source_hashes": _source_hashes(),
        "runtimes": runtimes,
        "max_absolute_first_pair_error": parity,
        "parity_envelope": {
            "rust": 1.0e-15,
            "go": 3.0e-15,
            "julia": 1.0e-15,
            "mojo": 2.0e-10,
        },
        "load_average_end": list(os.getloadavg()),
        "interpretation": (
            "Local single-affinity regression evidence only. The host has no isolated CPU; "
            "native runtimes use different call boundaries, so latency is not a universal speed claim."
        ),
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
