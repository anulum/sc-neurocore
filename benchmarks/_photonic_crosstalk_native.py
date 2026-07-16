# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native photonic crosstalk benchmark runners

"""Bounded Rust, Go, Julia, and Mojo benchmark adapters."""

from __future__ import annotations

import ctypes
import gc
import re
import statistics

# The benchmark invokes fixed toolchain commands without a shell.
import subprocess  # nosec B404
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from sc_neurocore.accel.mojo.isa_baseline import pin_isa

REPOSITORY = Path(__file__).resolve().parents[1]
RUST_SOURCE = REPOSITORY / "src/sc_neurocore/accel/rust/safety/photonic_emitter.rs"
GO_ROOT = REPOSITORY / "src/sc_neurocore/accel/go"
JULIA_SOURCE = REPOSITORY / "src/sc_neurocore/accel/julia/optics/photonic_emitter.jl"
MOJO_SOURCE = REPOSITORY / "src/sc_neurocore/accel/mojo/kernels/photonic_emitter.mojo"


def _run(command: list[str], *, cwd: Path = REPOSITORY, timeout: int = 180) -> str:
    """Run a bounded command and return standard output."""
    # Callers supply only the fixed toolchain argv declared in this module.
    completed = subprocess.run(  # nosec B603
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n{completed.stderr}"
        )
    return completed.stdout


def tool_version(command: list[str]) -> str:
    """Return the first non-empty version line."""
    # Callers supply only the fixed version argv declared in the harness.
    completed = subprocess.run(  # nosec B603
        command,
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    output = completed.stdout or completed.stderr
    return next(line.strip() for line in output.splitlines() if line.strip())


def _required_match(pattern: str, output: str, description: str) -> re.Match[str]:
    """Return a required benchmark-output match or reject malformed output."""
    match = re.search(pattern, output)
    if match is None:
        raise RuntimeError(f"benchmark emitted no {description}")
    return match


def measure(call: Callable[[], object], *, iterations: int, samples: int = 9) -> dict[str, float]:
    """Measure warm-cache batch latency with GC disabled inside each sample."""
    for _ in range(3):
        call()
    timings: list[float] = []
    for _ in range(samples):
        gc.collect()
        gc.disable()
        start = time.perf_counter_ns()
        for _ in range(iterations):
            call()
        elapsed = time.perf_counter_ns() - start
        gc.enable()
        timings.append(elapsed / iterations)
    return {
        "median_ns_per_batch": statistics.median(timings),
        "minimum_ns_per_batch": min(timings),
        "maximum_ns_per_batch": max(timings),
        "samples": float(samples),
        "iterations_per_sample": float(iterations),
    }


def rust_benchmark(temp: Path, pair_count: int, iterations: int) -> dict[str, object]:
    """Compile and measure the stand-alone fail-closed Rust batch."""
    program = temp / "photonic_bench.rs"
    binary = temp / "photonic_bench"
    program.write_text(
        f'''#[path = r#"{RUST_SOURCE}"#]
mod photonic_kernel;
use photonic_kernel::{{analyze_pairs, PairSpec}};
use std::hint::black_box;
use std::time::Instant;

fn main() {{
    let pairs: Vec<PairSpec> = (0..{pair_count}).map(|i| PairSpec {{
        index_a: i, index_b: i + 1,
        gap_nm: 180.0 + (i % 64) as f64,
        coupling_length_um: 8.0 + (i % 17) as f64,
    }}).collect();
    let first = analyze_pairs(&pairs, 1550.0, 3.48, 1.45).unwrap();
    for _ in 0..3 {{ black_box(analyze_pairs(&pairs, 1550.0, 3.48, 1.45).unwrap()); }}
    let start = Instant::now();
    let mut checksum = 0.0;
    for _ in 0..{iterations} {{
        let output = analyze_pairs(black_box(&pairs), 1550.0, 3.48, 1.45).unwrap();
        checksum += output[0].coupling_ratio;
        black_box(output);
    }}
    println!("RUST_NS {{:.9}}", start.elapsed().as_nanos() as f64 / {iterations}.0);
    println!("RUST_FIRST {{:.17}} {{:.17}} {{:.17}}", first[0].coupling_coefficient_per_um, first[0].coupling_ratio, first[0].isolation_db);
    println!("RUST_CHECKSUM {{:.17}}", checksum);
}}
''',
        encoding="utf-8",
    )
    _run(["rustc", "--edition", "2021", "-O", str(program), "-o", str(binary)])
    output = _run([str(binary)])
    latency = float(_required_match(r"RUST_NS ([0-9.]+)", output, "Rust latency").group(1))
    first = [
        float(value)
        for value in _required_match(r"RUST_FIRST (.+)", output, "Rust parity record")
        .group(1)
        .split()
    ]
    return {
        "median_ns_per_batch": latency,
        "mode": "native optimised Rust safety batch",
        "first_pair": first,
    }


def go_benchmark() -> dict[str, object]:
    """Run the maintained Go benchmark for the identical batch size."""
    output = _run(
        [
            "go",
            "test",
            "-v",
            "./services/photonic_emitter",
            "-run",
            "^$",
            "-bench",
            "BenchmarkAnalyzePairs",
            "-benchtime=2s",
            "-count=3",
        ],
        cwd=GO_ROOT,
    )
    values = [
        float(value)
        for value in re.findall(r"BenchmarkAnalyzePairs(?:-\d+)?\s+\d+\s+([0-9.]+)\s+ns/op", output)
    ]
    if not values:
        raise RuntimeError("Go benchmark emitted no ns/op samples")
    match = re.search(r"GO_FIRST ([0-9.eE+ -]+)", output)
    if match is None:
        raise RuntimeError("Go benchmark emitted no parity record")
    first = [float(value) for value in match.group(1).split()]
    return {
        "median_ns_per_batch": statistics.median(values),
        "samples": float(len(values)),
        "mode": "native Go package batch",
        "first_pair": first,
    }


def julia_benchmark(pair_count: int, iterations: int) -> dict[str, object]:
    """Measure the Julia module after compilation warm-up."""
    expression = f'''
include(raw"{JULIA_SOURCE}"); using .PhotonicEmitterAccel; using Statistics
pairs = [PairSpec(i, i + 1, 180.0 + mod(i, 64), 8.0 + mod(i, 17)) for i in 0:{pair_count - 1}]
first = analyze_pairs(pairs)[1]
analyze_pairs(pairs)
times = Float64[]
for _ in 1:7
    elapsed = @elapsed for _ in 1:{iterations}; analyze_pairs(pairs); end
    push!(times, elapsed * 1e9 / {iterations})
end
println("JULIA_NS ", median(times))
println("JULIA_FIRST ", first.coupling_coefficient_per_um, " ", first.coupling_ratio, " ", first.isolation_db)
'''
    output = _run(["julia", "--startup-file=no", "--project=@stdlib", "-e", expression])
    latency = float(_required_match(r"JULIA_NS ([0-9.eE+-]+)", output, "Julia latency").group(1))
    first = [
        float(value)
        for value in _required_match(r"JULIA_FIRST (.+)", output, "Julia parity record")
        .group(1)
        .split()
    ]
    return {
        "median_ns_per_batch": latency,
        "mode": "warm native Julia module batch",
        "first_pair": first,
    }


def mojo_benchmark(
    temp: Path,
    gaps: np.ndarray[Any, Any],
    lengths: np.ndarray[Any, Any],
    iterations: int,
) -> dict[str, object]:
    """Build and measure the Mojo batch C ABI from a warm loaded library."""
    library_path = temp / "libphotonic_emitter.so"
    _run(
        pin_isa(
            [
                "mojo",
                "build",
                "--emit",
                "shared-lib",
                "-o",
                str(library_path),
                str(MOJO_SOURCE),
            ]
        )
    )
    library = ctypes.CDLL(str(library_path))
    kernel = library.photonic_crosstalk_batch_c
    kernel.argtypes = [
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_longlong,
    ]
    kernel.restype = ctypes.c_longlong
    output = np.empty(gaps.size * 3, dtype=np.float64)

    def call() -> int:
        status = int(
            kernel(
                gaps.ctypes.data,
                lengths.ctypes.data,
                gaps.size,
                1550.0,
                3.48,
                1.45,
                output.ctypes.data,
            )
        )
        if status != 0:
            raise RuntimeError("Mojo batch rejected a valid benchmark workload")
        return status

    call()
    timings = measure(call, iterations=iterations)
    return {
        **timings,
        "mode": "warm Mojo shared-library C ABI batch",
        "first_pair": output[:3].tolist(),
    }
