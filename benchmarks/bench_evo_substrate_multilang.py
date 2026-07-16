# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Evolutionary substrate multi-language benchmark evidence

"""Measure equivalent evolutionary kernels across Rust, Julia, Go, Mojo, and Python."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import platform
import re
import shutil
import statistics
import subprocess
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from _benchmark_context import load_average, measurement_context
from numpy.typing import NDArray

from sc_neurocore.accel.mojo.isa_baseline import pin_isa

_SCHEMA_VERSION = "sc-neurocore.evo-substrate-multilang-benchmark.v2"
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OUTPUT = _REPO_ROOT / "benchmarks" / "results" / "bench_evo_substrate_multilang.json"
_ITERS = 100_000
_DIM = 19
_EPSILON = 1e-10
_KERNELS = (
    "genomic_distance_ns_per_call",
    "crossover_uniform_ns_per_call",
    "point_mutation_ns_per_call",
)
_BACKENDS = ("rust", "julia", "go", "mojo", "python")
Samples = dict[str, float]


class RustKernels(Protocol):
    """Typed boundary for the optional PyO3 evolutionary kernels."""

    def py_genomic_distance(
        self,
        left: NDArray[np.float64],
        right: NDArray[np.float64],
    ) -> float:
        """Return normalised genomic distance."""
        ...

    def py_crossover_uniform(
        self,
        left: NDArray[np.float64],
        right: NDArray[np.float64],
        mask: NDArray[np.uint8],
    ) -> NDArray[np.float64]:
        """Apply uniform crossover."""
        ...

    def py_point_mutation(
        self,
        gene: NDArray[np.float64],
        mask: NDArray[np.uint8],
        noise: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Apply masked multiplicative point mutation."""
        ...


def _parser() -> argparse.ArgumentParser:
    """Build the benchmark command parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Record unavailable backends instead of failing.",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    """Reject invalid sample and warm-up counts."""
    if args.samples <= 0:
        raise ValueError("samples must be positive")
    if args.warmups < 0:
        raise ValueError("warmups must be non-negative")


def _vectors() -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.uint8],
    NDArray[np.float64],
]:
    """Return the fixed 19-dimensional benchmark vectors."""
    left = (np.arange(_DIM, dtype=np.float64) + 1) * 0.1
    right = (np.arange(_DIM, dtype=np.float64) + 1) * 0.2
    mask = (np.arange(_DIM, dtype=np.uint8) % 2).astype(np.uint8)
    noise = np.full(_DIM, 0.01, dtype=np.float64)
    return left, right, mask, noise


def _ns_per_call(function: Callable[[], object], iterations: int = _ITERS) -> float:
    """Return nanoseconds per call across a fixed inner loop."""
    started = time.perf_counter_ns()
    for _ in range(iterations):
        function()
    return (time.perf_counter_ns() - started) / iterations


def _python_distance(
    left: NDArray[np.float64],
    right: NDArray[np.float64],
) -> float:
    """Return the NumPy reference genomic distance."""
    return float(np.mean(np.abs(left - right) / (np.abs(left) + np.abs(right) + _EPSILON)))


def _python_crossover(
    left: NDArray[np.float64],
    right: NDArray[np.float64],
    mask: NDArray[np.uint8],
) -> NDArray[np.float64]:
    """Return the NumPy reference uniform crossover."""
    return np.where(mask.astype(bool), left, right)


def _python_mutation(
    gene: NDArray[np.float64],
    mask: NDArray[np.uint8],
    noise: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return the NumPy reference masked point mutation."""
    output = gene.copy()
    selected = mask.astype(bool)
    output[selected] += noise[selected] * (np.abs(output[selected]) + 1e-8)
    return output


def _rust() -> Samples | None:
    try:
        module = importlib.import_module("sc_neurocore.evo_substrate.evo_substrate_core")
    except ImportError:
        return None
    kernels = cast(RustKernels, module)
    left, right, mask, noise = _vectors()
    kernels.py_genomic_distance(left, right)
    kernels.py_crossover_uniform(left, right, mask)
    kernels.py_point_mutation(left, mask, noise)
    return {
        "genomic_distance_ns_per_call": _ns_per_call(
            lambda: kernels.py_genomic_distance(left, right)
        ),
        "crossover_uniform_ns_per_call": _ns_per_call(
            lambda: kernels.py_crossover_uniform(left, right, mask)
        ),
        "point_mutation_ns_per_call": _ns_per_call(
            lambda: kernels.py_point_mutation(left, mask, noise)
        ),
    }


def _parse_lines(output: str) -> Samples | None:
    """Parse language-runner key/value timing lines."""
    results: Samples = {}
    for line in output.splitlines():
        match = re.fullmatch(
            r"(\S+_ns_per_call)\s+([\d.]+)(?:\s+ns)?",
            line.strip(),
        )
        if match:
            results[match.group(1)] = float(match.group(2))
    return results if set(results) == set(_KERNELS) else None


def _julia() -> Samples | None:
    julia = shutil.which("julia") or str(Path.home() / ".juliaup" / "bin" / "julia")
    if not Path(julia).is_file():
        return None
    directory = _REPO_ROOT / "src" / "sc_neurocore" / "accel" / "julia" / "evo_substrate"
    completed = subprocess.run(
        [julia, f"--project={directory}", str(directory / "evo_substrate_bench.jl")],
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    return _parse_lines(completed.stdout) if completed.returncode == 0 else None


def _go() -> Samples | None:
    directory = _REPO_ROOT / "src" / "sc_neurocore" / "accel" / "go" / "evo_substrate"
    binary = directory / "evo_substrate_bench"
    if not binary.exists():
        go = shutil.which("go")
        if go is None:
            return None
        built = subprocess.run(
            [go, "build", "-o", str(binary), "."],
            cwd=directory,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        if built.returncode != 0:
            return None
    completed = subprocess.run(
        [str(binary)],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    if completed.returncode != 0:
        return None
    try:
        parsed = json.loads(completed.stderr.splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return None
    if not isinstance(parsed, dict):
        return None
    results = {
        key: float(value)
        for key, value in parsed.items()
        if key in _KERNELS and isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    return results if set(results) == set(_KERNELS) else None


def _mojo() -> Samples | None:
    pixi = shutil.which("pixi")
    directory = _REPO_ROOT / "src" / "sc_neurocore" / "accel" / "mojo"
    if pixi is None or not (directory / "pixi.toml").is_file():
        return None
    completed = subprocess.run(
        pin_isa([pixi, "run", "mojo", "run", "kernels/evo_substrate_bench.mojo"]),
        cwd=directory,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    return _parse_lines(completed.stdout) if completed.returncode == 0 else None


def _python() -> Samples:
    left, right, mask, noise = _vectors()
    return {
        "genomic_distance_ns_per_call": _ns_per_call(lambda: _python_distance(left, right)),
        "crossover_uniform_ns_per_call": _ns_per_call(lambda: _python_crossover(left, right, mask)),
        "point_mutation_ns_per_call": _ns_per_call(lambda: _python_mutation(left, mask, noise)),
    }


def _source_digest() -> tuple[str, int]:
    """Hash all benchmarked kernel and runner source files."""
    roots = (
        _REPO_ROOT / "src" / "sc_neurocore" / "evo_substrate",
        _REPO_ROOT / "crates" / "evo_substrate_core",
        _REPO_ROOT / "src" / "sc_neurocore" / "accel" / "go" / "evo_substrate",
        _REPO_ROOT / "src" / "sc_neurocore" / "accel" / "julia" / "evo_substrate",
    )
    files: list[Path] = []
    for root in roots:
        files.extend(
            path
            for path in root.rglob("*")
            if path.is_file()
            and "target" not in path.parts
            and path.suffix in {".py", ".rs", ".go", ".jl", ".toml"}
        )
    mojo = _REPO_ROOT / "src" / "sc_neurocore" / "accel" / "mojo" / "kernels"
    files.extend(
        mojo / name
        for name in ("evo_substrate.mojo", "evo_substrate_bench.mojo", "evo_runner.mojo")
    )
    unique = sorted(set(files))
    digest = hashlib.sha256()
    for path in unique:
        digest.update(path.relative_to(_REPO_ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest(), len(unique)


def _summarise(values: list[float]) -> dict[str, object]:
    """Retain raw samples and compute descriptive timing statistics."""
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise RuntimeError("multi-language benchmark produced invalid samples")
    return {
        "sample_count": len(values),
        "samples_ns_per_call": values,
        "min_ns_per_call": min(values),
        "median_ns_per_call": statistics.median(values),
        "mean_ns_per_call": statistics.fmean(values),
        "max_ns_per_call": max(values),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    """Run interleaved backend samples and return source-bound evidence."""
    _validate_args(args)
    samplers: dict[str, Callable[[], Samples | None]] = {
        "rust": _rust,
        "julia": _julia,
        "go": _go,
        "mojo": _mojo,
        "python": _python,
    }
    unavailable: set[str] = set()
    for _ in range(args.warmups):
        for name, sampler in samplers.items():
            if sampler() is None:
                unavailable.add(name)
    raw: dict[str, dict[str, list[float]]] = {
        backend: {kernel: [] for kernel in _KERNELS} for backend in _BACKENDS
    }
    load_before = load_average()
    for _ in range(args.samples):
        for name, sampler in samplers.items():
            sample = sampler()
            if sample is None:
                unavailable.add(name)
                continue
            for kernel in _KERNELS:
                raw[name][kernel].append(sample[kernel])
    if unavailable and not args.allow_missing:
        raise RuntimeError(f"required backends unavailable: {sorted(unavailable)}")
    summaries = {
        backend: {kernel: _summarise(values) for kernel, values in kernels.items() if values}
        for backend, kernels in raw.items()
        if any(kernels.values())
    }
    digest, file_count = _source_digest()
    return {
        "schema_version": _SCHEMA_VERSION,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "command": (
            "JULIA_DEPOT_PATH=build/julia-depot PYTHONPATH=src taskset -c <cpu> "
            ".venv/bin/python benchmarks/bench_evo_substrate_multilang.py"
        ),
        "protocol": {
            "samples": args.samples,
            "warmups": args.warmups,
            "inner_iterations": _ITERS,
            "dimension": _DIM,
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "source": {
            "git_head": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=_REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip(),
            "source_sha256": digest,
            "source_file_count": file_count,
        },
        "measurement_context": measurement_context(load_before),
        "backends": summaries,
        "unavailable": sorted(unavailable),
    }


def _print_summary(payload: dict[str, object]) -> None:
    """Print median kernel latency for every measured backend."""
    backends = payload["backends"]
    if not isinstance(backends, dict):
        raise RuntimeError("benchmark payload has no backend summaries")
    print(f"\n{'Kernel':<28} " + " ".join(f"{name:>12}" for name in _BACKENDS))
    print("-" * 94)
    for kernel in _KERNELS:
        row = [kernel.removesuffix("_ns_per_call")]
        for backend in _BACKENDS:
            backend_data = backends.get(backend)
            value = None
            if isinstance(backend_data, dict):
                kernel_data = backend_data.get(kernel)
                if isinstance(kernel_data, dict):
                    value = kernel_data.get("median_ns_per_call")
            row.append("—" if not isinstance(value, (int, float)) else f"{value:.1f}")
        print(f"{row[0]:<28} " + " ".join(f"{value:>12}" for value in row[1:]))


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark and write its JSON evidence file."""
    args = _parser().parse_args(argv)
    payload = run(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _print_summary(payload)
    print(f"Results written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
