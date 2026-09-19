# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

"""Source/binary-bound five-runtime SC stochastic-adaptation benchmark."""

from __future__ import annotations
import argparse
import hashlib
import importlib
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any
import numpy as np
import numpy.typing as npt
from sc_neurocore.accel.sc_stochastic_rate_adaptation import (
    PARITY_ATOL,
    backend_available,
    simulate_sc_stochastic_rate_adaptation,
)

STEPS = 20_000
REPEATS = 5
BACKENDS = ("python", "rust", "julia", "go", "mojo")
ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = {
    "benchmark": "benchmarks/bench_model_sc_stochastic_rate_adaptation.py",
    "python": "src/sc_neurocore/neurons/models/sc_stochastic_rate_adaptation.py",
    "dispatch": "src/sc_neurocore/accel/sc_stochastic_rate_adaptation.py",
    "rust_model": "engine/src/neurons/simple_spiking/sc_stochastic_rate_adaptation.rs",
    "rust_binding": "engine/src/bindings/stochastic/sc_stochastic_rate_adaptation.rs",
    "julia": "src/sc_neurocore/accel/julia/neurons/sc_stochastic_rate_adaptation.jl",
    "go": "src/sc_neurocore/accel/go/sc_stochastic_rate_adaptation/sc_stochastic_rate_adaptation.go",
    "go_loader": "src/sc_neurocore/accel/go/sc_stochastic_rate_adaptation/__init__.py",
    "mojo": "src/sc_neurocore/accel/mojo/kernels/sc_stochastic_rate_adaptation.mojo",
    "mojo_abi": "src/sc_neurocore/accel/mojo/sc_stochastic_rate_adaptation/sc_stochastic_rate_adaptation_abi.mojo",
    "mojo_loader": "src/sc_neurocore/accel/mojo/sc_stochastic_rate_adaptation/__init__.py",
}
BINARY_FILES = {
    "go_binary": "src/sc_neurocore/accel/go/sc_stochastic_rate_adaptation/libsc_stochastic_rate_adaptation.so",
    "mojo_binary": "src/sc_neurocore/accel/mojo/sc_stochastic_rate_adaptation/libsc_stochastic_rate_adaptation.so",
}


def _hashes() -> dict[str, str]:
    """Bind the result to the measured source and native library bytes."""
    hashes = {
        key: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for key, path in {**SOURCE_FILES, **BINARY_FILES}.items()
    }
    rust = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")
    rust_file = rust.__file__
    if rust_file is None:
        raise RuntimeError("Rust backend has no binary path")
    hashes["rust_binary"] = hashlib.sha256(Path(rust_file).read_bytes()).hexdigest()
    return hashes


def main(argv: list[str] | None = None) -> int:
    """Measure all five public backends and reject incomplete parity evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--allow-unpinned", action="store_true")
    args = parser.parse_args(argv)
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) != 1 and not args.allow_unpinned:
        print(f"Refusing unpinned benchmark; affinity is {affinity}")
        return 2
    missing = [backend for backend in BACKENDS if not backend_available(backend)]
    if missing:
        print("Missing required backend(s): " + ", ".join(missing))
        return 2
    drive = np.tile(np.array([0.0, 10.0, 25.0, 50.0]), STEPS // 4)
    uniforms = np.random.default_rng(42).random(STEPS)
    summary: dict[str, dict[str, Any]] = {}
    reference_events: npt.NDArray[np.int64] | None = None
    reference_adaptation: npt.NDArray[np.float64] | None = None
    trace_digest: str | None = None
    for backend in BACKENDS:
        timings: list[int] = []
        last: dict[str, object] = {}
        simulate_sc_stochastic_rate_adaptation(drive[:4], uniforms[:4], backend=backend)
        for _ in range(REPEATS):
            start = time.perf_counter_ns()
            last = simulate_sc_stochastic_rate_adaptation(drive, uniforms, backend=backend)
            timings.append(time.perf_counter_ns() - start)
        events = np.asarray(last["events"], dtype=np.int64)
        adaptation = np.asarray(last["adaptation"], dtype=np.float64)
        digest = hashlib.sha256(events.tobytes()).hexdigest()
        if reference_events is None:
            reference_events = events
            reference_adaptation = adaptation
            trace_digest = digest
            maximum_error = 0.0
        else:
            if reference_adaptation is None:
                raise RuntimeError("Python reference adaptation is missing")
            maximum_error = float(np.max(np.abs(adaptation - reference_adaptation)))
            if not np.array_equal(events, reference_events) or maximum_error > PARITY_ATOL[backend]:
                print(f"Parity failure for {backend}: maximum error {maximum_error}")
                return 3
        summary[backend] = {
            "median_ns": int(statistics.median(timings)),
            "events": int(events.sum()),
            "max_adaptation_error": maximum_error,
        }
    record = {
        "benchmark": "SC retained stochastic rate-adaptation recurrence",
        "evidence_class": "local_regression_non_isolated",
        "hardware_measurement_claimed": False,
        "production_speed_claim": False,
        "steps": STEPS,
        "repeats": REPEATS,
        "source_hashes": _hashes(),
        "backend_summary": summary,
        "event_trace_sha256": trace_digest,
        "measurement_affinity": affinity,
    }
    encoded = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if args.json is None:
        print(encoded, end="")
    else:
        args.json.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
