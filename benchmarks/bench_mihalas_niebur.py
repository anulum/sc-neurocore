# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Python/Rust benchmark for Mihalas-Niebur RK4 dynamics

"""Measure Mihalas-Niebur Python reference and Rust PyO3 class throughput."""

from __future__ import annotations

import importlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

from sc_neurocore.neurons.models.mihalas_niebur import MihalasNieburNeuron

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_STEPS = 100_000
PARITY_STEPS = 10_000
CURRENT = 0.002
INTEGRATOR = "source_equations_2_1_2_2_sampled_rk4"


def _fixed(value: float, digits: int) -> float:
    return float(f"{value:.{digits}f}")


def _cpu_model() -> str:
    try:
        with Path("/proc/cpuinfo").open(encoding="utf-8") as file:
            for line in file:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _state_dict(neuron: MihalasNieburNeuron) -> dict[str, float]:
    return {
        "v": neuron.v,
        "theta": neuron.theta,
        "i1": neuron.i1,
        "i2": neuron.i2,
    }


def _run_python(n_steps: int) -> tuple[float, dict[str, float | int]]:
    neuron = MihalasNieburNeuron()
    spikes = 0
    start = time.perf_counter()
    for _ in range(n_steps):
        spikes += neuron.step(CURRENT)
    wall = time.perf_counter() - start
    return wall, {"spikes": spikes, **_state_dict(neuron)}


def _probe_rust_class() -> type[Any] | None:
    try:
        module = importlib.import_module("sc_neurocore_engine")
    except ImportError:
        return None
    cls = getattr(module, "MihalasNieburNeuron", None)
    return cls if isinstance(cls, type) else None


def _run_rust(n_steps: int) -> tuple[float, dict[str, float | int]] | None:
    cls = _probe_rust_class()
    if cls is None:
        return None
    neuron = cls()
    spikes = 0
    start = time.perf_counter()
    for _ in range(n_steps):
        spikes += int(neuron.step(CURRENT))
    wall = time.perf_counter() - start
    state = neuron.get_state()
    return wall, {
        "spikes": spikes,
        "v": float(state["v"]),
        "theta": float(state["theta"]),
        "i1": float(state["i1"]),
        "i2": float(state["i2"]),
    }


def _parity() -> dict[str, float | str]:
    rust = _run_rust(PARITY_STEPS)
    if rust is None:
        return {"status": "skipped"}
    _, py_state = _run_python(PARITY_STEPS)
    _, rust_state = rust
    fields = ("v", "theta", "i1", "i2")
    return {
        "status": "measured",
        "max_abs_delta": max(
            abs(float(py_state[name]) - float(rust_state[name])) for name in fields
        ),
        "spikes_delta": abs(int(py_state["spikes"]) - int(rust_state["spikes"])),
    }


def main() -> int:
    py_wall, py_state = _run_python(N_STEPS)
    backends: dict[str, dict[str, Any]] = {
        "python": {
            "available": True,
            "used": True,
            "wall_ms": _fixed(py_wall * 1e3, 3),
            "steps_per_s": _fixed(N_STEPS / py_wall, 0),
            "state": py_state,
        }
    }
    rust = _run_rust(N_STEPS)
    if rust is None:
        backends["rust"] = {
            "available": False,
            "used": False,
            "reason": "sc_neurocore_engine missing",
        }
    else:
        rust_wall, rust_state = rust
        backends["rust"] = {
            "available": True,
            "used": True,
            "wall_ms": _fixed(rust_wall * 1e3, 3),
            "steps_per_s": _fixed(N_STEPS / rust_wall, 0),
            "speedup_over_python": _fixed(py_wall / rust_wall, 2),
            "state": rust_state,
        }

    parity = _parity()
    header = f"{'Backend':<10} {'Steps/s':>15} {'Wall (ms)':>12} {'Speedup':>10}"
    print(header)
    print("-" * len(header))
    for name in ("python", "rust"):
        backend = backends[name]
        if not backend.get("used", False):
            print(f"{name:<10} {'MISSING':>15} {'':>12} {'':>10}")
            continue
        speedup = backend.get("speedup_over_python", 1.0)
        print(
            f"{name:<10} {int(backend['steps_per_s']):>15,} "
            f"{float(backend['wall_ms']):>12.2f} {float(speedup):>9.2f}x"
        )
    print(f"Parity status: {parity}")

    meta = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cpu": _cpu_model(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "n_steps": N_STEPS,
        "parity_steps": PARITY_STEPS,
        "current": CURRENT,
        "integrator": INTEGRATOR,
    }
    out_path = RESULTS_DIR / "bench_mihalas_niebur.json"
    with out_path.open("w", encoding="utf-8") as file:
        json.dump({"meta": meta, "backends": backends, "parity": parity}, file, indent=2)
    print(f"Results -> {out_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
