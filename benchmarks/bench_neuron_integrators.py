# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Cross-language RK4 neuron integrator parity benchmark

"""Cross-language RK4 parity and timing harness for priority neurons."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sc_neurocore.neurons.models.adex import AdExNeuron
from sc_neurocore.neurons.models.hodgkin_huxley import HodgkinHuxleyNeuron
from sc_neurocore.neurons.sc_izhikevich import SCIzhikevichNeuron

RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"

PARITY_STEPS = 1_000
BENCH_STEPS = 10_000
DEFAULT_REPEATS = 3
BACKEND_NAMES = ("python", "rust", "julia", "go", "mojo")

BackendFn = Callable[[str, np.ndarray, float | None], dict[str, Any]]
ProbeResult = tuple[BackendFn | None, str]


@dataclass(frozen=True)
class ModelCase:
    """Deterministic RK4 trace case for one neuron model."""

    model_name: str
    display_name: str
    dt: float
    state_keys: tuple[str, ...]
    tolerance: float

    def currents(self, n_steps: int) -> np.ndarray:
        phase = np.linspace(0.0, 2.0 * np.pi, n_steps, dtype=np.float64)
        if self.model_name == "izhikevich":
            return np.full(n_steps, 1.0, dtype=np.float64)
        if self.model_name == "hodgkin_huxley":
            return 10.0 + 1.5 * np.sin(phase) + 0.5 * np.cos(3.0 * phase)
        if self.model_name == "adex":
            return 450.0 + 90.0 * np.sin(phase) + 30.0 * np.cos(2.0 * phase)
        raise ValueError(f"unsupported RK4 neuron model {self.model_name!r}")


MODEL_CASES = (
    ModelCase("izhikevich", "Izhikevich", 1.0, ("v", "u"), 1e-9),
    ModelCase("hodgkin_huxley", "Hodgkin-Huxley", 0.02, ("v", "m", "h", "n"), 1e-8),
    ModelCase("adex", "AdEx", 0.2, ("v", "w"), 1e-9),
)


def _normalise_model_name(model_name: str) -> str:
    return "".join(ch.lower() for ch in model_name if ch.isalnum())


def _python_reference(
    model_name: str,
    currents: np.ndarray,
    dt: float | None,
) -> dict[str, Any]:
    model = _normalise_model_name(model_name)
    if model in {"izhikevich", "scizhikevichneuron", "izhikevichneuron"}:
        izh_neuron = SCIzhikevichNeuron(
            dt=1.0 if dt is None else dt,
            noise_std=0.0,
            integrator="rk4",
        )
        izh_v: list[float] = []
        izh_u: list[float] = []
        izh_spikes: list[int] = []
        for idx, current in enumerate(currents):
            if izh_neuron.step(float(current)):
                izh_spikes.append(idx)
            state = izh_neuron.get_state()
            izh_v.append(state["v"])
            izh_u.append(state["u"])
        return {
            "v": np.asarray(izh_v, dtype=np.float64),
            "u": np.asarray(izh_u, dtype=np.float64),
            "spikes": np.asarray(izh_spikes, dtype=np.uint64),
            "n_steps": currents.size,
        }

    if model in {"hodgkinhuxley", "hodgkinhuxleyneuron"}:
        hh_neuron = HodgkinHuxleyNeuron(dt=0.01 if dt is None else dt, integrator="rk4")
        hh_v = np.empty(currents.size, dtype=np.float64)
        hh_m = np.empty(currents.size, dtype=np.float64)
        hh_h = np.empty(currents.size, dtype=np.float64)
        hh_gate_n = np.empty(currents.size, dtype=np.float64)
        hh_spikes: list[int] = []
        for idx, current in enumerate(currents):
            if hh_neuron.step(float(current)):
                hh_spikes.append(idx)
            hh_v[idx] = hh_neuron.v
            hh_m[idx] = hh_neuron.m
            hh_h[idx] = hh_neuron.h
            hh_gate_n[idx] = hh_neuron.n
        return {
            "v": hh_v,
            "m": hh_m,
            "h": hh_h,
            "n": hh_gate_n,
            "spikes": np.asarray(hh_spikes, dtype=np.uint64),
            "n_steps": currents.size,
        }

    if model in {"adex", "adexneuron"}:
        adex_neuron = AdExNeuron(dt=0.1 if dt is None else dt, integrator="rk4")
        adex_v = np.empty(currents.size, dtype=np.float64)
        adex_w = np.empty(currents.size, dtype=np.float64)
        adex_spikes: list[int] = []
        for idx, current in enumerate(currents):
            if adex_neuron.step(float(current)):
                adex_spikes.append(idx)
            adex_v[idx] = adex_neuron.v
            adex_w[idx] = adex_neuron.w
        return {
            "v": adex_v,
            "w": adex_w,
            "spikes": np.asarray(adex_spikes, dtype=np.uint64),
            "n_steps": currents.size,
        }

    raise ValueError(f"unsupported RK4 neuron model {model_name!r}")


def _probe_rust() -> ProbeResult:
    try:
        module = importlib.import_module("sc_neurocore_engine")
    except ImportError as exc:
        return None, f"missing: {exc}"
    fn = getattr(module, "py_rk4_neuron_simulate", None)
    if not callable(fn):
        return None, "engine wheel built without RK4 neuron simulator binding"
    return fn, "available"


def _probe_julia() -> ProbeResult:
    try:
        module = importlib.import_module("sc_neurocore.accel.julia.neurons")
    except ImportError as exc:
        return None, f"missing: {exc}"
    if not bool(getattr(module, "_HAS_JULIA_NEURONS", False)):
        return None, "juliacall not installed"
    fn = getattr(module, "simulate_rk4_neuron", None)
    if not callable(fn):
        return None, "Julia RK4 neuron wrapper missing"
    return fn, "available"


def _probe_go() -> ProbeResult:
    try:
        module = importlib.import_module("sc_neurocore.accel.go.rk4_neurons")
    except ImportError as exc:
        return None, f"missing: {exc}"
    if not bool(getattr(module, "_HAS_GO_RK4_NEURONS", False)):
        return None, "librk4_neurons.so not built with go build"
    fn = getattr(module, "simulate_rk4_neuron", None)
    if not callable(fn):
        return None, "Go RK4 neuron wrapper missing"
    return fn, "available"


def _probe_mojo() -> ProbeResult:
    try:
        module = importlib.import_module("sc_neurocore.accel.mojo.rk4_neurons")
    except ImportError as exc:
        return None, f"missing: {exc}"
    if not bool(getattr(module, "_HAS_MOJO_RK4_NEURONS", False)):
        return None, "librk4_neurons.so not built with mojo build"
    fn = getattr(module, "simulate_rk4_neuron", None)
    if not callable(fn):
        return None, "Mojo RK4 neuron wrapper missing"
    return fn, "available"


def discover_backends() -> dict[str, dict[str, Any]]:
    """Return all five backend slots with callable functions where available."""
    probes: dict[str, Callable[[], ProbeResult]] = {
        "rust": _probe_rust,
        "julia": _probe_julia,
        "go": _probe_go,
        "mojo": _probe_mojo,
    }
    discovered: dict[str, dict[str, Any]] = {
        "python": {"available": True, "reason": "primary reference", "fn": _python_reference}
    }
    for name, probe in probes.items():
        fn, reason = probe()
        discovered[name] = {"available": fn is not None, "reason": reason, "fn": fn}
    return discovered


def _compare_outputs(
    reference: dict[str, Any],
    actual: dict[str, Any],
    state_keys: Iterable[str],
) -> dict[str, Any]:
    spikes_ref = np.asarray(reference["spikes"], dtype=np.uint64)
    spikes_actual = np.asarray(actual["spikes"], dtype=np.uint64)
    spikes_equal = bool(np.array_equal(spikes_ref, spikes_actual))
    state_deltas: dict[str, float] = {}
    state_bit_exact: dict[str, bool] = {}
    for key in state_keys:
        ref_arr = np.asarray(reference[key], dtype=np.float64)
        actual_arr = np.asarray(actual[key], dtype=np.float64)
        state_deltas[key] = float(np.max(np.abs(ref_arr - actual_arr))) if ref_arr.size else 0.0
        state_bit_exact[key] = bool(np.array_equal(ref_arr, actual_arr))
    max_abs_delta = max(state_deltas.values(), default=0.0)
    return {
        "n_steps": int(actual["n_steps"]),
        "spike_indices_equal": spikes_equal,
        "state_max_abs_delta": state_deltas,
        "max_abs_delta": max_abs_delta,
        "bit_exact": bool(spikes_equal and all(state_bit_exact.values())),
    }


def run_parity_suite(n_steps: int = PARITY_STEPS) -> dict[str, Any]:
    """Run deterministic 1 000-step parity traces against the Python RK4 reference."""
    backends = discover_backends()
    model_results: dict[str, Any] = {}

    for case in MODEL_CASES:
        currents = case.currents(n_steps)
        reference = _python_reference(case.model_name, currents, case.dt)
        backend_results: dict[str, Any] = {}
        for backend_name in BACKEND_NAMES:
            backend = backends[backend_name]
            fn = backend["fn"]
            if not backend["available"] or fn is None:
                backend_results[backend_name] = {
                    "available": False,
                    "used": False,
                    "reason": backend["reason"],
                }
                continue
            actual = (
                reference if backend_name == "python" else fn(case.model_name, currents, case.dt)
            )
            comparison = _compare_outputs(reference, actual, case.state_keys)
            backend_results[backend_name] = {
                "available": True,
                "used": True,
                "reason": backend["reason"],
                "within_tolerance": bool(
                    comparison["spike_indices_equal"]
                    and comparison["max_abs_delta"] <= case.tolerance
                ),
                **comparison,
            }
        model_results[case.model_name] = {
            "display_name": case.display_name,
            "dt": case.dt,
            "n_steps": n_steps,
            "tolerance": case.tolerance,
            "backends": backend_results,
        }

    return {
        "n_steps": n_steps,
        "backend_order": BACKEND_NAMES,
        "models": model_results,
    }


def _time_call(
    fn: BackendFn, case: ModelCase, currents: np.ndarray, repeats: int
) -> dict[str, float]:
    fn(case.model_name, currents[: min(128, currents.size)], case.dt)
    timings: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn(case.model_name, currents, case.dt)
        timings.append(time.perf_counter() - start)
    best = min(timings)
    return {
        "best_wall_ms": round(best * 1e3, 3),
        "steps_per_s": round(currents.size / best, 0),
    }


def run_benchmark(
    n_steps: int = BENCH_STEPS,
    repeats: int = DEFAULT_REPEATS,
) -> dict[str, Any]:
    """Measure available RK4 backends on deterministic traces."""
    backends = discover_backends()
    parity = run_parity_suite(PARITY_STEPS)
    bench_results: dict[str, Any] = {}

    for case in MODEL_CASES:
        currents = case.currents(n_steps)
        case_results: dict[str, Any] = {}
        python_time: float | None = None
        for backend_name in BACKEND_NAMES:
            backend = backends[backend_name]
            fn = backend["fn"]
            if not backend["available"] or fn is None:
                case_results[backend_name] = {
                    "available": False,
                    "used": False,
                    "reason": backend["reason"],
                }
                continue
            timing = _time_call(fn, case, currents, repeats)
            if backend_name == "python":
                python_time = timing["best_wall_ms"]
                speedup = 1.0
            elif python_time is not None and timing["best_wall_ms"] > 0:
                speedup = round(python_time / timing["best_wall_ms"], 3)
            else:
                speedup = 0.0
            case_results[backend_name] = {
                "available": True,
                "used": True,
                "reason": backend["reason"],
                **timing,
                "speedup_over_python": speedup,
            }
        bench_results[case.model_name] = case_results

    return {
        "meta": {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "cpu": _cpu_model(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "n_steps": n_steps,
            "parity_steps": PARITY_STEPS,
            "repeats": repeats,
        },
        "benchmark": bench_results,
        "parity": parity,
    }


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _print_parity(parity: dict[str, Any]) -> None:
    print(f"Parity trace length: {parity['n_steps']} steps")
    for model_name, model in parity["models"].items():
        print(f"\n{model['display_name']} ({model_name}, dt={model['dt']})")
        for backend_name in BACKEND_NAMES:
            result = model["backends"][backend_name]
            if not result.get("used", False):
                print(f"  {backend_name:<6} missing  {result['reason']}")
                continue
            marker = "ok" if result["within_tolerance"] else "FAIL"
            exact = "bit-exact" if result["bit_exact"] else f"max Δ={result['max_abs_delta']:.3e}"
            print(f"  {backend_name:<6} {marker:<4} {exact}")


def _print_benchmark(payload: dict[str, Any]) -> None:
    print("\nTiming best-of repeats")
    header = f"{'Model':<16} {'Backend':<8} {'Steps/s':>14} {'Wall ms':>12} {'Speedup':>10}"
    print(header)
    print("-" * len(header))
    for model_name, backends in payload["benchmark"].items():
        for backend_name in BACKEND_NAMES:
            result = backends[backend_name]
            if not result.get("used", False):
                print(f"{model_name:<16} {backend_name:<8} {'missing':>14} {'':>12} {'':>10}")
                continue
            print(
                f"{model_name:<16} {backend_name:<8} "
                f"{int(result['steps_per_s']):>14,} "
                f"{result['best_wall_ms']:>12.3f} "
                f"{result['speedup_over_python']:>9.3f}x"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=BENCH_STEPS)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--parity-only", action="store_true")
    parser.add_argument("--json", type=Path, default=RESULTS_DIR / "bench_neuron_integrators.json")
    args = parser.parse_args(argv)

    if args.parity_only:
        parity = run_parity_suite(PARITY_STEPS)
        _print_parity(parity)
        payload = {"parity": parity}
    else:
        payload = run_benchmark(args.steps, args.repeats)
        _print_parity(payload["parity"])
        _print_benchmark(payload)

    args.json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.json, "w") as handle:
        json.dump(payload, handle, indent=2)
    try:
        result_path = args.json.relative_to(REPO_ROOT)
    except ValueError:
        result_path = args.json
    print(f"\nResults -> {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
