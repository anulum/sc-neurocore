# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Equation program benchmark: cost of running checked expressions

"""Measure equation programs against the same arithmetic written as Python functions.

An equation program is a tree of plain function calls translated from a checked
expression (:mod:`sc_neurocore.neurons.equation_program`). The reference for
each case is the same arithmetic as an ordinary Python function reading the
same bindings, which the interpreter runs as bytecode. Both must return the
same value of the same type before anything is timed. The ratio says what
running a checked tree costs over bytecode; it is not a comparison with the
removed ``eval`` path, which no longer exists to be measured.

Whole neuron steps are timed as absolute figures for three configurations. A
step also builds its bindings and draws noise, so its time is not the sum of
its expressions.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import timeit
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from benchmarks._benchmark_context import load_average, measurement_context

from sc_neurocore.neurons.equation_builder import EquationNeuron  # noqa: E402
from sc_neurocore.neurons.equation_namespace import build_eval_namespace  # noqa: E402
from sc_neurocore.neurons.equation_program import compile_expression  # noqa: E402

DEFAULT_OUTPUT = ROOT / "benchmarks/results/bench_equation_program.json"
SOURCE_PATHS = (
    "benchmarks/bench_equation_program.py",
    "benchmarks/_benchmark_context.py",
    "src/sc_neurocore/neurons/equation_builder.py",
    "src/sc_neurocore/neurons/equation_evaluation.py",
    "src/sc_neurocore/neurons/equation_namespace.py",
    "src/sc_neurocore/neurons/equation_program.py",
    "src/sc_neurocore/neurons/equation_safety.py",
)
BINDINGS: dict[str, float] = {
    "v": -60.0,
    "E_L": -65.0,
    "R": 10.0,
    "I": 1.5,
    "tau": 10.0,
    "g_L": 30.0,
    "delta_T": 2.0,
    "V_T": -50.0,
    "w": 0.1,
    "C": 281.0,
    "m": 0.05,
    "delta": 1.0,
    "r": 0.2,
    "v_th": -50.0,
}
ADEX = "(-g_L*(v-E_L) + g_L*delta_T*exp((v-V_T)/delta_T) - w + I)/C"
Reference = Callable[[dict[str, Any]], Any]
EXPRESSIONS: dict[str, tuple[str, Reference]] = {
    "hypot": ("hypot(v, I)", lambda e: e["hypot"](e["v"], e["I"])),
    "round": ("round(I)", lambda e: e["round"](e["I"])),
    "heaviside": (
        "heaviside(v, I)",
        lambda e: e["heaviside"](e["v"], e["I"]),
    ),
    "leaky_integrator": (
        "(-(v - E_L) + R * I) / tau",
        lambda e: (-(e["v"] - e["E_L"]) + e["R"] * e["I"]) / e["tau"],
    ),
    "adaptive_exponential": (
        ADEX,
        lambda e: (
            (
                -e["g_L"] * (e["v"] - e["E_L"])
                + e["g_L"] * e["delta_T"] * e["exp"]((e["v"] - e["V_T"]) / e["delta_T"])
                - e["w"]
                + e["I"]
            )
            / e["C"]
        ),
    ),
    "gating_variable": (
        "0.1*(v+40.0)/(1.0-exp(-(v+40.0)/10.0))*(1.0-m) - 4.0*exp(-(v+65.0)/18.0)*m",
        lambda e: (
            0.1 * (e["v"] + 40.0) / (1.0 - e["exp"](-(e["v"] + 40.0) / 10.0)) * (1.0 - e["m"])
            - 4.0 * e["exp"](-(e["v"] + 65.0) / 18.0) * e["m"]
        ),
    ),
    "mean_field_rate": (
        "delta/(3.141592653589793*tau**2) + 2.0*r*v/tau",
        lambda e: (
            e["delta"] / (3.141592653589793 * e["tau"] ** 2) + 2.0 * e["r"] * e["v"] / e["tau"]
        ),
    ),
    "threshold": ("v >= v_th", lambda e: e["v"] >= e["v_th"]),
    "conditional": ("v + 1 if v < 10 else -v", lambda e: e["v"] + 1 if e["v"] < 10 else -e["v"]),
    "angle": ("atan2(v, I)", lambda e: e["atan2"](e["v"], e["I"])),
    "softplus": ("softplus(v)", lambda e: e["softplus"](e["v"])),
}
ADEX_NEURON: dict[str, Any] = {
    "equations": {"v": ADEX, "w": "(a*(v-E_L) - w)/tau_w"},
    "state": {"v": -65.0, "w": 0.0},
    "parameters": {
        "g_L": 30.0,
        "E_L": -70.6,
        "delta_T": 2.0,
        "V_T": -50.4,
        "C": 281.0,
        "a": 4.0,
        "tau_w": 144.0,
    },
    "threshold": "v > 20",
    "reset": {"v": "-70.6", "w": "w + 80.5"},
    "dt": 0.1,
}
NEURONS: dict[str, dict[str, Any]] = {
    "leaky_integrator_euler": {
        "equations": {"v": "(-(v - E_L) + R * I) / tau"},
        "state": {"v": -65.0},
        "parameters": {"E_L": -65.0, "R": 10.0, "tau": 10.0},
        "threshold": "v > -50",
        "reset": {"v": "-65.0"},
        "dt": 0.1,
    },
    "adaptive_exponential_euler": ADEX_NEURON,
    "adaptive_exponential_rk4": {**ADEX_NEURON, "method": "rk4"},
}


def _per_call_ns(function: Callable[[], Any], calls: int, repeats: int) -> list[float]:
    """Return the time per call, in nanoseconds, of each repeat of ``calls`` calls."""
    return [
        1e9 * sample / calls for sample in timeit.repeat(function, number=calls, repeat=repeats)
    ]


def measure_expression(name: str, calls: int, repeats: int) -> dict[str, Any]:
    """Time one expression as a program and as the reference function.

    Parameters
    ----------
    name : str
        Key of :data:`EXPRESSIONS`.
    calls : int
        Calls per timed repeat.
    repeats : int
        Timed repeats; the minimum and the median are both recorded.

    Returns
    -------
    dict
        The expression, both sample lists, their minima and medians, and the
        ratio of the minima.

    Raises
    ------
    ValueError
        The program and the reference return different values or types.
    """
    expression, reference = EXPRESSIONS[name]
    program = compile_expression(expression)
    bindings: dict[str, Any] = {**build_eval_namespace(), **BINDINGS}
    value, expected = program(bindings), reference(bindings)
    if value != expected or type(value) is not type(expected):
        raise ValueError(f"program and reference disagree for {name}")
    program_ns = _per_call_ns(lambda: program(bindings), calls, repeats)
    reference_ns = _per_call_ns(lambda: reference(bindings), calls, repeats)
    return {
        "expression": expression,
        "value": repr(value),
        "program_ns": program_ns,
        "reference_ns": reference_ns,
        "program_min_ns": min(program_ns),
        "reference_min_ns": min(reference_ns),
        "program_median_ns": statistics.median(program_ns),
        "reference_median_ns": statistics.median(reference_ns),
        "ratio_of_minima": min(program_ns) / min(reference_ns),
    }


def measure_neuron(name: str, calls: int, repeats: int) -> dict[str, Any]:
    """Time whole steps of one neuron configuration.

    Parameters
    ----------
    name : str
        Key of :data:`NEURONS`.
    calls : int
        Steps per timed repeat.
    repeats : int
        Timed repeats.

    Returns
    -------
    dict
        Sample list, minimum and median of the time per step in nanoseconds,
        and the state after the last step.
    """
    neuron = EquationNeuron(**NEURONS[name])
    step_ns = _per_call_ns(lambda: neuron.step(I=500.0), calls, repeats)
    return {
        "method": NEURONS[name].get("method", "euler"),
        "step_ns": step_ns,
        "step_min_ns": min(step_ns),
        "step_median_ns": statistics.median(step_ns),
        "final_state": {key: float(value) for key, value in neuron.state.items()},
    }


def build_payload(calls: int, steps: int, repeats: int) -> dict[str, Any]:
    """Measure every expression and neuron and bind the result to its sources.

    Parameters
    ----------
    calls : int
        Expression calls per timed repeat.
    steps : int
        Neuron steps per timed repeat.
    repeats : int
        Timed repeats of each measurement.

    Returns
    -------
    dict
        The receipt: configuration, environment, source digests, one entry
        per expression and per neuron, and the range of the ratios.

    Raises
    ------
    ValueError
        A count is not positive, or a program disagrees with its reference.
    """
    if min(calls, steps, repeats) <= 0:
        raise ValueError("calls, steps and repeats must be positive")
    load_before = load_average()
    expressions = {name: measure_expression(name, calls, repeats) for name in EXPRESSIONS}
    ratios = [entry["ratio_of_minima"] for entry in expressions.values()]
    return {
        "schema_version": "sc-neurocore.equation-program-benchmark.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evidence_class": "local_regression",
        "production_speed_claimed": False,
        "interpretation": (
            "Loaded-host local-regression evidence. The reference is the same arithmetic as a "
            "Python function, not the removed eval path."
        ),
        "configuration": {"calls": calls, "steps": steps, "repeats": repeats},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor() or "unknown",
            "affinity": sorted(os.sched_getaffinity(0)),
            "load_average": list(os.getloadavg()),
        },
        "measurement_context": measurement_context(load_before),
        "source_sha256": {
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in SOURCE_PATHS
        },
        "expressions": expressions,
        "neurons": {name: measure_neuron(name, steps, repeats) for name in NEURONS},
        "ratio_of_minima_range": [min(ratios), max(ratios)],
        "all_values_equal_reference": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark and write its receipt.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments; the process arguments when omitted.

    Returns
    -------
    int
        Zero after the receipt is written.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calls", type=int, default=200_000)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args(argv)
    payload = build_payload(arguments.calls, arguments.steps, arguments.repeats)
    arguments.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    low, high = payload["ratio_of_minima_range"]
    print(f"program / reference, ratio of minima: {low:.2f} to {high:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
