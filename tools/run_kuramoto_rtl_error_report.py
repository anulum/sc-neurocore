#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Kuramoto RTL fixed-point error evidence report

"""Generate a deterministic Kuramoto RTL fixed-point error report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from sc_neurocore.hdl_gen import KuramotoEmitter

SCHEMA_VERSION = 1
REPORT_TYPE = "kuramoto_rtl_fixed_point_error"
EVIDENCE_CLASS = "deterministic_fixed_point_reference"
EVIDENCE_BOUNDARY = (
    "Compares the research Kuramoto RTL emitter fixed-point arithmetic against "
    "the float Kuramoto Euler step and the maintained Rust Kuramoto solver for "
    "the bounded noiseless all-to-all scalar-coupling regime. This is not FPGA "
    "timing or board evidence."
)


def _case(
    *,
    case_name: str,
    phase_regime: str,
    oscillator_count: int,
    omegas: list[float],
    initial_phases: list[float],
    coupling: float,
    dt: float,
    data_width: int,
    fraction: int,
    lut_size: int,
    steps: int,
    max_abs_phase_error_rad: float,
    rms_phase_error_rad: float,
    rust_max_abs_phase_error_rad: float,
    rust_order_parameter_error: float,
) -> dict[str, Any]:
    return {
        "case_name": case_name,
        "config": {
            "phase_regime": phase_regime,
            "oscillator_count": oscillator_count,
            "omegas": omegas,
            "initial_phases": initial_phases,
            "coupling": coupling,
            "dt": dt,
            "data_width": data_width,
            "fraction": fraction,
            "lut_size": lut_size,
            "steps": steps,
        },
        "thresholds": {
            "max_abs_phase_error_rad": max_abs_phase_error_rad,
            "rms_phase_error_rad": rms_phase_error_rad,
            "rust_max_abs_phase_error_rad": rust_max_abs_phase_error_rad,
            "rust_order_parameter_error": rust_order_parameter_error,
        },
    }


DEFAULT_CASES: tuple[dict[str, Any], ...] = (
    _case(
        case_name="higher_coupling_quartet_short",
        phase_regime="nominal_higher_coupling",
        oscillator_count=4,
        omegas=[0.88, 0.96, 1.04, 1.12],
        initial_phases=[0.05, 0.9, 1.7, 2.6],
        coupling=1.0,
        dt=2.5e-4,
        data_width=24,
        fraction=16,
        lut_size=128,
        steps=32,
        max_abs_phase_error_rad=0.01,
        rms_phase_error_rad=0.005,
        rust_max_abs_phase_error_rad=0.01,
        rust_order_parameter_error=0.001,
    ),
    _case(
        case_name="low_coupling_quartet_short",
        phase_regime="nominal_low_coupling",
        oscillator_count=4,
        omegas=[0.8, 1.0, 1.1, 1.3],
        initial_phases=[0.0, 0.3, 0.6, 0.9],
        coupling=0.12,
        dt=5e-3,
        data_width=24,
        fraction=16,
        lut_size=128,
        steps=32,
        max_abs_phase_error_rad=0.015,
        rms_phase_error_rad=0.01,
        rust_max_abs_phase_error_rad=0.015,
        rust_order_parameter_error=0.001,
    ),
    _case(
        case_name="no_coupling_single_oscillator",
        phase_regime="single_oscillator_no_coupling",
        oscillator_count=1,
        omegas=[1.0],
        initial_phases=[0.0],
        coupling=0.0,
        dt=0.01,
        data_width=24,
        fraction=16,
        lut_size=64,
        steps=32,
        max_abs_phase_error_rad=0.001,
        rms_phase_error_rad=0.001,
        rust_max_abs_phase_error_rad=0.001,
        rust_order_parameter_error=0.001,
    ),
    _case(
        case_name="wrap_boundary_pair",
        phase_regime="phase_modulus_wrap_boundary",
        oscillator_count=2,
        omegas=[1.0, 0.9],
        initial_phases=[6.27, 0.01],
        coupling=0.4,
        dt=1e-3,
        data_width=24,
        fraction=16,
        lut_size=128,
        steps=32,
        max_abs_phase_error_rad=0.01,
        rms_phase_error_rad=0.005,
        rust_max_abs_phase_error_rad=0.01,
        rust_order_parameter_error=0.001,
    ),
    _case(
        case_name="near_antiphase_pair",
        phase_regime="near_antiphase_circular_error",
        oscillator_count=2,
        omegas=[1.05, 0.95],
        initial_phases=[0.001, 3.140592653589793],
        coupling=0.8,
        dt=5e-4,
        data_width=24,
        fraction=16,
        lut_size=128,
        steps=32,
        max_abs_phase_error_rad=0.01,
        rms_phase_error_rad=0.005,
        rust_max_abs_phase_error_rad=0.01,
        rust_order_parameter_error=0.001,
    ),
)


def build_report(cases: tuple[dict[str, Any], ...] = DEFAULT_CASES) -> dict[str, Any]:
    """Build a deterministic fixed-point error report for committed evidence."""
    evaluated_cases = [_evaluate_case(case) for case in cases]
    failed_cases = sum(not case["passed"] for case in evaluated_cases)
    return {
        "schema_version": SCHEMA_VERSION,
        "SPDX-License-Identifier": "AGPL-3.0-or-later",
        "report_type": REPORT_TYPE,
        "evidence_class": EVIDENCE_CLASS,
        "evidence_boundary": EVIDENCE_BOUNDARY,
        "hardware_measurement_claimed": False,
        "total_cases": len(evaluated_cases),
        "passed_cases": len(evaluated_cases) - failed_cases,
        "failed_cases": failed_cases,
        "cases": evaluated_cases,
    }


def _evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    config = case["config"]
    emitter = KuramotoEmitter(
        n_oscillators=config["oscillator_count"],
        omegas=config["omegas"],
        initial_phases=config["initial_phases"],
        coupling=config["coupling"],
        dt=config["dt"],
        data_width=config["data_width"],
        fraction=config["fraction"],
        lut_size=config["lut_size"],
    )
    summary = emitter.fixed_point_error_summary(steps=config["steps"])
    rust_parity = _rust_solver_parity(config=config, summary=summary)
    thresholds = case["thresholds"]
    passed = (
        summary["max_abs_phase_error_rad"] < thresholds["max_abs_phase_error_rad"]
        and summary["rms_phase_error_rad"] < thresholds["rms_phase_error_rad"]
        and rust_parity["max_abs_phase_error_rad"] < thresholds["rust_max_abs_phase_error_rad"]
        and rust_parity["order_parameter_error"] < thresholds["rust_order_parameter_error"]
    )
    return {
        "case_name": case["case_name"],
        "config": config,
        "thresholds": thresholds,
        "summary": summary,
        "rust_solver_parity": rust_parity,
        "passed": passed,
    }


def _rust_solver_parity(*, config: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    try:
        from sc_neurocore_engine import KuramotoSolver
    except ImportError as exc:
        raise RuntimeError("sc_neurocore_engine is required for Kuramoto RTL Rust parity") from exc

    n_oscillators = int(config["oscillator_count"])
    coupling = float(config["coupling"])
    solver = KuramotoSolver(
        list(config["omegas"]),
        [coupling] * (n_oscillators * n_oscillators),
        list(config["initial_phases"]),
        noise_amp=0.0,
    )
    solver.run(int(config["steps"]), float(config["dt"]), seed=0)
    rust_phases = [float(phase) for phase in solver.phases]
    fixed_phases = summary["final_fixed_phases_rad"]
    assert isinstance(fixed_phases, list)
    phase_errors = [
        abs(KuramotoEmitter._circular_phase_error(fixed, rust))
        for fixed, rust in zip(fixed_phases, rust_phases)
    ]
    fixed_order = _order_parameter(fixed_phases)
    rust_order = float(solver.order_parameter())
    return {
        "available": True,
        "engine": "sc_neurocore_engine.KuramotoSolver",
        "regime": "noiseless_all_to_all_scalar_coupling",
        "max_abs_phase_error_rad": max(phase_errors),
        "rms_phase_error_rad": (sum(error * error for error in phase_errors) / len(phase_errors))
        ** 0.5,
        "order_parameter_fixed": fixed_order,
        "order_parameter_rust": rust_order,
        "order_parameter_error": abs(fixed_order - rust_order),
        "final_rust_phases_rad": rust_phases,
    }


def _order_parameter(phases: list[float]) -> float:
    import math

    n_inv = 1.0 / len(phases)
    mean_cos = sum(math.cos(phase) for phase in phases) * n_inv
    mean_sin = sum(math.sin(phase) for phase in phases) * n_inv
    return (mean_cos * mean_cos + mean_sin * mean_sin) ** 0.5


def write_report(output: Path) -> Path:
    """Write the canonical report JSON and return its path."""
    payload = build_report()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="Output report JSON path")
    args = parser.parse_args(argv)

    try:
        output = write_report(args.output)
        payload = json.loads(output.read_text(encoding="utf-8"))
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"kuramoto rtl fixed-point error report invalid: {exc}", file=sys.stderr)
        return 1
    print(output)
    return 0 if payload["failed_cases"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
