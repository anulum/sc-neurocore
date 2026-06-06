# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — HIL calibration protocol

"""Hardware-in-the-loop (HIL) calibration protocol generation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class HILCalibration:
    """Hardware-in-the-loop calibration protocol.

    Attributes
    ----------
    protocol_steps : list[str]
    num_parameters : int
    sweep_ranges : dict[str, tuple[float, float]]
    """

    protocol_steps: list[str]
    num_parameters: int
    sweep_ranges: dict[str, tuple[float, float]]
    design_matrix: list[dict[str, float]] = field(default_factory=list)
    sample_count: int = 0
    repetitions: int = 1
    settle_cycles: int = 0
    acceptance_tolerance: float = 1.0 / 256.0
    correction_model: str = "weighted_least_squares"
    observables: tuple[str, ...] = field(default_factory=tuple)


def _validate_hil_contract(
    module_name: str,
    equations: dict[str, str],
    parameters: dict[str, tuple[float, float]],
    sample_points: int,
    repetitions: int,
    settle_cycles: int,
    acceptance_tolerance: float,
) -> dict[str, tuple[float, float]]:
    if not module_name.strip():
        raise ValueError("module_name must be non-empty")
    if not equations:
        raise ValueError("equations must contain at least one state variable")
    if sample_points < 2:
        raise ValueError("sample_points must be >= 2")
    if repetitions < 1:
        raise ValueError("repetitions must be >= 1")
    if settle_cycles < 0:
        raise ValueError("settle_cycles must be >= 0")
    if not math.isfinite(acceptance_tolerance) or acceptance_tolerance <= 0.0:
        raise ValueError("acceptance_tolerance must be finite and > 0")

    clean: dict[str, tuple[float, float]] = {}
    for name, bounds in parameters.items():
        if len(bounds) != 2:
            raise ValueError(f"parameter {name!r} must define exactly two bounds")
        lo, hi = float(bounds[0]), float(bounds[1])
        if not (math.isfinite(lo) and math.isfinite(hi)):
            raise ValueError(f"parameter {name!r} bounds must be finite")
        if lo >= hi:
            raise ValueError(f"parameter {name!r} requires lower < upper")
        clean[str(name)] = (lo, hi)
    if not clean:
        raise ValueError("parameters must contain at least one sweep range")
    return clean


def _coprime_stride(sample_points: int, start: int) -> int:
    stride = max(1, start)
    while math.gcd(stride, sample_points) != 1:
        stride += 1
    return stride


def _latin_hypercube_design(
    parameters: dict[str, tuple[float, float]],
    sample_points: int,
) -> list[dict[str, float]]:
    names = list(parameters)
    design: list[dict[str, float]] = []
    for sample_idx in range(sample_points):
        point: dict[str, float] = {}
        for dim_idx, name in enumerate(names):
            lo, hi = parameters[name]
            stride = _coprime_stride(sample_points, 2 * dim_idx + 1)
            slot = (sample_idx * stride + dim_idx) % sample_points
            fraction = (slot + 0.5) / sample_points
            point[name] = lo + fraction * (hi - lo)
        design.append(point)
    return design


def generate_hil_calibration(
    module_name: str,
    equations: dict[str, str],
    *,
    parameters: dict[str, tuple[float, float]] | None = None,
    sample_points: int = 10,
    repetitions: int = 3,
    settle_cycles: int = 32,
    acceptance_tolerance: float = 1.0 / 256.0,
    correction_model: str = "weighted_least_squares",
    observables: tuple[str, ...] | None = None,
) -> HILCalibration:
    """Generate hardware-in-the-loop calibration protocol."""
    if parameters is None:
        parameters = {var: (0.0, 1.0) for var in equations}
    sweep_ranges = _validate_hil_contract(
        module_name,
        equations,
        parameters,
        sample_points,
        repetitions,
        settle_cycles,
        acceptance_tolerance,
    )
    observable_names = tuple(observables or equations.keys())
    unknown_observables = [name for name in observable_names if name not in equations]
    if unknown_observables:
        raise ValueError(f"observables are not present in equations: {unknown_observables}")

    design_matrix = _latin_hypercube_design(sweep_ranges, sample_points)
    sample_count = len(design_matrix) * repetitions
    steps = [
        f"1. Deploy {module_name} bitstream to target hardware",
        "2. Connect telemetry capture for "
        + ", ".join(observable_names)
        + " and lock the hardware/software timestep",
        f"3. Initialise all {len(equations)} state variables to zero",
        f"4. Execute {len(design_matrix)} deterministic Latin-hypercube design points "
        f"with {repetitions} repetitions each",
    ]
    step_num = 5
    for param, (lo, hi) in parameters.items():
        steps.append(
            f"{step_num}. Calibrate '{param}' over [{lo}, {hi}] using the shared design matrix"
        )
        step_num += 1
    steps.extend(
        [
            f"{step_num}. At each point, settle {settle_cycles} cycles before sampling",
            f"{step_num + 1}. Compare measured {', '.join(observable_names)} traces to the software golden model",
            f"{step_num + 2}. Fit {correction_model} coefficients to residuals and measured variance",
            f"{step_num + 3}. Program correction coefficients and rerun the design matrix",
            f"{step_num + 4}. Accept only if max absolute drift <= {acceptance_tolerance:g}",
        ]
    )

    return HILCalibration(
        protocol_steps=steps,
        num_parameters=len(sweep_ranges),
        sweep_ranges=sweep_ranges,
        design_matrix=design_matrix,
        sample_count=sample_count,
        repetitions=repetitions,
        settle_cycles=settle_cycles,
        acceptance_tolerance=acceptance_tolerance,
        correction_model=correction_model,
        observables=observable_names,
    )
