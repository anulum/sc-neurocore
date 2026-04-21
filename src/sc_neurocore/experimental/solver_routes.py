# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Real solver routes for the safe alternative-path harness

from __future__ import annotations

import numpy as np

from sc_neurocore.solvers import ExactLIFSolver, RK4Solver

from .alternative_path import AlternativePathRoute


def _lif_rhs(
    _t: float,
    y: np.ndarray,
    *,
    tau: float,
    v_rest: float,
    current: float,
    r_m: float,
) -> np.ndarray:
    return np.array([(-(y[0] - v_rest) + r_m * current) / tau], dtype=np.float64)


def _lif_rk4_baseline(
    v0: float,
    current: float,
    horizon: float,
    *,
    tau: float = 20.0,
    v_rest: float = -65.0,
    v_thresh: float = -50.0,
    r_m: float = 1.0,
    dt: float = 1e-2,
) -> dict[str, float | bool | None]:
    steps = max(1, int(round(horizon / dt)))
    solver = RK4Solver()
    y = np.array([v0], dtype=np.float64)
    t = 0.0

    for _ in range(steps):
        y, dt_used = solver.step(
            lambda time, state: _lif_rhs(
                time,
                state,
                tau=tau,
                v_rest=v_rest,
                current=current,
                r_m=r_m,
            ),
            y,
            t,
            dt,
        )
        t += dt_used

    voltage = float(y[0])
    return {
        "voltage": voltage,
        "distance_to_threshold": float(v_thresh - voltage),
        "subthreshold": bool(voltage < v_thresh),
        "predicted_spike_time": None,
    }


def _lif_exact_candidate(
    v0: float,
    current: float,
    horizon: float,
    *,
    tau: float = 20.0,
    v_rest: float = -65.0,
    v_thresh: float = -50.0,
    r_m: float = 1.0,
    dt: float = 1e-2,
) -> dict[str, float | bool | None]:
    del dt
    solver = ExactLIFSolver(
        tau=tau,
        v_rest=v_rest,
        v_thresh=v_thresh,
        v_reset=v_rest,
        r_m=r_m,
    )
    voltage = float(solver.evolve_to_time(v0, horizon, current))
    spike_time = solver.next_spike_time(v0, current)
    return {
        "voltage": voltage,
        "distance_to_threshold": float(v_thresh - voltage),
        "subthreshold": bool(voltage < v_thresh),
        "predicted_spike_time": spike_time if spike_time is None else float(spike_time),
    }


def make_lif_subthreshold_exact_route() -> AlternativePathRoute[dict[str, float | bool | None]]:
    """Route subthreshold LIF integration against the analytical solution."""

    return AlternativePathRoute(
        name="solver.lif.subthreshold-exact",
        baseline=_lif_rk4_baseline,
        candidate=_lif_exact_candidate,
        summary="RK4 baseline vs exact LIF solution in the subthreshold regime",
        expected_behavior=(
            "For constant subthreshold current, the analytical candidate should "
            "match the RK4 baseline while remaining below spike threshold"
        ),
    )
