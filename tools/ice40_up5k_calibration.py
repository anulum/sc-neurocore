# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — iCE40 UP5K structural calibration: fit, check and generate

"""Fit the iCE40 UP5K structural estimate from its measurements and generate the engine constants.

The measurements (``benchmarks/results/ice40_up5k_calibration/designs.json``)
hold every calibration design: its role, family (dense layer, LIF neuron,
synapse stream), parameters, status, routed core logic cells and per-seed
critical path. LIF and synapse are fitted on role ``fit`` and held out as
``held`` and ``held2``. Dense designs of roles ``fit``, ``held``, ``held2`` and
``held3`` inside the domain are all fitted: ``held``, ``held2`` and ``held3``
exceeded the bound of the fit-only model by up to 2.5 % (input counts off the
fit grid), so they joined the fit, and ``held4`` is the fresh held-out set that
validates the refit. ``folded`` dense designs used a weight pattern with about
half of the weights at zero, which synthesis folds away, so they are a measured
lower case, checked against the bound but never fitted. A design whose
five-pin harness did not fit the device still has its core size: the packed
logic cells nextpnr reports before placement, minus the harness flip-flops.
Dense fitted and held-out designs have every weight non-zero, the measured
worst case:
``weight_probes`` records that a zero (or clamped negative) weight only
removes logic and that every non-zero weight costs the same. This is measured,
not proven. This tool is the only place the model is computed:

- logic cells per family are a non-negative least-squares fit on the fit set,
  scaled by the largest measured/fit ratio of that set and by the family's
  explicit margin, and rounded up;
- the critical paths are non-negative least-squares fits of the worst-seed
  path (dense: 1, n_in, n_in²; LIF: 1, width; synapse: a constant), scaled by
  the largest measured/fit ratio of the fit set times the largest
  seed-to-seed spread of the fitted and folded designs. The dense regressors
  were chosen by a rule on the fitted designs alone (every left-out design
  covered, then the tightest mean bound, then the fewest regressors). They
  were chosen after the held4 margins of every candidate had been seen; every
  candidate passes held4, and ``--check`` prints that table.

Synapse streams are measured at every calibrated length (the seven powers of
two from 64 to 4096), so their bound is checked on the whole domain rather
than validated on held-out lengths.

``--check`` refits, checks that every fit, held-out and folded design inside
the domain is within its bound, that every over-capacity design is judged not
to fit, that the weight probes show zero weights only reducing logic, and that
the generated Rust file is current to floating-point rounding
(:func:`constants_match`). ``--write`` regenerates it.

Usage::

    python tools/ice40_up5k_calibration.py --check
    python tools/ice40_up5k_calibration.py --write
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.optimize import nnls

REPO_ROOT = Path(__file__).resolve().parents[1]
MEASUREMENTS = REPO_ROOT / "benchmarks" / "results" / "ice40_up5k_calibration" / "designs.json"
GENERATED = REPO_ROOT / "engine" / "src" / "ir" / "ice40_calibration.rs"

#: Tolerance, relative and absolute, of ``--check`` for each generated float.
#: NNLS and the BLAS kernels under it round differently on different CPUs: an
#: AVX-512 workstation and an AVX-only Xeon disagree in the last one or two
#: digits of eight fitted values, and so do hosted CI runners.
GENERATED_FLOAT_TOLERANCE = 1e-9

#: A floating-point literal of the generated file, not part of a longer word.
_FLOAT_LITERAL = re.compile(r"(?<![\w.])-?\d+(?:\.\d+(?:[eE][-+]?\d+)?|[eE][-+]?\d+)(?![\w.])")

#: Explicit extra logic-cell margin for synapse streams. Synthesis of the
#: power-of-two streams moves by a few cells without following the length
#: (513 at 128, 520 at 512, 522 at 2048 against 4 fit points), which the fit
#: cannot follow; the held-out stream at 512 exceeded the bare bound by 1 cell,
#: and this margin was set after that result was seen.
SYNAPSE_LOGIC_CELL_MARGIN = 1.03

#: Explicit extra logic-cell margin for dense layers. The four smooth
#: regressors cannot follow input counts off the fit grid (3, 7, 9, 10 inputs
#: sit 2-4 % above the grid trend); with every fitted design left out in turn,
#: the refitted bound misses the left-out design by at most 1.2 %
#: (:func:`dense_logic_cell_leave_one_out`, printed by ``--check``).
DENSE_LOGIC_CELL_MARGIN = 1.03

#: Roles fitted per family; every other role is held out or a lower case.
FIT_ROLES: dict[str, tuple[str, ...]] = {
    "dense": ("fit", "held", "held2", "held3"),
    "lif": ("fit",),
    "synapse": ("fit",),
}

#: Calibrated domain of each family: parameter -> inclusive range.
DOMAIN: dict[str, dict[str, tuple[int, int]]] = {
    "dense": {"n_inputs": (2, 12), "n_neurons": (2, 8), "data_width": (12, 24)},
    "lif": {"data_width": (12, 32)},
    "synapse": {"stream_length": (64, 4096)},
}


def logic_cell_features(family: str, x: dict[str, int]) -> list[float]:
    """Return the logic-cell regressors of one design.

    Parameters
    ----------
    family : str
        ``dense``, ``lif`` or ``synapse``.
    x : dict
        Design parameters.

    Returns
    -------
    list of float
        Dense: n_in, n_out, n_in·n_out, n_out·width. LIF: width, width².
        Synapse: 1, log2(stream length).
    """
    if family == "dense":
        n_in, n_out, width = x["n_inputs"], x["n_neurons"], x["data_width"]
        return [n_in, n_out, n_in * n_out, n_out * width]
    if family == "lif":
        return [x["data_width"], x["data_width"] ** 2]
    return [1.0, math.log2(x["stream_length"])]


def path_features(family: str, x: dict[str, int]) -> list[float]:
    """Return the critical-path regressors of one design.

    Dense: 1, n_in, n_in². LIF: 1, width. Synapse: 1.
    """
    if family == "dense":
        return [1.0, x["n_inputs"], x["n_inputs"] ** 2]
    return [1.0, x["data_width"]] if family == "lif" else [1.0]


def in_domain(family: str, x: dict[str, int]) -> bool:
    """Say whether a design lies inside its family's calibrated domain.

    A synapse stream length must also be a power of two: the held-out lengths
    100, 700 and 3000 took 645-889 logic cells and 93-206 ns against 512-527
    cells and under 43 ns for the powers of two, because the decoder's division
    is then a shift.
    """
    if family == "synapse" and x["stream_length"] & (x["stream_length"] - 1):
        return False
    return all(low <= x[key] <= high for key, (low, high) in DOMAIN[family].items())


def core_cells(design: dict[str, Any]) -> int | None:
    """Return the measured core logic cells: routed, or packed when the harness did not fit."""
    if design["status"] == "routed":
        return int(design["core_logic_cells"])
    packed = design.get("packed_core_logic_cells")
    return None if packed is None else int(packed)


def worst_path(design: dict[str, Any]) -> float:
    """Return the worst critical path over the design's seeds, in ns."""
    return float(max(design["critical_path_ns_per_seed"]))


def fit_model(designs: list[dict[str, Any]]) -> dict[str, Any]:
    """Fit every family from the routed designs of the fit set.

    Parameters
    ----------
    designs : list of dict
        All measured designs.

    Returns
    -------
    dict
        ``seed_spread`` and, per family, logic-cell coefficients and bound
        factor, and the critical-path constant or coefficients and factor.
    """
    # The seed spread comes from fitted and folded designs only, so that no
    # held-out design shapes the bound it is checked against.
    routed = [
        d
        for d in designs
        if d["status"] == "routed"
        and (d["role"] in FIT_ROLES[d["family"]] or d["role"] == "folded")
    ]
    spread = max(
        max(d["critical_path_ns_per_seed"]) / min(d["critical_path_ns_per_seed"]) for d in routed
    )
    model: dict[str, Any] = {"seed_spread": spread, "families": {}}
    for family in DOMAIN:
        sized = [
            d
            for d in designs
            if d["family"] == family
            and d["role"] in FIT_ROLES[family]
            and in_domain(family, d["parameters"])
            and core_cells(d) is not None
        ]
        fit = [d for d in sized if d["status"] == "routed"]
        a = np.array([logic_cell_features(family, d["parameters"]) for d in sized], dtype=float)
        y = np.array([core_cells(d) for d in sized], dtype=float)
        coefficients, _ = nnls(a, y)
        factor = float(np.max(y / (a @ coefficients)))
        if family == "synapse":
            factor *= SYNAPSE_LOGIC_CELL_MARGIN
        if family == "dense":
            factor *= DENSE_LOGIC_CELL_MARGIN
        entry: dict[str, Any] = {
            "logic_cell_coefficients": [float(c) for c in coefficients],
            "logic_cell_factor": factor,
        }
        paths = np.array([worst_path(d) for d in fit], dtype=float)
        b = np.array([path_features(family, d["parameters"]) for d in fit], dtype=float)
        path_coefficients, _ = nnls(b, paths)
        entry["path_coefficients"] = [float(c) for c in path_coefficients]
        entry["path_factor"] = float(np.max(paths / (b @ path_coefficients))) * spread
        model["families"][family] = entry
    return model


def bound(model: dict[str, Any], family: str, x: dict[str, int]) -> tuple[int, float]:
    """Return the logic-cell and critical-path bounds the model gives a design."""
    entry = model["families"][family]
    cells = entry["logic_cell_factor"] * float(
        np.dot(logic_cell_features(family, x), entry["logic_cell_coefficients"])
    )
    path = entry["path_factor"] * float(
        np.dot(path_features(family, x), entry["path_coefficients"])
    )
    return math.ceil(cells), path


def coverage(
    model: dict[str, Any], designs: list[dict[str, Any]]
) -> tuple[list[str], dict[tuple[str, str], dict[str, float]]]:
    """Check every in-domain design against its bound.

    Returns
    -------
    tuple
        Failures (one line each) and, per family and role, the smallest
        relative margin of logic cells and of critical path over the routed
        designs checked.
    """
    failures: list[str] = []
    margins: dict[tuple[str, str], dict[str, float]] = {}
    for d in designs:
        family, x = d["family"], d["parameters"]
        if d["status"] == "invalid type" or not in_domain(family, x):
            continue
        cells, path = bound(model, family, x)
        if d["status"] == "over capacity":
            packed = core_cells(d)
            if packed is None:
                failures.append(f"{d['name']}: over capacity without a packed core size")
            elif packed > cells:
                failures.append(f"{d['name']}: packed core {packed} logic cells > bound {cells}")
            continue
        if d["core_logic_cells"] > cells:
            failures.append(f"{d['name']}: {d['core_logic_cells']} logic cells > bound {cells}")
        if worst_path(d) > path:
            failures.append(f"{d['name']}: {worst_path(d):.3f} ns > bound {path:.3f} ns")
        entry = margins.setdefault(
            (family, d["role"]), {"logic_cells": math.inf, "critical_path": math.inf}
        )
        entry["logic_cells"] = min(entry["logic_cells"], (cells - d["core_logic_cells"]) / cells)
        entry["critical_path"] = min(entry["critical_path"], (path - worst_path(d)) / path)
    return failures, margins


def probe_failures(model: dict[str, Any], probes: list[dict[str, Any]]) -> list[str]:
    """Check that zero weights only reduce logic and non-zero weights stay within the bound.

    For each geometry and width, every probe whose codes are all zero must use
    no more logic cells than any other probe, every probe with some zero codes
    no more than any all-non-zero probe, and every all-non-zero probe inside the
    domain must be within the dense bound.

    Returns
    -------
    list of str
        One line per violated relation.
    """
    failures: list[str] = []
    groups: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for row in probes:
        key = (row["n_inputs"], row["n_neurons"], row["data_width"])
        groups.setdefault(key, []).append(row)
    for key, rows in sorted(groups.items()):
        zero = [r["core_logic_cells"] for r in rows if r["q8_codes"] == "all 0"]
        full = [r for r in rows if r["q8_codes"].startswith("all") and r["q8_codes"] != "all 0"]
        partial = [r["core_logic_cells"] for r in rows if not r["q8_codes"].startswith("all")]
        if not full:
            failures.append(f"probe {key}: no all-non-zero weight set to compare with")
            continue
        smallest_full = min(r["core_logic_cells"] for r in full)
        if zero and max(zero) > min(r["core_logic_cells"] for r in rows):
            failures.append(f"probe {key}: an all-zero weight set is not the smallest")
        if partial and max(partial) > smallest_full:
            failures.append(f"probe {key}: a partly zero weight set exceeds an all-non-zero one")
        x = dict(zip(("n_inputs", "n_neurons", "data_width"), key))
        if in_domain("dense", x):
            cells, _ = bound(model, "dense", x)
            for r in full:
                if r["core_logic_cells"] > cells:
                    failures.append(
                        f"probe {key} {r['weights']}: {r['core_logic_cells']} > bound {cells}"
                    )
    return failures


#: Dense critical-path regressor sets compared when the path model was chosen.
PATH_CANDIDATES: dict[str, Callable[[dict[str, int]], list[float]]] = {
    "constant": lambda x: [1.0],
    "[1, n_in]": lambda x: [1.0, x["n_inputs"]],
    "[1, n_in, n_in^2]": lambda x: [1.0, x["n_inputs"], x["n_inputs"] ** 2],
}


def _fitted_dense(designs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the sized in-domain dense designs of the fitted roles."""
    return [
        d
        for d in designs
        if d["family"] == "dense"
        and d["role"] in FIT_ROLES["dense"]
        and in_domain("dense", d["parameters"])
        and core_cells(d) is not None
    ]


def _bounded_fit(a: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Return NNLS coefficients and the largest measured/fit ratio."""
    coefficients, _ = nnls(a, y)
    return coefficients, float(np.max(y / (a @ coefficients)))


def dense_logic_cell_leave_one_out(designs: list[dict[str, Any]]) -> tuple[float, str]:
    """Return the worst miss of the bare dense bound on each left-out fitted design.

    Each fitted design is left out in turn; the others are refitted and scaled
    by their own largest ratio, without :data:`DENSE_LOGIC_CELL_MARGIN`. The
    result is the largest measured/bound ratio of a left-out design and its name,
    the basis of that margin.
    """
    sized = _fitted_dense(designs)
    a = np.array([logic_cell_features("dense", d["parameters"]) for d in sized], dtype=float)
    y = np.array([core_cells(d) for d in sized], dtype=float)
    worst, name = 0.0, ""
    for i in range(len(sized)):
        keep = np.arange(len(sized)) != i
        coefficients, factor = _bounded_fit(a[keep], y[keep])
        ratio = float(y[i] / (factor * (a[i] @ coefficients)))
        if ratio > worst:
            worst, name = ratio, sized[i]["name"]
    return worst, name


def dense_path_selection(
    designs: list[dict[str, Any]], spread: float
) -> dict[str, dict[str, float]]:
    """Compare the dense path regressor sets by the rule that chose the model.

    Rule, on the fitted designs only: every left-out design covered, then the
    smallest mean bound/measured ratio, then the fewest regressors. ``held4``
    is reported beside it; the regressors were chosen after its margins had
    been seen, so every candidate's held4 result is printed too.

    Returns
    -------
    dict
        Per candidate: ``loo_worst`` and ``loo_fails`` (left-out ratio to bound),
        ``mean_bound_ratio`` (bound/measured over the fitted designs) and
        ``held4_worst`` (largest held4 measured/bound ratio).
    """
    routed = [d for d in _fitted_dense(designs) if d["status"] == "routed"]
    held4 = [
        d
        for d in designs
        if d["family"] == "dense"
        and d["role"] == "held4"
        and d["status"] == "routed"
        and in_domain("dense", d["parameters"])
    ]
    y = np.array([worst_path(d) for d in routed], dtype=float)
    table: dict[str, dict[str, float]] = {}
    for name, features in PATH_CANDIDATES.items():
        a = np.array([features(d["parameters"]) for d in routed], dtype=float)
        ratios = []
        for i in range(len(routed)):
            keep = np.arange(len(routed)) != i
            coefficients, factor = _bounded_fit(a[keep], y[keep])
            ratios.append(float(y[i] / (factor * spread * (a[i] @ coefficients))))
        coefficients, factor = _bounded_fit(a, y)
        bounds = factor * spread * (a @ coefficients)
        held = [
            worst_path(d)
            / (factor * spread * float(np.dot(features(d["parameters"]), coefficients)))
            for d in held4
        ]
        table[name] = {
            "loo_worst": max(ratios),
            "loo_fails": float(sum(r > 1.0 for r in ratios)),
            "mean_bound_ratio": float(np.mean(bounds / y)),
            "held4_worst": max(held) if held else math.nan,
        }
    return table


def render(model: dict[str, Any], measured: dict[str, Any]) -> str:
    """Render the generated Rust constants."""
    dense, lif, synapse = (model["families"][f] for f in ("dense", "lif", "synapse"))

    def array(name: str, values: list[float]) -> list[str]:
        head = f"pub const {name}: [f64; {len(values)}] = "
        one_line = head + "[" + ", ".join(repr(v) for v in values) + "];"
        if len(one_line) <= 100:
            return [one_line]
        return [head + "["] + [f"    {v!r}," for v in values] + ["];"]

    # The embedded SPDX text belongs to the generated Rust file.
    # REUSE-IgnoreStart
    lines = [
        "// SPDX-License-Identifier: AGPL-3.0-or-later",
        "// Commercial license available",
        "// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.",
        "// © Code 2020–2026 Miroslav Šotek. All rights reserved.",
        "// ORCID: 0009-0009-3560-0851",
        "// Contact: www.anulum.li | protoscience@anulum.li",
        "// SC-NeuroCore — iCE40 UP5K structural calibration constants (generated)",
        # REUSE-IgnoreEnd
        "",
        "//! Generated by `tools/ice40_up5k_calibration.py --write` from",
        "//! `benchmarks/results/ice40_up5k_calibration/designs.json`. Do not edit.",
        "",
        "/// Toolchain the calibration was measured with.",
        f"pub const TOOLCHAIN: &str = {json.dumps(measured['toolchain']['yosys'] + '; nextpnr-ice40 ' + measured['toolchain']['nextpnr_ice40'])};",
        "/// Clock of the calibration runs in MHz.",
        f"pub const CLOCK_MHZ: u32 = {measured['clock_mhz']};",
        "/// Dense layer logic cells: n_in, n_out, n_in·n_out, n_out·width.",
        *array("DENSE_LOGIC_CELLS", dense["logic_cell_coefficients"]),
        "/// Largest measured/fit logic-cell ratio of the dense fit set times its explicit margin.",
        f"pub const DENSE_LOGIC_CELL_FACTOR: f64 = {dense['logic_cell_factor']!r};",
        "/// Dense layer critical path in ns: 1, n_in, n_in².",
        *array("DENSE_PATH", dense["path_coefficients"]),
        "/// Largest measured/fit path ratio of the dense fit set times the seed spread.",
        f"pub const DENSE_PATH_FACTOR: f64 = {dense['path_factor']!r};",
        "/// LIF neuron logic cells: width, width².",
        *array("LIF_LOGIC_CELLS", lif["logic_cell_coefficients"]),
        "/// Largest measured/fit logic-cell ratio of the LIF fit set.",
        f"pub const LIF_LOGIC_CELL_FACTOR: f64 = {lif['logic_cell_factor']!r};",
        "/// LIF neuron critical path in ns: 1, width.",
        *array("LIF_PATH", lif["path_coefficients"]),
        "/// Largest measured/fit path ratio of the LIF fit set times the seed spread.",
        f"pub const LIF_PATH_FACTOR: f64 = {lif['path_factor']!r};",
        "/// Synapse stream logic cells: 1, log2(stream length).",
        *array("SYNAPSE_LOGIC_CELLS", synapse["logic_cell_coefficients"]),
        "/// Largest measured/fit logic-cell ratio of the synapse fit set times its explicit margin.",
        f"pub const SYNAPSE_LOGIC_CELL_FACTOR: f64 = {synapse['logic_cell_factor']!r};",
        "/// Synapse stream critical path in ns.",
        *array("SYNAPSE_PATH", synapse["path_coefficients"]),
        "/// Largest measured/fit path ratio of the synapse fit set times the seed spread.",
        f"pub const SYNAPSE_PATH_FACTOR: f64 = {synapse['path_factor']!r};",
        "/// Calibrated dense inputs, inclusive.",
        f"pub const DENSE_INPUTS: (usize, usize) = {DOMAIN['dense']['n_inputs']};",
        "/// Calibrated dense neurons, inclusive.",
        f"pub const DENSE_NEURONS: (usize, usize) = {DOMAIN['dense']['n_neurons']};",
        "/// Calibrated dense data width, inclusive.",
        f"pub const DENSE_WIDTH: (u32, u32) = {DOMAIN['dense']['data_width']};",
        "/// Calibrated LIF data width, inclusive.",
        f"pub const LIF_WIDTH: (u32, u32) = {DOMAIN['lif']['data_width']};",
        "/// Calibrated synapse stream length, inclusive; only powers of two are calibrated.",
        f"pub const SYNAPSE_LENGTH: (usize, usize) = {DOMAIN['synapse']['stream_length']};",
        "",
    ]
    return "\n".join(lines)


def constants_match(committed: str, generated: str) -> bool:
    """Return whether two renderings differ only in floating-point rounding.

    Parameters
    ----------
    committed : str
        Text of the committed Rust file.
    generated : str
        Text that :func:`render` produced on this machine.

    Returns
    -------
    bool
        True when the text outside the floating-point literals is the same,
        apart from whitespace and the trailing comma before ``]`` (an array
        moves between one line and several with the length of its literals),
        and every literal is within :data:`GENERATED_FLOAT_TOLERANCE` of its
        counterpart.
    """

    def skeleton(text: str) -> str:
        return re.sub(r"\s+", "", _FLOAT_LITERAL.sub("#", text)).replace(",]", "]")

    if skeleton(committed) != skeleton(generated):
        return False
    return all(
        math.isclose(
            float(a),
            float(b),
            rel_tol=GENERATED_FLOAT_TOLERANCE,
            abs_tol=GENERATED_FLOAT_TOLERANCE,
        )
        for a, b in zip(
            _FLOAT_LITERAL.findall(committed), _FLOAT_LITERAL.findall(generated), strict=True
        )
    )


def main(argv: list[str] | None = None) -> int:
    """Run the check or regenerate the constants.

    Parameters
    ----------
    argv : list of str or None
        Command-line arguments.

    Returns
    -------
    int
        0 when the calibration holds and the generated file is current
        (``--check``) or was written (``--write``); 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    measured = json.loads(MEASUREMENTS.read_text(encoding="utf-8"))
    model = fit_model(measured["designs"])
    failures, margins = coverage(model, measured["designs"])
    failures += probe_failures(model, measured["weight_probes"]["rows"])
    for (family, role), margin in sorted(margins.items()):
        print(
            f"{family} {role}: smallest margin logic cells {margin['logic_cells']:.4f}, "
            f"critical path {margin['critical_path']:.4f}"
        )
    worst, name = dense_logic_cell_leave_one_out(measured["designs"])
    print(
        f"dense logic cells, bare bound with each fitted design left out: worst "
        f"measured/bound {worst:.4f} ({name}); explicit margin {DENSE_LOGIC_CELL_MARGIN}"
    )
    print("dense path regressors (leave-one-out on fitted designs; held4 beside):")
    for candidate, row in dense_path_selection(measured["designs"], model["seed_spread"]).items():
        print(
            f"  {candidate}: LOO worst {row['loo_worst']:.3f}, LOO fails {row['loo_fails']:.0f}, "
            f"mean bound/measured {row['mean_bound_ratio']:.3f}, held4 worst {row['held4_worst']:.3f}"
        )
    if failures:
        print("calibration does not hold:")
        for failure in failures:
            print(f"  {failure}")
        return 1
    text = render(model, measured)
    if args.write:
        GENERATED.write_text(text, encoding="utf-8")
        print(f"wrote {GENERATED.relative_to(REPO_ROOT)}")
        return 0
    if not GENERATED.exists() or not constants_match(GENERATED.read_text(encoding="utf-8"), text):
        print(f"{GENERATED.relative_to(REPO_ROOT)} is stale; run with --write")
        return 1
    print("calibration holds and the generated constants are current")
    return 0


if __name__ == "__main__":
    sys.exit(main())
