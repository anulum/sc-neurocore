# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Equation source and numerical provenance

"""Bind ordered model semantics and numerical choices to generated native graphs."""

from __future__ import annotations

from dataclasses import asdict
import json

from ...neurons.equation_builder import EquationNeuron
from ..verilog_compiler_config import Q88


def model_source(
    neuron: EquationNeuron,
    q: Q88,
    module_name: str,
    pipeline_stages: int,
    pipeline_points: tuple[str, ...],
) -> str:
    """Serialize initial model semantics, retaining declaration and reset order.

    Parameters
    ----------
    neuron : EquationNeuron
        Authored model before intrinsic preparation. Live state and random
        counters are deliberately absent: compilation starts at initial state.
    q : Q88
        Complete fixed-point, function and random-precision configuration.
    module_name : str
        Actual native hardware module name.
    pipeline_stages : int
        Requested global product-register selection.
    pipeline_points : tuple of str
        Explicit ordered product-register selection.

    Returns
    -------
    str
        Canonical UTF-8 JSON with ordered pairs for semantically ordered maps.
        Strict models retain both runtime base units and state display units.

    Raises
    ------
    ValueError
        A numerical model value is nonfinite and cannot be a source contract.
    """
    payload = {
        "schema": "sc-equation-source.v1",
        "module_name": module_name,
        "model": {
            "equations": list(neuron.equations.items()),
            "parameters": list(neuron.parameters.items()),
            "constants": list(neuron.constants.items()),
            "initial_state": list(neuron.initial_state.items()),
            "threshold": neuron.threshold_expr,
            "reset": list(neuron.reset_rules.items()),
            "spike_value": neuron.spike_value_expr,
            "dt": neuron.dt,
            "method": neuron.method,
            "detection": neuron.detection,
            "substeps": neuron.substeps,
            "rate_expression": neuron.rate_expression,
            "probability_expression": neuron.probability_expression,
            "rng_seed": neuron.random_seed,
            "random_stream": neuron.random_stream,
            "threshold_rng": neuron.threshold_rng,
            "units": neuron.units,
            "runtime_units": {name: str(unit) for name, unit in neuron._runtime_units.items()},
            "base_state_units": {
                name: str(unit) for name, unit in neuron._base_state_units.items()
            },
            "display_state_units": {
                name: str(unit) for name, unit in neuron._display_state_units.items()
            },
        },
        "numeric": asdict(q),
        "pipeline": {"stages": pipeline_stages, "points": pipeline_points},
        "semantics": {
            "state_overflow": q.overflow,
            "event_overflow": "wrap" if q.overflow == "trap" else q.overflow,
            "division": "encoded-nonzero-or-source-guard",
            "reset": "asynchronous-active-low-initial-state",
            "rng": "model-seed64-stream32-site-counter",
        },
    }
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
