# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Canonical equation source replay

"""Reconstruct validated equation inputs from the native graph's bound source."""

from __future__ import annotations

from collections.abc import Iterator
import json
import math

from ...neurons.equation_builder import EquationNeuron
from ..verilog_compiler_config import Q88
from .source import model_source


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or any(type(key) is not str for key in value):
        raise ValueError("equation source requires an object with string keys")
    return dict(value)


def _text(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("equation source requires a string")
    return value


def _optional_text(value: object) -> str | None:
    return None if value is None else _text(value)


def _integer(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("equation source requires an integer")
    return value


def _number(value: object) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError("equation source requires a real number")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("equation source requires finite numbers")
    return value


def _pairs(value: object) -> Iterator[tuple[str, object]]:
    if not isinstance(value, list):
        raise ValueError("equation source requires ordered name/value pairs")
    seen: set[str] = set()
    for pair in value:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError("equation source requires two-element pairs")
        name = _text(pair[0])
        if name in seen:
            raise ValueError(f"duplicate equation source name: {name!r}")
        seen.add(name)
        yield name, pair[1]


def _texts(value: object) -> dict[str, str]:
    return {name: _text(item) for name, item in _pairs(value)}


def _numbers(value: object) -> dict[str, float]:
    return {name: _number(item) for name, item in _pairs(value)}


def restore_source(payload: str) -> tuple[EquationNeuron, Q88, str, int, list[str]]:
    """Reconstruct the exact canonical initial model and numerical compiler request.

    Parameters
    ----------
    payload : str
        Canonical ``sc-equation-source.v1`` payload produced by the public
        compiler. Ordered model maps, all modes and unit metadata are retained.

    Returns
    -------
    tuple
        Initial model, fixed-point configuration, module name, global pipeline
        request and explicit product selections. Source replay neither imports
        code from the payload nor restores a caller's running trajectory.

    Raises
    ------
    ValueError
        The source schema, types, units, model or canonical encoding is invalid.
    ImportError
        A strict-units model requires the optional dimensional-analysis package.
    """
    try:
        raw: object = json.loads(payload)
        root = _mapping(raw)
        if root["schema"] != "sc-equation-source.v1":
            raise ValueError("unsupported equation source schema")
        model, numeric, pipeline = (
            _mapping(root[name]) for name in ("model", "numeric", "pipeline")
        )
        signed = numeric["signed"]
        if not isinstance(signed, bool):
            raise ValueError("equation source signedness must be a boolean")
        q = Q88(
            _integer(numeric["data_width"]),
            _integer(numeric["fraction"]),
            signed,
            _text(numeric["overflow"]),
            _text(numeric["rounding"]),
            _text(numeric["function_lowering"]),
            _text(numeric["random_precision"]),
        )
        name, stages = _text(root["module_name"]), _integer(pipeline["stages"])
        raw_points = pipeline["points"]
        if not isinstance(raw_points, list):
            raise ValueError("equation source pipeline points must be a list")
        points = [_text(point) for point in raw_points]
        parameters, state, constants = (
            _numbers(model[field]) for field in ("parameters", "initial_state", "constants")
        )
        dt = _number(model["dt"])
        units = _text(model["units"])
        runtime_units = _mapping(model["runtime_units"])
        input_unit = None
        if units == "strict":
            from ...neurons._units import UNIT_REGISTRY, require_pint

            require_pint()
            parameters = {
                key: value * UNIT_REGISTRY(_text(runtime_units[key]))
                for key, value in parameters.items()
            }
            state_units = _mapping(model["base_state_units"])
            state = {
                key: value * UNIT_REGISTRY(_text(state_units[key])) for key, value in state.items()
            }
            constants = {
                key: value * UNIT_REGISTRY(_text(runtime_units[key]))
                for key, value in constants.items()
            }
            dt = dt * UNIT_REGISTRY.second
            if "I" in runtime_units:
                input_unit = UNIT_REGISTRY(_text(runtime_units["I"]))
        neuron = EquationNeuron(
            _texts(model["equations"]),
            parameters=parameters,
            state=state,
            threshold=_optional_text(model["threshold"]),
            reset=_texts(model["reset"]),
            constants=constants,
            dt=dt,
            method=_text(model["method"]),
            units=units,
            input_unit=input_unit,
            detection=_text(model["detection"]),
            substeps=_integer(model["substeps"]),
            rate_expression=_optional_text(model["rate_expression"]),
            probability_expression=_optional_text(model["probability_expression"]),
            rng_seed=_integer(model["rng_seed"]),
            random_stream=_integer(model["random_stream"]),
            spike_value=_optional_text(model["spike_value"]),
            threshold_rng=_text(model["threshold_rng"]),
        )
        if units == "strict":
            display = _mapping(model["display_state_units"])
            neuron._display_state_units = {
                key: UNIT_REGISTRY(_text(unit)).units for key, unit in display.items()
            }
        if model_source(neuron, q, name, stages, tuple(points)) != payload:
            raise ValueError("equation source is noncanonical or contains unsupported fields")
        return neuron, q, name, stages, points
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid equation source contract") from exc
