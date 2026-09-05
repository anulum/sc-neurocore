# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Float64 versus bit-true fixed-point comparison of an equation run

"""Compare a float64 equation run with its genuine fixed-point execution.

Three runs of the same experiment are reported and kept apart:

* ``float_result`` — the float64 explicit-Euler reference (the playground
  run itself);
* ``fixed_result`` — the bit-true fixed-point run of the maintained C kernel
  (:mod:`sc_neurocore.studio.bit_true_execution`), whose arithmetic the
  ``arithmetic`` block states from the kernel generator; every state word and
  spike of every step is decoded and compared;
* ``parameter_quantisation_result`` — float64 with the parameters, initial
  state and time step rounded to the word resolution, so the reader can
  separate the sensitivity to parameter rounding from the effect of
  fixed-point operations (wrap-truncate multiplies, saturation, look-up
  tables).

A value the word cannot hold (a parameter, a constant in an expression, the
initial state, the time step or an input sample) is a rejection with the
field named, never a silent clamp: a clamped parameter is a different model,
not a precision effect.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np

from sc_neurocore.compiler.c_fixed_emitter import signed_q
from sc_neurocore.compiler.intelligence.bit_true_kernel import kernel_arithmetic_contract
from sc_neurocore.compiler.q_format import QFormat
from sc_neurocore.compiler.verilog_compiler_config import Q88
from sc_neurocore.neurons.equation_builder import EquationNeuron, from_equations
from sc_neurocore.studio.analysis_contract import (
    MODEL_DEFINED_UNIT,
    MetricContract,
    attach_contract,
)
from sc_neurocore.studio.bit_true_execution import BitTrueTrace, run_bittrue_kernel
from sc_neurocore.studio.model_run_contract import ModelInputError
from sc_neurocore.studio.simulation import (
    MAX_STEPS,
    ODE_MODEL_NAME,
    current_trace,
    simulate,
    spike_statistics,
)
from sc_neurocore.studio.state_layout import equation_state, observe_layout, snapshot
from sc_neurocore.studio.trace_projection import (
    MAX_PLOT_POINTS,
    RAW_ELEMENT_BUDGET,
    custody_payload,
    full_state_traces,
)

PRECISION_COMPARE_SCHEMA_VERSION = "studio.precision-compare.v2"
SUPPORTED_OVERFLOW: tuple[str, ...] = ("saturate", "wrap")
SUPPORTED_ROUNDING: tuple[str, ...] = ("truncate", "nearest")
MIN_DATA_WIDTH = 8
MAX_DATA_WIDTH = 32

OverflowMode = Literal["saturate", "wrap"]
RoundingMode = Literal["truncate", "nearest"]


def _input_error(field: str, reason: str) -> ModelInputError:
    return ModelInputError(model=ODE_MODEL_NAME, field=field, reason=reason)


def resolve_word_format(q_format: str) -> tuple[int, int]:
    """Parse a Studio Q-format label into ``(data_width, fraction)``.

    ``Q8.8`` is 16 bits with 8 fractional bits; the kernel supports total
    widths from 8 to 32 bits with at least one integer bit.
    """
    try:
        parsed = QFormat.from_string(q_format)
    except (ValueError, TypeError) as exc:
        raise _input_error("q_format", f"{q_format!r} is not a Q<int>.<frac> label") from exc
    data_width = parsed.total_bits
    fraction = parsed.fraction_bits
    if not MIN_DATA_WIDTH <= data_width <= MAX_DATA_WIDTH:
        raise _input_error(
            "q_format",
            f"{q_format} has {data_width} bits; the bit-true kernel supports "
            f"{MIN_DATA_WIDTH} to {MAX_DATA_WIDTH}",
        )
    if fraction >= data_width:
        raise _input_error("q_format", f"{q_format} leaves no integer bit")
    return data_width, fraction


def _encode(q: Q88, value: float) -> int:
    return signed_q(q, value)


def _representable(q: Q88, value: float) -> bool:
    return math.isfinite(value) and q.min_value <= value <= q.max_value


def _right_hand_side(text: str) -> str:
    """Return the expression part of ``dx/dt = f`` / ``x = g`` (or ``text`` itself)."""
    head, separator, tail = text.partition("=")
    if separator and not tail.startswith("=") and not head.rstrip().endswith(("<", ">", "!")):
        return tail.strip()
    return text.strip()


def _expression_constants(expression: str) -> list[tuple[float, str]]:
    """Return ``(value, role)`` of every numeric literal the emitter encodes.

    A literal dividing a sub-expression is encoded as its reciprocal, a
    modulo period as itself; every other literal is encoded directly.
    """
    try:
        tree = ast.parse(_right_hand_side(expression), mode="eval")
    except SyntaxError:
        return []
    found: list[tuple[float, str]] = []
    reciprocal_nodes: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            right = node.right
            if isinstance(right, ast.Constant) and isinstance(right.value, (int, float)):
                reciprocal_nodes.add(id(right))
                if float(right.value) != 0.0:
                    found.append((1.0 / float(right.value), f"reciprocal of {right.value!r}"))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)
            and id(node) not in reciprocal_nodes
        ):
            found.append((float(node.value), f"literal {node.value!r}"))
    return found


def _reject_unrepresentable(
    q: Q88,
    q_label: str,
    neuron: EquationNeuron,
    equations: Sequence[str],
    threshold: str | None,
    reset: str | None,
    dt: float,
) -> None:
    span = f"[{q.min_value}, {q.max_value}]"
    for name, value in {**neuron.parameters, **neuron.constants}.items():
        if not _representable(q, float(value)):
            # The DSL stores a constant reset value as the parameter ``<var>_reset_val``;
            # report it under the request field the user wrote.
            field = "reset" if name.endswith("_reset_val") else f"params.{name}"
            raise _input_error(field, f"{value!r} is not representable in {q_label} {span}")
    for name, value in neuron.initial_state.items():
        if not _representable(q, float(value)):
            raise _input_error(
                f"init.{name}", f"{value!r} is not representable in {q_label} {span}"
            )
    if not _representable(q, dt):
        raise _input_error("dt", f"{dt!r} is not representable in {q_label} {span}")
    if _encode(q, dt) == 0:
        raise _input_error("dt", f"{dt!r} underflows the {q_label} resolution {q.resolution}")
    sources: list[tuple[str, str]] = [("equations", text) for text in equations]
    if threshold:
        sources.append(("threshold", threshold))
    if reset:
        sources.append(("reset", reset))
    for field, text in sources:
        for value, role in _expression_constants(text):
            if not _representable(q, value):
                raise _input_error(
                    field, f"{role} in {text!r} is not representable in {q_label} {span}"
                )


def _quantised(q: Q88, values: Mapping[str, float]) -> dict[str, float]:
    scale = float(1 << q.fraction)
    return {name: _encode(q, float(value)) / scale for name, value in values.items()}


def _encoding_rows(q: Q88, values: Mapping[str, float]) -> dict[str, dict[str, float | int]]:
    scale = float(1 << q.fraction)
    rows: dict[str, dict[str, float | int]] = {}
    for name, value in values.items():
        word = _encode(q, float(value))
        rows[name] = {
            "requested": float(value),
            "word": word,
            "quantised": word / scale,
            "abs_error": abs(float(value) - word / scale),
        }
    return rows


def _sha256_floats(values: np.ndarray[Any, Any]) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype=np.float64).tobytes()).hexdigest()


def _sha256_json(payload: object) -> str:
    encoded = json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _fixed_result(
    trace: BitTrueTrace,
    *,
    dt: float,
    drive: np.ndarray[Any, Any],
    quantised_init: Mapping[str, float],
) -> dict[str, Any]:
    """Assemble the custody payload of the kernel run from its decoded words."""
    decoded = trace.decoded()
    n_steps = trace.n_steps
    names = list(trace.variables)
    layout = observe_layout(
        dict(quantised_init),
        "equations",
        "",
        equation_state(names, quantised_init),
        n_steps=n_steps,
        element_budget=RAW_ELEMENT_BUDGET,
    )
    final = {name: float(decoded[name][-1]) for name in names}
    spikes = trace.spike_steps()
    payload = custody_payload(
        dt=dt,
        n_steps=n_steps,
        layout=layout,
        initial_state=snapshot(dict(quantised_init), layout),
        final_state=snapshot(final, layout),
        scalar_traces={name: decoded[name] for name in names},
        vector_traces={},
        vector_omitted=(),
        drive=drive,
        spikes=spikes,
        stats=spike_statistics(spikes, dt, n_steps),
        max_points=MAX_PLOT_POINTS,
    )
    payload["backend"] = "bit-true-kernel"
    payload["words"] = {
        "schema_version": "studio.bit-true-words.v1",
        "fraction": trace.fraction,
        "state": {name: trace.words[:, index].tolist() for index, name in enumerate(names)},
        "drive": trace.drive_words.tolist(),
    }
    return payload


def _variable_comparison(
    reference: np.ndarray[Any, Any],
    candidate: np.ndarray[Any, Any],
    *,
    resolution: float,
    display_index: np.ndarray[Any, Any],
) -> dict[str, Any]:
    error = np.abs(reference - candidate)
    beyond = np.flatnonzero(error > resolution / 2.0)
    return {
        "max_abs_error": float(error.max()) if error.size else 0.0,
        "mean_abs_error": float(error.mean()) if error.size else 0.0,
        "rms_error": float(np.sqrt(np.mean(error**2))) if error.size else 0.0,
        "final_abs_error": float(error[-1]) if error.size else 0.0,
        "first_divergence_step": int(beyond[0]) if beyond.size else None,
        "divergence_tolerance": resolution / 2.0,
        "trace": error.tolist(),
        "display": error[display_index].tolist(),
    }


def _event_comparison(reference: Sequence[int], candidate: Sequence[int]) -> dict[str, Any]:
    ref = [int(step) for step in reference]
    cand = [int(step) for step in candidate]
    paired = min(len(ref), len(cand))
    offsets = [abs(ref[index] - cand[index]) for index in range(paired)]
    first: dict[str, Any] | None = None
    for index in range(max(len(ref), len(cand))):
        ref_step = ref[index] if index < len(ref) else None
        cand_step = cand[index] if index < len(cand) else None
        if ref_step != cand_step:
            first = {"index": index, "reference_step": ref_step, "candidate_step": cand_step}
            break
    return {
        "identical": ref == cand,
        "reference_count": len(ref),
        "candidate_count": len(cand),
        "paired": paired,
        "max_paired_step_offset": max(offsets) if offsets else 0,
        "first_divergence": first,
    }


def _saturation(trace: BitTrueTrace, data_width: int) -> dict[str, Any]:
    max_word = (1 << (data_width - 1)) - 1
    min_word = -(1 << (data_width - 1))
    rows: dict[str, dict[str, int]] = {}
    for index, name in enumerate(trace.variables):
        column = trace.words[:, index]
        rows[name] = {
            "steps_at_max": int(np.count_nonzero(column == max_word)),
            "steps_at_min": int(np.count_nonzero(column == min_word)),
        }
    return {"max_word": max_word, "min_word": min_word, "per_variable": rows}


def precision_compare(
    equations: Sequence[str],
    threshold: str | None,
    reset: str | None,
    params: Mapping[str, float] | None,
    init: Mapping[str, float] | None,
    dt: float,
    duration: float,
    current: float,
    *,
    protocol: str = "constant",
    frequency_hz: float = 10.0,
    q_format: str = "Q8.8",
    overflow: str = "saturate",
    rounding: str = "truncate",
    max_steps: int = MAX_STEPS,
) -> dict[str, Any]:
    """Run float64, bit-true fixed-point and parameter-quantised comparisons.

    Parameters
    ----------
    equations, threshold, reset, params, init:
        The equation system exactly as the playground runs it.
    dt, duration, current, protocol, frequency_hz:
        The experiment; the same drive samples feed all three runs (encoded
        to words for the kernel).
    q_format:
        Word format label (``Q8.8`` = 16 bits, 8 fractional).
    overflow, rounding:
        Kernel accumulate overflow and product rounding policies.
    max_steps:
        Synchronous step ceiling of the reference run.

    Returns
    -------
    dict
        ``float_result``, ``fixed_result`` and
        ``parameter_quantisation_result`` (each a complete custody payload),
        the ``arithmetic`` statement of the kernel, the ``encoding`` of every
        value, the per-variable and event ``comparison`` for both candidate
        runs, the ``contract`` and the legacy ``error`` / ``quantized_params``
        summary (bit-true error of the first declared variable).

    Raises
    ------
    ModelInputError
        For an unsupported format or mode, a stochastic system, an
        integrator the kernel does not mirror, an unrepresentable value or
        an invalid protocol.
    ModelSimulationFailure
        When the float64 reference run fails numerically.
    NativeToolUnavailable
        When no C compiler is installed.
    """
    if overflow not in SUPPORTED_OVERFLOW:
        raise _input_error(
            "overflow", f"{overflow!r} is not one of {', '.join(SUPPORTED_OVERFLOW)}"
        )
    if rounding not in SUPPORTED_ROUNDING:
        raise _input_error(
            "rounding", f"{rounding!r} is not one of {', '.join(SUPPORTED_ROUNDING)}"
        )
    data_width, fraction = resolve_word_format(q_format)
    q = Q88(data_width=data_width, fraction=fraction, overflow=overflow, rounding=rounding)
    q_label = f"Q{data_width - fraction}.{fraction}"
    try:
        arithmetic = kernel_arithmetic_contract(
            data_width=data_width,
            fraction=fraction,
            overflow=overflow,
            rounding=rounding,
            method="euler",
        )
    except ValueError as exc:
        raise _input_error("q_format", str(exc)[:300]) from exc

    try:
        neuron = from_equations(
            *equations,
            threshold=threshold,
            reset=reset if reset else None,
            params=dict(params) if params else None,
            init=dict(init) if init else None,
            dt=dt,
        )
    except (ValueError, TypeError, SyntaxError, KeyError) as exc:
        raise _input_error("equations", str(exc)[:300]) from exc
    if neuron.uses_diffusion_noise:
        raise _input_error(
            "equations", "the diffusion-noise symbol xi has no bit-true fixed-point arithmetic"
        )
    if neuron.method != "euler":
        raise _input_error("equations", f"integrator {neuron.method!r} is not mirrored bit-true")
    _reject_unrepresentable(q, q_label, neuron, equations, threshold, reset, dt)

    n_steps = int(duration / dt)
    if n_steps > max_steps:
        n_steps = max_steps
    if n_steps < 1:
        raise _input_error("duration", f"{duration!r} with dt {dt!r} yields no complete step")
    drive = current_trace(protocol, float(current), n_steps, dt=dt, frequency_hz=frequency_hz)
    if not _representable(q, float(drive.max())) or not _representable(q, float(drive.min())):
        raise _input_error(
            "current",
            f"drive samples in [{float(drive.min())!r}, {float(drive.max())!r}] are not "
            f"representable in {q_label} [{q.min_value}, {q.max_value}]",
        )
    drive_words = np.rint(drive * float(1 << fraction)).astype(np.int64)
    scale = float(1 << fraction)
    quantised_drive = drive_words.astype(np.float64) / scale

    float_result = simulate(
        list(equations),
        threshold=threshold,
        reset=reset,
        params=dict(params) if params else None,
        init=dict(init) if init else None,
        dt=dt,
        duration=duration,
        current=float(current),
        protocol=protocol,
        frequency_hz=frequency_hz,
        max_steps=max_steps,
    )
    if int(float_result["n_steps"]) != n_steps:
        raise _input_error("duration", "reference run step count differs from the experiment")

    kernel_neuron = from_equations(
        *equations,
        threshold=threshold,
        reset=reset if reset else None,
        params=dict(params) if params else None,
        init=dict(init) if init else None,
        dt=dt,
    )
    trace = run_bittrue_kernel(
        kernel_neuron,
        data_width=data_width,
        fraction=fraction,
        overflow=overflow,
        rounding=rounding,
        drive_words=drive_words,
    )
    quantised_init = _quantised(
        q, {name: float(neuron.initial_state.get(name, 0.0)) for name in neuron.equations}
    )
    fixed_result = _fixed_result(trace, dt=dt, drive=quantised_drive, quantised_init=quantised_init)

    # Parameters and initial state rounded to the word resolution; the requested
    # dt is kept so every run receives the identical drive samples. The time-step
    # word the kernel applies is reported under encoding.dt.
    quantised_params = _quantised(
        q, {name: float(value) for name, value in neuron.parameters.items()}
    )
    parameter_result = simulate(
        list(equations),
        threshold=threshold,
        reset=reset,
        params=quantised_params or None,
        init=quantised_init,
        dt=dt,
        duration=duration,
        current=float(current),
        protocol=protocol,
        frequency_hz=frequency_hz,
        max_steps=max_steps,
    )
    if int(parameter_result["n_steps"]) != n_steps:
        raise _input_error(
            "duration", "parameter-quantised run step count differs from the experiment"
        )

    display_index = np.asarray(float_result["display"]["sample_index"], dtype=np.int64)
    reference_traces = {
        name: np.asarray(values, dtype=np.float64)
        for name, values in full_state_traces(float_result).items()
    }
    fixed_traces = trace.decoded()
    parameter_traces = {
        name: np.asarray(values, dtype=np.float64)
        for name, values in full_state_traces(parameter_result).items()
    }
    bit_true_variables = {
        name: _variable_comparison(
            reference_traces[name],
            fixed_traces[name],
            resolution=q.resolution,
            display_index=display_index,
        )
        for name in neuron.equations
    }
    parameter_variables = {
        name: _variable_comparison(
            reference_traces[name],
            parameter_traces[name],
            resolution=q.resolution,
            display_index=display_index,
        )
        for name in neuron.equations
    }
    comparison = {
        "bit_true": {
            "candidate": "fixed_result",
            "variables": bit_true_variables,
            "events": _event_comparison(float_result["spikes"], fixed_result["spikes"]),
            "saturation": _saturation(trace, data_width),
        },
        "parameter_quantisation": {
            "candidate": "parameter_quantisation_result",
            "variables": parameter_variables,
            "events": _event_comparison(float_result["spikes"], parameter_result["spikes"]),
        },
    }

    primary = next(iter(neuron.equations))
    primary_metrics = bit_true_variables[primary]
    contract = MetricContract(
        kind="precision-compare",
        definition=(
            "absolute difference per step between the float64 explicit-Euler reference "
            "and (a) the bit-true fixed-point kernel run and (b) a float64 run with "
            "parameters and initial state rounded to the word resolution; spike steps "
            "compared in order"
        ),
        units={
            **{name: MODEL_DEFINED_UNIT for name in neuron.equations},
            "time": "ms",
            "current": MODEL_DEFINED_UNIT,
            "rate_hz": "Hz",
        },
        applicability=(
            "explicit-Euler equation systems without diffusion noise",
            f"values representable in {q_label} [{q.min_value}, {q.max_value}]",
            "identical drive: the float runs receive the requested samples, the kernel "
            "their word encoding (reported under encoding.drive)",
            "the fixed-point run is the generated bit-true kernel executed natively; "
            "its arithmetic is stated under arithmetic",
        ),
        limitations=(
            "the parameter-quantisation run isolates value rounding only; the operation "
            "effects (wrap-truncate multiply, saturation, look-up tables) appear in the "
            "bit-true run alone",
            "the time-step word (encoding.dt) is applied by the kernel only; the "
            "parameter-quantisation run keeps the requested dt so all three runs share "
            "the same drive samples",
            "first_divergence_step is the first step whose error exceeds half a word resolution",
        ),
        domain="complete",
    )
    payload: dict[str, Any] = {
        "schema_version": PRECISION_COMPARE_SCHEMA_VERSION,
        "float_result": float_result,
        "fixed_result": fixed_result,
        "parameter_quantisation_result": parameter_result,
        "arithmetic": {
            **arithmetic,
            "dt_word": _encode(q, dt),
            "kernel_sha256": trace.kernel_sha256,
            "harness_sha256": trace.harness_sha256,
            "compiler": trace.compiler,
        },
        "encoding": {
            "q_format": q_label,
            "data_width": data_width,
            "fraction": fraction,
            "resolution": q.resolution,
            "params": _encoding_rows(q, {**neuron.parameters, **neuron.constants}),
            "init": _encoding_rows(
                q, {name: float(neuron.initial_state.get(name, 0.0)) for name in neuron.equations}
            ),
            "dt": _encoding_rows(q, {"dt": dt})["dt"],
            "drive": {
                "protocol": protocol,
                "frequency_hz": frequency_hz,
                "n_steps": n_steps,
                "sha256_requested": _sha256_floats(drive),
                "sha256_words": _sha256_json(drive_words.tolist()),
                "max_abs_error": float(np.max(np.abs(drive - quantised_drive))),
                "min": float(drive.min()),
                "max": float(drive.max()),
            },
        },
        "comparison": comparison,
        "error": {
            "kind": "bit-true-vs-float64",
            "variable": primary,
            "max_error": round(primary_metrics["max_abs_error"], 6),
            "mean_error": round(primary_metrics["mean_abs_error"], 6),
            "rms_error": round(primary_metrics["rms_error"], 6),
            "first_divergence_step": primary_metrics["first_divergence_step"],
            "trace": primary_metrics["trace"],
            "display": primary_metrics["display"],
        },
        "quantized_params": quantised_params,
        "quantized_init": quantised_init,
    }
    return attach_contract(payload, contract)


__all__ = [
    "MAX_DATA_WIDTH",
    "MIN_DATA_WIDTH",
    "PRECISION_COMPARE_SCHEMA_VERSION",
    "SUPPORTED_OVERFLOW",
    "SUPPORTED_ROUNDING",
    "OverflowMode",
    "RoundingMode",
    "precision_compare",
    "resolve_word_format",
]
