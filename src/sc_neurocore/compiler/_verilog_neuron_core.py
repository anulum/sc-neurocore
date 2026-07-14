# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Shared equation-neuron Verilog core

"""Build shared fixed-point next-state, event, and reset fragments."""

from __future__ import annotations

from dataclasses import dataclass

from ..hdl_gen._ident import sanitize_ident
from ..neurons.equation_builder import EquationNeuron
from ._verilog_integrators import (
    _emit_euler_deriv_wires,
    _emit_exp_euler_deriv_wires,
    _emit_gauss_seidel_deriv_wires,
    _emit_map_deriv_wires,
    _emit_rk4_deriv_wires,
)
from .verilog_compiler_config import Q88
from .verilog_expr_emitter import _emit_expr


@dataclass
class _NeuronCore:
    """Shared combinational building blocks for one neuron's step.

    Both the registered per-instance module and the combinational datapath PE are
    assembled from these identical fragments, so their arithmetic is bit-for-bit
    the same. Derivatives read ``<safe_var>_reg``; threshold and reset expressions
    read the integrated ``<safe_var>_next`` candidate. The per-instance module
    declares the former as registers, while the datapath PE declares them as input
    ports.
    """

    state_var_map: dict[str, str]
    param_map: dict[str, str]
    param_decls: list[str]
    intermediates: list[str]
    pipeline_regs: list[str]
    deriv_wires: list[str]
    next_wires: list[str]
    threshold_verilog: str
    escape_probability_verilog: str
    reset_expressions: dict[str, str]

    @property
    def total_pipeline_latency(self) -> int:
        return len(self.pipeline_regs)


def _escape_threshold_wires(
    probability_expression: str,
    sample_expression: str,
    *,
    data_width: int,
    fraction: int,
) -> list[str]:
    """Lower one Q-format probability and one LFSR sample to a spike bit.

    The 17-bit threshold mirrors ``probability_to_lfsr16_threshold`` exactly:
    zero never fires, one always fires, and an interior probability realises
    ``floor(p*65535)/65535`` under the strict ``sample < threshold`` compare.
    """
    one = 1 << fraction
    product_width = data_width + 16
    return [
        (f"wire signed [{data_width - 1}:0] _escape_probability = {probability_expression};"),
        (
            f"wire [{data_width - 1}:0] _escape_probability_clamped = "
            f"_escape_probability[{data_width - 1}] ? {data_width}'d0 : "
            f"((_escape_probability >= {data_width}'sd{one}) ? "
            f"{data_width}'d{one} : _escape_probability);"
        ),
        (
            f"wire [{product_width - 1}:0] _escape_threshold_product = "
            "$unsigned(_escape_probability_clamped) * 16'd65535;"
        ),
        (f"wire [15:0] _escape_threshold_floor = _escape_threshold_product >> {fraction};"),
        (
            "wire [16:0] _escape_threshold = "
            f"(_escape_probability <= {data_width}'sd0) ? 17'd0 : "
            f"((_escape_probability >= {data_width}'sd{one}) ? 17'd65536 : "
            "({1'b0, _escape_threshold_floor} + 17'd1));"
        ),
        f"wire _escape_spike = {{1'b0, {sample_expression}}} < _escape_threshold;",
    ]


def _build_neuron_core(
    neuron: EquationNeuron,
    q: Q88,
    *,
    data_width: int,
    fraction: int,
    pipeline_stages: int,
    pipeline_points: list[str] | None,
) -> _NeuronCore:
    """Emit the combinational next-state + threshold + reset fragments.

    This is the logic shared verbatim by :func:`compile_to_verilog` and
    :func:`compile_to_datapath`; neither wraps nor mutates it differently, which
    is what guarantees bit-exact agreement between the per-instance module and the
    folded datapath.
    """
    if q.rounding not in {"truncate", "nearest", "bankers", "stochastic"}:
        raise ValueError(f"Unknown rounding mode: {q.rounding!r}")
    if q.rounding == "stochastic":
        raise NotImplementedError(
            "stochastic product rounding is not supported by the equation-to-Verilog "
            "compiler because the emitted datapath has no rounding LFSR; use truncate, "
            "nearest, or bankers"
        )
    if type(pipeline_stages) is not int:
        raise TypeError(f"pipeline_stages must be an integer, got {pipeline_stages!r}")
    if pipeline_stages < 0:
        raise ValueError(f"pipeline_stages must be non-negative, got {pipeline_stages}")
    if pipeline_points is not None:
        if type(pipeline_points) is not list:
            raise TypeError(f"pipeline_points must be a list of strings, got {pipeline_points!r}")
        if any(type(point) is not str for point in pipeline_points):
            raise TypeError("pipeline_points entries must all be strings")
        if len(pipeline_points) != len(set(pipeline_points)):
            raise ValueError("pipeline_points must not contain duplicates")
        if pipeline_stages > 0 and pipeline_points:
            raise ValueError(
                "pipeline_stages and pipeline_points are mutually exclusive; "
                "choose global or explicit multiply pipelining"
            )
    state_var_map = {var: sanitize_ident(var, context="state variable") for var in neuron.equations}

    param_map: dict[str, str] = {}
    param_decls: list[str] = []
    used_vnames: set[str] = set()
    for pname, pval in {**neuron.parameters, **neuron.constants}.items():
        safe_pname = sanitize_ident(pname, context="parameter name")
        vname = f"P_{safe_pname.upper()}"
        if vname in used_vnames:
            # ``str.upper()`` is not injective, so two case-distinct parameter names
            # (e.g. the Izhikevich 2007 capacitance ``C`` vs the reset voltage ``c``)
            # would collapse to the same ``P_C`` port and iverilog rejects the redeclare.
            # Verilog identifiers are case-sensitive, so fall back to a case-preserving
            # identifier — and, only if that is also taken, a numeric suffix — keeping the
            # parameter port map injective while leaving every single-case name unchanged.
            candidate = f"P_{safe_pname}"
            suffix = 2
            while candidate in used_vnames:
                candidate = f"P_{safe_pname}_{suffix}"
                suffix += 1
            vname = candidate
        used_vnames.add(vname)
        param_map[pname] = vname
        q_val = q.encode(pval)
        param_decls.append(
            f"    parameter signed [{data_width - 1}:0] {vname} = {data_width}'sd{q_val}"
        )

    all_intermediates: list[str] = []
    all_pipeline_regs: list[str] = []
    _mc = 0
    _tc = 0

    use_pipeline = pipeline_stages > 0
    pp_set = set(pipeline_points) if pipeline_points and not use_pipeline else set()

    method = getattr(neuron, "method", "euler")
    if method == "rk4":
        deriv_wires, deriv_intermediates, deriv_regs, _mc, _tc = _emit_rk4_deriv_wires(
            neuron,
            state_var_map,
            param_map,
            q,
            data_width=data_width,
            fraction=fraction,
            use_pipeline=use_pipeline,
            pp_set=pp_set,
            mul_start=_mc,
            trunc_start=_tc,
        )
    elif method == "exp_euler":
        deriv_wires, deriv_intermediates, deriv_regs, _mc, _tc = _emit_exp_euler_deriv_wires(
            neuron,
            state_var_map,
            param_map,
            q,
            data_width=data_width,
            use_pipeline=use_pipeline,
            pp_set=pp_set,
            mul_start=_mc,
            trunc_start=_tc,
        )
    elif method == "map":
        deriv_wires, deriv_intermediates, deriv_regs, _mc, _tc = _emit_map_deriv_wires(
            neuron,
            state_var_map,
            param_map,
            q,
            data_width=data_width,
            use_pipeline=use_pipeline,
            pp_set=pp_set,
            mul_start=_mc,
            trunc_start=_tc,
        )
    elif method == "gauss_seidel":
        deriv_wires, deriv_intermediates, deriv_regs, _mc, _tc = _emit_gauss_seidel_deriv_wires(
            neuron,
            state_var_map,
            param_map,
            q,
            data_width=data_width,
            fraction=fraction,
            use_pipeline=use_pipeline,
            pp_set=pp_set,
            mul_start=_mc,
            trunc_start=_tc,
        )
    else:
        deriv_wires, deriv_intermediates, deriv_regs, _mc, _tc = _emit_euler_deriv_wires(
            neuron,
            state_var_map,
            param_map,
            q,
            data_width=data_width,
            fraction=fraction,
            use_pipeline=use_pipeline,
            pp_set=pp_set,
            mul_start=_mc,
            trunc_start=_tc,
        )
    all_intermediates.extend(deriv_intermediates)
    all_pipeline_regs.extend(deriv_regs)

    max_val = (1 << (data_width - 1)) - 1
    min_val = -(1 << (data_width - 1))

    next_wires: list[str] = []
    for var in neuron.equations:
        safe_var = state_var_map[var]
        raw = f"{safe_var}_raw"
        if method == "map":
            # A discrete map's next state is f(state) itself (held in d<var>);
            # saturate it directly rather than adding the current state, which would
            # risk a full-scale overflow before the saturating clamp can recover it.
            next_wires.append(f"wire signed [{data_width}:0] {raw} = d{safe_var};")
        else:
            next_wires.append(f"wire signed [{data_width}:0] {raw} = {safe_var}_reg + d{safe_var};")

        if q.overflow == "saturate":
            abs_min = abs(min_val)
            next_wires.append(
                f"wire signed [{data_width - 1}:0] {safe_var}_next = "
                f"({raw} > {data_width + 1}'sd{max_val}) ? {data_width}'sd{max_val} : "
                f"({raw} < (-{data_width + 1}'sd{abs_min})) ? (-{data_width}'sd{abs_min}) : "
                f"{raw}[{data_width - 1}:0];"
            )
        elif q.overflow == "wrap":
            next_wires.append(
                f"wire signed [{data_width - 1}:0] {safe_var}_next = {raw}[{data_width - 1}:0];"
            )
        elif q.overflow == "trap":
            abs_min = abs(min_val)
            next_wires.append(
                f"wire signed [{data_width - 1}:0] {safe_var}_next = {raw}[{data_width - 1}:0];"
            )
            next_wires.append("// synthesis translate_off")
            next_wires.append(
                f"always @(*) if ({raw} > {data_width + 1}'sd{max_val} || "
                f"{raw} < (-{data_width + 1}'sd{abs_min})) "
                f'$fatal(1, "OVERFLOW TRAP: {safe_var}_raw=%0d", {raw});'
            )
            next_wires.append("// synthesis translate_on")
        else:
            raise ValueError(f"Unknown overflow mode: {q.overflow!r}")

    threshold_verilog = ""
    escape_probability_verilog = ""
    if getattr(neuron, "_stochastic_threshold_enabled", False):
        stochastic_param_map = dict(param_map)
        for var in neuron.equations:
            safe_var = state_var_map[var]
            stochastic_param_map[var] = f"{safe_var}_next"
            stochastic_param_map[f"{var}_prev"] = f"{safe_var}_reg"
        if getattr(neuron, "_poisson_enabled", False):
            probability_expression = getattr(neuron, "probability_expression", None)
            if not probability_expression:
                raise ValueError("Poisson compilation requires a probability expression")
        else:
            rate_expression = getattr(neuron, "rate_expression", None)
            if not rate_expression:
                raise ValueError("escape-rate compilation requires a rate expression")
            probability_expression = f"1.0 - exp(-(({rate_expression}) * {neuron.dt!r}))"
        (
            escape_probability_verilog,
            probability_intermediates,
            _mc,
            _tc,
            probability_pregs,
        ) = _emit_expr(
            probability_expression,
            {},
            stochastic_param_map,
            q,
            mul_start=_mc,
            trunc_start=_tc,
        )
        all_intermediates.extend(probability_intermediates)
        all_pipeline_regs.extend(probability_pregs)
    elif neuron.threshold_expr:
        thr_param_map = dict(param_map)
        for var in neuron.equations:
            safe_var = state_var_map[var]
            thr_param_map[var] = f"{safe_var}_next"
            thr_param_map[f"{var}_prev"] = f"{safe_var}_reg"
        threshold_verilog, thr_intermediates, _mc, _tc, thr_pregs = _emit_expr(
            neuron.threshold_expr,
            {},
            thr_param_map,
            q,
            mul_start=_mc,
            trunc_start=_tc,
        )
        all_intermediates.extend(thr_intermediates)
        all_pipeline_regs.extend(thr_pregs)

    reset_param_map = dict(param_map)
    reset_param_map.update({var: f"{safe_var}_next" for var, safe_var in state_var_map.items()})
    reset_param_map.update(
        {f"{var}_prev": f"{safe_var}_reg" for var, safe_var in state_var_map.items()}
    )
    reset_expressions: dict[str, str] = {}
    for var, expr_str in neuron.reset_rules.items():
        safe_var = state_var_map[var]
        rexpr, r_intermediates, _mc, _tc, r_pregs = _emit_expr(
            expr_str,
            {},
            reset_param_map,
            q,
            mul_start=_mc,
            trunc_start=_tc,
        )
        all_intermediates.extend(r_intermediates)
        all_pipeline_regs.extend(r_pregs)
        reset_expressions[safe_var] = rexpr

    return _NeuronCore(
        state_var_map=state_var_map,
        param_map=param_map,
        param_decls=param_decls,
        intermediates=all_intermediates,
        pipeline_regs=all_pipeline_regs,
        deriv_wires=deriv_wires,
        next_wires=next_wires,
        threshold_verilog=threshold_verilog,
        escape_probability_verilog=escape_probability_verilog,
        reset_expressions=reset_expressions,
    )
