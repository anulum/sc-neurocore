# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Verilog compiler implementation

"""Compile EquationNeuron to synthesizable Verilog RTL.

Two emitters share one combinational core (:func:`_build_neuron_core`):

* :func:`compile_to_verilog` — the per-instance module: internal state registers
  advanced in an ``always`` block. This is the bit-true, golden path used by the
  direct/AER interconnects.
* :func:`compile_to_datapath` — the same arithmetic as a **combinational
  processing element** with state carried on ports (``<var>_reg`` inputs,
  ``<var>_next_out`` outputs). One PE is time-multiplexed across many neurons in
  the folded interconnect, so the next-state arithmetic must be — and is, by
  construction — identical to the per-instance module.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..hdl_gen._ident import sanitize_ident
from ..neurons.equation_builder import EquationNeuron
from .verilog_compiler_config import Q88
from .verilog_expr_emitter import _emit_expr


@dataclass
class _NeuronCore:
    """Shared combinational building blocks for one neuron's step.

    Both the registered per-instance module and the combinational datapath PE are
    assembled from these identical fragments, so their arithmetic is bit-for-bit
    the same. State variables are referenced as ``<safe_var>_reg`` throughout
    (the per-instance module declares those as registers; the datapath PE declares
    them as input ports).
    """

    state_var_map: dict[str, str]
    param_map: dict[str, str]
    param_decls: list[str]
    intermediates: list[str]
    pipeline_regs: list[str]
    deriv_wires: list[str]
    next_wires: list[str]
    threshold_verilog: str
    reset_assignments: list[str]

    @property
    def total_pipeline_latency(self) -> int:
        return len(self.pipeline_regs)


def _emit_euler_deriv_wires(
    neuron: EquationNeuron,
    state_var_map: dict[str, str],
    param_map: dict[str, str],
    q: Q88,
    *,
    data_width: int,
    fraction: int,
    use_pipeline: bool,
    pp_set: set[str],
    mul_start: int,
    trunc_start: int,
) -> tuple[list[str], list[str], list[str], int, int]:
    """Emit the per-variable ``d<var> = f(state)·dt`` forward-Euler increment wires.

    This is the original single-step update path, extracted verbatim so the method
    dispatch in :func:`_build_neuron_core` can select it or the RK4 path without
    changing its byte-for-byte output. Returns ``(deriv_wires, intermediates,
    pipeline_regs, mul_count, trunc_count)``.
    """
    deriv_wires: list[str] = []
    all_intermediates: list[str] = []
    all_pipeline_regs: list[str] = []
    _mc = mul_start
    _tc = trunc_start
    for var, expr_str in neuron.equations.items():
        safe_var = state_var_map[var]
        vexpr, intermediates, _mc, _tc, p_regs = _emit_expr(
            expr_str,
            state_var_map,
            param_map,
            q,
            mul_start=_mc,
            trunc_start=_tc,
            pipeline=use_pipeline,
            pipeline_points=pp_set,
        )
        all_intermediates.extend(intermediates)
        all_pipeline_regs.extend(p_regs)
        dt_literal = q.encode_signed_literal(neuron.dt)
        dt_tmp = f"_dt_mul_{safe_var}"
        dt_should_pipe = use_pipeline or dt_tmp in pp_set
        all_intermediates.append(
            f"wire signed [{2 * data_width - 1}:0] {dt_tmp} = ({vexpr}) * {dt_literal};"
        )
        if dt_should_pipe:
            dt_reg = f"{dt_tmp}_r"
            all_pipeline_regs.append(f"reg signed [{2 * data_width - 1}:0] {dt_reg};")
            deriv_name = f"d{safe_var}"
            deriv_trunc = f"_dt_trunc_{safe_var}"
            all_intermediates.append(
                f"wire signed [{data_width - 1}:0] {deriv_trunc} = ({dt_reg} >>> {fraction});"
            )
        else:
            deriv_name = f"d{safe_var}"
            deriv_trunc = f"_dt_trunc_{safe_var}"
            all_intermediates.append(
                f"wire signed [{data_width - 1}:0] {deriv_trunc} = ({dt_tmp} >>> {fraction});"
            )
        deriv_wires.append(f"wire signed [{data_width - 1}:0] {deriv_name} = {deriv_trunc};")
    return deriv_wires, all_intermediates, all_pipeline_regs, _mc, _tc


def _emit_map_deriv_wires(
    neuron: EquationNeuron,
    state_var_map: dict[str, str],
    param_map: dict[str, str],
    q: Q88,
    *,
    data_width: int,
    use_pipeline: bool,
    pp_set: set[str],
    mul_start: int,
    trunc_start: int,
) -> tuple[list[str], list[str], list[str], int, int]:
    """Emit the per-variable next-state wires for a discrete-time map (``method="map"``).

    A map assigns ``state_{n+1} = f(state_n)`` directly — no ``dt`` scaling and no
    ``+ state`` term — unlike the ODE integrators. Each ``d<var>`` wire therefore
    holds the full next-step value ``f(state)`` (not an increment), exactly like the
    exponential-Euler emitter. The map branch of the state update in
    :func:`_build_neuron_core` feeds it straight into the saturating ``<var>_next``
    path (``next = saturate(f(state))``) instead of adding it to the current state,
    so no full-scale ``f(state) - state`` subtraction can overflow the data word.
    Returns ``(deriv_wires, intermediates, pipeline_regs, mul_count, trunc_count)``.
    """
    deriv_wires: list[str] = []
    all_intermediates: list[str] = []
    all_pipeline_regs: list[str] = []
    _mc = mul_start
    _tc = trunc_start
    for var, expr_str in neuron.equations.items():
        safe_var = state_var_map[var]
        vexpr, intermediates, _mc, _tc, p_regs = _emit_expr(
            expr_str,
            state_var_map,
            param_map,
            q,
            mul_start=_mc,
            trunc_start=_tc,
            pipeline=use_pipeline,
            pipeline_points=pp_set,
        )
        all_intermediates.extend(intermediates)
        all_pipeline_regs.extend(p_regs)
        deriv_wires.append(f"wire signed [{data_width - 1}:0] d{safe_var} = {vexpr};")
    return deriv_wires, all_intermediates, all_pipeline_regs, _mc, _tc


def _emit_rk4_deriv_wires(
    neuron: EquationNeuron,
    state_var_map: dict[str, str],
    param_map: dict[str, str],
    q: Q88,
    *,
    data_width: int,
    fraction: int,
    use_pipeline: bool,
    pp_set: set[str],
    mul_start: int,
    trunc_start: int,
) -> tuple[list[str], list[str], list[str], int, int]:
    """Emit the per-variable ``d<var>`` increment wires for one classical RK4 step.

    Mirrors the Python golden in :meth:`EquationNeuron.step` (``method="rk4"``)::

        k1 = f(s0);   s1 = s0 + k1·dt/2
        k2 = f(s1);   s2 = s0 + k2·dt/2
        k3 = f(s2);   s3 = s0 + k3·dt
        k4 = f(s3)
        d<var> = (k1 + 2·k2 + 2·k3 + k4)·dt/6

    Every derivative evaluation reuses :func:`_emit_expr` — the same fixed-point
    expression emitter the Euler path uses — so the whole RK4 stage graph is emitted
    in whatever ``q`` format is requested. The integrator is therefore agnostic to
    the number representation: all Q-formats, rounding modes and overflow handling
    are inherited without special-casing (one integrator × N representations). The
    state at each stage is supplied through ``param_map`` (state variables render as
    the stage wire names), exactly as the threshold emitter substitutes ``<var>_next``.
    Targets deterministic models (no stochastic ``xi`` term). When ``use_pipeline`` (or an
    explicit ``pp_set``) is set, every derivative-evaluation multiply across the four stages is
    registered by :func:`_emit_expr` (the cheap constant ``dt``-scalings stay combinational); the
    fill-counter FSM in :func:`compile_to_verilog` holds the state steady until the whole stage
    graph drains, so the recurrence stays bit-true regardless of pipeline depth. Returns
    ``(deriv_wires, intermediates, pipeline_regs, mul_count, trunc_count)``.
    """
    inter: list[str] = []
    regs: list[str] = []
    mc, tc = mul_start, trunc_start
    variables = list(neuron.equations)
    dt2_lit = q.encode_signed_literal(neuron.dt / 2.0)
    dt_lit = q.encode_signed_literal(neuron.dt)
    dt6_lit = q.encode_signed_literal(neuron.dt / 6.0)

    def eval_stage(stage_map: dict[str, str], tag: str) -> dict[str, str]:
        """Emit ``k<tag>_<var> = f_<var>(stage state)`` for every variable."""
        nonlocal mc, tc
        k_wires: dict[str, str] = {}
        for var in variables:
            safe_var = state_var_map[var]
            vexpr, ints, mc, tc, pregs = _emit_expr(
                neuron.equations[var],
                {},
                stage_map,
                q,
                mul_start=mc,
                trunc_start=tc,
                pipeline=use_pipeline,
                pipeline_points=pp_set,
            )
            inter.extend(ints)
            regs.extend(pregs)
            k_name = f"_k{tag}_{safe_var}"
            inter.append(f"wire signed [{data_width - 1}:0] {k_name} = {vexpr};")
            k_wires[var] = k_name
        return k_wires

    def advance(k_wires: dict[str, str], scale_lit: str, tag: str) -> dict[str, str]:
        """Emit ``s<tag>_<var> = s0_<var> + k·scale`` and return the stage state map."""
        stage_map = dict(param_map)
        for var in variables:
            safe_var = state_var_map[var]
            mul = f"_rk{tag}mul_{safe_var}"
            inter.append(
                f"wire signed [{2 * data_width - 1}:0] {mul} = {k_wires[var]} * {scale_lit};"
            )
            trunc = f"_rk{tag}tr_{safe_var}"
            inter.append(f"wire signed [{data_width - 1}:0] {trunc} = ({mul} >>> {fraction});")
            s_wire = f"_s{tag}_{safe_var}"
            inter.append(f"wire signed [{data_width - 1}:0] {s_wire} = {safe_var}_reg + {trunc};")
            stage_map[var] = s_wire
        return stage_map

    s0_map = {**param_map, **{var: f"{state_var_map[var]}_reg" for var in variables}}
    k1 = eval_stage(s0_map, "1")
    k2 = eval_stage(advance(k1, dt2_lit, "1"), "2")
    k3 = eval_stage(advance(k2, dt2_lit, "2"), "3")
    k4 = eval_stage(advance(k3, dt_lit, "3"), "4")

    deriv_wires: list[str] = []
    for var in variables:
        safe_var = state_var_map[var]
        weighted = f"_rkw_{safe_var}"
        inter.append(
            f"wire signed [{data_width + 2}:0] {weighted} = "
            f"{k1[var]} + {k2[var]} + {k2[var]} + {k3[var]} + {k3[var]} + {k4[var]};"
        )
        scaled = f"_rk6mul_{safe_var}"
        inter.append(f"wire signed [{2 * data_width + 2}:0] {scaled} = {weighted} * {dt6_lit};")
        deriv_wires.append(
            f"wire signed [{data_width - 1}:0] d{safe_var} = ({scaled} >>> {fraction});"
        )
    return deriv_wires, inter, regs, mc, tc


def _emit_exp_euler_deriv_wires(
    neuron: EquationNeuron,
    state_var_map: dict[str, str],
    param_map: dict[str, str],
    q: Q88,
    *,
    data_width: int,
    use_pipeline: bool,
    pp_set: set[str],
    mul_start: int,
    trunc_start: int,
) -> tuple[list[str], list[str], list[str], int, int]:
    """Emit the per-variable ``d<var>`` increment wires for one exponential-Euler step.

    Mirrors the Python golden in :meth:`EquationNeuron.step` (``method="exp_euler"``),
    the linearised exponential Euler (Rush–Larsen) update::

        d<var> = f(state) · dt · exprel(A·dt),   A = ∂f/∂x

    ``exprel(z) = (e**z − 1)/z`` is reused so the zero-Jacobian limit ``A→0`` collapses
    to the exact forward-Euler increment ``f·dt``, and the update is exact on the gating
    form ``dx/dt = (x_inf − x)/tau`` where forward Euler drifts. The diagonal Jacobian
    ``A`` is the *same* symbolic derivative string the golden compiled
    (``neuron.jacobian_expressions[var]``): one derivative expression drives the golden
    and the Verilog, so the two stay consistent by construction rather than by a parallel
    re-derivation.

    The whole increment — ``f``, ``A``, both ``dt`` scalings and the ``exprel`` hardware
    LUT — is lowered by the same :func:`_emit_expr` the Euler and RK4 paths use, applied
    to a single composed expression per variable, so exp-Euler inherits every Q-format and
    the transcendental LUT for free (one integrator × N representations). State renders as
    ``<var>_reg`` — the pre-step value — so all increments are computed from the pre-step
    state (a forward, not Gauss–Seidel, update), exactly as the golden applies them.
    Targets deterministic models (no stochastic ``xi`` term). When ``use_pipeline`` (or an
    explicit ``pp_set``) is set every multiply in the composed increment is registered by
    :func:`_emit_expr`; the fill-counter FSM in :func:`compile_to_verilog` holds the state
    steady until those stages drain, so the recurrence stays bit-true regardless of depth.
    Returns ``(deriv_wires, intermediates, pipeline_regs, mul_count, trunc_count)``.
    """
    inter: list[str] = []
    regs: list[str] = []
    mc, tc = mul_start, trunc_start
    # repr keeps the exact float; const_float quantises it in Q-format like every literal,
    # so the dt scaling matches the rest of the emitted arithmetic.
    dt_lit = repr(neuron.dt)
    deriv_wires: list[str] = []
    for var, f_expr in neuron.equations.items():
        safe_var = state_var_map[var]
        a_expr = neuron.jacobian_expressions[var]
        # increment = (f · dt) · exprel(A · dt); the parentheses fix the golden's
        # associativity ((f*dt)*exprel(A*dt)) and isolate f/A as their own sub-trees.
        increment_expr = f"(({f_expr}) * {dt_lit}) * exprel((({a_expr})) * {dt_lit})"
        vexpr, ints, mc, tc, pregs = _emit_expr(
            increment_expr,
            state_var_map,
            param_map,
            q,
            mul_start=mc,
            trunc_start=tc,
            pipeline=use_pipeline,
            pipeline_points=pp_set,
        )
        inter.extend(ints)
        regs.extend(pregs)
        deriv_wires.append(f"wire signed [{data_width - 1}:0] d{safe_var} = {vexpr};")
    return deriv_wires, inter, regs, mc, tc


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
    state_var_map = {var: sanitize_ident(var, context="state variable") for var in neuron.equations}

    param_map: dict[str, str] = {}
    param_decls: list[str] = []
    for pname, pval in {**neuron.parameters, **neuron.constants}.items():
        safe_pname = sanitize_ident(pname, context="parameter name")
        vname = f"P_{safe_pname.upper()}"
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

    sign_kw = "signed " if q.signed else ""
    if q.signed:
        max_val = (1 << (data_width - 1)) - 1
        min_val = -(1 << (data_width - 1))
    else:
        max_val = (1 << data_width) - 1
        min_val = 0

    next_wires: list[str] = []
    for var in neuron.equations:
        safe_var = state_var_map[var]
        raw = f"{safe_var}_raw"
        if method == "map":
            # A discrete map's next state is f(state) itself (held in d<var>);
            # saturate it directly rather than adding the current state, which would
            # risk a full-scale overflow before the saturating clamp can recover it.
            next_wires.append(f"wire {sign_kw}[{data_width}:0] {raw} = d{safe_var};")
        else:
            next_wires.append(
                f"wire {sign_kw}[{data_width}:0] {raw} = {safe_var}_reg + d{safe_var};"
            )

        if q.overflow == "saturate":
            abs_min = abs(min_val)
            if q.signed:
                next_wires.append(
                    f"wire {sign_kw}[{data_width - 1}:0] {safe_var}_next = "
                    f"({raw} > {data_width + 1}'sd{max_val}) ? {data_width}'sd{max_val} : "
                    f"({raw} < (-{data_width + 1}'sd{abs_min})) ? (-{data_width}'sd{abs_min}) : "
                    f"{raw}[{data_width - 1}:0];"
                )
            else:
                next_wires.append(
                    f"wire [{data_width - 1}:0] {safe_var}_next = "
                    f"({raw} > {data_width + 1}'d{max_val}) ? {data_width}'d{max_val} : "
                    f"({raw}[{data_width}]) ? {data_width}'d0 : "
                    f"{raw}[{data_width - 1}:0];"
                )
        elif q.overflow == "wrap":
            next_wires.append(
                f"wire {sign_kw}[{data_width - 1}:0] {safe_var}_next = {raw}[{data_width - 1}:0];"
            )
        elif q.overflow == "trap":
            abs_min = abs(min_val)
            if q.signed:
                next_wires.append(
                    f"wire {sign_kw}[{data_width - 1}:0] {safe_var}_next = "
                    f"{raw}[{data_width - 1}:0];"
                )
                next_wires.append("// synthesis translate_off")
                next_wires.append(
                    f"always @(*) if ({raw} > {data_width + 1}'sd{max_val} || "
                    f"{raw} < (-{data_width + 1}'sd{abs_min})) "
                    f'$fatal(1, "OVERFLOW TRAP: {safe_var}_raw=%0d", {raw});'
                )
                next_wires.append("// synthesis translate_on")
            else:
                next_wires.append(
                    f"wire [{data_width - 1}:0] {safe_var}_next = {raw}[{data_width - 1}:0];"
                )
                next_wires.append("// synthesis translate_off")
                next_wires.append(
                    f"always @(*) if ({raw}[{data_width}]) "
                    f'$fatal(1, "OVERFLOW TRAP: {safe_var}_raw=%0d", {raw});'
                )
                next_wires.append("// synthesis translate_on")
        else:
            raise ValueError(f"Unknown overflow mode: {q.overflow!r}")

    threshold_verilog = ""
    if neuron.threshold_expr:
        thr_param_map = dict(param_map)
        for var in neuron.equations:
            safe_var = state_var_map[var]
            thr_param_map[var] = f"{safe_var}_next"
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

    reset_assignments: list[str] = []
    for var, expr_str in neuron.reset_rules.items():
        safe_var = state_var_map[var]
        rexpr, r_intermediates, _mc, _tc, r_pregs = _emit_expr(
            expr_str,
            state_var_map,
            param_map,
            q,
            mul_start=_mc,
            trunc_start=_tc,
        )
        all_intermediates.extend(r_intermediates)
        all_pipeline_regs.extend(r_pregs)
        reset_assignments.append(f"            {safe_var}_reg <= {rexpr};")

    return _NeuronCore(
        state_var_map=state_var_map,
        param_map=param_map,
        param_decls=param_decls,
        intermediates=all_intermediates,
        pipeline_regs=all_pipeline_regs,
        deriv_wires=deriv_wires,
        next_wires=next_wires,
        threshold_verilog=threshold_verilog,
        reset_assignments=reset_assignments,
    )


def compile_to_verilog(
    neuron: EquationNeuron,
    module_name: str = "sc_equation_neuron",
    data_width: int = 16,
    fraction: int = 8,
    *,
    signed: bool = True,
    overflow: str = "saturate",
    rounding: str = "truncate",
    pipeline_stages: int = 0,
    pipeline_points: list[str] | None = None,
) -> str:
    """Compile an EquationNeuron to synthesizable Verilog RTL.

    Parameters
    ----------
    neuron : EquationNeuron
        The neuron defined by arbitrary ODE strings.
    module_name : str
        Name of the generated Verilog module.
    data_width : int
        Bit width for fixed-point arithmetic (default 16 = Q8.8).
    fraction : int
        Number of fractional bits (default 8).
    signed : bool
        True for signed two's complement (default), False for unsigned.
    overflow : str
        Overflow mode: ``"saturate"`` (default), ``"wrap"``, or ``"trap"``.
    rounding : str
        Rounding mode: ``"truncate"`` (default), ``"nearest"``,
        ``"bankers"``, or ``"stochastic"``.
    pipeline_stages : int
        Number of pipeline register stages to insert at multiply outputs.
    pipeline_points : list[str], optional
        Explicit list of intermediate signal names.
    """
    q = Q88(
        data_width=data_width,
        fraction=fraction,
        signed=signed,
        overflow=overflow,
        rounding=rounding,
    )

    if neuron.dt != 0.0:
        dt_quantised = int(round(neuron.dt * (1 << fraction)))
        if dt_quantised == 0:
            min_representable = 1.0 / (1 << fraction)
            raise ValueError(
                f"dt={neuron.dt} underflows in Q{data_width - fraction}.{fraction}: "
                f"smallest representable non-zero value is {min_representable}. "
                f"Use dt=1.0 or another value >= {min_representable}, "
                "or increase fractional precision, e.g. fraction=12."
            )

    safe_module_name = sanitize_ident(module_name, context="module name")
    core = _build_neuron_core(
        neuron,
        q,
        data_width=data_width,
        fraction=fraction,
        pipeline_stages=pipeline_stages,
        pipeline_points=pipeline_points,
    )
    state_var_map = core.state_var_map
    param_decls = core.param_decls
    all_intermediates = core.intermediates
    all_pipeline_regs = core.pipeline_regs
    deriv_wires = core.deriv_wires
    next_wires = core.next_wires
    threshold_verilog = core.threshold_verilog
    reset_assignments = core.reset_assignments
    total_pipeline_latency = core.total_pipeline_latency

    lines = [
        "// Auto-generated by SC-NeuroCore equation compiler",
        f"// Source: {neuron!r}",
        f"// Fixed-point: Q{data_width - fraction}.{fraction} ({data_width}-bit signed)",
        "`timescale 1ns / 1ps",
        "",
        f"module {safe_module_name} #(",
    ]
    if param_decls:
        lines.append(",\n".join(param_decls))
        lines.append(")(")
    else:
        # No parameters: an empty #() clause is malformed; drop it entirely.
        lines[-1] = f"module {safe_module_name} ("
    lines.append("    input wire clk,")
    lines.append("    input wire rst_n,")
    lines.append(f"    input wire signed [{data_width - 1}:0] I_t,")
    lines.append("    output reg spike_out,")

    for var in neuron.equations:
        safe_var = state_var_map[var]
        lines.append(f"    output reg signed [{data_width - 1}:0] {safe_var}_out,")

    if total_pipeline_latency > 0:
        lat_w = max(1, (total_pipeline_latency).bit_length())
        lines.append(f"    output wire [{lat_w - 1}:0] latency,")

    lines[-1] = lines[-1].rstrip(",")
    lines.append(");")
    lines.append("")

    if total_pipeline_latency > 0:
        lat_w = max(1, (total_pipeline_latency).bit_length())
        lines.append(f"// Pipeline latency: {total_pipeline_latency} cycle(s)")
        lines.append(f"assign latency = {lat_w}'d{total_pipeline_latency};")
        lines.append("")

    for var in neuron.equations:
        safe_var = state_var_map[var]
        lines.append(f"reg signed [{data_width - 1}:0] {safe_var}_reg;")

    lines.append("")

    if all_pipeline_regs:
        lines.append("// Pipeline registers for multiply outputs")
        for reg_decl in all_pipeline_regs:
            lines.append(reg_decl)
        lines.append("")

    for wire in all_intermediates:
        lines.append(wire)
    lines.append("")

    for wire in deriv_wires:
        lines.append(wire)
    lines.append("")

    for wire in next_wires:
        lines.append(wire)
    lines.append("")

    if all_pipeline_regs:
        # Reset the staging registers to 0 so an unfilled pipeline never injects X into the
        # state feedback; the fill counter below guarantees the state only advances once these
        # stages have drained, so the reset value is never read as a live increment.
        lines.append("// Pipeline register stage — register multiply outputs (reset to 0)")
        lines.append("always @(posedge clk or negedge rst_n) begin")
        lines.append("    if (!rst_n) begin")
        for reg_decl in all_pipeline_regs:
            reg_name = reg_decl.split()[-1].rstrip(";")
            lines.append(f"        {reg_name} <= 0;")
        lines.append("    end else begin")
        for reg_decl in all_pipeline_regs:
            reg_name = reg_decl.split()[-1].rstrip(";")
            wire_name = reg_name[:-2] if reg_name.endswith("_r") else reg_name
            lines.append(f"        {reg_name} <= {wire_name};")
        lines.append("    end")
        lines.append("end")
        lines.append("")

    # Build the single-step body once (8-space base indent). When pipelined it is gated behind
    # the fill counter so the recurrence only advances after the register stages have drained.
    step_lines: list[str] = []
    if threshold_verilog:
        step_lines.append(f"        if ({threshold_verilog}) begin")
        step_lines.append("            spike_out <= 1'b1;")
        for assign in reset_assignments:
            step_lines.append(assign)
        for var in neuron.equations:
            safe_var = state_var_map[var]
            if var not in neuron.reset_rules:
                step_lines.append(f"            {safe_var}_reg <= {safe_var}_next;")
        for var in neuron.equations:
            safe_var = state_var_map[var]
            step_lines.append(f"            {safe_var}_out <= {safe_var}_reg;")
        step_lines.append("        end else begin")
        step_lines.append("            spike_out <= 1'b0;")
        for var in neuron.equations:
            safe_var = state_var_map[var]
            step_lines.append(f"            {safe_var}_reg <= {safe_var}_next;")
            step_lines.append(f"            {safe_var}_out <= {safe_var}_next;")
        step_lines.append("        end")
    else:
        step_lines.append("        spike_out <= 1'b0;")
        for var in neuron.equations:
            step_lines.append(f"        {var}_reg <= {var}_next;")
            step_lines.append(f"        {var}_out <= {var}_next;")

    # Fill counter: a self-recurrent step whose increment takes ``total_pipeline_latency``
    # register stages must hold the state steady for that many cycles, else the increment
    # applied would reflect stale (mid-fill) products and break bit-exactness with the golden.
    # ``total_pipeline_latency`` (the register count) upper-bounds the true pipeline depth, so
    # holding for that many cycles guarantees every stage has drained before the state advances
    # — one logical step every ``latency + 1`` clocks, spikes only on the valid cycle.
    if total_pipeline_latency > 0:
        cnt_w = max(1, total_pipeline_latency.bit_length())
        lines.append(f"reg [{cnt_w - 1}:0] _pl_cnt;")
        lines.append(f"wire _pl_valid = (_pl_cnt == {cnt_w}'d{total_pipeline_latency});")
        lines.append("")

    lines.append("always @(posedge clk or negedge rst_n) begin")
    lines.append("    if (!rst_n) begin")
    for var in neuron.equations:
        safe_var = state_var_map[var]
        init_val = q.encode_signed_literal(neuron.initial_state.get(var, 0.0))
        lines.append(f"        {safe_var}_reg <= {init_val};")
        lines.append(f"        {safe_var}_out <= {init_val};")
    lines.append("        spike_out <= 1'b0;")
    if total_pipeline_latency > 0:
        lines.append(f"        _pl_cnt <= {cnt_w}'d0;")
    lines.append("    end else begin")

    if total_pipeline_latency > 0:
        lines.append(f"        _pl_cnt <= _pl_valid ? {cnt_w}'d0 : (_pl_cnt + {cnt_w}'d1);")
        lines.append("        if (_pl_valid) begin")
        for sline in step_lines:
            lines.append("    " + sline)
        lines.append("        end else begin")
        lines.append("            spike_out <= 1'b0;")
        lines.append("        end")
    else:
        lines.extend(step_lines)

    lines.append("    end")
    lines.append("end")
    lines.append("")
    lines.append("endmodule")

    return "\n".join(lines)


def compile_to_datapath(
    neuron: EquationNeuron,
    module_name: str = "sc_equation_neuron_pe",
    data_width: int = 16,
    fraction: int = 8,
    *,
    signed: bool = True,
    overflow: str = "saturate",
    rounding: str = "truncate",
    param_ports: Sequence[str] = (),
) -> str:
    """Compile an EquationNeuron to a **combinational** processing element.

    State is carried on ports — ``<var>_reg`` inputs and ``<var>_next_out``
    outputs — instead of internal registers, so one PE can be time-multiplexed
    across many neurons (the folded interconnect stores per-neuron state in BRAM
    and streams it through this PE one neuron per cycle). ``spike_out`` and each
    ``<var>_next_out`` are the post-threshold, post-reset next values, computed by
    the same fragments as :func:`compile_to_verilog` — so the folded datapath is
    bit-for-bit identical to the per-instance module.

    ``param_ports`` names the parameters to carry on **input ports** instead of
    baking them as module ``parameter`` defaults. A folded population whose neurons
    have heterogeneous parameters drives these ports from a per-neuron parameter ROM
    (one value per neuron, streamed by the sequencer), the parameter-space analogue of
    the state BRAM. Because the arithmetic body references each parameter by the same
    ``P_<NAME>`` identifier whether it is a ``parameter`` or an ``input wire``, moving a
    parameter to a port changes only its declaration — the datapath stays bit-for-bit
    identical. Every name must be a real neuron parameter/constant; the rest stay baked.

    Pipelining is not supported here (a combinational PE has no register stages);
    the folded sequencer provides the one-cycle-per-neuron timing instead.
    """
    q = Q88(
        data_width=data_width,
        fraction=fraction,
        signed=signed,
        overflow=overflow,
        rounding=rounding,
    )

    if neuron.dt != 0.0:
        dt_quantised = int(round(neuron.dt * (1 << fraction)))
        if dt_quantised == 0:
            min_representable = 1.0 / (1 << fraction)
            raise ValueError(
                f"dt={neuron.dt} underflows in Q{data_width - fraction}.{fraction}: "
                f"smallest representable non-zero value is {min_representable}."
            )

    safe_module_name = sanitize_ident(module_name, context="module name")
    core = _build_neuron_core(
        neuron,
        q,
        data_width=data_width,
        fraction=fraction,
        pipeline_stages=0,
        pipeline_points=None,
    )
    state_var_map = core.state_var_map

    # Partition parameters into those baked as module ``parameter`` defaults and those
    # carried on input ports (``param_ports``), so a folded population can stream
    # per-neuron parameters through the PE. Both keep the same ``P_<NAME>`` identifier,
    # so only the declaration differs — the arithmetic body is untouched.
    unknown = [p for p in param_ports if p not in core.param_map]
    if unknown:
        raise ValueError(f"param_ports names are not parameters of {neuron!r}: {sorted(unknown)}")
    port_vnames = [core.param_map[p] for p in param_ports]
    baked_decls = [
        decl for decl in core.param_decls if not any(f" {vn} =" in decl for vn in port_vnames)
    ]

    lines = [
        "// Auto-generated by SC-NeuroCore equation compiler (folded datapath PE)",
        f"// Source: {neuron!r}",
        f"// Fixed-point: Q{data_width - fraction}.{fraction} ({data_width}-bit signed)",
        "// Combinational next-state: state carried on <var>_reg inputs / <var>_next_out outputs.",
        "`timescale 1ns / 1ps",
        "",
        f"module {safe_module_name} #(",
    ]
    if baked_decls:
        lines.append(",\n".join(baked_decls))
        lines.append(")(")
    else:
        lines[-1] = f"module {safe_module_name} ("
    lines.append(f"    input wire signed [{data_width - 1}:0] I_t,")
    # Heterogeneous parameters carried on input ports, streamed from a per-neuron ROM.
    for vname in port_vnames:
        lines.append(f"    input wire signed [{data_width - 1}:0] {vname},")
    # State carried in on ports (named <var>_reg so the shared core wires match).
    for var in neuron.equations:
        safe_var = state_var_map[var]
        lines.append(f"    input wire signed [{data_width - 1}:0] {safe_var}_reg,")
    lines.append("    output wire spike_out,")
    for var in neuron.equations:
        safe_var = state_var_map[var]
        lines.append(f"    output wire signed [{data_width - 1}:0] {safe_var}_next_out,")
    lines[-1] = lines[-1].rstrip(",")
    lines.append(");")
    lines.append("")

    for wire in core.intermediates:
        lines.append(wire)
    lines.append("")
    for wire in core.deriv_wires:
        lines.append(wire)
    lines.append("")
    for wire in core.next_wires:
        lines.append(wire)
    lines.append("")

    # Spike is the combinational threshold over the candidate next state.
    if core.threshold_verilog:
        lines.append(f"assign spike_out = ({core.threshold_verilog});")
    else:
        lines.append("assign spike_out = 1'b0;")

    # Next-state output: on spike, apply reset rules (or hold <var>_next when the
    # variable has no reset rule); otherwise advance to <var>_next. This mirrors
    # the per-instance always block exactly.
    reset_map: dict[str, str] = {}
    for assign in core.reset_assignments:
        # Each entry is "            <safe_var>_reg <= <expr>;"
        body = assign.strip().rstrip(";")
        lhs, rhs = body.split("<=", 1)
        safe_var = lhs.strip().removesuffix("_reg")
        reset_map[safe_var] = rhs.strip()

    for var in neuron.equations:
        safe_var = state_var_map[var]
        on_spike = reset_map.get(safe_var, f"{safe_var}_next")
        if core.threshold_verilog:
            lines.append(
                f"assign {safe_var}_next_out = spike_out ? ({on_spike}) : {safe_var}_next;"
            )
        else:
            lines.append(f"assign {safe_var}_next_out = {safe_var}_next;")

    lines.append("")
    lines.append("endmodule")

    return "\n".join(lines)
