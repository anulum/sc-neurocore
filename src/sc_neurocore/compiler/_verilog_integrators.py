# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Fixed-point Verilog integrator lowering

"""Lower equation-neuron integration methods to fixed-point Verilog wires."""

from __future__ import annotations

from ..neurons.equation_builder import EquationNeuron
from .verilog_compiler_config import Q88
from .verilog_expr_emitter import _emit_expr


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


def _emit_gauss_seidel_deriv_wires(
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
    """Emit the per-variable ``d<var> = f(state)·dt`` sequential (Gauss-Seidel) increments.

    Identical to :func:`_emit_euler_deriv_wires` except for which register each state
    variable resolves to inside a later variable's derivative expression. In the
    simultaneous Euler path every derivative reads the pre-step ``<var>_reg``; here each
    derivative is emitted in declaration order and the earlier-declared variables resolve
    to their freshly-committed ``<var>_next`` wire (the saturated ``<var>_reg + d<var>``),
    while this variable and the later ones still read ``<var>_reg``. A later variable
    therefore consumes the already-updated earlier variables within the same sub-step,
    lowering the Python golden's Gauss-Seidel commit order (Wang-Buzsaki 1996 updates the
    gating variables ``h``/``n`` before the membrane voltage ``v``) as a commit-before-read
    combinational dependency chain — no cycle, since ``h_next``/``n_next`` depend only on
    the registers, not on ``v_next``. The substitution reuses the same mechanism the
    threshold emitter uses to reference ``<var>_next``; state variables are supplied
    through ``param_map`` (``state_vars`` is empty) so the emitter renders the exact wire
    chosen per variable. Returns ``(deriv_wires, intermediates, pipeline_regs, mul_count,
    trunc_count)``.
    """
    deriv_wires: list[str] = []
    all_intermediates: list[str] = []
    all_pipeline_regs: list[str] = []
    _mc = mul_start
    _tc = trunc_start
    variables = list(neuron.equations)
    for idx, var in enumerate(variables):
        safe_var = state_var_map[var]
        # Earlier-declared variables resolve to their committed ``<var>_next`` value;
        # this variable and later ones keep the pre-step ``<var>_reg``.
        render_map = dict(param_map)
        for position, other in enumerate(variables):
            render_map[other] = (
                f"{state_var_map[other]}_next" if position < idx else f"{state_var_map[other]}_reg"
            )
        vexpr, intermediates, _mc, _tc, p_regs = _emit_expr(
            neuron.equations[var],
            {},
            render_map,
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
        deriv_name = f"d{safe_var}"
        deriv_trunc = f"_dt_trunc_{safe_var}"
        if dt_should_pipe:
            dt_reg = f"{dt_tmp}_r"
            all_pipeline_regs.append(f"reg signed [{2 * data_width - 1}:0] {dt_reg};")
            all_intermediates.append(
                f"wire signed [{data_width - 1}:0] {deriv_trunc} = ({dt_reg} >>> {fraction});"
            )
        else:
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
