# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bit-true simulation kernel

"""Bit-true fixed-point simulation kernels for the equation compiler.

The generated C / Rust reproduces the exact two's-complement arithmetic of the
Verilog RTL (see :mod:`sc_neurocore.compiler.verilog_compiler`): wrap-truncate
fixed-point multiply, saturating (or wrapping) explicit-Euler or discrete-map
commit, positive-period modulo, and transcendental look-up tables with the RTL's
index arithmetic — all lowered by :mod:`sc_neurocore.compiler.c_fixed_emitter`.

Two entry points:

* :func:`generate_bittrue_kernel` — a lightweight kernel from a bare
  ``{state_var: derivative}`` mapping. It advances each variable by one
  unit-``dt`` explicit-Euler step (``next = sat(reg + dt·f)`` with ``dt = 1``);
  free identifiers and, if referenced, the input current become step arguments.
  This form has no per-instance input/threshold contract, so it exercises the
  bit-true *primitives* rather than a whole neuron.
* :func:`generate_bittrue_kernel_from_neuron` — a full per-neuron kernel that is
  **bit-for-bit identical to the Verilog produced by**
  :func:`sc_neurocore.compiler.verilog_compiler.compile_to_verilog`: same dt,
  threshold, reset rules, parameter encoding and single ``I`` input. That
  identity is *proven*, not asserted, by the iverilog co-simulation in
  ``tests/test_bit_true_cosim.py``, which drives the compiled RTL and this
  kernel with the same stimulus and asserts equal per-cycle state traces.
"""

from __future__ import annotations

from ...neurons.equation_builder import EquationNeuron
from ...hdl_gen._ident import sanitize_ident
from ..c_fixed_emitter import emit_c_fixed_expr, signed_q
from ..verilog_compiler_config import Q88

_NATIVE_WIDTHS = frozenset({8, 16, 32, 64})


def _ctype(data_width: int) -> str:
    """C integer type name holding a ``data_width``-bit word (native or widened)."""
    if data_width in _NATIVE_WIDTHS:
        return f"int{data_width}_t"
    return "int32_t" if data_width <= 32 else "int64_t"


def _rtype(data_width: int) -> str:
    """Rust integer type name holding a ``data_width``-bit word (native or widened)."""
    if data_width in _NATIVE_WIDTHS:
        return f"i{data_width}"
    return "i32" if data_width <= 32 else "i64"


def _validate_modes(q: Q88) -> None:
    """Reject overflow / rounding modes the bit-true kernel does not mirror."""
    if q.rounding not in {"truncate", "nearest"}:
        raise ValueError(
            f"bit-true kernel supports rounding 'truncate' or 'nearest', got {q.rounding!r} "
            "('bankers' is not yet mirrored; 'stochastic' needs an LFSR and is not "
            "deterministically co-simulable)."
        )
    if q.overflow not in {"saturate", "wrap"}:
        raise ValueError(
            f"bit-true kernel supports overflow 'saturate' or 'wrap', got {q.overflow!r} "
            "('trap' emits a Verilog $fatal, which has no value-trace equivalent)."
        )


def _preamble_c(q: Q88) -> list[str]:
    """Emit the shared C helpers: ``sc_wrap``, ``sat``, ``fxmul`` and constants."""
    dw, frac = q.data_width, q.fraction
    ctype = _ctype(dw)
    max_val = (1 << (dw - 1)) - 1
    min_val = -(1 << (dw - 1))
    if q.rounding == "nearest":
        prod = "sc_wrap(sc_wrap(a * b, WIDE_BITS) + (1 << (FRAC_BITS - 1)), WIDE_BITS)"
    else:
        prod = "sc_wrap(a * b, WIDE_BITS)"
    return [
        "#include <stdint.h>",
        "",
        f"#define FRAC_BITS {frac}",
        f"#define WIDE_BITS {2 * dw}",
        f"#define WORD_BITS {dw}",
        f"#define MAX_VAL {max_val}",
        f"#define MIN_VAL ({min_val})",
        "",
        "static inline int64_t sc_wrap(int64_t x, int bits) {",
        "    if (bits >= 64) { return x; }",
        "    uint64_t mask = ((uint64_t)1 << bits) - 1u;",
        "    uint64_t v = (uint64_t)x & mask;",
        "    if (v & ((uint64_t)1 << (bits - 1))) { v |= ~mask; }",
        "    return (int64_t)v;",
        "}",
        "",
        f"static inline {ctype} sat(int64_t x) {{",
        f"    if (x > MAX_VAL) {{ return ({ctype})MAX_VAL; }}",
        f"    if (x < MIN_VAL) {{ return ({ctype})MIN_VAL; }}",
        f"    return ({ctype})x;",
        "}",
        "",
        f"static inline {ctype} fxmul(int64_t a, int64_t b) {{",
        f"    return ({ctype})sc_wrap({prod} >> FRAC_BITS, WORD_BITS);",
        "}",
        "",
        f"static inline {ctype} fxmod(int64_t value, int64_t period) {{",
        "    int64_t dividend = sc_wrap(value, WORD_BITS);",
        "    int64_t remainder = dividend % period;",
        "    if (remainder < 0) { remainder += period; }",
        f"    return ({ctype})remainder;",
        "}",
        "",
    ]


def _preamble_rust(q: Q88) -> list[str]:
    """Emit the shared Rust helpers: ``sc_wrap``, ``sat``, ``fxmul`` and constants."""
    dw, frac = q.data_width, q.fraction
    rtype = _rtype(dw)
    max_val = (1 << (dw - 1)) - 1
    min_val = -(1 << (dw - 1))
    if q.rounding == "nearest":
        prod = "sc_wrap(sc_wrap(a * b, WIDE_BITS) + (1 << (FRAC_BITS - 1)), WIDE_BITS)"
    else:
        prod = "sc_wrap(a * b, WIDE_BITS)"
    return [
        f"const FRAC_BITS: u32 = {frac};",
        f"const WIDE_BITS: u32 = {2 * dw};",
        f"const WORD_BITS: u32 = {dw};",
        f"const MAX_VAL: i64 = {max_val};",
        f"const MIN_VAL: i64 = {min_val};",
        "",
        "fn sc_wrap(x: i64, bits: u32) -> i64 {",
        "    if bits >= 64 { return x; }",
        "    let mask: u64 = (1u64 << bits) - 1;",
        "    let mut v: u64 = (x as u64) & mask;",
        "    if v & (1u64 << (bits - 1)) != 0 { v |= !mask; }",
        "    v as i64",
        "}",
        "",
        f"fn sat(x: i64) -> {rtype} {{",
        f"    x.clamp(MIN_VAL, MAX_VAL) as {rtype}",
        "}",
        "",
        f"fn fxmul(a: i64, b: i64) -> {rtype} {{",
        f"    sc_wrap({prod} >> FRAC_BITS, WORD_BITS) as {rtype}",
        "}",
        "",
        f"fn fxmod(value: i64, period: i64) -> {rtype} {{",
        "    let dividend = sc_wrap(value, WORD_BITS);",
        "    let mut remainder = dividend % period;",
        "    if remainder < 0 { remainder += period; }",
        f"    remainder as {rtype}",
        "}",
        "",
    ]


def _format_tables_c(tables: dict[str, list[int]], data_width: int) -> list[str]:
    """Declare each accumulated LUT as a ``static const`` C array."""
    ctype = _ctype(data_width)
    lines: list[str] = []
    for name, entries in tables.items():
        body = ", ".join(str(v) for v in entries)
        lines.append(f"static const {ctype} {name}[{len(entries)}] = {{{body}}};")
    if lines:
        lines.append("")
    return lines


def _format_tables_rust(tables: dict[str, list[int]], data_width: int) -> list[str]:
    """Declare each accumulated LUT as a Rust ``const`` array."""
    rtype = _rtype(data_width)
    lines: list[str] = []
    for name, entries in tables.items():
        body = ", ".join(str(v) for v in entries)
        lines.append(f"const {name}: [{rtype}; {len(entries)}] = [{body}];")
    if lines:
        lines.append("")
    return lines


def _accumulate_bias(x: str, overflow: str) -> str:
    """Wrap a raw ``reg + d`` accumulate expression per the overflow mode."""
    if overflow == "wrap":
        return f"sc_wrap({x}, WORD_BITS)"
    return f"sat({x})"


def generate_bittrue_kernel(
    module_name: str,
    equations: dict[str, str],
    *,
    data_width: int = 16,
    fraction: int = 8,
    language: str = "c",
) -> str:
    """Generate a bit-true fixed-point kernel from a ``{var: derivative}`` mapping.

    Each state variable advances by one unit-``dt`` explicit-Euler step using the
    same wrap-truncate multiply and saturating accumulate as the RTL datapath.
    Identifiers in the derivatives that are not state variables become step
    arguments (the input current ``I`` maps to ``I_t`` when referenced), so the
    kernel exercises the bit-true primitives without a per-instance I/O contract.
    For a whole-neuron kernel proven bit-identical to the generated Verilog, use
    :func:`generate_bittrue_kernel_from_neuron`.

    Parameters
    ----------
    module_name : str
        Base name for the generated struct / functions.
    equations : dict
        Mapping from state-variable name to its derivative expression string.
    data_width : int
        Fixed-point total width (default 16 → Q8.8).
    fraction : int
        Fractional bits (default 8).
    language : str
        ``"c"`` (default) or ``"rust"``.

    Returns
    -------
    str
        Bit-true kernel source code.
    """
    if language not in {"c", "rust"}:
        raise ValueError(f"language must be 'c' or 'rust', got {language!r}")
    q = Q88(data_width=data_width, fraction=fraction)
    _validate_modes(q)

    safe = {v: sanitize_ident(v, context="state variable") for v in equations}
    ref = "self." if language == "rust" else "s->"
    state_map = {v: f"{ref}{safe[v]}" for v in equations}
    dt_q = signed_q(q, 1.0)

    tables: dict[str, list[int]] = {}
    free_vars: list[str] = []
    input_used = False
    lut = 0
    derivs: dict[str, str] = {}
    stmts: list[str] = []
    for var, expr in equations.items():
        result, s, t, fv, lut, used = emit_c_fixed_expr(
            expr, state_map, {}, q, lang=language, lut_start=lut
        )
        derivs[var] = result
        stmts.extend(s)
        tables.update(t)
        for name in fv:
            if name not in free_vars:
                free_vars.append(name)
        input_used = input_used or used

    if language == "c":
        return _emit_simple_c(
            module_name, equations, safe, q, dt_q, derivs, stmts, tables, free_vars, input_used
        )
    return _emit_simple_rust(
        module_name, equations, safe, q, dt_q, derivs, stmts, tables, free_vars, input_used
    )


def _step_args_c(input_used: bool, free_vars: list[str], data_width: int) -> str:
    """Comma-joined C parameter list for a simple kernel's extra inputs."""
    ctype = _ctype(data_width)
    args = ""
    if input_used:
        args += f", {ctype} I_t"
    for name in free_vars:
        args += f", {ctype} {sanitize_ident(name, context='free identifier')}"
    return args


def _emit_simple_c(
    module_name: str,
    equations: dict[str, str],
    safe: dict[str, str],
    q: Q88,
    dt_q: int,
    derivs: dict[str, str],
    stmts: list[str],
    tables: dict[str, list[int]],
    free_vars: list[str],
    input_used: bool,
) -> str:
    """Render the simple mapping kernel as C."""
    ctype = _ctype(q.data_width)
    int_bits = q.data_width - q.fraction - 1
    lines = [
        f"/* Bit-true simulation kernel for {module_name} */",
        f"/* SC-NeuroCore — Q{int_bits}.{q.fraction} ({q.data_width}-bit), "
        f"{q.overflow} overflow, {q.rounding} rounding */",
        "/* Wrap-truncate fixed-point multiply + saturating unit-dt Euler accumulate, */",
        "/* matching the RTL datapath primitives of compile_to_verilog. */",
        "",
    ]
    lines.extend(_preamble_c(q))
    lines.extend(_format_tables_c(tables, q.data_width))
    lines.append("typedef struct {")
    for var in equations:
        lines.append(f"    {ctype} {safe[var]};")
    lines.append(f"}} {module_name}_state_t;")
    lines.append("")
    args = _step_args_c(input_used, free_vars, q.data_width)
    lines.append(f"void {module_name}_step({module_name}_state_t *s{args}) {{")
    for stmt in stmts:
        lines.append(f"    {stmt}")
    for var in equations:
        acc = _accumulate_bias(
            f"((int64_t)s->{safe[var]}) + fxmul({derivs[var]}, {dt_q})", q.overflow
        )
        lines.append(f"    {ctype} _next_{safe[var]} = {acc};")
    for var in equations:
        lines.append(f"    s->{safe[var]} = _next_{safe[var]};")
    lines.append("}")
    return "\n".join(lines)


def _emit_simple_rust(
    module_name: str,
    equations: dict[str, str],
    safe: dict[str, str],
    q: Q88,
    dt_q: int,
    derivs: dict[str, str],
    stmts: list[str],
    tables: dict[str, list[int]],
    free_vars: list[str],
    input_used: bool,
) -> str:
    """Render the simple mapping kernel as Rust."""
    rtype = _rtype(q.data_width)
    int_bits = q.data_width - q.fraction - 1
    lines = [
        f"// Bit-true simulation kernel for {module_name}",
        f"// SC-NeuroCore — Q{int_bits}.{q.fraction} ({q.data_width}-bit), "
        f"{q.overflow} overflow, {q.rounding} rounding",
        "",
    ]
    lines.extend(_preamble_rust(q))
    lines.extend(_format_tables_rust(tables, q.data_width))
    struct = f"{module_name.capitalize()}State"
    lines.append(f"pub struct {struct} {{")
    for var in equations:
        lines.append(f"    pub {safe[var]}: {rtype},")
    lines.append("}")
    lines.append("")
    lines.append(f"impl {struct} {{")
    args = ""
    if input_used:
        args += f", I_t: {rtype}"
    for name in free_vars:
        args += f", {sanitize_ident(name, context='free identifier')}: {rtype}"
    lines.append(f"    pub fn step(&mut self{args}) {{")
    for stmt in stmts:
        lines.append(f"        {stmt}")
    for var in equations:
        acc = _accumulate_bias(
            f"(self.{safe[var]} as i64) + (fxmul({derivs[var]}, {dt_q}) as i64)", q.overflow
        )
        lines.append(f"        let _next_{safe[var]}: {rtype} = {acc};")
    for var in equations:
        lines.append(f"        self.{safe[var]} = _next_{safe[var]};")
    lines.append("    }")
    lines.append("}")
    return "\n".join(lines)


def generate_bittrue_kernel_from_neuron(
    neuron: EquationNeuron,
    module_name: str = "sc_neuron",
    *,
    data_width: int = 16,
    fraction: int = 8,
    signed: bool = True,
    overflow: str = "saturate",
    rounding: str = "truncate",
    language: str = "c",
) -> str:
    """Generate a whole-neuron kernel bit-identical to the compiled Verilog.

    Mirrors :func:`sc_neurocore.compiler.verilog_compiler.compile_to_verilog`
    exactly for explicit-Euler and discrete-map neurons — parameter/constant
    Q-encoding, wrap-truncate arithmetic, the same overflow handling, and the
    threshold / reset / spike sequencing of the RTL ``always`` block. Threshold
    and reset expressions may read ``<state>_prev`` aliases for the pre-step
    register while ordinary state names resolve to the integrated candidate.
    Both state and output fields take the same post-reset value on a spike. The
    resulting ``<module>_step`` therefore produces the identical per-cycle state
    trace as the RTL, which the iverilog co-simulation proves.

    Parameters
    ----------
    neuron : EquationNeuron
        The neuron whose ODEs, parameters, threshold and reset rules are lowered.
    module_name : str
        Base name for the generated struct / functions.
    data_width, fraction : int
        Fixed-point format (default Q8.8).
    signed : bool
        Signed two's complement (only ``True`` is supported; unsigned neuron state
        is not part of the RTL contract).
    overflow : str
        ``"saturate"`` (default) or ``"wrap"``.
    rounding : str
        ``"truncate"`` (default) or ``"nearest"``.
    language : str
        ``"c"`` (default) or ``"rust"``.

    Returns
    -------
    str
        Bit-true whole-neuron kernel source code.
    """
    if language not in {"c", "rust"}:
        raise ValueError(f"language must be 'c' or 'rust', got {language!r}")
    if not signed:
        raise ValueError("bit-true neuron kernel requires signed=True (the RTL neuron contract)")
    q = Q88(
        data_width=data_width, fraction=fraction, signed=True, overflow=overflow, rounding=rounding
    )
    _validate_modes(q)
    if neuron.method not in {"euler", "map"}:
        raise ValueError(
            "bit-true neuron kernel currently supports method='euler' or method='map', "
            f"got {neuron.method!r}"
        )

    dt_q = signed_q(q, neuron.dt)
    if neuron.dt != 0.0 and dt_q == 0:
        raise ValueError(
            f"dt={neuron.dt} underflows in Q{data_width - fraction}.{fraction}; "
            "increase fraction or dt."
        )

    safe = {v: sanitize_ident(v, context="state variable") for v in neuron.equations}
    ref = "self." if language == "rust" else "s->"
    state_map = {v: f"{ref}{safe[v]}" for v in neuron.equations}
    param_map = {p: signed_q(q, v) for p, v in {**neuron.parameters, **neuron.constants}.items()}

    tables: dict[str, list[int]] = {}
    lut = 0
    deriv_stmts: list[str] = []
    derivs: dict[str, str] = {}
    for var, expr in neuron.equations.items():
        result, s, t, _fv, lut, _used = emit_c_fixed_expr(
            expr, state_map, param_map, q, lang=language, lut_start=lut
        )
        derivs[var] = result
        deriv_stmts.extend(s)
        tables.update(t)

    next_map = {v: f"_next_{safe[v]}" for v in neuron.equations}
    candidate_map = dict(next_map)
    candidate_map.update({f"{v}_prev": state_map[v] for v in neuron.equations})
    threshold_expr = ""
    threshold_stmts: list[str] = []
    if neuron.threshold_expr:
        threshold_expr, threshold_stmts, t, _fv, lut, _used = emit_c_fixed_expr(
            neuron.threshold_expr, candidate_map, param_map, q, lang=language, lut_start=lut
        )
        tables.update(t)

    reset_exprs: dict[str, str] = {}
    reset_stmts: list[str] = []
    for var, expr in neuron.reset_rules.items():
        result, s, t, _fv, lut, _used = emit_c_fixed_expr(
            expr, candidate_map, param_map, q, lang=language, lut_start=lut
        )
        reset_exprs[var] = result
        reset_stmts.extend(s)
        tables.update(t)

    init_q = {v: signed_q(q, neuron.initial_state.get(v, 0.0)) for v in neuron.equations}
    ctx = _KernelContext(
        module_name=module_name,
        neuron=neuron,
        q=q,
        safe=safe,
        dt_q=dt_q,
        derivs=derivs,
        deriv_stmts=deriv_stmts,
        threshold_expr=threshold_expr,
        threshold_stmts=threshold_stmts,
        reset_exprs=reset_exprs,
        reset_stmts=reset_stmts,
        tables=tables,
        init_q=init_q,
    )
    if language == "c":
        return _emit_neuron_c(ctx)
    return _emit_neuron_rust(ctx)


class _KernelContext:
    """Everything the neuron-kernel renderers need, assembled once per generation."""

    def __init__(
        self,
        *,
        module_name: str,
        neuron: EquationNeuron,
        q: Q88,
        safe: dict[str, str],
        dt_q: int,
        derivs: dict[str, str],
        deriv_stmts: list[str],
        threshold_expr: str,
        threshold_stmts: list[str],
        reset_exprs: dict[str, str],
        reset_stmts: list[str],
        tables: dict[str, list[int]],
        init_q: dict[str, int],
    ) -> None:
        self.module_name = module_name
        self.neuron = neuron
        self.q = q
        self.safe = safe
        self.dt_q = dt_q
        self.derivs = derivs
        self.deriv_stmts = deriv_stmts
        self.threshold_expr = threshold_expr
        self.threshold_stmts = threshold_stmts
        self.reset_exprs = reset_exprs
        self.reset_stmts = reset_stmts
        self.tables = tables
        self.init_q = init_q


def _emit_neuron_c(ctx: _KernelContext) -> str:
    """Render the whole-neuron kernel as C, mirroring the RTL always block."""
    q = ctx.q
    ctype = _ctype(q.data_width)
    m = ctx.module_name
    int_bits = q.data_width - q.fraction - 1
    lines = [
        f"/* Bit-true neuron kernel for {m} */",
        f"/* SC-NeuroCore — Q{int_bits}.{q.fraction} ({q.data_width}-bit), "
        f"{q.overflow} overflow, {q.rounding} rounding */",
        "/* Bit-identical to compile_to_verilog (proven by iverilog co-simulation). */",
        "",
    ]
    lines.extend(_preamble_c(q))
    lines.extend(_format_tables_c(ctx.tables, q.data_width))
    lines.append("typedef struct {")
    for var in ctx.neuron.equations:
        lines.append(f"    {ctype} {ctx.safe[var]};")
        lines.append(f"    {ctype} {ctx.safe[var]}_out;")
    lines.append("    int spike_out;")
    lines.append(f"}} {m}_state_t;")
    lines.append("")
    lines.append(f"static void {m}_reset({m}_state_t *s) {{")
    for var in ctx.neuron.equations:
        lines.append(f"    s->{ctx.safe[var]} = {ctx.init_q[var]};")
        lines.append(f"    s->{ctx.safe[var]}_out = {ctx.init_q[var]};")
    lines.append("    s->spike_out = 0;")
    lines.append("}")
    lines.append("")
    lines.append(f"int {m}_step({m}_state_t *s, {ctype} I_t) {{")
    for stmt in ctx.deriv_stmts:
        lines.append(f"    {stmt}")
    for var in ctx.neuron.equations:
        if ctx.neuron.method == "map":
            acc = _accumulate_bias(ctx.derivs[var], q.overflow)
        else:
            acc = _accumulate_bias(
                f"((int64_t)s->{ctx.safe[var]}) + fxmul({ctx.derivs[var]}, {ctx.dt_q})",
                q.overflow,
            )
        lines.append(f"    {ctype} _next_{ctx.safe[var]} = {acc};")
    lines.extend(_neuron_commit_c(ctx))
    lines.append("}")
    return "\n".join(lines)


def _neuron_commit_c(ctx: _KernelContext) -> list[str]:
    """Emit the threshold / reset / spike commit block for the C kernel."""
    lines: list[str] = []
    eqs = ctx.neuron.equations
    if not ctx.threshold_expr:
        for var in eqs:
            lines.append(f"    s->{ctx.safe[var]} = _next_{ctx.safe[var]};")
            lines.append(f"    s->{ctx.safe[var]}_out = _next_{ctx.safe[var]};")
        lines.append("    s->spike_out = 0;")
        lines.append("    return 0;")
        return lines
    for stmt in ctx.threshold_stmts:
        lines.append(f"    {stmt}")
    lines.append(f"    int _spk = ({ctx.threshold_expr}) ? 1 : 0;")
    for var, expr in ctx.reset_exprs.items():
        lines.append(f"    {_ctype(ctx.q.data_width)} _rst_{ctx.safe[var]} = sat({expr});")
    lines.append("    if (_spk) {")
    for var in eqs:
        rhs = f"_rst_{ctx.safe[var]}" if var in ctx.reset_exprs else f"_next_{ctx.safe[var]}"
        lines.append(f"        s->{ctx.safe[var]} = {rhs};")
        lines.append(f"        s->{ctx.safe[var]}_out = {rhs};")
    lines.append("    } else {")
    for var in eqs:
        lines.append(f"        s->{ctx.safe[var]} = _next_{ctx.safe[var]};")
        lines.append(f"        s->{ctx.safe[var]}_out = _next_{ctx.safe[var]};")
    lines.append("    }")
    lines.append("    s->spike_out = _spk;")
    lines.append("    return _spk;")
    return lines


def _emit_neuron_rust(ctx: _KernelContext) -> str:
    """Render the whole-neuron kernel as Rust, mirroring the RTL always block."""
    q = ctx.q
    rtype = _rtype(q.data_width)
    m = ctx.module_name
    struct = f"{m.capitalize()}State"
    int_bits = q.data_width - q.fraction - 1
    lines = [
        f"// Bit-true neuron kernel for {m}",
        f"// SC-NeuroCore — Q{int_bits}.{q.fraction} ({q.data_width}-bit), "
        f"{q.overflow} overflow, {q.rounding} rounding",
        "",
    ]
    lines.extend(_preamble_rust(q))
    lines.extend(_format_tables_rust(ctx.tables, q.data_width))
    lines.append(f"pub struct {struct} {{")
    for var in ctx.neuron.equations:
        lines.append(f"    pub {ctx.safe[var]}: {rtype},")
        lines.append(f"    pub {ctx.safe[var]}_out: {rtype},")
    lines.append("    pub spike_out: i32,")
    lines.append("}")
    lines.append("")
    lines.append(f"impl {struct} {{")
    lines.append("    pub fn reset(&mut self) {")
    for var in ctx.neuron.equations:
        lines.append(f"        self.{ctx.safe[var]} = {ctx.init_q[var]};")
        lines.append(f"        self.{ctx.safe[var]}_out = {ctx.init_q[var]};")
    lines.append("        self.spike_out = 0;")
    lines.append("    }")
    lines.append("")
    lines.append(f"    pub fn step(&mut self, I_t: {rtype}) -> i32 {{")
    for stmt in ctx.deriv_stmts:
        lines.append(f"        {stmt}")
    for var in ctx.neuron.equations:
        if ctx.neuron.method == "map":
            acc = _accumulate_bias(ctx.derivs[var], q.overflow)
        else:
            acc = _accumulate_bias(
                f"(self.{ctx.safe[var]} as i64) + (fxmul({ctx.derivs[var]}, {ctx.dt_q}) as i64)",
                q.overflow,
            )
        lines.append(f"        let _next_{ctx.safe[var]}: {rtype} = {acc};")
    lines.extend(_neuron_commit_rust(ctx))
    lines.append("    }")
    lines.append("}")
    return "\n".join(lines)


def _neuron_commit_rust(ctx: _KernelContext) -> list[str]:
    """Emit the threshold / reset / spike commit block for the Rust kernel."""
    lines: list[str] = []
    eqs = ctx.neuron.equations
    if not ctx.threshold_expr:
        for var in eqs:
            lines.append(f"        self.{ctx.safe[var]} = _next_{ctx.safe[var]};")
            lines.append(f"        self.{ctx.safe[var]}_out = _next_{ctx.safe[var]};")
        lines.append("        self.spike_out = 0;")
        lines.append("        return 0;")
        return lines
    for stmt in ctx.threshold_stmts:
        lines.append(f"        {stmt}")
    lines.append(f"        let _spk: i32 = if ({ctx.threshold_expr}) {{ 1 }} else {{ 0 }};")
    for var, expr in ctx.reset_exprs.items():
        lines.append(f"        let _rst_{ctx.safe[var]}: {_rtype(ctx.q.data_width)} = sat({expr});")
    lines.append("        if _spk != 0 {")
    for var in eqs:
        rhs = f"_rst_{ctx.safe[var]}" if var in ctx.reset_exprs else f"_next_{ctx.safe[var]}"
        lines.append(f"            self.{ctx.safe[var]} = {rhs};")
        lines.append(f"            self.{ctx.safe[var]}_out = {rhs};")
    lines.append("        } else {")
    for var in eqs:
        lines.append(f"            self.{ctx.safe[var]} = _next_{ctx.safe[var]};")
        lines.append(f"            self.{ctx.safe[var]}_out = _next_{ctx.safe[var]};")
    lines.append("        }")
    lines.append("        self.spike_out = _spk;")
    lines.append("        _spk")
    return lines
