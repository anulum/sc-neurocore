# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — bit-exact integer fixed-point expression emitter

"""Lower an ODE expression to integer C/Rust that matches the Verilog RTL bit-for-bit.

Unlike :mod:`sc_neurocore.compiler.c_expr_emitter` — which targets Vitis
``ap_fixed`` and lets the HLS type carry the Q-format scaling — this emitter
reproduces the *exact* two's-complement arithmetic of
:mod:`sc_neurocore.compiler.verilog_expr_emitter`. Every value is carried as a
64-bit integer, and the two width-collapse points of the RTL datapath are
mirrored precisely:

* a multiply wraps its product to the ``2*data_width``-bit intermediate wire and
  then arithmetic-right-shifts by ``fraction`` and wraps to ``data_width`` bits —
  i.e. it *wraps*, it does not saturate (this is emitted as the ``fxmul`` helper);
* transcendental look-up tables replicate the RTL index arithmetic of
  :meth:`_VerilogExprEmitter._emit_lut_call` (argument truncated to ``data_width``,
  offset, power-of-two shift, then a saturating clamp to ``[0, N-1]``).

Because Verilog resolves the width of ``+``/``-`` sub-expressions from the
enclosing (wider) assignment context, carrying everything at 64 bits and
collapsing only at the ``fxmul`` / LUT points reproduces that behaviour for any
``2*data_width <= 64`` (Q8.8 and Q16.16 both qualify). Wider words are rejected
because the intermediate would exceed 64 bits. The saturating Euler accumulate
(``reg + d``) and the reset/threshold logic are added by the kernel generator in
:mod:`sc_neurocore.compiler.intelligence.bit_true_kernel`; this module only lowers
the right-hand-side expression grammar. Bit-exactness against the RTL is proven,
not asserted, by the iverilog co-simulation in ``tests/test_bit_true_cosim.py``.
"""

from __future__ import annotations

import ast
import math
from collections.abc import Callable

from . import expr_lut_tables
from .expr_lut_tables import SUPPORTED_FUNCTIONS, const_float
from .verilog_compiler_config import Q88

# Per-function LUT geometry, matching the calls in
# ``_VerilogExprEmitter.visit_Call`` / ``visit_BinOp`` exactly. ``sqrt`` (and
# the ``**0.5`` power) retains its [-8, 8) unit-step geometry; ``log`` uses the
# shared strictly-positive grid; every other transcendental uses [-16, 16).
_SYMMETRIC_MIN = -16.0
_SYMMETRIC_STEP = 0.125
_UNIT_MIN = -8.0
_UNIT_STEP = 1.0


def signed_q(q: Q88, value: float) -> int:
    """Encode ``value`` as the signed integer its Verilog ``'sd`` literal denotes.

    This is the two's-complement bit pattern of ``round(value * 2**fraction)``
    truncated to ``data_width`` bits, reinterpreted as a signed integer — exactly
    what :meth:`Q88.encode_signed_literal` writes into the RTL, but as a Python
    ``int`` suitable for a C/Rust literal.
    """
    raw = int(round(value * (1 << q.fraction)))
    mask = (1 << q.data_width) - 1
    raw &= mask
    if raw >= (1 << (q.data_width - 1)):
        raw -= 1 << q.data_width
    return int(raw)


_LUT_GEOMETRY: dict[str, tuple[float, float]] = {
    "exp": (_SYMMETRIC_MIN, _SYMMETRIC_STEP),
    "log": (expr_lut_tables.LOG_LUT_MIN, expr_lut_tables.LOG_LUT_STEP),
    "sqrt": (_UNIT_MIN, _UNIT_STEP),
    "tanh": (_SYMMETRIC_MIN, _SYMMETRIC_STEP),
    "cosh": (_SYMMETRIC_MIN, _SYMMETRIC_STEP),
    "exprel": (_SYMMETRIC_MIN, _SYMMETRIC_STEP),
    "sigmoid": (_SYMMETRIC_MIN, _SYMMETRIC_STEP),
    "sin": (_SYMMETRIC_MIN, _SYMMETRIC_STEP),
    "cos": (_SYMMETRIC_MIN, _SYMMETRIC_STEP),
    "cbrt": (_SYMMETRIC_MIN, _SYMMETRIC_STEP),
}


class _CFixedExprEmitter(ast.NodeVisitor):
    """Walk a Python AST and emit a bit-exact integer C/Rust expression.

    Every ``visit_*`` returns a source string that evaluates to a 64-bit signed
    integer holding a Q-format value. Multiplies, divisions, powers and LUT calls
    collapse to the word width through the generated ``fxmul`` / ``sc_wrap``
    helpers, so the composed expression reproduces the RTL datapath verbatim.

    Parameters
    ----------
    state_map : dict
        Maps ODE variable names to the source lvalue that reads their current
        register value (e.g. ``"s->v"`` in C, ``"self.v"`` in Rust).
    param_map : dict
        Maps parameter names to their already-Q-encoded *signed* integer value.
    q : Q88
        Fixed-point configuration (width, fraction, rounding).
    lang : str
        ``"c"`` or ``"rust"`` — selects the small syntactic differences (cast,
        conditional expression, array index).
    input_ref : str
        Source expression that reads the input current ``I`` (default ``"I_t"``).

    Attributes
    ----------
    statements : list of str
        Helper statements (LUT argument/index locals) that must be emitted before
        the returned expression is used, in order.
    tables : dict
        LUT variable name to its integer entries, for the generator to declare.
    free_vars : list of str
        Identifiers that are neither state, parameter nor input, in first-seen
        order — the generator declares them as extra function arguments.
    """

    def __init__(
        self,
        state_map: dict[str, str],
        param_map: dict[str, int],
        q: Q88,
        *,
        lang: str = "c",
        input_ref: str = "I_t",
        lut_start: int = 0,
    ) -> None:
        if lang not in {"c", "rust"}:
            raise ValueError(f"lang must be 'c' or 'rust', got {lang!r}")
        if 2 * q.data_width > 64:
            raise ValueError(
                f"data_width={q.data_width} needs a {2 * q.data_width}-bit intermediate; "
                "the bit-exact integer emitter supports 2*data_width <= 64 "
                "(i.e. data_width <= 32, covering Q8.8 and Q16.16)."
            )
        self.state_map = state_map
        self.param_map = param_map
        self.q = q
        self.lang = lang
        self.input_ref = input_ref
        self.statements: list[str] = []
        self.tables: dict[str, list[int]] = {}
        self.free_vars: list[str] = []
        self.input_used = False
        self._lut_count = lut_start

    # ── small language-dependent building blocks ──────────────────────────

    def _wide(self, expr: str) -> str:
        """Cast a source expression to the 64-bit signed accumulator type."""
        if self.lang == "rust":
            return f"(({expr}) as i64)"
        return f"((int64_t)({expr}))"

    def _tern(self, cond: str, when_true: str, when_false: str) -> str:
        """Emit a conditional expression (C ``?:`` / Rust ``if``-expression)."""
        if self.lang == "rust":
            return f"(if ({cond}) {{ {when_true} }} else {{ {when_false} }})"
        return f"(({cond}) ? ({when_true}) : ({when_false}))"

    def _index(self, table: str, idx: str) -> str:
        """Index a LUT table (Rust needs a ``usize`` cast)."""
        if self.lang == "rust":
            return f"{table}[({idx}) as usize]"
        return f"{table}[{idx}]"

    def _local(self, name: str, expr: str) -> str:
        """Declare a 64-bit signed local holding ``expr`` (no indentation)."""
        if self.lang == "rust":
            return f"let {name}: i64 = {expr};"
        return f"int64_t {name} = {expr};"

    # ── Q-format helpers ──────────────────────────────────────────────────

    def _q_signed(self, value: float) -> int:
        """Encode ``value`` as the signed integer its Verilog ``'sd`` literal denotes."""
        return signed_q(self.q, value)

    def _fxmul(self, left: str, right: str) -> str:
        """Multiply two wide operands with RTL wrap-truncate semantics."""
        return self._wide(f"fxmul({left}, {right})")

    # ── AST visitors ──────────────────────────────────────────────────────

    def visit_BinOp(self, node: ast.BinOp) -> str:
        """Emit a binary op (add, sub, mul, div, positive modulo, pow)."""
        left: str = self.visit(node.left)
        if isinstance(node.op, ast.Add):
            return f"({left} + {self.visit(node.right)})"
        if isinstance(node.op, ast.Sub):
            return f"({left} - {self.visit(node.right)})"
        if isinstance(node.op, ast.Mult):
            return self._fxmul(left, self.visit(node.right))
        if isinstance(node.op, ast.Div):
            return self._emit_div(node, left)
        if isinstance(node.op, ast.Mod):
            return self._emit_mod(node, left)
        if isinstance(node.op, ast.Pow):
            return self._emit_pow(node, left)
        raise ValueError(f"Unsupported binary op: {type(node.op).__name__}")

    def _emit_div(self, node: ast.BinOp, left: str) -> str:
        """Emit division: by a constant it becomes a reciprocal multiply, else a shift-divide."""
        if isinstance(node.right, ast.Constant) and isinstance(node.right.value, (int, float)):
            recip = self._q_signed(1.0 / float(node.right.value))
            return self._fxmul(left, str(recip))
        right: str = self.visit(node.right)
        dw = self.q.data_width
        wide = 2 * dw
        # num << fraction (wrapped to the 2*dw intermediate), integer-divided by
        # den, then the quotient truncated to dw bits — matching the RTL wires.
        shifted = f"sc_wrap(({left}) << {self.q.fraction}, {wide})"
        quotient = f"({shifted} / ({right}))"
        return self._wide(f"sc_wrap({quotient}, {dw})")

    def _emit_mod(self, node: ast.BinOp, left: str) -> str:
        """Emit Python-compatible modulo by one positive numeric literal.

        The generated ``fxmod`` helper first collapses the dividend to the RTL
        word width, takes the C/Rust signed remainder, then adds one positive
        period when that remainder is negative. This is exactly the correction
        used by the Verilog emitter to reproduce Python's floored ``x % p``.
        """
        if (
            not isinstance(node.right, ast.Constant)
            or isinstance(node.right.value, bool)
            or not isinstance(node.right.value, (int, float))
        ):
            raise ValueError("Modulo divisor must be a positive numeric literal")
        period = float(node.right.value)
        if not math.isfinite(period) or period <= 0.0:
            raise ValueError("Modulo divisor must be a finite positive numeric literal")
        if period > self.q.max_value:
            raise ValueError(
                f"Modulo divisor {period} exceeds fixed-point maximum {self.q.max_value}"
            )
        period_q = int(round(period * (1 << self.q.fraction)))
        if period_q <= 0:
            raise ValueError(f"Modulo divisor {period} underflows at fraction={self.q.fraction}")
        return self._wide(f"fxmod({left}, {period_q})")

    def _emit_pow(self, node: ast.BinOp, left: str) -> str:
        """Emit a power: integer 2-8 as repeated wrap-multiply, 1/2 and 1/3 as LUT roots."""
        right = node.right
        if (
            isinstance(right, ast.Constant)
            and isinstance(right.value, int)
            and 2 <= right.value <= 8
        ):
            acc = left
            for _ in range(right.value - 1):
                acc = self._fxmul(acc, left)
            return acc
        frac_exp = const_float(right)
        if frac_exp is not None and abs(frac_exp - 0.5) < 1e-6:
            return self._emit_lut("sqrt", left)
        if frac_exp is not None and abs(frac_exp - 1.0 / 3.0) < 1e-6:
            return self._emit_lut("cbrt", left)
        raise ValueError(f"Only integer powers 2-8 and 1/2, 1/3 supported, got {ast.dump(right)}")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> str:
        """Emit a unary op (negate, positive)."""
        operand: str = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return f"(-({operand}))"
        if isinstance(node.op, ast.UAdd):
            return operand
        raise ValueError(f"Unsupported unary op: {type(node.op).__name__}")

    def visit_Name(self, node: ast.Name) -> str:
        """Resolve a name to a wide read expression, recording free variables."""
        name = node.id
        if name in self.state_map:
            return self._wide(self.state_map[name])
        if name in self.param_map:
            return self._wide(str(self.param_map[name]))
        if name == "I":
            self.input_used = True
            return self._wide(self.input_ref)
        if name not in self.free_vars:
            self.free_vars.append(name)
        return self._wide(name)

    def visit_Constant(self, node: ast.Constant) -> str:
        """Emit a numeric constant as its signed Q-format integer."""
        val = float(node.value) if isinstance(node.value, (int, float)) else 0.0
        return self._wide(str(self._q_signed(val)))

    def visit_Compare(self, node: ast.Compare) -> str:
        """Emit comparison operators (>, >=, <, <=), chained with logical AND."""
        left: str = self.visit(node.left)
        results: list[str] = []
        for op, comp in zip(node.ops, node.comparators):
            right: str = self.visit(comp)
            if isinstance(op, ast.Gt):
                results.append(f"({left} > {right})")
            elif isinstance(op, ast.GtE):
                results.append(f"({left} >= {right})")
            elif isinstance(op, ast.Lt):
                results.append(f"({left} < {right})")
            elif isinstance(op, ast.LtE):
                results.append(f"({left} <= {right})")
            else:
                raise ValueError(f"Unsupported comparison: {type(op).__name__}")
            left = right
        return " && ".join(results)

    def visit_Call(self, node: ast.Call) -> str:
        """Emit a supported function call (transcendental LUT, abs, clip, max/min)."""
        if not isinstance(node.func, ast.Name):
            raise ValueError(f"Only named function calls supported, got {ast.dump(node.func)}")
        fname = node.func.id
        if fname not in SUPPORTED_FUNCTIONS and fname not in {"abs", "clip", "max", "min"}:
            raise ValueError(f"Unsupported function '{fname}' in bit-exact compilation.")
        if not node.args:
            raise ValueError(f"Function {fname} requires at least 1 argument")
        arg: str = self.visit(node.args[0])

        lut_name = {"sigmoid": "sigmoid", "expit": "sigmoid"}.get(fname, fname)
        if lut_name in _LUT_GEOMETRY:
            return self._emit_lut(lut_name, arg)
        if fname == "abs":
            return self._tern(f"{arg} < 0", f"-({arg})", arg)
        if fname == "clip":
            if len(node.args) == 3:
                lo: str = self.visit(node.args[1])
                hi: str = self.visit(node.args[2])
                inner = self._tern(f"{arg} > {hi}", hi, arg)
                return self._tern(f"{arg} < {lo}", lo, inner)
            return arg
        # max / min
        if len(node.args) >= 2:
            other: str = self.visit(node.args[1])
            cmp = ">" if fname == "max" else "<"
            return self._tern(f"{arg} {cmp} {other}", arg, other)
        return arg

    def _lut_entries(self, name: str) -> list[int]:
        """Return the quantised LUT entries for ``name`` at this word/fraction."""
        dw, frac = self.q.data_width, self.q.fraction
        generators: dict[str, Callable[[], list[int]]] = {
            "exp": lambda: expr_lut_tables.exp_lut_entries(dw, frac),
            "log": lambda: expr_lut_tables.log_lut_entries(frac),
            "sqrt": lambda: expr_lut_tables.sqrt_lut_entries(frac),
            "tanh": lambda: expr_lut_tables.tanh_lut_entries(frac),
            "cosh": lambda: expr_lut_tables.cosh_lut_entries(dw, frac),
            "exprel": lambda: expr_lut_tables.exprel_lut_entries(dw, frac),
            "sigmoid": lambda: expr_lut_tables.sigmoid_lut_entries(frac),
            "sin": lambda: expr_lut_tables.sin_lut_entries(frac),
            "cos": lambda: expr_lut_tables.cos_lut_entries(frac),
            "cbrt": lambda: expr_lut_tables.cbrt_lut_entries(frac),
        }
        return generators[name]()

    def _emit_lut(self, name: str, arg: str) -> str:
        """Emit a LUT lookup mirroring ``_VerilogExprEmitter._emit_lut_call``."""
        entries = self._lut_entries(name)
        lut_min, lut_step = _LUT_GEOMETRY[name]
        n = len(entries)
        dw, frac = self.q.data_width, self.q.fraction
        dwp1 = dw + 1
        shift = frac - round(-math.log2(lut_step))
        min_q = round(lut_min * (1 << frac))
        table = f"_{name}_lut{self._lut_count}"
        argv = f"{table}_arg"
        raw = f"{table}_raw"
        idx = f"{table}_idx"
        self._lut_count += 1
        self.tables[table] = entries

        # Argument first truncated to the word width (RTL hoists it to a dw wire).
        self.statements.append(self._local(argv, f"sc_wrap({arg}, {dw})"))
        op = "-" if min_q >= 0 else "+"
        offset = f"sc_wrap(({argv} {op} {abs(min_q)}), {dwp1})"
        shifted = f"{offset} >> {shift}" if shift >= 0 else f"{offset} << {-shift}"
        # Offset+shift both happen in the (dw+1)-bit domain of the RTL raw wire.
        self.statements.append(self._local(raw, f"sc_wrap({shifted}, {dwp1})"))
        clamp = self._tern(f"{raw} < 0", "0", self._tern(f"{raw} > {n - 1}", str(n - 1), raw))
        self.statements.append(self._local(idx, clamp))
        return self._wide(self._index(table, idx))

    def generic_visit(self, node: ast.AST) -> str:
        """Raise for any unsupported AST node type."""
        raise ValueError(f"Unsupported AST node for bit-exact C/Rust: {type(node).__name__}")


def emit_c_fixed_expr(
    expr_str: str,
    state_map: dict[str, str],
    param_map: dict[str, int],
    q: Q88,
    *,
    lang: str = "c",
    input_ref: str = "I_t",
    lut_start: int = 0,
) -> tuple[str, list[str], dict[str, list[int]], list[str], int, bool]:
    """Lower an ODE expression to a bit-exact integer C/Rust expression.

    Parameters
    ----------
    expr_str : str
        Python-syntax ODE right-hand-side expression.
    state_map : dict
        ODE variable name to its register-read source lvalue.
    param_map : dict
        Parameter name to its signed Q-format integer value.
    q : Q88
        Fixed-point configuration.
    lang : str
        ``"c"`` or ``"rust"``.
    input_ref : str
        Source expression reading the input current ``I``.
    lut_start : int
        Starting index for LUT table naming, so several expressions of one kernel
        get unique table names.

    Returns
    -------
    tuple
        ``(expr, statements, tables, free_vars, lut_count, input_used)`` — the
        64-bit-integer expression string, the helper statements it depends on, the
        LUT tables it references, the free identifiers it introduced (first-seen
        order), the next free LUT index, and whether the input current was read.
    """
    tree = ast.parse(expr_str, mode="eval")
    emitter = _CFixedExprEmitter(
        state_map, param_map, q, lang=lang, input_ref=input_ref, lut_start=lut_start
    )
    result: str = emitter.visit(tree.body)
    return (
        result,
        emitter.statements,
        emitter.tables,
        emitter.free_vars,
        emitter._lut_count,
        emitter.input_used,
    )
