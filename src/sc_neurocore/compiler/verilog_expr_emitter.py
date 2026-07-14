# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Verilog expression emitter

"""Walk a Python AST and emit equivalent Verilog fixed-point expressions."""

from __future__ import annotations

import ast
import math

from ..hdl_gen._ident import sanitize_ident
from . import expr_lut_tables
from .verilog_compiler_config import Q88


class _VerilogExprEmitter(ast.NodeVisitor):
    """Walk a Python AST and emit equivalent Verilog fixed-point expressions.

    Handles: +, -, *, /, signed floor division by a positive integer
    power-of-two literal, positive-literal %, **, unary minus, comparisons,
    names, constants.
    Multiplications emit wide product with arithmetic right shift.
    """

    def __init__(
        self,
        state_vars: dict[str, str],
        param_map: dict[str, str],
        q: Q88,
        *,
        mul_start: int = 0,
        trunc_start: int = 0,
    ):
        """Initialise the Verilog expression emitter.

        Parameters
        ----------
        state_vars : dict
            Mapping from ODE variable names to sanitised Verilog identifiers.
        param_map : dict
            Mapping from parameter names to Verilog parameter identifiers.
        q : Q88
            Fixed-point format configuration (width, fraction, rounding, overflow).
        mul_start : int
            Starting counter for multiply wire naming (for chaining emitters).
        trunc_start : int
            Starting counter for truncation wire naming.
        """
        self.state_vars = state_vars
        self.param_map = param_map
        self.q = q
        self._mul_count = mul_start
        self._trunc_count = trunc_start
        self.intermediates: list[str] = []
        self._pipeline = False
        self._pipeline_points: set[str] = set()  # user-specified insertion points
        self._pipeline_regs: list[str] = []  # registered intermediates

    @staticmethod
    def _const_float(node: ast.AST) -> float | None:
        """Constant-fold a literal or simple literal arithmetic node to a float.

        Delegates to the shared :func:`expr_lut_tables.const_float` so every
        backend recognises the same compile-time constants (e.g. ``1.0 / 3.0``).
        """
        return expr_lut_tables.const_float(node)

    def _trunc(self, wide_name: str) -> str:
        """Emit an intermediate wire for fixed-point truncation with rounding."""
        trunc_name = f"_t{self._trunc_count}"
        self._trunc_count += 1
        dw = self.q.data_width
        frac = self.q.fraction
        sign = "signed " if self.q.signed else ""
        rounding = self.q.rounding

        if rounding == "truncate":
            self.intermediates.append(
                f"wire {sign}[{dw - 1}:0] {trunc_name} = ({wide_name} >>> {frac});"
            )
        elif rounding == "nearest":
            half = f"_rnd_half{self._trunc_count}"
            self.intermediates.append(
                f"wire {sign}[{2 * dw - 1}:0] {half} = {wide_name} + {1 << (frac - 1)};"
            )
            self.intermediates.append(
                f"wire {sign}[{dw - 1}:0] {trunc_name} = ({half} >>> {frac});"
            )
        elif rounding == "bankers":
            biased = f"_rnd_biased{self._trunc_count}"
            guard = f"_rnd_guard{self._trunc_count}"
            self.intermediates.append(
                f"wire {sign}[{2 * dw - 1}:0] {biased} = {wide_name} + {1 << (frac - 1)};"
            )
            self.intermediates.append(
                f"wire {guard} = ({wide_name}[{frac - 1}:0] == {1 << (frac - 1)});"
            )
            self.intermediates.append(
                f"wire {sign}[{dw - 1}:0] {trunc_name} = "
                f"({biased} >>> {frac}) & "
                f"(({guard}) ? ~{dw}'d1 : {{{dw}{{1'b1}}}});"
            )
        elif rounding == "stochastic":
            stoch = f"_rnd_stoch{self._trunc_count}"
            self.intermediates.append(
                f"wire {sign}[{2 * dw - 1}:0] {stoch} = "
                f"{wide_name} + {{{{({2 * dw - frac}){{1'b0}}}}, _lfsr[{frac - 1}:0]}};"
            )
            self.intermediates.append(
                f"wire {sign}[{dw - 1}:0] {trunc_name} = ({stoch} >>> {frac});"
            )
        else:
            raise ValueError(f"Unknown rounding mode: {rounding!r}")

        return trunc_name

    def visit_BinOp(self, node: ast.BinOp) -> str:
        """Emit Verilog for a binary operation, including narrow floor division."""
        if isinstance(node.op, ast.FloorDiv) and not self.q.signed:
            raise ValueError("Floor divisor requires signed fixed-point state")
        left: str = self.visit(node.left)
        right: str = self.visit(node.right)

        if isinstance(node.op, ast.Add):
            return f"({left} + {right})"
        elif isinstance(node.op, ast.Sub):
            return f"({left} - {right})"
        elif isinstance(node.op, ast.Mult):
            tmp = f"_mul{self._mul_count}"
            self._mul_count += 1
            should_pipe = (
                self._pipeline
                or tmp in self._pipeline_points
                or f"mul{self._mul_count - 1}" in self._pipeline_points
            )
            if should_pipe:
                self._pipeline_regs.append(f"reg signed [{2 * self.q.data_width - 1}:0] {tmp}_r;")
                self.intermediates.append(
                    f"wire signed [{2 * self.q.data_width - 1}:0] {tmp} = {left} * {right};"
                )
                return self._trunc(f"{tmp}_r")
            else:
                self.intermediates.append(
                    f"wire signed [{2 * self.q.data_width - 1}:0] {tmp} = {left} * {right};"
                )
                return self._trunc(tmp)
        elif isinstance(node.op, ast.Div):
            if isinstance(node.right, ast.Constant) and isinstance(node.right.value, (int, float)):
                recip = 1.0 / float(node.right.value)
                recip_q = self.q.encode_signed_literal(recip)
                tmp = f"_mul{self._mul_count}"
                self._mul_count += 1
                self.intermediates.append(
                    f"wire signed [{2 * self.q.data_width - 1}:0] {tmp} = {left} * {recip_q};"
                )
                return self._trunc(tmp)
            dw = self.q.data_width
            frac = self.q.fraction
            wide = 2 * dw
            num_wire = f"_dop{self._mul_count}"
            den_wire = f"_dden{self._mul_count}"
            ext_name = f"_dnum{self._mul_count}"
            div_tmp = f"_div{self._mul_count}"
            self._mul_count += 1
            # Hoist both operands to named wires: a bit-select (msb for sign
            # extension) is only valid on a net/reg, not on a compound expression
            # such as ``(a - b)``, so ``right`` must be assigned before extension.
            self.intermediates.append(f"wire signed [{dw - 1}:0] {num_wire} = {left};")
            self.intermediates.append(f"wire signed [{dw - 1}:0] {den_wire} = {right};")
            self.intermediates.append(
                f"wire signed [{wide - 1}:0] {ext_name} = "
                f"$signed({{{{{dw}{{{num_wire}[{dw - 1}]}}}}, {num_wire}}}) <<< {frac};"
            )
            res_name = f"_dres{self._mul_count - 1}"
            self.intermediates.append(
                f"wire signed [{wide - 1}:0] {div_tmp} = "
                f"{ext_name} / $signed({{{{{dw}{{{den_wire}[{dw - 1}]}}}}, {den_wire}}});"
            )
            self.intermediates.append(
                f"wire signed [{dw - 1}:0] {res_name} = {div_tmp}[{dw - 1}:0];"
            )
            return res_name
        elif isinstance(node.op, ast.FloorDiv):
            if (
                not isinstance(node.right, ast.Constant)
                or isinstance(node.right.value, bool)
                or not isinstance(node.right.value, int)
            ):
                raise ValueError("Floor divisor must be a positive integer power-of-two literal")
            divisor = node.right.value
            if divisor <= 0 or divisor & (divisor - 1):
                raise ValueError("Floor divisor must be a positive integer power-of-two literal")
            if divisor > self.q.max_value:
                raise ValueError(
                    f"Floor divisor {divisor} exceeds fixed-point maximum {self.q.max_value}"
                )

            op_index = self._mul_count
            self._mul_count += 1
            dw = self.q.data_width
            frac = self.q.fraction
            shift = frac + divisor.bit_length() - 1
            dividend = f"_floordiv{op_index}_dividend"
            quotient = f"_floordiv{op_index}_integer"
            result = f"_floordiv{op_index}"
            # A Q-format dividend stores x*2**frac. Python ``x // 2**k``
            # first floors to a whole integer and then represents that integer
            # in the same Q format. An arithmetic shift by frac+k performs the
            # signed floor; shifting back by frac restores the scale.
            self.intermediates.append(f"wire signed [{dw - 1}:0] {dividend} = {left};")
            self.intermediates.append(
                f"wire signed [{dw - 1}:0] {quotient} = $signed({dividend}) >>> {shift};"
            )
            if frac == 0:
                self.intermediates.append(f"wire signed [{dw - 1}:0] {result} = {quotient};")
            else:
                self.intermediates.append(
                    f"wire signed [{dw - 1}:0] {result} = {quotient} <<< {frac};"
                )
            return result
        elif isinstance(node.op, ast.Mod):
            if not self.q.signed:
                raise ValueError("Positive-literal modulo requires signed fixed-point state")
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
                raise ValueError(
                    f"Modulo divisor {period} underflows at fraction={self.q.fraction}"
                )

            op_index = self._mul_count
            self._mul_count += 1
            dw = self.q.data_width
            dividend = f"_mod{op_index}_dividend"
            remainder = f"_mod{op_index}_remainder"
            result = f"_mod{op_index}"
            period_literal = f"{dw}'sd{period_q}"
            # Python's ``x % p`` with p > 0 is a floored remainder in [0, p),
            # whereas Verilog's remainder keeps the dividend's sign. Hoist the
            # compound dividend to one data-width word, take the hardware
            # remainder, then correct its negative branch by one period.
            self.intermediates.append(f"wire signed [{dw - 1}:0] {dividend} = {left};")
            self.intermediates.append(
                f"wire signed [{dw - 1}:0] {remainder} = "
                f"$signed({dividend}) % $signed({period_literal});"
            )
            self.intermediates.append(
                f"wire signed [{dw - 1}:0] {result} = "
                f"({remainder} < 0) ? ({remainder} + {period_literal}) : {remainder};"
            )
            return result
        elif isinstance(node.op, ast.Pow):
            if isinstance(node.right, ast.Constant) and node.right.value == 2:
                tmp = f"_mul{self._mul_count}"
                self._mul_count += 1
                self.intermediates.append(
                    f"wire signed [{2 * self.q.data_width - 1}:0] {tmp} = {left} * {left};"
                )
                return self._trunc(tmp)
            elif isinstance(node.right, ast.Constant) and node.right.value == 3:
                sq = f"_mul{self._mul_count}"
                self._mul_count += 1
                self.intermediates.append(
                    f"wire signed [{2 * self.q.data_width - 1}:0] {sq} = {left} * {left};"
                )
                sq_trunc = self._trunc(sq)
                cu = f"_mul{self._mul_count}"
                self._mul_count += 1
                self.intermediates.append(
                    f"wire signed [{2 * self.q.data_width - 1}:0] {cu} = {sq_trunc} * {left};"
                )
                return self._trunc(cu)
            elif (
                isinstance(node.right, ast.Constant)
                and isinstance(node.right.value, int)
                and 4 <= node.right.value <= 8
            ):
                exp: int = node.right.value
                prev: str = left
                for step in range(exp - 1):
                    tmp = f"_mul{self._mul_count}"
                    self._mul_count += 1
                    self.intermediates.append(
                        f"wire signed [{2 * self.q.data_width - 1}:0] {tmp} = {prev} * {left};"
                    )
                    prev = self._trunc(tmp)
                return prev
            frac_exp = self._const_float(node.right)
            if frac_exp is not None and abs(frac_exp - 1.0 / 3.0) < 1e-6:
                return self._emit_lut_call("_cbrt_lut", left, self._cbrt_lut_entries())
            if frac_exp is not None and abs(frac_exp - 0.5) < 1e-6:
                return self._emit_lut_call(
                    "_sqrt_lut", left, self._sqrt_lut_entries(), lut_min=-8.0, lut_step=1.0
                )
            raise ValueError(
                f"Only integer powers 2-8 and 1/2, 1/3 supported in Verilog, "
                f"got {ast.dump(node.right)}"
            )
        raise ValueError(f"Unsupported binary op: {type(node.op).__name__}")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> str:
        """Emit Verilog for a unary operation (negate, positive)."""
        operand: str = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return f"(-{operand})"
        if isinstance(node.op, ast.UAdd):
            return str(operand)
        raise ValueError(f"Unsupported unary op: {type(node.op).__name__}")

    def visit_Name(self, node: ast.Name) -> str:
        """Resolve a Python name to its Verilog equivalent."""
        name = node.id
        if name in self.state_vars:
            return f"{self.state_vars[name]}_reg"
        if name in self.param_map:
            return self.param_map[name]
        if name == "I":
            return "I_t"
        return str(sanitize_ident(name, context="expression identifier"))

    def visit_Constant(self, node: ast.Constant) -> str:
        """Encode a numeric constant as a Verilog signed literal in Q-format."""
        val: float = float(node.value) if isinstance(node.value, (int, float)) else 0.0
        return str(self.q.encode_signed_literal(val))

    def visit_Compare(self, node: ast.Compare) -> str:
        """Emit Verilog for comparison operators (>, >=, <, <=)."""
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

    def visit_IfExp(self, node: ast.IfExp) -> str:
        """Emit Verilog for a conditional expression ``body if test else orelse``.

        Lowers a Python piecewise/ternary expression to a Verilog ``?:`` select,
        reusing :meth:`visit_Compare` for the boolean test. This lets a schema
        express a piecewise map branch (for example the Rulkov 2002 fast map) in the
        same combinational datapath the arithmetic nodes already use.
        """
        test: str = self.visit(node.test)
        body: str = self.visit(node.body)
        orelse: str = self.visit(node.orelse)
        return f"(({test}) ? ({body}) : ({orelse}))"

    def visit_Call(self, node: ast.Call) -> str:
        """Emit Verilog for function calls."""
        if not isinstance(node.func, ast.Name):
            raise ValueError(f"Only named function calls supported, got {ast.dump(node.func)}")
        fname = node.func.id
        if len(node.args) < 1:
            raise ValueError(f"Function {fname} requires at least 1 argument")
        arg: str = self.visit(node.args[0])

        if fname == "exp":
            return self._emit_lut_call("_exp_lut", arg, self._exp_lut_entries())
        elif fname == "log":
            return self._emit_lut_call(
                "_log_lut",
                arg,
                self._log_lut_entries(),
                lut_min=expr_lut_tables.LOG_LUT_MIN,
                lut_step=expr_lut_tables.LOG_LUT_STEP,
            )
        elif fname == "sqrt":
            return self._emit_lut_call(
                "_sqrt_lut", arg, self._sqrt_lut_entries(), lut_min=-8.0, lut_step=1.0
            )
        elif fname == "tanh":
            return self._emit_lut_call("_tanh_lut", arg, self._tanh_lut_entries())
        elif fname == "cosh":
            return self._emit_lut_call("_cosh_lut", arg, self._cosh_lut_entries())
        elif fname == "exprel":
            return self._emit_lut_call("_exprel_lut", arg, self._exprel_lut_entries())
        elif fname in ("sigmoid", "expit"):
            return self._emit_lut_call("_sigmoid_lut", arg, self._sigmoid_lut_entries())
        elif fname == "sin":
            return self._emit_lut_call("_sin_lut", arg, self._sin_lut_entries())
        elif fname == "cos":
            return self._emit_lut_call("_cos_lut", arg, self._cos_lut_entries())
        elif fname == "abs":
            return f"(({arg} < 0) ? (-{arg}) : {arg})"
        elif fname == "clip":
            if len(node.args) == 3:
                lo: str = self.visit(node.args[1])
                hi: str = self.visit(node.args[2])
                return f"(({arg} < {lo}) ? {lo} : (({arg} > {hi}) ? {hi} : {arg}))"
            return arg
        elif fname in ("max", "min"):
            if len(node.args) >= 2:
                b: str = self.visit(node.args[1])
                if fname == "max":
                    return f"(({arg} > {b}) ? {arg} : {b})"
                return f"(({arg} < {b}) ? {arg} : {b})"
            return arg
        raise ValueError(f"Unsupported function '{fname}' in Verilog compilation.")

    def _emit_lut_call(
        self,
        lut_name: str,
        arg: str,
        entries: list[int],
        *,
        lut_min: float = -16.0,
        lut_step: float = 0.125,
    ) -> str:
        """Emit an ``len(entries)``-entry LUT over ``[lut_min, lut_min + N*step)``.

        ``lut_step`` must be a power of two so the index reduces to a shift. The
        index saturates to ``[0, N-1]``, so arguments outside the table clamp to its
        endpoints instead of wrapping (essential for steep functions whose argument
        leaves the tabulated range, e.g. Hodgkin-Huxley gating).
        """
        import math

        n = len(entries)
        idx_bits = max(1, (n - 1).bit_length())
        lut_id = f"{lut_name}{self._mul_count}"
        self._mul_count += 1
        dw = self.q.data_width
        frac = self.q.fraction
        s = round(-math.log2(lut_step))  # lut_step == 2**(-s)
        shift = frac - s
        min_q = round(lut_min * (1 << frac))
        self.intermediates.append(
            f"// {lut_name} lookup table ({n} entries over "
            f"[{lut_min}, {lut_min + n * lut_step}), step {lut_step})"
        )
        # Hoist the argument to a wire so the sign-extension bit-select is valid even
        # when the argument is a compound expression.
        arg_w = f"{lut_id}_arg"
        self.intermediates.append(f"wire signed [{dw - 1}:0] {arg_w} = {arg};")
        ext = f"{{{arg_w}[{dw - 1}]}}, {arg_w}"  # (dw+1)-bit sign extension
        op = "-" if min_q >= 0 else "+"
        shift_op = f">>> {shift}" if shift >= 0 else f"<<< {-shift}"
        raw = f"{lut_id}_raw"
        self.intermediates.append(
            f"wire signed [{dw}:0] {raw} = (({{{ext}}}) {op} {dw + 1}'sd{abs(min_q)}) {shift_op};"
        )
        idx_wire = f"{lut_id}_idx"
        self.intermediates.append(
            f"wire [{idx_bits - 1}:0] {idx_wire} = "
            f"({raw} < 0) ? {idx_bits}'d0 : "
            f"(({raw} > {dw + 1}'sd{n - 1}) ? {idx_bits}'d{n - 1} : {raw}[{idx_bits - 1}:0]);"
        )
        result_wire = f"{lut_id}_out"
        lines = [f"reg signed [{dw - 1}:0] {result_wire};"]
        lines.append(f"always @(*) case ({idx_wire})")
        for i, val in enumerate(entries):
            # A negative value must be written -W'sdN, not W'sd-N (malformed Verilog).
            literal = f"-{dw}'sd{-val}" if val < 0 else f"{dw}'sd{val}"
            lines.append(f"    {idx_bits}'d{i}: {result_wire} = {literal};")
        lines.append(f"    default: {result_wire} = {dw}'sd0;")
        lines.append("endcase")
        for line in lines:
            self.intermediates.append(line)
        return result_wire

    def _sym_points(self) -> list[float]:
        """Return the shared 256-point symmetric sample grid over [-16, 16)."""
        return expr_lut_tables.symmetric_sample_points()

    def _exp_lut_entries(self) -> list[int]:
        """Quantised ``exp`` LUT for this word's width and fraction."""
        return expr_lut_tables.exp_lut_entries(self.q.data_width, self.q.fraction)

    def _log_lut_entries(self) -> list[int]:
        """Quantised ``log`` LUT for this word's fraction."""
        return expr_lut_tables.log_lut_entries(self.q.fraction)

    def _sqrt_lut_entries(self) -> list[int]:
        """Quantised ``sqrt`` LUT for this word's fraction."""
        return expr_lut_tables.sqrt_lut_entries(self.q.fraction)

    def _tanh_lut_entries(self) -> list[int]:
        """Quantised ``tanh`` LUT for this word's fraction."""
        return expr_lut_tables.tanh_lut_entries(self.q.fraction)

    def _cosh_lut_entries(self) -> list[int]:
        """Quantised ``cosh`` LUT for this word's width and fraction."""
        return expr_lut_tables.cosh_lut_entries(self.q.data_width, self.q.fraction)

    def _cbrt_lut_entries(self) -> list[int]:
        """Quantised cube-root LUT for this word's fraction."""
        return expr_lut_tables.cbrt_lut_entries(self.q.fraction)

    def _exprel_lut_entries(self) -> list[int]:
        """Quantised ``exprel`` LUT for this word's width and fraction."""
        return expr_lut_tables.exprel_lut_entries(self.q.data_width, self.q.fraction)

    def _sigmoid_lut_entries(self) -> list[int]:
        """Quantised logistic-sigmoid LUT for this word's fraction."""
        return expr_lut_tables.sigmoid_lut_entries(self.q.fraction)

    def _sin_lut_entries(self) -> list[int]:
        """Quantised ``sin`` LUT for this word's fraction."""
        return expr_lut_tables.sin_lut_entries(self.q.fraction)

    def _cos_lut_entries(self) -> list[int]:
        """Quantised ``cos`` LUT for this word's fraction."""
        return expr_lut_tables.cos_lut_entries(self.q.fraction)

    def generic_visit(self, node: ast.AST) -> str:
        """Raise an error for any unsupported AST node type."""
        raise ValueError(f"Unsupported AST node for Verilog: {type(node).__name__}")


def _emit_expr(
    expr_str: str,
    state_vars: dict[str, str],
    param_map: dict[str, str],
    q: Q88,
    *,
    mul_start: int = 0,
    trunc_start: int = 0,
    pipeline: bool = False,
    pipeline_points: set[str] | None = None,
) -> tuple[str, list[str], int, int, list[str]]:
    """Parse a Python expression string and return Verilog."""
    tree = ast.parse(expr_str, mode="eval")
    emitter = _VerilogExprEmitter(
        state_vars,
        param_map,
        q,
        mul_start=mul_start,
        trunc_start=trunc_start,
    )
    emitter._pipeline = pipeline
    if pipeline_points:
        emitter._pipeline_points = pipeline_points
    result = emitter.visit(tree.body)
    return (
        result,
        emitter.intermediates,
        emitter._mul_count,
        emitter._trunc_count,
        emitter._pipeline_regs,
    )
