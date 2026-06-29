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

from ..hdl_gen._ident import sanitize_ident
from .verilog_compiler_config import Q88


class _VerilogExprEmitter(ast.NodeVisitor):
    """Walk a Python AST and emit equivalent Verilog fixed-point expressions.

    Handles: +, -, *, /, **, unary minus, comparisons, names, constants.
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
        """Emit Verilog for a binary operation (add, sub, mul, div)."""
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
            raise ValueError(
                f"Only integer powers 2-8 supported in Verilog, got {ast.dump(node.right)}"
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
        return sanitize_ident(name, context="expression identifier")

    def visit_Constant(self, node: ast.Constant) -> str:
        """Encode a numeric constant as a Verilog signed literal in Q-format."""
        val: float = float(node.value) if isinstance(node.value, (int, float)) else 0.0
        return self.q.encode_signed_literal(val)

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
        return " && ".join(results)

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
            return self._emit_lut_call("_log_lut", arg, self._log_lut_entries())
        elif fname == "sqrt":
            return self._emit_lut_call("_sqrt_lut", arg, self._sqrt_lut_entries())
        elif fname == "tanh":
            return self._emit_lut_call("_tanh_lut", arg, self._tanh_lut_entries())
        elif fname == "cosh":
            return self._emit_lut_call("_cosh_lut", arg, self._cosh_lut_entries())
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

    def _emit_lut_call(self, lut_name: str, arg: str, entries: list[int]) -> str:
        """Emit a 16-entry LUT indexed by top 4 bits of the input."""
        lut_id = f"{lut_name}{self._mul_count}"
        self._mul_count += 1
        dw = self.q.data_width
        self.intermediates.append(
            f"// {lut_name} lookup table (16 entries, Q{dw - self.q.fraction}.{self.q.fraction})"
        )
        offset = 8 << self.q.fraction
        idx_wire = f"{lut_id}_idx"
        self.intermediates.append(
            f"wire [3:0] {idx_wire} = ({arg} + {dw}'sd{offset}) >>> {self.q.fraction + 4 - 4};"
        )
        result_wire = f"{lut_id}_out"
        lines = [f"reg signed [{dw - 1}:0] {result_wire};"]
        lines.append(f"always @(*) case ({idx_wire})")
        for i, val in enumerate(entries):
            # A negative value must be written -W'sdN, not W'sd-N (malformed Verilog).
            literal = f"-{dw}'sd{-val}" if val < 0 else f"{dw}'sd{val}"
            lines.append(f"    4'd{i}: {result_wire} = {literal};")
        lines.append(f"    default: {result_wire} = {dw}'sd0;")
        lines.append("endcase")
        for line in lines:
            self.intermediates.append(line)
        return result_wire

    def _exp_lut_entries(self) -> list[int]:
        import math

        points = [(-8 + i) for i in range(16)]
        return [min(int(round(math.exp(x) * (1 << self.q.fraction))), 32767) for x in points]

    def _log_lut_entries(self) -> list[int]:
        import math

        return [
            int(round(math.log(max(0.06 + i * 0.5, 0.001)) * (1 << self.q.fraction)))
            for i in range(16)
        ]

    def _sqrt_lut_entries(self) -> list[int]:
        import math

        return [int(round(math.sqrt(max(i * 0.5, 0)) * (1 << self.q.fraction))) for i in range(16)]

    def _tanh_lut_entries(self) -> list[int]:
        import math

        points = [(-8 + i) for i in range(16)]
        return [int(round(math.tanh(x) * (1 << self.q.fraction))) for x in points]

    def _cosh_lut_entries(self) -> list[int]:
        import math

        # cosh grows fast; saturate at the signed 16-bit max like the exp LUT so
        # large arguments clamp rather than overflow the fixed-point word.
        points = [(-8 + i) for i in range(16)]
        return [min(int(round(math.cosh(x) * (1 << self.q.fraction))), 32767) for x in points]

    def _sigmoid_lut_entries(self) -> list[int]:
        import math

        points = [(-8 + i) for i in range(16)]
        return [int(round(1.0 / (1.0 + math.exp(-x)) * (1 << self.q.fraction))) for x in points]

    def _sin_lut_entries(self) -> list[int]:
        import math

        points = [(-8 + i) for i in range(16)]
        return [int(round(math.sin(x) * (1 << self.q.fraction))) for x in points]

    def _cos_lut_entries(self) -> list[int]:
        import math

        points = [(-8 + i) for i in range(16)]
        return [int(round(math.cos(x) * (1 << self.q.fraction))) for x in points]

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
