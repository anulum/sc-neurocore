# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — C/C++ expression emitter for HLS lowering

"""Walk a Python AST and emit an equivalent C/C++ (Vitis HLS) expression.

Companion to :mod:`sc_neurocore.compiler.verilog_expr_emitter`: it lowers the
same ODE expression grammar (see
:data:`sc_neurocore.compiler.expr_lut_tables.SUPPORTED_FUNCTIONS`) to a C++
expression string over an ``ap_fixed`` fixed-point type. ``ap_fixed`` overloads
the arithmetic operators, so unlike the Verilog backend no explicit Q-format
scaling is emitted; the fractional-bit width lives in the type. Transcendentals
map to the Vitis ``hls_math`` library, except ``sigmoid``/``expit`` and
``exprel`` which are emitted as calls to small inline helpers (defined by the
HLS exporter) so the generated code is self-contained.

Free identifiers — names that are neither a state variable, a mapped parameter,
nor the input current ``I`` — are recorded on :attr:`CExprEmitter.free_vars` so
the exporter can declare them as function inputs, keeping the generated function
compilable.
"""

from __future__ import annotations

import ast

from .expr_lut_tables import SUPPORTED_FUNCTIONS, const_float

# Transcendentals that map directly to a single hls_math / <cmath> function.
_DIRECT_MATH_FUNCTIONS: dict[str, str] = {
    "exp": "exp",
    "log": "log",
    "sqrt": "sqrt",
    "tanh": "tanh",
    "cosh": "cosh",
    "sin": "sin",
    "cos": "cos",
}


class CExprEmitter(ast.NodeVisitor):
    """Lower a Python expression AST to a C++ (``ap_fixed``) expression string.

    Parameters
    ----------
    state_vars : set of str
        ODE state-variable names; emitted verbatim (they are struct members /
        locals in the generated function).
    param_map : dict, optional
        Mapping from parameter names to their C++ identifiers.
    math_ns : str
        Namespace prefix for transcendental calls (``"hls"`` for Vitis HLS math,
        ``"std"`` for a portable ``<cmath>`` build).
    fp_type : str
        Fixed-point type name used to cast numeric literals.

    Attributes
    ----------
    free_vars : list of str
        Identifiers referenced by the expression that are not state variables,
        parameters, or the input current — collected in first-seen order for the
        exporter to declare as inputs.
    """

    def __init__(
        self,
        state_vars: set[str],
        param_map: dict[str, str] | None = None,
        *,
        math_ns: str = "hls",
        fp_type: str = "fp_t",
    ) -> None:
        self.state_vars = state_vars
        self.param_map = param_map or {}
        self.math_ns = math_ns
        self.fp_type = fp_type
        self.free_vars: list[str] = []

    def visit_BinOp(self, node: ast.BinOp) -> str:
        """Emit C++ for a binary operation (add, sub, mul, div, pow)."""
        left: str = self.visit(node.left)
        if isinstance(node.op, ast.Add):
            return f"({left} + {self.visit(node.right)})"
        if isinstance(node.op, ast.Sub):
            return f"({left} - {self.visit(node.right)})"
        if isinstance(node.op, ast.Mult):
            return f"({left} * {self.visit(node.right)})"
        if isinstance(node.op, ast.Div):
            return f"({left} / {self.visit(node.right)})"
        if isinstance(node.op, ast.Pow):
            return self._emit_pow(node, left)
        raise ValueError(f"Unsupported binary op: {type(node.op).__name__}")

    def _emit_pow(self, node: ast.BinOp, left: str) -> str:
        """Emit C++ for a power: integer 2-8 as repeated multiply, 1/2, 1/3 as roots."""
        if isinstance(node.right, ast.Constant) and isinstance(node.right.value, int):
            exp = node.right.value
            if 2 <= exp <= 8:
                return "(" + " * ".join([left] * exp) + ")"
        frac_exp = const_float(node.right)
        if frac_exp is not None and abs(frac_exp - 0.5) < 1e-6:
            return f"{self.math_ns}::sqrt({left})"
        if frac_exp is not None and abs(frac_exp - 1.0 / 3.0) < 1e-6:
            return f"{self.math_ns}::cbrt({left})"
        raise ValueError(
            f"Only integer powers 2-8 and 1/2, 1/3 supported, got {ast.dump(node.right)}"
        )

    def visit_UnaryOp(self, node: ast.UnaryOp) -> str:
        """Emit C++ for a unary operation (negate, positive)."""
        operand: str = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return f"(-{operand})"
        if isinstance(node.op, ast.UAdd):
            return operand
        raise ValueError(f"Unsupported unary op: {type(node.op).__name__}")

    def visit_Name(self, node: ast.Name) -> str:
        """Resolve a Python name to its C++ identifier, recording free variables."""
        name = node.id
        if name in self.state_vars:
            return name
        if name in self.param_map:
            return self.param_map[name]
        if name == "I":
            return "I_t"
        if name not in self.free_vars:
            self.free_vars.append(name)
        return name

    def visit_Constant(self, node: ast.Constant) -> str:
        """Emit a numeric constant cast to the fixed-point type."""
        val = float(node.value) if isinstance(node.value, (int, float)) else 0.0
        return f"{self.fp_type}({val!r})"

    def visit_Compare(self, node: ast.Compare) -> str:
        """Emit C++ for comparison operators (>, >=, <, <=)."""
        left: str = self.visit(node.left)
        results: list[str] = []
        for op, comp in zip(node.ops, node.comparators):
            right = self.visit(comp)
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
        """Emit C++ for a supported function call."""
        if not isinstance(node.func, ast.Name):
            raise ValueError(f"Only named function calls supported, got {ast.dump(node.func)}")
        fname = node.func.id
        if fname not in SUPPORTED_FUNCTIONS:
            raise ValueError(f"Unsupported function '{fname}' in C/C++ compilation.")
        if not node.args:
            raise ValueError(f"Function {fname} requires at least 1 argument")
        arg: str = self.visit(node.args[0])

        if fname in _DIRECT_MATH_FUNCTIONS:
            return f"{self.math_ns}::{_DIRECT_MATH_FUNCTIONS[fname]}({arg})"
        if fname in ("sigmoid", "expit"):
            return f"sc_sigmoid({arg})"
        if fname == "exprel":
            return f"sc_exprel({arg})"
        if fname == "abs":
            return f"{self.math_ns}::abs({arg})"
        if fname == "clip":
            if len(node.args) == 3:
                lo: str = self.visit(node.args[1])
                hi: str = self.visit(node.args[2])
                return f"(({arg} < {lo}) ? {lo} : (({arg} > {hi}) ? {hi} : {arg}))"
            return arg
        if fname in ("max", "min"):
            if len(node.args) >= 2:
                other: str = self.visit(node.args[1])
                cmp = ">" if fname == "max" else "<"
                return f"(({arg} {cmp} {other}) ? {arg} : {other})"
            return arg
        raise ValueError(f"Unsupported function '{fname}' in C/C++ compilation.")

    def generic_visit(self, node: ast.AST) -> str:
        """Raise for any unsupported AST node type."""
        raise ValueError(f"Unsupported AST node for C/C++: {type(node).__name__}")


def emit_c_expr(
    expr_str: str,
    state_vars: set[str],
    param_map: dict[str, str] | None = None,
    *,
    math_ns: str = "hls",
    fp_type: str = "fp_t",
) -> tuple[str, list[str]]:
    """Parse an ODE expression and return its C++ form plus its free variables.

    Parameters
    ----------
    expr_str : str
        Python-syntax ODE expression.
    state_vars : set of str
        ODE state-variable names.
    param_map : dict, optional
        Parameter-name to C++-identifier mapping.
    math_ns : str
        Namespace prefix for transcendental calls.
    fp_type : str
        Fixed-point type used to cast numeric literals.

    Returns
    -------
    tuple of (str, list of str)
        The C++ expression string and the free identifiers it references (in
        first-seen order).
    """
    tree = ast.parse(expr_str, mode="eval")
    emitter = CExprEmitter(state_vars, param_map, math_ns=math_ns, fp_type=fp_type)
    result = emitter.visit(tree.body)
    return result, emitter.free_vars
