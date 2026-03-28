# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Arbitrary-equation neuron builder — define any model

"""Arbitrary-equation neuron builder — define any model from strings.

Eliminates the last competitive gap with Brian2: users specify ODEs
as strings, and EquationNeuron compiles them into a working model.

Usage:
    from sc_neurocore.neurons.equation_builder import EquationNeuron

    # Define a custom neuron with string equations
    neuron = EquationNeuron(
        equations={
            "v": "-(v - v_rest) / tau + R * I",
            "w": "epsilon * (v + a - b * w)",
        },
        parameters={"v_rest": -65.0, "tau": 20.0, "R": 1.0, "epsilon": 0.08, "a": 0.7, "b": 0.8},
        state={"v": -65.0, "w": 0.0},
        threshold="v > v_threshold",
        reset={"v": "v_reset"},
        constants={"v_threshold": -50.0, "v_reset": -65.0},
        dt=0.1,
        method="euler",
    )

    for t in range(10000):
        spike = neuron.step(I=10.0)

    # Or use the factory for common patterns
    from sc_neurocore.neurons.equation_builder import from_equations

    lif = from_equations(
        "dv/dt = -(v - E_L)/tau_m + I/C",
        threshold="v > -50",
        reset="v = -65",
        params=dict(E_L=-65, tau_m=10, C=1),
        init=dict(v=-65),
    )
"""

from __future__ import annotations

import ast
import math
import re
from copy import deepcopy
from typing import Any

import numpy as np


class EquationNeuron:
    """Neuron defined by arbitrary ODE equations as strings.

    Each equation is a right-hand-side expression for dX/dt.
    Variables can reference other state variables, parameters,
    and the special variable `I` (input current).
    """

    def __init__(
        self,
        equations: dict[str, str],
        parameters: dict[str, float] | None = None,
        state: dict[str, float] | None = None,
        threshold: str | None = None,
        reset: dict[str, str] | None = None,
        constants: dict[str, float] | None = None,
        dt: float = 0.1,
        method: str = "euler",
    ):
        self.equations = equations
        self.parameters = parameters or {}
        self.state = state or {k: 0.0 for k in equations}
        self.initial_state = deepcopy(self.state)
        self.threshold_expr = threshold
        self.reset_rules = reset or {}
        self.constants = constants or {}
        self.dt = dt
        self.method = method

        def _sigmoid(x: float) -> Any:
            return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

        self._namespace = {
            "exp": np.exp,
            "log": np.log,
            "sqrt": np.sqrt,
            "abs": abs,
            "sin": np.sin,
            "cos": np.cos,
            "tanh": np.tanh,
            "cosh": np.cosh,
            "sinh": np.sinh,
            "sigmoid": _sigmoid,
            "pi": math.pi,
            "clip": np.clip,
            "max": max,
            "min": min,
        }
        self._noise_scale = np.sqrt(self.dt)

        all_exprs = list(self.equations.values()) + list(self.reset_rules.values())
        if self.threshold_expr:
            all_exprs.append(self.threshold_expr)
        for expr in all_exprs:
            self._validate_expr(expr)

        self._compiled_eqs = {
            var: compile(expr, f"<eq:{var}>", "eval") for var, expr in self.equations.items()
        }
        self._compiled_threshold = (
            compile(self.threshold_expr, "<threshold>", "eval") if self.threshold_expr else None
        )
        self._compiled_reset = {
            var: compile(expr, f"<reset:{var}>", "eval") for var, expr in self.reset_rules.items()
        }

    _ALLOWED_AST_NODES = {
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Compare,
        ast.BoolOp,
        ast.IfExp,
        ast.Call,
        ast.Name,
        ast.Constant,
        ast.Attribute,
        ast.Subscript,
        ast.Index,
        ast.Slice,
        ast.Load,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.Mod,
        ast.FloorDiv,
        ast.USub,
        ast.UAdd,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.And,
        ast.Or,
        ast.Not,
        ast.Tuple,
        ast.List,
    }

    _BLOCKED_NAMES = {
        "__import__",
        "eval",
        "exec",
        "compile",
        "globals",
        "locals",
        "getattr",
        "setattr",
        "delattr",
        "open",
        "input",
        "breakpoint",
        "__builtins__",
        "__class__",
        "__subclasses__",
    }

    def _validate_expr(self, expr: str) -> None:
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError as e:
            raise ValueError(f"Invalid equation syntax: {expr!r}") from e
        for node in ast.walk(tree):
            if type(node) not in self._ALLOWED_AST_NODES:
                raise ValueError(f"Unsafe AST node {type(node).__name__} in equation: {expr!r}")
            if isinstance(node, ast.Name) and node.id in self._BLOCKED_NAMES:
                raise ValueError(f"Blocked function {node.id!r} in equation: {expr!r}")
            if isinstance(node, ast.Attribute) and node.attr in self._BLOCKED_NAMES:
                raise ValueError(f"Blocked attribute {node.attr!r} in equation: {expr!r}")

    def _build_env(self, **kwargs: float) -> dict[str, object]:
        env: dict[str, object] = dict(self._namespace)
        # Euler-Maruyama: noise scaled by sqrt(dt)/dt so that after deriv*dt
        # the net noise is noise_scale * sqrt(dt) * N(0,1)
        env["xi"] = self._noise_scale * np.random.randn() / max(self.dt, 1e-12) ** 0.5
        env.update(self.parameters)
        env.update(self.constants)
        env.update(self.state)
        env.update(kwargs)
        return env

    def step(self, I: float = 0.0, **kwargs: float) -> int:
        kwargs["I"] = I
        env = self._build_env(**kwargs)

        if self.method == "euler":
            derivatives = {}
            for var, code in self._compiled_eqs.items():
                derivatives[var] = float(eval(code, {"__builtins__": {}}, env))
            for var in self.equations:
                self.state[var] += derivatives[var] * self.dt

        elif self.method == "rk4":
            s0 = deepcopy(self.state)

            xi_sample = self._noise_scale * np.random.randn() / max(self.dt, 1e-12) ** 0.5

            def eval_derivs(state_override: dict[str, float]) -> dict[str, float]:
                e: dict[str, object] = dict(self._namespace)
                e.update(self.parameters)
                e.update(self.constants)
                e.update(state_override)
                e.update(kwargs)
                e["xi"] = xi_sample
                return {
                    var: float(eval(code, {"__builtins__": {}}, e))
                    for var, code in self._compiled_eqs.items()
                }

            k1 = eval_derivs(s0)
            s1 = {v: s0[v] + k1[v] * self.dt / 2 for v in self.equations}
            k2 = eval_derivs(s1)
            s2 = {v: s0[v] + k2[v] * self.dt / 2 for v in self.equations}
            k3 = eval_derivs(s2)
            s3 = {v: s0[v] + k3[v] * self.dt for v in self.equations}
            k4 = eval_derivs(s3)
            for v in self.equations:
                self.state[v] = s0[v] + (k1[v] + 2 * k2[v] + 2 * k3[v] + k4[v]) * self.dt / 6

        spike = 0
        if self._compiled_threshold:
            env_post = self._build_env(**kwargs)
            if eval(self._compiled_threshold, {"__builtins__": {}}, env_post):
                spike = 1
                reset_env = self._build_env(**kwargs)
                for var, code in self._compiled_reset.items():
                    self.state[var] = float(eval(code, {"__builtins__": {}}, reset_env))

        return spike

    def get_state(self) -> dict[str, float]:
        return dict(self.state)

    def reset(self) -> None:
        self.state = deepcopy(self.initial_state)

    def __repr__(self) -> str:
        eqs = ", ".join(f"d{k}/dt = {v}" for k, v in self.equations.items())
        return f"EquationNeuron({eqs})"


def from_equations(
    *equation_strings: str,
    threshold: str | None = None,
    reset: str | None = None,
    params: dict[str, float] | None = None,
    init: dict[str, float] | None = None,
    dt: float = 0.1,
    method: str = "euler",
) -> EquationNeuron:
    """Factory: build EquationNeuron from Brian2-style equation strings.

    Example:
        lif = from_equations(
            "dv/dt = -(v - E_L)/tau_m + I/C",
            threshold="v > -50",
            reset="v = -65",
            params=dict(E_L=-65, tau_m=10, C=1),
            init=dict(v=-65),
        )
    """
    equations = {}
    for eq_str in equation_strings:
        eq_str = eq_str.strip()
        m = re.match(r"d(\w+)/dt\s*=\s*(.+)", eq_str)
        if m:
            var_name = m.group(1)
            rhs = m.group(2).strip()
            equations[var_name] = rhs
        else:
            raise ValueError(f"Cannot parse equation: {eq_str!r}. Expected 'd<var>/dt = <expr>'")

    reset_rules = {}
    constants = {}
    if reset:
        for part in reset.split(";"):
            part = part.strip()
            if not part:
                continue
            m = re.match(r"(\w+)\s*=\s*(.+)", part)
            if m:
                var = m.group(1)
                val_str = m.group(2).strip()
                try:
                    constants[f"{var}_reset_val"] = float(val_str)
                    reset_rules[var] = f"{var}_reset_val"
                except ValueError:
                    reset_rules[var] = val_str

    threshold_expr = None
    if threshold:
        threshold = threshold.strip()
        threshold_expr = threshold

    state = init or {k: 0.0 for k in equations}

    return EquationNeuron(
        equations=equations,
        parameters=params or {},
        state=state,
        threshold=threshold_expr,
        reset=reset_rules,
        constants=constants,
        dt=dt,
        method=method,
    )
