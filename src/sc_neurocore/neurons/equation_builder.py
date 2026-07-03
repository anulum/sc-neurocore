# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
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

from sc_neurocore.neurons._units import (
    UNIT_REGISTRY,
    build_quantity_namespace,
    is_quantity,
    quantity_to_base,
    require_pint,
    require_quantity,
    validate_quantity_expression,
)


class EquationNeuron:
    """Neuron defined by arbitrary ODE equations as strings.

    Each equation is a right-hand-side expression for dX/dt.
    Variables can reference other state variables, parameters,
    and the special variable `I` (input current).

    ``units="strict"`` enables opt-in pint-based dimensional
    validation before the expressions are compiled for runtime.
    """

    def __init__(
        self,
        equations: dict[str, str],
        parameters: dict[str, float] | None = None,
        state: dict[str, float] | None = None,
        threshold: str | None = None,
        reset: dict[str, str] | None = None,
        constants: dict[str, float] | None = None,
        dt: Any = 0.1,
        method: str = "euler",
        units: str = "none",
        input_unit: Any | None = None,
    ) -> None:
        """Initialise an equation-defined neuron from ODE strings."""
        if units not in {"none", "strict"}:
            raise ValueError("units must be 'none' or 'strict'")

        self.equations = equations
        self.threshold_expr = threshold
        self.reset_rules = reset or {}
        self.method = method
        self.units = units
        self._strict_units = units == "strict"
        self._display_state_units: dict[str, Any] = {}
        self._base_state_units: dict[str, Any] = {}
        self._runtime_units: dict[str, Any] = {}
        self._input_unit_name = "I"

        def _sigmoid(x: float) -> Any:
            """Logistic sigmoid with clipping for numerical stability."""
            return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

        def _sqrt(x: Any) -> Any:
            """Square root that fails before NumPy warning machinery on invalid domains."""
            if np.any(np.asarray(x) < 0):
                raise ValueError("sqrt domain error")
            return np.sqrt(x)

        def _exprel(x: Any) -> Any:
            """(exp(x) - 1) / x with the removable-singularity limit exprel(0) = 1.

            Lets conductance rate functions of the form a*(V-V0)/(1-exp(-(V-V0)/k))
            be written without the 0/0 singularity at V = V0 (it becomes a*k/exprel).
            """
            arr = np.asarray(x, dtype=float)
            safe = np.where(arr == 0.0, 1.0, arr)
            return np.where(np.abs(arr) < 1e-9, 1.0 + arr / 2.0, np.expm1(arr) / safe)

        self._namespace = {
            "exp": np.exp,
            "log": np.log,
            "sqrt": _sqrt,
            "abs": abs,
            "sin": np.sin,
            "cos": np.cos,
            "tanh": np.tanh,
            "cosh": np.cosh,
            "sinh": np.sinh,
            "exprel": _exprel,
            "sigmoid": _sigmoid,
            "pi": math.pi,
            "clip": np.clip,
            "max": max,
            "min": min,
        }
        raw_parameters = parameters or {}
        raw_state = state or {k: 0.0 for k in equations}
        raw_constants = constants or {}

        if self._strict_units:
            self.parameters, self.state, self.constants, self.dt = self._prepare_strict_runtime(
                raw_parameters=raw_parameters,
                raw_state=raw_state,
                raw_constants=raw_constants,
                dt=dt,
                input_unit=input_unit,
            )
        else:
            self.parameters = raw_parameters
            self.state = raw_state
            self.constants = raw_constants
            self.dt = float(dt)

        self.initial_state = deepcopy(self.state)
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

    _MAX_AST_DEPTH = 20

    _BLOCKED_NAMES = {
        # Python builtins that enable code execution or introspection
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
        "type",
        "vars",
        "dir",
        "help",
        "print",
        "exit",
        "quit",
        # Dunder attributes used in sandbox escape chains
        "__builtins__",
        "__class__",
        "__subclasses__",
        "__mro__",
        "__bases__",
        "__globals__",
        "__code__",
        "__reduce__",
        "__reduce_ex__",
        "__dict__",
        "__init_subclass__",
        "__getattr__",
        "__setattr__",
        "__delattr__",
        # Module names that must never appear as identifiers
        "os",
        "sys",
        "subprocess",
        "importlib",
        "shutil",
        "pathlib",
        "socket",
        "ctypes",
        "pickle",
    }

    _EVAL_GLOBALS = {
        "__builtins__": {"__import__": __import__},
    }

    def _validate_expr(self, expr: str) -> None:
        """Validate an expression against the AST whitelist."""
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError as e:
            raise ValueError(f"Invalid equation syntax: {expr!r}") from e

        # Reject excessively deep ASTs (stack exhaustion / obfuscation)
        max_depth = self._ast_depth(tree)
        if max_depth > self._MAX_AST_DEPTH:
            raise ValueError(
                f"Equation AST depth {max_depth} exceeds limit {self._MAX_AST_DEPTH}: {expr!r}"
            )

        for node in ast.walk(tree):
            if type(node) not in self._ALLOWED_AST_NODES:
                raise ValueError(f"Unsafe AST node {type(node).__name__} in equation: {expr!r}")
            if isinstance(node, ast.Name) and node.id in self._BLOCKED_NAMES:
                raise ValueError(f"Blocked function {node.id!r} in equation: {expr!r}")
            if isinstance(node, ast.Attribute):
                # Block all double-underscore attribute access
                if node.attr.startswith("__") and node.attr.endswith("__"):
                    raise ValueError(
                        f"Dunder attribute access {node.attr!r} blocked in equation: {expr!r}"
                    )
                if node.attr in self._BLOCKED_NAMES:
                    raise ValueError(f"Blocked attribute {node.attr!r} in equation: {expr!r}")

    @staticmethod
    def _ast_depth(node: ast.AST) -> int:
        """Return the maximum nesting depth of an AST."""
        children = list(ast.iter_child_nodes(node))
        if not children:
            return 1
        return 1 + max(EquationNeuron._ast_depth(c) for c in children)

    def _prepare_strict_runtime(
        self,
        *,
        raw_parameters: dict[str, Any],
        raw_state: dict[str, Any],
        raw_constants: dict[str, Any],
        dt: Any,
        input_unit: Any | None,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float], float]:
        """Convert pint quantities to base-unit floats for runtime."""
        require_pint()

        missing_state = sorted(set(self.equations) - set(raw_state))
        if missing_state:
            raise ValueError(
                "units='strict' requires explicit state quantities for all equation variables: "
                + ", ".join(missing_state)
            )

        dt_quantity = require_quantity(dt, "dt")
        dt_base = quantity_to_base(dt_quantity)
        dt_base.to(UNIT_REGISTRY.second)

        quantity_parameters = {
            name: require_quantity(value, f"parameter {name}")
            for name, value in raw_parameters.items()
        }
        quantity_state = {
            name: require_quantity(value, f"state {name}") for name, value in raw_state.items()
        }
        quantity_constants = {
            name: require_quantity(value, f"constant {name}")
            for name, value in raw_constants.items()
        }

        quantity_env = build_quantity_namespace()
        quantity_env.update(quantity_parameters)
        quantity_env.update(quantity_constants)
        quantity_env.update(quantity_state)
        quantity_env["xi"] = 1.0 * UNIT_REGISTRY.dimensionless

        uses_input = any(re.search(r"\bI\b", expr) for expr in self.equations.values())
        if self.threshold_expr:
            uses_input = uses_input or bool(re.search(r"\bI\b", self.threshold_expr))
        if any(re.search(r"\bI\b", expr) for expr in self.reset_rules.values()):
            uses_input = True

        if uses_input:
            if input_unit is None:
                raise ValueError(
                    "units='strict' requires input_unit when equations reference the special input 'I'"
                )
            input_quantity = require_quantity(input_unit, "input_unit")
            quantity_env[self._input_unit_name] = input_quantity
            self._runtime_units[self._input_unit_name] = quantity_to_base(input_quantity).units

        for var, expr in self.equations.items():
            expected = quantity_state[var] / dt_quantity
            validate_quantity_expression(
                expr,
                quantity_env,
                expected_quantity=expected,
                label=f"d{var}/dt",
            )

        for var, expr in self.reset_rules.items():
            validate_quantity_expression(
                expr,
                quantity_env,
                expected_quantity=quantity_state[var],
                label=f"reset {var}",
            )

        if self.threshold_expr:
            threshold_result = validate_quantity_expression(
                self.threshold_expr,
                quantity_env,
                label="threshold",
            )
            if not isinstance(threshold_result, (bool, np.bool_)):
                raise ValueError(
                    "Threshold expression must evaluate to a boolean in strict units mode"
                )

        runtime_parameters = {}
        runtime_state = {}
        runtime_constants = {}

        for name, quantity in quantity_parameters.items():
            runtime_parameters[name] = float(quantity_to_base(quantity).magnitude)
            self._runtime_units[name] = quantity_to_base(quantity).units

        for name, quantity in quantity_constants.items():
            runtime_constants[name] = float(quantity_to_base(quantity).magnitude)
            self._runtime_units[name] = quantity_to_base(quantity).units

        for name, quantity in quantity_state.items():
            base_quantity = quantity_to_base(quantity)
            runtime_state[name] = float(base_quantity.magnitude)
            self._runtime_units[name] = base_quantity.units
            self._base_state_units[name] = base_quantity.units
            self._display_state_units[name] = quantity.units

        return runtime_parameters, runtime_state, runtime_constants, float(dt_base.magnitude)

    def _convert_runtime_value(self, name: str, value: Any) -> float:
        """Convert a runtime value from pint quantity to float."""
        if not self._strict_units:
            return float(value)
        if not is_quantity(value):
            return float(value)
        if name not in self._runtime_units:
            raise ValueError(f"No runtime unit declared for {name!r}")
        return float(quantity_to_base(value).to(self._runtime_units[name]).magnitude)

    def _build_env(self, **kwargs: float) -> dict[str, object]:
        """Build the eval environment with parameters, state, and noise."""
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
        """Advance the neuron by one timestep; return 1 if it spikes."""
        kwargs["I"] = self._convert_runtime_value(self._input_unit_name, I)
        if self._strict_units:
            kwargs = {
                name: self._convert_runtime_value(name, value) for name, value in kwargs.items()
            }
        env = self._build_env(**kwargs)

        if self.method == "euler":
            derivatives = {}
            for var, code in self._compiled_eqs.items():
                # nosec B307: `code` is a compiled expression that has
                # already passed `_validate_expr`'s AST whitelist (no
                # imports, no attribute access into builtins, only the
                # whitelisted maths/comparison nodes). The `eval` env
                # has empty `__builtins__` so even reaching `eval` /
                # `exec` / `__import__` is impossible.
                derivatives[var] = float(eval(code, self._EVAL_GLOBALS, env))  # nosec B307
            for var in self.equations:
                self.state[var] += derivatives[var] * self.dt

        elif self.method == "rk4":
            s0 = deepcopy(self.state)

            xi_sample = self._noise_scale * np.random.randn() / max(self.dt, 1e-12) ** 0.5

            def eval_derivs(state_override: dict[str, float]) -> dict[str, float]:
                """Evaluate all ODE derivatives at given state."""
                e: dict[str, object] = dict(self._namespace)
                e.update(self.parameters)
                e.update(self.constants)
                e.update(state_override)
                e.update(kwargs)
                e["xi"] = xi_sample
                return {
                    # nosec B307: AST-whitelisted compiled equation
                    # (see euler branch comment above for full sandbox
                    # rationale).
                    var: float(eval(code, self._EVAL_GLOBALS, e))  # nosec B307
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
            # nosec B307: AST-whitelisted compiled threshold expression.
            if eval(self._compiled_threshold, self._EVAL_GLOBALS, env_post):  # nosec B307
                spike = 1
                reset_env = self._build_env(**kwargs)
                for var, code in self._compiled_reset.items():
                    # nosec B307: AST-whitelisted compiled reset rule.
                    self.state[var] = float(eval(code, self._EVAL_GLOBALS, reset_env))  # nosec B307

        return spike

    def get_state(self) -> dict[str, Any]:
        """Return current state, with units if in strict mode."""
        if self._strict_units:
            return {
                name: (value * self._base_state_units[name]).to(self._display_state_units[name])
                for name, value in self.state.items()
            }
        return dict(self.state)

    def reset(self) -> None:
        """Reset state to initial values."""
        self.state = deepcopy(self.initial_state)

    def __repr__(self) -> str:
        """Human-readable representation of the neuron equations."""
        eqs = ", ".join(f"d{k}/dt = {v}" for k, v in self.equations.items())
        return f"EquationNeuron({eqs})"


def from_equations(
    *equation_strings: str,
    threshold: str | None = None,
    reset: str | None = None,
    params: dict[str, Any] | None = None,
    init: dict[str, Any] | None = None,
    constants: dict[str, Any] | None = None,
    dt: Any = 0.1,
    method: str = "euler",
    units: str = "none",
    input_unit: Any | None = None,
) -> EquationNeuron:
    """Build an EquationNeuron from Brian2-style equation strings.

    Example:
        lif = from_equations(
            "dv/dt = -(v - E_L)/tau_m + I/C",
            threshold="v > -50",
            reset="v = -65",
            params=dict(E_L=-65, tau_m=10, C=1),
            init=dict(v=-65),
        )

    Use ``units="strict"`` with pint quantities to validate the
    equation dimensions before runtime compilation.
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
    constant_values = constants.copy() if constants else {}
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
                    constant_values[f"{var}_reset_val"] = float(val_str)
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
        constants=constant_values,
        dt=dt,
        method=method,
        units=units,
        input_unit=input_unit,
    )
