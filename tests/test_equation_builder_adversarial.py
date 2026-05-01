# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Adversarial test suite for equation builder AST sandbox

"""Adversarial test suite for the equation builder's AST sandbox.

Every test in this module provides a malicious equation string that MUST
be rejected by the sandbox.  If any test fails (i.e. the expression is
accepted), there is a sandbox escape vulnerability.

Categories covered:
- Direct code execution via builtins (__import__, eval, exec)
- Dunder attribute chain escapes (__class__.__bases__, __mro__, etc.)
- Module injection (os, sys, subprocess, importlib)
- AST depth exhaustion
- Obfuscated injection via string manipulation
"""

from __future__ import annotations

import pytest

from sc_neurocore.neurons.equation_builder import EquationNeuron, from_equations


class TestDirectCodeExecution:
    """Attempts to call dangerous builtins directly."""

    def test_import_os(self) -> None:
        with pytest.raises(ValueError, match="Blocked|Unsafe"):
            EquationNeuron(
                equations={"v": "__import__('os').system('id')"},
                state={"v": 0.0},
            )

    def test_eval_nested(self) -> None:
        with pytest.raises(ValueError, match="Blocked|Unsafe"):
            EquationNeuron(
                equations={"v": "eval('1+1')"},
                state={"v": 0.0},
            )

    def test_exec_call(self) -> None:
        with pytest.raises(ValueError, match="Blocked|Unsafe"):
            EquationNeuron(
                equations={"v": "exec('v=1')"},
                state={"v": 0.0},
            )

    def test_compile_call(self) -> None:
        with pytest.raises(ValueError, match="Blocked|Unsafe"):
            EquationNeuron(
                equations={"v": "compile('v=1','','exec')"},
                state={"v": 0.0},
            )

    def test_open_file(self) -> None:
        with pytest.raises(ValueError, match="Blocked|Unsafe"):
            EquationNeuron(
                equations={"v": "open('/etc/passwd').read()"},
                state={"v": 0.0},
            )

    def test_breakpoint_call(self) -> None:
        with pytest.raises(ValueError, match="Blocked|Unsafe"):
            EquationNeuron(
                equations={"v": "breakpoint()"},
                state={"v": 0.0},
            )


class TestDunderChainEscapes:
    """Attempts to escape the sandbox via dunder attribute chains."""

    def test_class_bases_subclasses(self) -> None:
        with pytest.raises(ValueError, match="Dunder|Blocked|Unsafe"):
            EquationNeuron(
                equations={"v": "().__class__.__bases__[0].__subclasses__()"},
                state={"v": 0.0},
            )

    def test_class_mro(self) -> None:
        with pytest.raises(ValueError, match="Dunder|Blocked|Unsafe"):
            EquationNeuron(
                equations={"v": "().__class__.__mro__[1]"},
                state={"v": 0.0},
            )

    def test_globals_access(self) -> None:
        with pytest.raises(ValueError, match="Dunder|Blocked|Unsafe"):
            EquationNeuron(
                equations={"v": "(lambda: 0).__globals__"},
                state={"v": 0.0},
            )

    def test_code_access(self) -> None:
        with pytest.raises(ValueError, match="Dunder|Blocked|Unsafe"):
            EquationNeuron(
                equations={"v": "(lambda: 0).__code__"},
                state={"v": 0.0},
            )

    def test_reduce_access(self) -> None:
        with pytest.raises(ValueError, match="Dunder|Blocked|Unsafe"):
            EquationNeuron(
                equations={"v": "().__reduce__()"},
                state={"v": 0.0},
            )

    def test_dict_access(self) -> None:
        with pytest.raises(ValueError, match="Dunder|Blocked|Unsafe"):
            EquationNeuron(
                equations={"v": "().__class__.__dict__"},
                state={"v": 0.0},
            )

    def test_init_subclass(self) -> None:
        with pytest.raises(ValueError, match="Dunder|Blocked|Unsafe"):
            EquationNeuron(
                equations={"v": "type.__init_subclass__()"},
                state={"v": 0.0},
            )

    def test_builtins_access(self) -> None:
        with pytest.raises(ValueError, match="Dunder|Blocked|Unsafe"):
            EquationNeuron(
                equations={"v": "__builtins__"},
                state={"v": 0.0},
            )


class TestModuleInjection:
    """Attempts to reference dangerous module names as identifiers."""

    @pytest.mark.parametrize(
        "module_name",
        ["os", "sys", "subprocess", "importlib", "shutil", "pathlib", "socket", "ctypes", "pickle"],
    )
    def test_blocked_module_as_name(self, module_name: str) -> None:
        with pytest.raises(ValueError, match="Blocked"):
            EquationNeuron(
                equations={"v": f"{module_name}"},
                state={"v": 0.0},
            )


class TestBuiltinIntrospection:
    """Attempts to use introspection builtins."""

    @pytest.mark.parametrize(
        "func_name",
        ["getattr", "setattr", "delattr", "globals", "locals", "vars", "dir", "type"],
    )
    def test_blocked_introspection_func(self, func_name: str) -> None:
        with pytest.raises(ValueError, match="Blocked"):
            EquationNeuron(
                equations={"v": f"{func_name}(v)"},
                state={"v": 0.0},
            )


class TestASTDepthExhaustion:
    """Attempts to exhaust the parser with deeply nested expressions."""

    def test_depth_limit_exceeded(self) -> None:
        # Build a deeply nested expression: (((((...(v)...)))))
        depth = 30
        expr = "v"
        for _ in range(depth):
            expr = f"({expr} + 1)"
        with pytest.raises(ValueError, match="AST depth"):
            EquationNeuron(
                equations={"v": expr},
                state={"v": 0.0},
            )

    def test_depth_just_under_limit_accepted(self) -> None:
        # A moderately nested expression should still work
        depth = 8
        expr = "v"
        for _ in range(depth):
            expr = f"({expr} + 1)"
        # Should NOT raise — this is a legitimate equation
        neuron = EquationNeuron(
            equations={"v": expr},
            state={"v": 0.0},
        )
        assert neuron is not None


class TestUnsafeASTNodes:
    """Attempts to use AST node types not in the whitelist."""

    def test_lambda(self) -> None:
        with pytest.raises(ValueError, match="Unsafe AST node"):
            EquationNeuron(
                equations={"v": "(lambda: 0)()"},
                state={"v": 0.0},
            )

    def test_generator_expression(self) -> None:
        with pytest.raises(ValueError, match="Unsafe AST node"):
            EquationNeuron(
                equations={"v": "sum(x for x in [1,2,3])"},
                state={"v": 0.0},
            )

    def test_dict_comprehension(self) -> None:
        with pytest.raises(ValueError, match="Unsafe AST node|Invalid equation"):
            EquationNeuron(
                equations={"v": "{k: v for k, v in [(1,2)]}"},
                state={"v": 0.0},
            )

    def test_starred(self) -> None:
        with pytest.raises(ValueError, match="Unsafe AST node|Invalid equation"):
            EquationNeuron(
                equations={"v": "*[1,2,3]"},
                state={"v": 0.0},
            )


class TestFromEquationsFactory:
    """Verify the from_equations() factory also rejects adversarial input."""

    def test_import_in_ode(self) -> None:
        with pytest.raises(ValueError, match="Blocked|Cannot parse"):
            from_equations(
                "dv/dt = __import__('os').system('id')",
                threshold="v > -50",
                reset="v = -65",
                init={"v": -65.0},
            )

    def test_dunder_in_threshold(self) -> None:
        with pytest.raises(ValueError, match="Dunder|Blocked"):
            from_equations(
                "dv/dt = -v / 10 + I",
                threshold="v.__class__.__mro__",
                init={"v": -65.0},
            )

    def test_dunder_in_reset(self) -> None:
        with pytest.raises(ValueError, match="Dunder|Blocked"):
            from_equations(
                "dv/dt = -v / 10 + I",
                threshold="v > -50",
                reset="v = __import__('os')",
                init={"v": -65.0},
            )


class TestLegitimateEquationsStillWork:
    """Ensure the hardening does not break legitimate neuron equations."""

    def test_lif_equation(self) -> None:
        neuron = from_equations(
            "dv/dt = -(v - E_L)/tau_m + I/C",
            threshold="v > -50",
            reset="v = -65",
            params={"E_L": -65.0, "tau_m": 10.0, "C": 1.0},
            init={"v": -65.0},
        )
        for _ in range(100):
            neuron.step(I=10.0)
        assert neuron.state["v"] != -65.0  # Should have integrated

    def test_fitzhugh_nagumo(self) -> None:
        neuron = EquationNeuron(
            equations={
                "v": "v - v**3 / 3 - w + I",
                "w": "0.08 * (v + 0.7 - 0.8 * w)",
            },
            state={"v": -1.0, "w": 0.0},
            dt=0.01,
        )
        for _ in range(1000):
            neuron.step(I=0.5)
        # Should have evolved from initial state
        assert neuron.state["v"] != -1.0

    def test_transcendental_functions(self) -> None:
        neuron = EquationNeuron(
            equations={"v": "-v + exp(-v) + tanh(v) + sin(v)"},
            state={"v": 1.0},
            dt=0.01,
        )
        for _ in range(100):
            neuron.step()
        assert neuron.state["v"] != 1.0

    def test_conditional_expression(self) -> None:
        neuron = EquationNeuron(
            equations={"v": "v + 1 if v < 10 else -v"},
            state={"v": 0.0},
            dt=0.1,
        )
        for _ in range(50):
            neuron.step()
        # Should not crash
        assert isinstance(neuron.state["v"], float)
