# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDirectCodeExecution from former test_equation_builder_adversarial.py

"""Focused suite: TestDirectCodeExecution from former test_equation_builder_adversarial.py."""

from __future__ import annotations

from tests.equation_builder_adversarial_support import *  # noqa: F403


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
