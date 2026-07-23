# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDunderChainEscapes from former test_equation_builder_adversarial.py

"""Focused suite: TestDunderChainEscapes from former test_equation_builder_adversarial.py."""

from __future__ import annotations

from tests.equation_builder_adversarial_support import *  # noqa: F403

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
