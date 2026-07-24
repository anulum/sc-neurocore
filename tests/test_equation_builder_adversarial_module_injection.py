# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestModuleInjection from former test_equation_builder_adversarial.py

"""Focused suite: TestModuleInjection from former test_equation_builder_adversarial.py."""

from __future__ import annotations

from tests.equation_builder_adversarial_support import *  # noqa: F403


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
