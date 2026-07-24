# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFromEquationsFactory from former test_equation_builder_adversarial.py

"""Focused suite: TestFromEquationsFactory from former test_equation_builder_adversarial.py."""

from __future__ import annotations

from tests.equation_builder_adversarial_support import *  # noqa: F403


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
