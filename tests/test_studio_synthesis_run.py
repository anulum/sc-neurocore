# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio synthesis run

"""Focused suite: TestRunSynthesis from former test_studio_synthesis.py."""

from __future__ import annotations

from typing import Any, cast

import pytest
from tests.studio_synthesis_support import (
    run_synthesis,
)


class TestRunSynthesis:
    """Running synthesis directly."""

    def test_unknown_target_raises(self) -> None:
        """An unknown target raises ValueError."""
        with pytest.raises(ValueError, match="Unknown target"):
            run_synthesis("module t(); endmodule", "nonexistent")

    def test_non_string_verilog_raises(self) -> None:
        """Non-string Verilog raises ValueError."""
        with pytest.raises(ValueError, match="verilog_source must be a string"):
            run_synthesis(cast(Any, 123), "ice40")

    def test_empty_verilog_raises(self) -> None:
        """Empty Verilog raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            run_synthesis("   ", "ice40")

    def test_returns_target_field(self) -> None:
        """A run reports its target."""
        result = run_synthesis("module t(); endmodule", "ice40")
        assert result["target"] == "ice40"
