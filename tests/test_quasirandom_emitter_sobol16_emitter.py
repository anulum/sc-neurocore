# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSobol16Emitter from former test_quasirandom_emitter.py

"""Focused suite: TestSobol16Emitter from former test_quasirandom_emitter.py."""

from __future__ import annotations

from tests.quasirandom_emitter_support import *  # noqa: F403

class TestSobol16Emitter:
    """Test Sobol-16 RTL generation."""

    def test_generates_valid_verilog(self) -> None:
        emitter = Sobol16Emitter()
        code = emitter.generate()
        assert "module sc_sobol16_source" in code
        assert "endmodule" in code
        assert "casez" in code

    def test_custom_module_name(self) -> None:
        emitter = Sobol16Emitter(module_name="my_sobol")
        code = emitter.generate()
        assert "module my_sobol" in code

    def test_seed_affects_first_sample(self) -> None:
        e1 = Sobol16Emitter(seed=0)
        e2 = Sobol16Emitter(seed=42)
        assert e1.generate() != e2.generate()
