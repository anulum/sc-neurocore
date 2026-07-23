# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBittrueKernel from former test_intelligence_verification_and_safety.py

"""Focused suite: TestBittrueKernel from former test_intelligence_verification_and_safety.py."""

from __future__ import annotations

from tests.intelligence_verification_and_safety_support import *  # noqa: F403

class TestBittrueKernel:
    def test_c_kernel(self):
        from sc_neurocore.compiler.intelligence import (
            generate_bittrue_kernel,
        )

        code = generate_bittrue_kernel("sc_lif", {"v": "a + b"})
        assert "#include <stdint.h>" in code
        assert "sc_lif_state_t" in code
        assert "sat(" in code
        assert "fxmul(" in code

    def test_rust_kernel(self):
        from sc_neurocore.compiler.intelligence import (
            generate_bittrue_kernel,
        )

        code = generate_bittrue_kernel(
            "sc_lif",
            {"v": "a + b"},
            language="rust",
        )
        assert "pub struct" in code
        assert "fn sat" in code
        assert "clamp" in code

    def test_multi_var(self):
        from sc_neurocore.compiler.intelligence import (
            generate_bittrue_kernel,
        )

        code = generate_bittrue_kernel(
            "sc_izh",
            {"v": "a * b", "u": "c + d"},
        )
        assert "int16_t v;" in code
        assert "int16_t u;" in code
