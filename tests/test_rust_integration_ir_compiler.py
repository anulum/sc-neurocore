# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestIRCompiler from former test_rust_integration.py

"""Focused suite: TestIRCompiler from former test_rust_integration.py."""

from __future__ import annotations

from tests.rust_integration_support import *  # noqa: F403

class TestIRCompiler:
    def test_build_verify_emit(self) -> None:
        b = engine.ScGraphBuilder("test_lif")
        i_in = b.input("current", "bool")
        leak = b.constant_i64(200, "i16")
        gain = b.constant_i64(256, "i16")
        noise = b.constant_i64(0, "i16")
        v_lif = b.lif_step(i_in, leak, gain, noise)
        b.output("spike", v_lif)
        graph = b.build()
        assert graph.verify() is None
        sv = graph.emit_sv()
        assert "module" in sv

    def test_ir_print(self) -> None:
        b = engine.ScGraphBuilder("print_test")
        v = b.input("x", "bool")
        b.output("y", v)
        graph = b.build()
        text = graph.to_text()
        assert "print_test" in text
