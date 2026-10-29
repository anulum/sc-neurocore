# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Installed native IR compiler contracts

"""Exercise installed native graph verification and SystemVerilog emission."""

from __future__ import annotations

from tests.rust_integration_support import engine, pytest


class TestIRCompiler:
    """Check numeric neuron inputs and Boolean graph values at the public API."""

    def test_build_verify_emit(self) -> None:
        """Emit a LIF graph whose current is a signed raw Q8 word."""
        b = engine.ScGraphBuilder("test_lif")
        i_in = b.input("current", "i16")
        leak = b.constant_i64(200, "i16")
        gain = b.constant_i64(256, "i16")
        noise = b.constant_i64(0, "i16")
        v_lif = b.lif_step(i_in, leak, gain, noise)
        b.output("spike", v_lif)
        graph = b.build()
        assert graph.verify() is None
        sv = graph.emit_sv()
        assert "module test_lif" in sv
        assert "input wire [15:0] current" in sv
        assert "output wire [15:0] spike" in sv

    def test_boolean_lif_current_rejected(self) -> None:
        """Reject Boolean current during both verification and RTL emission."""
        b = engine.ScGraphBuilder("invalid_lif")
        current = b.input("current", "bool")
        leak = b.constant_i64(200, "i16")
        gain = b.constant_i64(256, "i16")
        noise = b.constant_i64(0, "i16")
        spike = b.lif_step(current, leak, gain, noise)
        b.output("spike", spike)
        graph = b.build()

        errors = graph.verify()
        assert errors == [
            "op[4]: %0 requires a signed 16-bit Q8 word or an explicit signed raw word, got bool"
        ]
        with pytest.raises(ValueError, match="signed 16-bit Q8 word.*got bool"):
            graph.emit_sv()

    def test_ir_print(self) -> None:
        """Keep Boolean passthrough graphs valid independently of neuron inputs."""
        b = engine.ScGraphBuilder("print_test")
        v = b.input("x", "bool")
        b.output("y", v)
        graph = b.build()
        assert graph.verify() is None
        text = graph.to_text()
        assert "print_test" in text
        assert "module print_test" in graph.emit_sv()
