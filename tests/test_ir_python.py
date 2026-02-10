"""Tests for the IR Python bridge."""

from __future__ import annotations

import pytest

from sc_neurocore_engine.ir import ScGraphBuilder, ScGraph, parse_ir


class TestIRPythonBridge:
    """Verify IR construction, verification, and emission from Python."""

    def test_empty_graph(self):
        b = ScGraphBuilder("empty")
        g = b.build()
        assert len(g) == 0
        assert g.name == "empty"
        assert g.verify() is None

    def test_synapse_pipeline(self):
        b = ScGraphBuilder("synapse")
        x = b.input("x", "rate")
        w = b.input("w", "rate")
        x_enc = b.encode(x, length=1024, seed=0xACE1)
        w_enc = b.encode(w, length=1024, seed=0xBEEF)
        product = b.bitwise_and(x_enc, w_enc)
        count = b.popcount(product)
        rate = b.div_const(count, 1024)
        b.output("rate_out", rate)
        g = b.build()

        assert g.num_inputs == 2
        assert g.num_outputs == 1
        assert g.verify() is None

    def test_dense_layer_graph(self):
        b = ScGraphBuilder("dense")
        x = b.input("x", "rate")
        w = b.input("w", "rate")
        leak = b.input("leak", "i16")
        gain = b.input("gain", "i16")
        spikes = b.dense_forward(x, w, leak, gain, n_inputs=3, n_neurons=7)
        b.output("spikes", spikes)
        g = b.build()

        assert g.num_inputs == 4
        assert g.num_outputs == 1
        assert g.verify() is None

    def test_lif_step_graph(self):
        b = ScGraphBuilder("lif")
        current = b.input("current", "i16")
        leak = b.input("leak", "i16")
        gain = b.input("gain", "i16")
        noise = b.input("noise", "i16")
        spike = b.lif_step(current, leak, gain, noise)
        b.output("spike", spike)
        g = b.build()

        assert g.num_inputs == 4
        assert g.num_outputs == 1
        assert g.verify() is None

    def test_text_round_trip(self):
        b = ScGraphBuilder("roundtrip_test")
        x = b.input("x", "rate")
        enc = b.encode(x, length=512, seed=0xACE1)
        count = b.popcount(enc)
        b.output("count", count)
        g = b.build()

        text = g.to_text()
        parsed = parse_ir(text)
        assert parsed.to_text() == text

    def test_emit_sv_contains_module(self):
        b = ScGraphBuilder("sv_test")
        x = b.input("x", "rate")
        w = b.input("w", "rate")
        x_enc = b.encode(x, length=1024, seed=0xACE1)
        w_enc = b.encode(w, length=1024, seed=0xBEEF)
        product = b.bitwise_and(x_enc, w_enc)
        b.output("out", product)
        g = b.build()

        sv = g.emit_sv()
        assert "module" in sv
        assert "sv_test" in sv
        assert "endmodule" in sv
        assert "sc_bitstream_encoder" in sv
        assert "sc_bitstream_synapse" in sv

    def test_builder_consumed_after_build(self):
        b = ScGraphBuilder("consumed")
        b.input("x", "rate")
        b.build()
        with pytest.raises(Exception):
            b.input("y", "rate")

    def test_repr(self):
        b = ScGraphBuilder("repr_test")
        b.input("x", "rate")
        g = b.build()
        r = repr(g)
        assert "repr_test" in r

    def test_constant_f64(self):
        b = ScGraphBuilder("const_test")
        c = b.constant_f64(0.5, "rate")
        b.output("val", c)
        g = b.build()
        assert len(g) == 2  # constant + output
        assert g.verify() is None

    def test_scale_and_offset(self):
        b = ScGraphBuilder("arith_test")
        x = b.input("x", "rate")
        scaled = b.scale(x, 2.0)
        shifted = b.offset(scaled, 1.5)
        b.output("y", shifted)
        g = b.build()
        assert g.verify() is None
