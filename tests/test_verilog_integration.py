# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for the VerilogGenerator integration points

"""Test new VerilogGenerator emit methods and Halton IR resolution."""

from __future__ import annotations


from sc_neurocore.hdl_gen.verilog_generator import (
    VerilogGenerator,
    emit_sources_from_ir,
)


class TestHaltonIRResolution:
    """Test that Halton source type is correctly resolved from IR."""

    def test_halton_source_from_ir(self) -> None:
        ir = {"nodes": [{"type": "halton16", "name": "my_halton"}]}
        code = emit_sources_from_ir(ir)
        assert "module my_halton" in code
        assert "reversed" in code  # Halton uses bit-reversal

    def test_halton_by_source_type(self) -> None:
        ir = {
            "nodes": [{"type": "stochastic_source", "source_type": "halton", "name": "halton_src"}]
        }
        code = emit_sources_from_ir(ir)
        assert "module halton_src" in code

    def test_mixed_sources(self) -> None:
        ir = {
            "nodes": [
                {"type": "lfsr16", "name": "lfsr_src"},
                {"type": "sobol16", "name": "sobol_src"},
                {"type": "halton16", "name": "halton_src"},
            ]
        }
        code = emit_sources_from_ir(ir)
        assert "module lfsr_src" in code
        assert "module sobol_src" in code
        assert "module halton_src" in code


class TestVerilogGeneratorNewMethods:
    """Test the new emit methods on VerilogGenerator."""

    def test_emit_halton16_source(self) -> None:
        gen = VerilogGenerator()
        code = gen.emit_halton16_source()
        assert "module sc_halton16_source" in code
        assert "reversed" in code

    def test_emit_quasirandom_source_sobol(self) -> None:
        gen = VerilogGenerator()
        code = gen.emit_quasirandom_source(method="sobol")
        assert "module sc_sobol16_source" in code
        assert "casez" in code

    def test_emit_quasirandom_source_halton(self) -> None:
        gen = VerilogGenerator()
        code = gen.emit_quasirandom_source(method="halton")
        assert "module sc_halton16_source" in code

    def test_emit_decorrelator(self) -> None:
        gen = VerilogGenerator()
        code = gen.emit_decorrelator(num_streams=4, stream_width=8)
        assert "sc_decorrelator" in code
        assert "NUM_STREAMS(4)" in code
        assert "STREAM_WIDTH(8)" in code

    def test_emit_edt_controller(self) -> None:
        gen = VerilogGenerator()
        code = gen.emit_edt_controller(data_width=16, margin=0x0080, stable_cycles=4)
        assert "sc_edt_controller" in code
        assert "DATA_WIDTH(16)" in code
        assert "MARGIN(16'h0080)" in code
        assert "STABLE_CYCLES(4)" in code

    def test_emit_tmr_wrapper(self) -> None:
        gen = VerilogGenerator()
        code = gen.emit_tmr_wrapper(
            module_name="sc_aer_router",
            inputs=[("clk", 1), ("rst_n", 1)],
            outputs=[("packet_out", 32)],
        )
        assert "module sc_aer_router_tmr" in code
        assert "replica_0" in code
        assert "replica_1" in code
        assert "replica_2" in code
        assert "packet_out_tmr_error" in code


class TestPublicAPIExports:
    """Test that new modules are accessible from public API."""

    def test_hdl_gen_exports(self) -> None:
        from sc_neurocore.hdl_gen import (
            QuasiRandomEmitter,
            Halton16Emitter,
            generate_tmr_wrapper,
        )

        assert QuasiRandomEmitter is not None
        assert Halton16Emitter is not None
        assert callable(generate_tmr_wrapper)

    def test_generate_tmr_wrapper_handles_multi_bit_input_and_single_bit_output(
        self,
    ) -> None:
        from sc_neurocore.hdl_gen.tmr_wrapper import generate_tmr_wrapper

        code = generate_tmr_wrapper(
            "sc_router",
            inputs=[("addr", 8)],
            outputs=[("done", 1)],
        )
        # A multi-bit input declares an explicit range; a single-bit output and
        # its triplicated replica wires stay scalar.
        assert "input  wire [7:0] addr" in code
        assert "output wire done" in code
        assert "wire rep0_done;" in code

    def test_neurons_exports(self) -> None:
        from sc_neurocore.neurons import UniversalNeuron, list_bundled_schemas

        assert UniversalNeuron is not None
        assert callable(list_bundled_schemas)
        assert len(list_bundled_schemas()) >= 9

    def test_universal_neuron_from_neurons_package(self) -> None:
        from sc_neurocore.neurons import UniversalNeuron

        neuron = UniversalNeuron.from_schema("lif")
        spike = neuron.step(I=30.0)
        assert isinstance(spike, int)
