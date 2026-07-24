# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPublicAPIExports from former test_verilog_integration.py

"""Focused suite: TestPublicAPIExports from former test_verilog_integration.py."""

from __future__ import annotations

from tests.verilog_integration_support import *  # noqa: F403


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
