# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNASVerilogEmitter from former test_sc_nas_engine.py

"""Focused suite: TestNASVerilogEmitter from former test_sc_nas_engine.py."""

from __future__ import annotations

from sc_nas_engine_support import *  # noqa: F403

class TestNASVerilogEmitter:
    def _make_candidate(self) -> SCCandidate:
        c = SCCandidate(
            layers=[
                LayerConfig(32, NeuronType.LIF, 256, DecorrelationStrategy.LFSR),
                LayerConfig(64, NeuronType.IZHIKEVICH, 512, DecorrelationStrategy.SOBOL),
            ],
            accuracy=0.95,
        )
        c.evaluate_resources()
        return c

    def test_emit_contains_module(self) -> None:
        c = self._make_candidate()
        v = NASVerilogEmitter.emit(c)
        assert "module sc_nas_network" in v
        assert "endmodule" in v

    def test_emit_has_parameters(self) -> None:
        c = self._make_candidate()
        v = NASVerilogEmitter.emit(c)
        assert "L0_NEURONS    = 32" in v
        assert "L1_NEURONS    = 64" in v
        assert "L0_BITSTREAM  = 256" in v

    def test_emit_has_neuron_modules(self) -> None:
        c = self._make_candidate()
        v = NASVerilogEmitter.emit(c)
        assert "sc_lif_neuron" in v
        assert "sc_izhikevich_neuron" in v

    def test_emit_has_resource_comment(self) -> None:
        c = self._make_candidate()
        v = NASVerilogEmitter.emit(c)
        assert "LUTs" in v
        assert "DSPs" in v
        assert "BRAM" in v

    def test_emit_custom_name(self) -> None:
        c = self._make_candidate()
        v = NASVerilogEmitter.emit(c, module_name="my_net")
        assert "module my_net" in v

    def test_emit_pareto(self) -> None:
        c1 = self._make_candidate()
        c2 = self._make_candidate()
        result = NASVerilogEmitter.emit_pareto([c1, c2])
        assert len(result) == 2
        assert "sc_nas_pareto_0" in result
        assert "sc_nas_pareto_1" in result

    def test_emit_all_neuron_types(self) -> None:
        for nt in NeuronType:
            c = SCCandidate(
                layers=[
                    LayerConfig(16, nt, 128, DecorrelationStrategy.LFSR),
                ],
                accuracy=0.8,
            )
            c.evaluate_resources()
            v = NASVerilogEmitter.emit(c)
            assert "module" in v
            assert "endmodule" in v
