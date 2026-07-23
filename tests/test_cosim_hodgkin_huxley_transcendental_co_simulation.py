# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTranscendentalCoSimulation from former test_cosim_hodgkin_huxley.py

"""Focused suite: TestTranscendentalCoSimulation from former test_cosim_hodgkin_huxley.py."""

from __future__ import annotations

from tests.cosim_hodgkin_huxley_support import *  # noqa: F403

@pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog not available")
class TestTranscendentalCoSimulation:
    """Auto model→RTL compile contracts for Hodgkin-Huxley."""

    @pytest.mark.parametrize("model_name", _TRANSCENDENTAL_COMPILE_MODELS)
    def test_transcendental_model_lowers_to_valid_verilog(self, model_name: str) -> None:
        """Non-baseline models lower to iverilog-valid Verilog without malformed literals.

        This is the emitter-fix verification: before the negative-LUT-literal,
        cosh, and empty-parameter fixes these models either raised
        "Unsupported function" or emitted malformed `W'sd-N` literals. The GLIF
        Q8.8 path is resolution-limited and the conductance-model look-up tables can
        be too coarse for the dedicated Q16.16 behavioural claims, so this assertion
        covers valid synthesisable RTL rather than spike parity.
        """
        verilog = UniversalNeuron.from_schema(model_name).to_verilog(module_name=f"sc_{model_name}")
        assert "'sd-" not in verilog  # no malformed negative literals
        assert _verilog_compiles(model_name)
