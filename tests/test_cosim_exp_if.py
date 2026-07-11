# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Exponential IF co-simulation contracts

"""Exponential IF compile-valid co-simulation classification."""

from __future__ import annotations

import pytest

from tests.cosim_support import HAS_IVERILOG, _verilog_compiles

_SCHEMA_GAP_COMPILE_ONLY = ["exp_if"]


@pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog not available")
class TestSchemaGapModelCosim:
    """WC-A5 Tier-A closure: every schema-gap model has an explicit cosim status.

    Exponential IF remains compile-only because its stiff exponential saturation
    does not have a source-faithful fixed-point parity contract at this precision.
    """

    @pytest.mark.parametrize("model_name", _SCHEMA_GAP_COMPILE_ONLY)
    def test_compile_valid_but_not_spike_parity(self, model_name: str) -> None:
        """Exponential IF lowers to iverilog-valid Verilog.

        Its stiff exponential saturation has no enrolled spike-count parity
        contract, so this asserts the fixed-point path rather than a behavioural
        result.
        """
        assert _verilog_compiles(model_name)
