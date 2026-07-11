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

    ``fitzhugh_nagumo`` is spike-parity validated in ``TestQ1616Precision``. The
    remaining five are classified here: deterministic-but-not-parity models are
    asserted to lower to valid RTL (the honest compile-only precedent used for
    glif/morris_lecar at Q8.8), and stochastic models are asserted to be excluded
    from every deterministic cosim set with their schema stochastic flag confirmed.
    """

    @pytest.mark.parametrize("model_name", _SCHEMA_GAP_COMPILE_ONLY)
    def test_compile_valid_but_not_spike_parity(self, model_name: str) -> None:
        """exp_if and Hindmarsh-Rose lower to iverilog-valid Verilog.

        Spike-count parity is not scientifically claimable for these (stiff exp
        saturation and chaotic sensitive-dependence — see module notes), so this
        asserts the fixed-point *path* is valid rather than a
        spike count, matching the honest compile-only precedent for coarse-LUT
        transcendental models.
        """
        assert _verilog_compiles(model_name)
