# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAllPrecisionPairs from former test_adaptive_runtime_precision.py

"""Focused suite: TestAllPrecisionPairs from former test_adaptive_runtime_precision.py."""

from __future__ import annotations

from tests.adaptive_runtime_precision_support import *  # noqa: F403


class TestAllPrecisionPairs:
    """Verify all canonical LP/HP pairs from PRECISION_PAIRS."""

    @pytest.mark.parametrize(
        "lp_hp",
        PRECISION_PAIRS,
        ids=[
            f"Q{lp[0] - lp[1] - 1}.{lp[1]}_to_Q{hp[0] - hp[1] - 1}.{hp[1]}"
            for lp, hp in PRECISION_PAIRS
        ],
    )
    def test_canonical_pair_compiles(self, lif_neuron, lp_hp):
        """Each canonical LP/HP pair must compile without error."""
        (lp_w, lp_f), (hp_w, hp_f) = lp_hp
        v = compile_adaptive_precision(
            lif_neuron,
            module_name="sc_lif_test",
            lp_width=lp_w,
            lp_frac=lp_f,
            hp_width=hp_w,
            hp_frac=hp_f,
        )
        assert "module sc_lif_test_lp" in v
        assert "module sc_lif_test_hp" in v
        assert v.count("endmodule") == 3
