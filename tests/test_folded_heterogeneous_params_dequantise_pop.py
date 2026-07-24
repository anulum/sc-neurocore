# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDequantisePop from former test_folded_heterogeneous_params.py

"""Focused suite: TestDequantisePop from former test_folded_heterogeneous_params.py."""

from __future__ import annotations

from tests.folded_heterogeneous_params_support import *  # noqa: F403


class TestDequantisePop:
    """De-quantising a population's parameters for real-valued PE compilation."""

    def test_rescales_quantised_integers_back_to_real_values(self) -> None:
        pop = _qgraph([10.0, 20.0, 30.0]).populations[0]
        # pop.params['tau'] holds q.encode(10/20/30) = 2560/5120/7680; de-quantising divides
        # by 2**fraction, recovering 10/20/30 (which re-encode losslessly).
        rescaled = _dequantised_pop(pop, 8)
        np.testing.assert_allclose(
            np.asarray(rescaled.params["tau"]).reshape(-1), [10.0, 20.0, 30.0]
        )
        # Round-trips: encoding the rescaled value returns the original quantised integer.
        assert [_Q.encode(v) for v in [10.0, 20.0, 30.0]] == [2560, 5120, 7680]

    def test_empty_params_returns_population_unchanged(self) -> None:
        pop = _qgraph([10.0, 10.0]).populations[0]
        pop.params.clear()
        assert _dequantised_pop(pop, 8) is pop
