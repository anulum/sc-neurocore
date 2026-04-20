# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for Cortical Column (Mojo Parity)

import numpy as np
import pytest

from sc_neurocore.network.cortical_column import (
    _HAS_MOJO_MULTI_SPMV,
    POPULATIONS,
    CorticalColumn,
)


@pytest.mark.skipif(
    not _HAS_MOJO_MULTI_SPMV,
    reason="Mojo kernel not loaded",
)
class TestPythonMojoParity:
    """The bitwise correctness of the Mojo batched SpMV against SciPy single-threaded."""

    @pytest.mark.parametrize("scale", [0.01])
    def test_spikes_identical(self, scale):
        col_py = CorticalColumn(
            scale=scale,
            seed=42,
            backend="python",
            use_block_csr=True,
            delay_distribution=True,
        )
        col_mojo = CorticalColumn(
            scale=scale,
            seed=42,
            backend="mojo",
            use_block_csr=True,
            delay_distribution=True,
        )

        for step_idx in range(50):
            spikes_py = col_py.step(0.1)
            spikes_mojo = col_mojo.step(0.1)
            for p in POPULATIONS:
                np.testing.assert_array_equal(
                    spikes_py[p],
                    spikes_mojo[p],
                    err_msg=f"Mismatch at step {step_idx} in pop {p}",
                )
