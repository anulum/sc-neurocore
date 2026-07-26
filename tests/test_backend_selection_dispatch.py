# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Focused backend selection contracts

"""Focused data-driven backend selection contracts."""

from .backend_selection_support import *


def test_auto_dispatch_matches_python_floor_after_reorder() -> None:
    # Reordering must not change results: the selector-driven "auto" path is
    # bit-identical to the always-available Python floor.
    from sc_neurocore.scpn.dcls_tent_kernel import dcls_max_forward_batch

    spikes = [1, 1, 1, 0, 1, 0]
    weights = [256, 128, -64, 512, -256, 64]
    centres = [256, 512]
    sigmas = [512, 768]
    n_taps = 3
    auto = dcls_max_forward_batch(spikes, weights, centres, sigmas, n_taps, backend="auto")
    floor = dcls_max_forward_batch(spikes, weights, centres, sigmas, n_taps, backend="python")
    npt.assert_array_equal(auto.outputs_q88, floor.outputs_q88)
    npt.assert_array_equal(auto.accumulators_q16_16, floor.accumulators_q16_16)
