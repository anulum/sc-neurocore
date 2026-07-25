# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCPN L7 symbolic test construction

from __future__ import annotations

from typing import Any

from sc_neurocore.scpn.layers.l7_symbolic import L7_StochasticParameters, L7_SymbolicLayer


def make_layer(**overrides: Any) -> L7_SymbolicLayer:
    """Construct a compact deterministic L7 symbolic layer for contract tests."""
    params = dict(n_symbols=16, n_meridians=4, n_acupoints=16, bitstream_length=16, rng_seed=5)
    params.update(overrides)
    return L7_SymbolicLayer(L7_StochasticParameters(**params))
