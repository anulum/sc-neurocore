# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Stochastic-computing error-bound test sampling

"""Sampling support tied to the production unipolar encoder."""

from __future__ import annotations

import numpy as np

from sc_neurocore.encoding.encoders import rate_encode


def empirical_unipolar_variance(p: float, n: int, trials: int, seed: int) -> float:
    """Measure ``Var(k/N)`` through the production Bernoulli encoder path."""
    streams = rate_encode(np.full(trials, p, dtype=np.float64), T=n, seed=seed)
    estimates = streams.mean(axis=0)
    return float(np.var(estimates))
