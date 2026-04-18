# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Spike encoding zoo + automatic optimizer

"""7 spike encoding schemes + automatic per-layer encoding selection."""

from .encoders import (
    rate_encode,
    latency_encode,
    delta_encode,
    phase_encode,
    burst_encode,
    rank_order_encode,
    sigma_delta_encode,
)
from .optimizer import EncodingOptimizer, EncodingRecommendation

__all__ = [
    "rate_encode",
    "latency_encode",
    "delta_encode",
    "phase_encode",
    "burst_encode",
    "rank_order_encode",
    "sigma_delta_encode",
    "EncodingOptimizer",
    "EncodingRecommendation",
]
