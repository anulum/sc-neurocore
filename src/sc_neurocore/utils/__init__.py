# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Utils Package Init

"""Expose shared utility helpers for stochastic-computing workflows."""

from .rng import RNG
from .bitstreams import (
    BitstreamEncoder,
    BitstreamAverager,
    generate_bernoulli_bitstream,
    generate_sobol_bitstream,
    bitstream_to_probability,
    value_to_unipolar_prob,
    unipolar_prob_to_value,
)
from .deprecation import deprecated
from .logging import configure_logging
from .profiling import estimate_memory
from .registry import registry

__all__ = [
    "RNG",
    "BitstreamEncoder",
    "BitstreamAverager",
    "generate_bernoulli_bitstream",
    "generate_sobol_bitstream",
    "bitstream_to_probability",
    "value_to_unipolar_prob",
    "unipolar_prob_to_value",
    "configure_logging",
    "deprecated",
    "estimate_memory",
    "registry",
]
