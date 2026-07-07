# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Datasets Package Init

"""Expose event-dataset loaders and spike-encoding helpers."""

from .encoding import latency_encode, poisson_encode
from .loaders import load_dvs_cifar10, load_nmnist, load_shd

__all__ = [
    "load_nmnist",
    "load_shd",
    "load_dvs_cifar10",
    "poisson_encode",
    "latency_encode",
]
