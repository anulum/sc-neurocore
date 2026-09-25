# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Datasets Package Init

"""Expose event-dataset loaders, manifests, group splits and declared encoders."""

from .encoders import (
    EventBinning,
    FirstSpikeLatency,
    PoissonRates,
    encoder_from_declaration,
)
from .encoding import latency_encode, poisson_encode
from .loaders import load_dvs_cifar10, load_nmnist, load_shd
from .manifest import (
    EventDatasetManifest,
    build_manifest,
    manifest_from_dict,
    verify_manifest,
)
from .splits import SplitPlan, group_overlap, group_split, leaked_groups, split_plan_from_dict

__all__ = [
    "load_nmnist",
    "load_shd",
    "load_dvs_cifar10",
    "poisson_encode",
    "latency_encode",
    "EventBinning",
    "PoissonRates",
    "FirstSpikeLatency",
    "encoder_from_declaration",
    "EventDatasetManifest",
    "build_manifest",
    "verify_manifest",
    "manifest_from_dict",
    "SplitPlan",
    "group_split",
    "group_overlap",
    "leaked_groups",
    "split_plan_from_dict",
]
