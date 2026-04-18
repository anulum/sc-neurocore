# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — sc_neurocore.hardware -- Tier: core (production)

"""sc_neurocore.hardware — Neuromorphic Hardware Abstraction Layer.

Provides device specifications, resource estimation, constraint
checking, neuron-to-core mapping, and deployment packaging for
Loihi, SpiNNaker, BrainScaleS, FPGA, and Akida targets.
"""

__tier__ = "core"

from .device import DeviceFamily, DeviceSpec, DEVICE_CATALOG, get_device
from .resource_estimator import ResourceEstimate, ResourceEstimator
from .constraints import Violation, HardwareConstraints, ConstraintChecker
from .mapping import NeuronPlacement, Mapper
from .deployment import DeploymentPackage, Deployer

__all__ = [
    "DeviceFamily",
    "DeviceSpec",
    "DEVICE_CATALOG",
    "get_device",
    "ResourceEstimate",
    "ResourceEstimator",
    "Violation",
    "HardwareConstraints",
    "ConstraintChecker",
    "NeuronPlacement",
    "Mapper",
    "DeploymentPackage",
    "Deployer",
]
