# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_datasets.py

from __future__ import annotations

from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from sc_neurocore.datasets.loaders import (
    _check_root,
    load_nmnist,
    load_shd,
    load_dvs_cifar10,
)
from sc_neurocore.datasets.encoding import poisson_encode, latency_encode

__all__ = ['MagicMock', 'patch', 'np', 'pytest', '_check_root', 'load_nmnist', 'load_shd', 'load_dvs_cifar10', 'poisson_encode', 'latency_encode']
