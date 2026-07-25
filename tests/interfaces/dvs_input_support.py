# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — DVS input test support

"""Shared imports and performance gate for DVS input tests."""

import os
import time
from typing import Any

import numpy as np
import pytest

from sc_neurocore.interfaces.dvs_input import DVSInputLayer

__all__ = ["Any", "DVSInputLayer", "_perf_enabled", "np", "pytest", "time"]


def _perf_enabled() -> bool:
    return os.environ.get("SC_NEUROCORE_PERF") == "1"
