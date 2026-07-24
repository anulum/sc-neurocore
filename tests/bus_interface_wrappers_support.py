# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_bus_interface_wrappers.py

from __future__ import annotations

"""Tests for historical bus-wrapper and register-map generation."""
from typing import cast
import pytest
from sc_neurocore.hdl_gen.bus_interface import (
    BusProtocol,
    generate_bus_wrapper,
    generate_register_map,
)

LIF_PARAMS = {"P_V_REST": 16, "P_V_THRESH": 16, "P_TAU_M": 16}

__all__ = [
    "cast",
    "pytest",
    "BusProtocol",
    "generate_bus_wrapper",
    "generate_register_map",
    "LIF_PARAMS",
]
