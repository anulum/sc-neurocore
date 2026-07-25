# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — MPR accelerator dispatch test support

"""Shared imports and parameters for MPR dispatch tests."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from sc_neurocore.accel import ermentrout_kopell_pop as backends
from sc_neurocore.accel.go import ermentrout_kopell_pop as go_backend
from sc_neurocore.accel.mojo import ermentrout_kopell_pop as mojo_backend
from tests.module_reload import preserve_module_identity

_PARAMETERS = (0.13, -1.7, 1.3, 0.8, -4.2, 12.5, 0.004)

__all__ = [
    "Any",
    "SimpleNamespace",
    "_PARAMETERS",
    "backends",
    "go_backend",
    "importlib",
    "mojo_backend",
    "np",
    "preserve_module_identity",
    "pytest",
]
