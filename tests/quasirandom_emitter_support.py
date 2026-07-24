# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_quasirandom_emitter.py

from __future__ import annotations

"""Test suite for QuasiRandomEmitter (Sobol + Halton backends)."""
import subprocess
import shutil
import pytest
from sc_neurocore.hdl_gen.quasirandom_emitter import (
    Halton16Emitter,
    QuasiRandomEmitter,
)
from sc_neurocore.hdl_gen.sobol16_emitter import Sobol16Emitter

__all__ = [
    "subprocess",
    "shutil",
    "pytest",
    "Halton16Emitter",
    "QuasiRandomEmitter",
    "Sobol16Emitter",
]
