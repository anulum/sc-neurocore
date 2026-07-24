# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_array_guards.py

from __future__ import annotations

"""Exhaustive tests for ``sc_neurocore._native.array_guards``.

The guard is a zero-copy gatekeeper between NumPy and the Rust/Mojo
native libraries. Every failure path is exercised (non-contiguous
view, unaligned buffer, wrong dtype, list input, tuple input,
multi-dim slices, empty arrays, Fortran order, read-only mmap).
Any regression here corrupts FFI calls silently, so the tests are
kept strict on exception type AND message fragments.
"""
import numpy as np
import pytest
from sc_neurocore._native.array_guards import require_c_contiguous

__all__ = ["np", "pytest", "require_c_contiguous"]
