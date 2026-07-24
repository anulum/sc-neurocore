# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_rust_integration.py

from __future__ import annotations

"""Integration tests exercising the Rust backend through the Python API.

Covers: NetworkRunner simulation, neuron step parity, IR compilation,
and spike data format. The Rust SIMD primitives are tested separately
in the 378 Rust-native tests (cargo test).
"""
import pytest

engine = pytest.importorskip("sc_neurocore_engine")

__all__ = ["pytest", "engine"]
