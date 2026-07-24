# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_compiler_export.py

from __future__ import annotations

"""Tests for the SSA compiler export module."""
from dataclasses import dataclass
import unittest
from sc_neurocore.export.compiler_export import CompilerExporter, SSAEnvironment, ShapeInference

ShapeMap = dict[str, tuple[int, ...]]


@dataclass(frozen=True)
class MockNode:
    """Typed SC-IR node fixture for compiler export tests."""

    type: str
    id: str
    inputs: tuple[str, ...]
    output: str
    threshold: float = 1.0
    leak: float = 0.9


@dataclass(frozen=True)
class MockGraph:
    """Typed SC-IR graph fixture exposing the public ``nodes`` contract."""

    nodes: tuple[MockNode, ...]


if __name__ == "__main__":
    unittest.main()

__all__ = [
    "dataclass",
    "unittest",
    "CompilerExporter",
    "SSAEnvironment",
    "ShapeInference",
    "ShapeMap",
    "MockNode",
    "MockGraph",
]
