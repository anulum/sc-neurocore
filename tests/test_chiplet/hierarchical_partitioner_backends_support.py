# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_hierarchical_partitioner_backends.py

from __future__ import annotations

"""Parity and performance contracts for maintained KL-refinement kernels."""
from collections.abc import Callable
import time
import pytest
from sc_neurocore.chiplet import HierarchicalPartitioner
from tests.test_chiplet.hierarchical_partitioner_support import build_graph as _build_graph

__all__ = ["Callable", "time", "pytest", "HierarchicalPartitioner", "_build_graph"]
