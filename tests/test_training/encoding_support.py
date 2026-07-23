# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_encoding.py

from __future__ import annotations

"""Behavioural contracts for training spike encoders."""
import pytest
torch = pytest.importorskip("torch")
from sc_neurocore.training.encoding import delta_encode, latency_encode, rate_encode

__all__ = ['pytest', 'torch', 'delta_encode', 'latency_encode', 'rate_encode']
