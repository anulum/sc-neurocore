# SPDX-License-Identifier: AGPL-3.0-or-later
"""Compatibility helpers for v2-style imports."""

from .layers import VectorizedSCLayer
from .neurons import FixedPointLIFNeuron

__all__ = ["VectorizedSCLayer", "FixedPointLIFNeuron"]
