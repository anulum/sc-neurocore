# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""
Acceleration backends for SC-NeuroCore bitstream operations.

Provides:
- ``gpu_backend``: CuPy/NumPy dual-path array module
- ``vector_ops``: Packed uint64 bitstream operations
- ``jit_kernels``: Numba JIT-compiled hot loops
"""

from .vector_ops import pack_bitstream, unpack_bitstream, vec_and, vec_popcount
from .gpu_backend import xp, HAS_CUPY, to_device, to_host

__all__ = [
    "pack_bitstream",
    "unpack_bitstream",
    "vec_and",
    "vec_popcount",
    "xp",
    "HAS_CUPY",
    "to_device",
    "to_host",
]
