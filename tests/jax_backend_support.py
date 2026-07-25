# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — JAX backend availability and exports

"""Require JAX once and expose the backend functions to focused suites."""

import pytest

pytest.importorskip("jax")

from sc_neurocore.accel.jax_backend import (
    jax_forward_pass,
    jax_lif_step,
    jax_pack_bitstream,
    jax_popcount,
    jax_vec_and,
    jax_vec_mac,
    to_host,
)

__all__ = [
    "jax_forward_pass",
    "jax_lif_step",
    "jax_pack_bitstream",
    "jax_popcount",
    "jax_vec_and",
    "jax_vec_mac",
    "to_host",
]
