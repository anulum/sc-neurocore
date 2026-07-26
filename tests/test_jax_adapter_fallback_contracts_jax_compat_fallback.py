# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestJaxCompatFallback from former test_jax_adapter_fallback_contracts.py

"""Focused suite: TestJaxCompatFallback from former test_jax_adapter_fallback_contracts.py."""

from __future__ import annotations

from sc_neurocore.adapters.holonomic import _jax_compat
from tests.jax_adapter_fallback_contracts_support import *  # noqa: F403


class TestJaxCompatFallback:
    def test_make_rng_no_jax(self):
        with patch.object(_jax_compat, "HAS_JAX", False):
            key = _jax_compat.make_rng(42)
            assert hasattr(key, "shape")
            assert key[-1] == 42

    def test_split_rng_no_jax(self):
        with patch.object(_jax_compat, "HAS_JAX", False):
            key = _jax_compat.make_rng(42)
            k1, k2 = _jax_compat.split_rng(key)
            assert k1[-1] != k2[-1]

    def test_uniform_no_jax(self):
        with patch.object(_jax_compat, "HAS_JAX", False):
            key = _jax_compat.make_rng(42)
            result = _jax_compat.uniform(key, (3,))
            assert result.shape == (3,)

    def test_normal_no_jax(self):
        with patch.object(_jax_compat, "HAS_JAX", False):
            key = _jax_compat.make_rng(42)
            result = _jax_compat.normal(key, (3,))
            assert result.shape == (3,)

    def test_maybe_jit_no_jax(self):
        with patch.object(_jax_compat, "HAS_JAX", False):

            def _inc(x):
                return x + 1

            result = _jax_compat.maybe_jit(_inc)
            assert result is _inc
