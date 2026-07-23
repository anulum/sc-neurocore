# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestJaxBackendFallback from former test_jax_adapter_fallback_contracts.py

"""Focused suite: TestJaxBackendFallback from former test_jax_adapter_fallback_contracts.py."""

from __future__ import annotations

from tests.jax_adapter_fallback_contracts_support import *  # noqa: F403

class TestJaxBackendFallback:
    def test_surrogate_paths_declared_without_jax(self):
        from sc_neurocore.accel.jax_backend import JAX_SURROGATE_PATHS

        assert JAX_SURROGATE_PATHS == ("custom_vjp", "legacy_stop_gradient")

    def test_to_jax_no_jax(self):
        with patch("sc_neurocore.accel.jax_backend.HAS_JAX", False):
            from sc_neurocore.accel.jax_backend import to_jax

            arr = np.array([1, 2, 3])
            result = to_jax(arr)
            assert result is arr
