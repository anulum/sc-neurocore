# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestImportFallback from former test_hierarchical_partitioner_runtime_cache.py

"""Focused suite: TestImportFallback from former test_hierarchical_partitioner_runtime_cache.py."""

from __future__ import annotations

import pytest


class TestImportFallback:
    """Import fallback when the engine is missing.

    Cover the module-level `try: from sc_neurocore_engine import
    py_kl_refine ... except (ImportError, AttributeError)` branch by
    reloading the module with the engine masked out of sys.modules.
    """

    def test_engine_missing_sets_rust_kl_refine_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With the engine masked, reloading the runtime clears its Rust refine entry."""
        import importlib
        import sys
        from sc_neurocore.chiplet import hierarchical_backend_runtime as hp_mod

        from tests.module_reload import restore_module_namespace, snapshot_module_namespace

        # Save the module's namespace for restoration; monkeypatch restores the engine.
        saved_namespace = snapshot_module_namespace(hp_mod)
        try:
            # Mask the engine module so the next import fails.
            monkeypatch.setitem(sys.modules, "sc_neurocore_engine", None)
            reloaded = importlib.reload(hp_mod)
            assert reloaded._HAS_RUST_KL_REFINE is False
            assert reloaded._rust_kl_refine is None
        finally:
            # Restore the module's original class identities so subsequent tests
            # (and their by-value imports) see the canonical classes.
            restore_module_namespace(hp_mod, saved_namespace)
