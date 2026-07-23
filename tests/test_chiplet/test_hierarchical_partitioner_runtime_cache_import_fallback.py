# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestImportFallback from former test_hierarchical_partitioner_runtime_cache.py

"""Focused suite: TestImportFallback from former test_hierarchical_partitioner_runtime_cache.py."""

from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent))
from hierarchical_partitioner_runtime_cache_support import *  # noqa: F403

class TestImportFallback:
    """Cover the module-level `try: from sc_neurocore_engine import
    py_kl_refine ... except (ImportError, AttributeError)` branch by
    reloading the module with the engine masked out of sys.modules."""

    def test_engine_missing_sets_rust_kl_refine_none(self) -> None:
        import importlib
        import sys
        from sc_neurocore.chiplet import hierarchical_backend_runtime as hp_mod

        from tests.module_reload import restore_module_namespace, snapshot_module_namespace

        # Save current state for restoration.
        saved_engine = sys.modules.get("sc_neurocore_engine")
        saved_namespace = snapshot_module_namespace(hp_mod)
        try:
            # Mask the engine module so the next import fails.
            sys.modules["sc_neurocore_engine"] = None  # type: ignore[assignment]
            reloaded = importlib.reload(hp_mod)
            assert reloaded._HAS_RUST_KL_REFINE is False
            assert reloaded._rust_kl_refine is None
        finally:
            # Restore engine + the module's original class identities so subsequent tests
            # (and their by-value imports) see the real engine and canonical classes.
            if saved_engine is not None:
                sys.modules["sc_neurocore_engine"] = saved_engine
            else:
                sys.modules.pop("sc_neurocore_engine", None)
            restore_module_namespace(hp_mod, saved_namespace)
