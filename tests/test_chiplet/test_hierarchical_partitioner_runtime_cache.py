# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hierarchical partitioner runtime cache tests

"""Cache and import-fallback contracts for backend discovery."""

from __future__ import annotations

import pytest


class TestProbeReturnsTrueOnSecondCall:
    """Each `_ensure_*_loaded` probe must short-circuit return True
    when called again — covers the `if X is not None: return True`
    branch. We force-load (first call) then assert (second call) so
    the test does not depend on the order of unrelated tests."""

    def test_julia_probe_second_call_short_circuits(self) -> None:
        from sc_neurocore.chiplet import hierarchical_backend_runtime as hp_mod

        try:
            import juliacall  # noqa: F401
        except ImportError:
            pytest.skip("juliacall not installed")
        if not hp_mod._ensure_julia_kl_refine_loaded():
            pytest.skip("Julia kl_refine.jl not loadable")
        # Now julia is loaded — second call must short-circuit.
        assert hp_mod._ensure_julia_kl_refine_loaded() is True

    def test_go_probe_second_call_short_circuits(self) -> None:
        from sc_neurocore.chiplet import hierarchical_backend_runtime as hp_mod

        if not hp_mod._ensure_go_kl_refine_loaded():
            pytest.skip("Go libpartition.so not built")
        assert hp_mod._ensure_go_kl_refine_loaded() is True

    def test_mojo_probe_second_call_short_circuits(self) -> None:
        from sc_neurocore.chiplet import hierarchical_backend_runtime as hp_mod

        if not hp_mod._ensure_mojo_kl_refine_loaded():
            pytest.skip("Mojo libpartition.so not built")
        assert hp_mod._ensure_mojo_kl_refine_loaded() is True


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
