# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hierarchical partitioner dispatch tests

"""Fail-closed dispatcher and direct-backend error contracts."""

from __future__ import annotations

import pytest

from sc_neurocore.chiplet import HierarchicalPartitioner
from tests.test_chiplet.hierarchical_partitioner_support import build_graph as _build_graph


class TestDispatcherMissingToolErrors:
    """The dispatcher must raise informative `RuntimeError` (with the
    exact build/install command) when a backend is requested but the
    underlying tool/.so is unavailable. These error paths are the
    user's only signal that a backend isn't wired."""

    def test_rust_missing_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from sc_neurocore.chiplet import hierarchical_backend_runtime as hp_mod

        monkeypatch.setattr(hp_mod, "_HAS_RUST_KL_REFINE", False)
        monkeypatch.setattr(hp_mod, "_rust_kl_refine", None)
        hp = HierarchicalPartitioner(num_partitions=2, refine_backend="rust")
        g = _build_graph(20, seed=1)
        with pytest.raises(RuntimeError, match="Rust KL refine requested"):
            hp.partition(g)

    def test_julia_missing_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from sc_neurocore.chiplet import hierarchical_backend_runtime as hp_mod

        monkeypatch.setattr(hp_mod, "_julia_kl_refine", None)
        monkeypatch.setattr(hp_mod, "_HAS_JULIA_KL_REFINE", False)
        monkeypatch.setattr(
            hp_mod,
            "_ensure_julia_kl_refine_loaded",
            lambda: False,
        )
        hp = HierarchicalPartitioner(num_partitions=2, refine_backend="julia")
        g = _build_graph(20, seed=1)
        with pytest.raises(RuntimeError, match="Julia KL refine requested"):
            hp.partition(g)

    def test_go_missing_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from sc_neurocore.chiplet import hierarchical_backend_runtime as hp_mod

        monkeypatch.setattr(hp_mod, "_go_kl_refine_lib", None)
        monkeypatch.setattr(hp_mod, "_HAS_GO_KL_REFINE", False)
        monkeypatch.setattr(
            hp_mod,
            "_ensure_go_kl_refine_loaded",
            lambda: False,
        )
        hp = HierarchicalPartitioner(num_partitions=2, refine_backend="go")
        g = _build_graph(20, seed=1)
        with pytest.raises(RuntimeError, match="Go KL refine requested"):
            hp.partition(g)

    def test_mojo_missing_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from sc_neurocore.chiplet import hierarchical_backend_runtime as hp_mod

        monkeypatch.setattr(hp_mod, "_mojo_kl_refine_lib", None)
        monkeypatch.setattr(hp_mod, "_HAS_MOJO_KL_REFINE", False)
        monkeypatch.setattr(
            hp_mod,
            "_ensure_mojo_kl_refine_loaded",
            lambda: False,
        )
        hp = HierarchicalPartitioner(num_partitions=2, refine_backend="mojo")
        g = _build_graph(20, seed=1)
        with pytest.raises(RuntimeError, match="Mojo KL refine requested"):
            hp.partition(g)

    def test_refine_rust_direct_call_without_backend_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The `_refine_rust` helper has its own `_rust_kl_refine is None`
        guard for callers that bypass the dispatcher."""
        from sc_neurocore.chiplet import hierarchical_backend_runtime as hp_mod

        monkeypatch.setattr(hp_mod, "_rust_kl_refine", None)
        hp = HierarchicalPartitioner(num_partitions=2, refine_backend="python")
        g = _build_graph(20, seed=1)
        with pytest.raises(RuntimeError, match="Rust KL refine backend"):
            hp._refine_rust([list(range(20))], g.adjacency(), g)

    def test_refine_julia_direct_call_without_backend_raises(self) -> None:
        from sc_neurocore.chiplet import hierarchical_backend_runtime as hp_mod

        # Save and clear the loaded handle to simulate "not loaded".
        saved = hp_mod._julia_kl_refine
        hp_mod._julia_kl_refine = None
        try:
            hp = HierarchicalPartitioner(num_partitions=2, refine_backend="python")
            g = _build_graph(20, seed=1)
            with pytest.raises(RuntimeError, match="Julia KL refine backend"):
                hp._refine_julia([list(range(20))], g.adjacency(), g)
        finally:
            hp_mod._julia_kl_refine = saved

    def test_refine_go_direct_call_without_lib_raises(self) -> None:
        from sc_neurocore.chiplet import hierarchical_backend_runtime as hp_mod

        saved = hp_mod._go_kl_refine_lib
        hp_mod._go_kl_refine_lib = None
        try:
            hp = HierarchicalPartitioner(num_partitions=2, refine_backend="python")
            g = _build_graph(20, seed=1)
            with pytest.raises(RuntimeError, match="Go KL refine"):
                hp._refine_go([list(range(20))], g.adjacency(), g)
        finally:
            hp_mod._go_kl_refine_lib = saved

    def test_refine_mojo_direct_call_without_lib_raises(self) -> None:
        from sc_neurocore.chiplet import hierarchical_backend_runtime as hp_mod

        saved = hp_mod._mojo_kl_refine_lib
        hp_mod._mojo_kl_refine_lib = None
        try:
            hp = HierarchicalPartitioner(num_partitions=2, refine_backend="python")
            g = _build_graph(20, seed=1)
            with pytest.raises(RuntimeError, match="Mojo KL refine"):
                hp._refine_mojo([list(range(20))], g.adjacency(), g)
        finally:
            hp_mod._mojo_kl_refine_lib = saved
