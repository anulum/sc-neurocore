# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Compilation-cache contracts

"""Contracts for deterministic compiler cache behaviour."""

from __future__ import annotations


class TestCompilationCache:
    def test_miss_then_hit(self) -> None:
        from sc_neurocore.compiler.intelligence import CompilationCache

        cache = CompilationCache()
        eqs = {"v": "a + b"}
        assert cache.get(eqs, "artix7") is None
        assert cache.misses == 1
        cache.put(eqs, "artix7", 16, 8, {"verilog": "..."})
        result = cache.get(eqs, "artix7")
        assert result is not None
        assert cache.hits == 1

    def test_different_target_misses(self) -> None:
        from sc_neurocore.compiler.intelligence import CompilationCache

        cache = CompilationCache()
        eqs = {"v": "a + b"}
        cache.put(eqs, "artix7", 16, 8, {"v": "data"})
        assert cache.get(eqs, "loihi2") is None

    def test_size(self) -> None:
        from sc_neurocore.compiler.intelligence import CompilationCache

        cache = CompilationCache()
        cache.put({"v": "a"}, "artix7", 16, 8, {})
        cache.put({"v": "b"}, "artix7", 16, 8, {})
        assert cache.size == 2
