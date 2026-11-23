# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSCNASEngineEdgeBranches from former test_sc_nas_engine.py

"""Focused suite: TestSCNASEngineEdgeBranches from former test_sc_nas_engine.py."""

from __future__ import annotations

import importlib
import pytest
import sys
from sc_nas_engine_support import (
    RustTournament,
)
from sc_neurocore.nas import (
    sc_nas_engine as nas_module,
)
from sc_neurocore.nas.sc_nas_engine import (
    DecorrelationStrategy,
    EvolutionaryNAS,
    FPGAResourceBudget,
    LayerConfig,
    NASObjective,
    NASReport,
    NeuronType,
    SCCandidate,
)
from types import (
    ModuleType,
)


class TestSCNASEngineEdgeBranches:
    """Edge branches of the SC-NAS engine.

    Resource-utilisation ratios, empty-front report accessors, and the
    Rust-evolution tournament path (flag + import branch).
    """

    @staticmethod
    def _candidate(fitness: float) -> SCCandidate:
        return SCCandidate(
            layers=[LayerConfig(32, NeuronType.LIF, 256, DecorrelationStrategy.LFSR)],
            fitness=fitness,
        )

    def test_resource_budget_utilisation_ratios(self) -> None:
        """Budget utilisation is each total over its limit."""
        budget = FPGAResourceBudget(max_luts=1000, max_ffs=2000, max_bram_kb=100, max_dsp=50)
        util = budget.utilisation(luts=500, ffs=500, bram=25, dsp=25)
        assert util["luts"] == 0.5
        assert util["ffs"] == 0.25
        assert util["bram"] == 0.25
        assert util["dsp"] == 0.5

    def test_report_accessors_on_empty_pareto_front(self) -> None:
        """An empty front reports zero accuracy and no efficient candidate."""
        report = NASReport(pareto_front=[], search_history=[])
        assert report.best_accuracy == 0.0
        assert report.most_efficient is None

    def test_tournament_select_uses_rust_evo_when_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Tournament selection uses the Rust kernel when it is present."""
        nas = EvolutionaryNAS(
            objective=NASObjective(),
            budget=FPGAResourceBudget(),
            population_size=24,
            num_generations=1,
            seed=7,
        )
        population = [self._candidate(float(index)) for index in range(24)]
        captured: dict[str, list[float]] = {}

        def _fake_evo(fitness: list[float], n: int, k: int, seed: int) -> list[int]:
            captured["fitness"] = list(fitness)
            return [3]

        # The Rust tournament path activates only when the flag is set AND the
        # population exceeds the 20-candidate threshold.
        monkeypatch.setattr(nas_module, "_HAS_RUST_EVO", True)
        monkeypatch.setattr(nas_module, "py_evo_tournament", _fake_evo, raising=False)

        chosen = nas._tournament_select(population, k=3)
        assert chosen is population[3]
        assert len(captured["fitness"]) == 24

    def test_module_import_enables_rust_evo_when_extension_is_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Importing the module enables the Rust evolution path when its extension loads."""
        captured: dict[str, list[float]] = {}

        def _fake_evo(fitness: list[float], n: int, k: int, seed: int) -> list[int]:
            captured["fitness"] = list(fitness)
            assert n == 1
            assert k == 3
            assert seed >= 0
            return [5]

        class _RustExtension(ModuleType):
            py_evo_tournament: RustTournament

        rust_extension = _RustExtension("sc_neurocore_engine")
        rust_extension.py_evo_tournament = _fake_evo
        monkeypatch.setitem(sys.modules, "sc_neurocore_engine", rust_extension)

        # importlib.reload rebinds every class the module defines (DecorrelationStrategy,
        # LayerConfig, ...), so a reloaded — or a second, restoring — reload leaves those
        # classes with fresh identities that leak to other tests in the session (a
        # LayerConfig.decorrelation built afterwards would no longer be `is`/`==` the
        # DecorrelationStrategy those tests imported at collection time). Snapshot the
        # pre-reload module dict and restore the exact original objects afterwards.
        _original_attrs = dict(nas_module.__dict__)
        importlib.reload(nas_module)
        try:
            assert nas_module._HAS_RUST_EVO is True
            nas = nas_module.EvolutionaryNAS(
                objective=nas_module.NASObjective(),
                budget=nas_module.FPGAResourceBudget(),
                population_size=24,
                num_generations=1,
                seed=11,
            )
            population = [self._candidate(float(index)) for index in range(24)]

            chosen = nas._tournament_select(population, k=3)

            assert chosen is population[5]
            assert captured["fitness"] == [candidate.fitness for candidate in population]
        finally:
            monkeypatch.delitem(sys.modules, "sc_neurocore_engine", raising=False)
            nas_module.__dict__.clear()
            nas_module.__dict__.update(_original_attrs)
