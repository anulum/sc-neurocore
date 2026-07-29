# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Focused photonic export and optimizer fallback coverage

"""Close optional GDS export and pure-Python annealing behavior branches."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from sc_neurocore.optics._photonic_compiler import CompilationResult
from sc_neurocore.optimizer import sc_optimizer
from sc_neurocore.optimizer.sc_optimizer import HardwareBudget, LayerProfile, SCOptimizer


class _Reference:
    """Minimal layout reference carrying an assignable x coordinate."""

    def __init__(self) -> None:
        self.x = 0.0


class _Component:
    """Minimal gdsfactory component double recording requested operations."""

    def __init__(self, *, kdb_cell: object) -> None:
        self.kdb_cell = kdb_cell
        self.labels: list[dict[str, object]] = []
        self.references: list[_Reference] = []
        self.filename = ""

    def add_label(self, **kwargs: object) -> None:
        self.labels.append(kwargs)

    def add_ref(self, _cell: object) -> _Reference:
        reference = _Reference()
        self.references.append(reference)
        return reference

    def write_gds(self, filename: str) -> None:
        self.filename = filename


def _fake_gdsfactory(*, legacy_pdk_fallback: bool) -> ModuleType:
    """Build a deterministic gdsfactory facade for both PDK activation APIs."""

    module = ModuleType("gdsfactory")
    gpdk_module = ModuleType("gdsfactory.gpdk")
    activated: list[str] = []

    class GenericPdk:
        def activate(self) -> None:
            activated.append("generic")

    gpdk_module.get_generic_pdk = lambda: GenericPdk()  # type: ignore[attr-defined]
    if legacy_pdk_fallback:
        pdk = object()
    else:

        class Pdk:
            @staticmethod
            def activate() -> None:
                activated.append("direct")

        pdk = Pdk

    def no_active_pdk() -> None:
        raise ValueError("no active PDK")

    module.get_active_pdk = no_active_pdk  # type: ignore[attr-defined]
    module.gpdk = SimpleNamespace(PDK=pdk)  # type: ignore[attr-defined]
    module.kcl = SimpleNamespace(  # type: ignore[attr-defined]
        create_cell=lambda _name, allow_duplicate: (allow_duplicate, "cell")
    )
    module.Component = _Component  # type: ignore[attr-defined]
    module.components = SimpleNamespace(  # type: ignore[attr-defined]
        mzi=lambda **kwargs: ("mzi", kwargs)
    )
    module._activated = activated  # type: ignore[attr-defined]
    sys.modules["gdsfactory.gpdk"] = gpdk_module
    return module


@pytest.mark.parametrize("legacy_pdk_fallback", (False, True))
def test_gds_export_activates_pdk_and_builds_complete_layout(
    monkeypatch: pytest.MonkeyPatch, legacy_pdk_fallback: bool
) -> None:
    """GDS export supports both current and legacy generic-PDK activation APIs."""

    gdsfactory = _fake_gdsfactory(legacy_pdk_fallback=legacy_pdk_fallback)
    monkeypatch.setitem(sys.modules, "gdsfactory", gdsfactory)
    result = CompilationResult(
        target="silicon",
        num_modulators=2,
        optical_power_mean_mw=1.0,
        phase_coverage_rad=0.5,
        netlist="logical-netlist",
    )

    receipt = result.to_gdsii("layout.gds", mzi_length_um=12.0, pitch_um=25.0)

    assert receipt == {
        "filename": "layout.gds",
        "n_modulators": 2,
        "mzi_length_um": 12.0,
        "pitch_um": 25.0,
        "total_length_um": 50.0,
        "target": "silicon",
    }
    assert gdsfactory._activated  # type: ignore[attr-defined]


def test_python_annealing_dispatch_and_search_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """Python annealing accepts improvements and returns a reproducible report."""

    monkeypatch.setattr(sc_optimizer, "_HAS_RUST", False)
    optimizer = SCOptimizer(HardwareBudget(max_luts=500_000, max_power_mw=5_000.0))
    network = [
        LayerProfile("critical", 50, is_critical_path=True),
        LayerProfile("ordinary", 40),
    ]

    report = optimizer.optimize_annealing(
        network,
        t_init=1.0,
        t_min=0.001,
        alpha=0.9,
        max_iter=200,
        seed=7,
    )

    assert report is not None
    assert set(report.config) == {"critical", "ordinary"}
    assert 0.0 < report.mean_accuracy <= 1.0
    assert report.pareto_frontier


def test_python_annealing_rejects_infeasible_start(monkeypatch: pytest.MonkeyPatch) -> None:
    """A budget below the cheapest candidate returns no annealing report."""

    monkeypatch.setattr(sc_optimizer, "_HAS_RUST", False)
    optimizer = SCOptimizer(HardwareBudget(max_luts=0, max_power_mw=0.0))

    assert optimizer.optimize_annealing([LayerProfile("layer", 10)], max_iter=1) is None


def test_python_annealing_skips_infeasible_trials(monkeypatch: pytest.MonkeyPatch) -> None:
    """A just-feasible starting budget cools past more expensive candidates."""

    monkeypatch.setattr(sc_optimizer, "_HAS_RUST", False)
    layer = LayerProfile("layer", 50)
    probe = SCOptimizer(HardwareBudget(max_luts=10**9, max_power_mw=10**9))
    cheapest = min(probe._generate_candidates(layer), key=lambda candidate: candidate.luts_used)
    optimizer = SCOptimizer(
        HardwareBudget(max_luts=cheapest.luts_used, max_power_mw=cheapest.power_used)
    )

    report = optimizer.optimize_annealing([layer], max_iter=100, seed=1)

    assert report is not None
    assert report.total_luts == cheapest.luts_used
