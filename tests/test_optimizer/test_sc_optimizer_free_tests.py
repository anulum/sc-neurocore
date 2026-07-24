# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Module-level tests from former test_sc_optimizer.py

"""Module-level tests from former test_sc_optimizer.py."""

from __future__ import annotations

from sc_optimizer_support import *  # noqa: F403


def test_is_feasible_rejects_power_over_budget():
    # LUTs fit comfortably but the power ceiling is below any candidate, so the
    # power branch of the feasibility check rejects the configuration.
    opt = SCOptimizer(HardwareBudget(max_luts=10**9, max_power_mw=1e-9))
    report = opt.optimize([LayerProfile(id="L0", mac_count=100)])
    assert report is None


def test_annealing_python_handles_infeasible_trials():
    # Pin the LUT ceiling to exactly the cheapest candidate: the annealing start
    # is feasible, but any pricier trial overshoots it and takes the
    # cool-and-continue rejection path. The feasible start is still returned.
    layer = LayerProfile(id="L0", mac_count=50)
    probe = SCOptimizer(HardwareBudget(max_luts=10**9, max_power_mw=10**9))
    min_luts = min(c.luts_used for c in probe._generate_candidates(layer))
    opt = SCOptimizer(HardwareBudget(max_luts=min_luts, max_power_mw=10**9))
    report = opt.optimize_annealing([layer], max_iter=400, seed=1)
    assert report is not None


def test_annealing_dispatches_to_rust_backend(monkeypatch):
    network = [LayerProfile(id="L0", mac_count=100), LayerProfile(id="L1", mac_count=200)]
    _install_fake_rust(monkeypatch, _fake_sa_result(len(network)))
    opt = SCOptimizer(HardwareBudget(max_luts=10**6, max_power_mw=10**4))
    report = opt.optimize_annealing(network)
    assert report is not None
    assert len(report.pareto_frontier) == 2


def test_annealing_rust_frontier_is_sorted_and_deduplicated(monkeypatch):
    # A Rust backend may emit non-dominated points in arbitrary order and with
    # exact duplicates; the report must still expose a LUT-sorted, deduplicated
    # frontier so it honours the same contract as the pure-Python fallback.
    network = [LayerProfile(id="L0", mac_count=100), LayerProfile(id="L1", mac_count=200)]
    result = _fake_sa_result(len(network))
    result["pareto_luts"] = [200, 100, 100]
    result["pareto_power"] = [2.0, 1.0, 1.0]
    result["pareto_score"] = [0.95, 0.9, 0.9]
    _install_fake_rust(monkeypatch, result)
    opt = SCOptimizer(HardwareBudget(max_luts=10**6, max_power_mw=10**4))
    report = opt.optimize_annealing(network)
    assert report is not None
    luts_vals = [p[0] for p in report.pareto_frontier]
    assert luts_vals == sorted(luts_vals)
    assert luts_vals == [100, 200]


def test_annealing_rust_infeasible_returns_none(monkeypatch):
    network = [LayerProfile(id="L0", mac_count=100)]
    _install_fake_rust(monkeypatch, _fake_sa_result(len(network), feasible=False))
    opt = SCOptimizer(HardwareBudget(max_luts=10**6, max_power_mw=10**4))
    assert opt.optimize_annealing(network) is None


def test_annealing_rust_without_pareto_points(monkeypatch):
    network = [LayerProfile(id="L0", mac_count=100)]
    _install_fake_rust(monkeypatch, _fake_sa_result(len(network), with_pareto=False))
    opt = SCOptimizer(HardwareBudget(max_luts=10**6, max_power_mw=10**4))
    report = opt.optimize_annealing(network)
    assert report is not None
    assert report.pareto_frontier == []


def test_module_detects_rust_engine_when_importable():
    # Drive the import-time `_HAS_RUST = True` branch by making a stand-in
    # sc_neurocore_engine importable, then restore the real (engine-absent) state.
    import importlib
    import sys
    import types

    import sc_neurocore.optimizer.sc_optimizer as mod

    from tests.module_reload import preserve_module_identity

    fake = types.ModuleType("sc_neurocore_engine")
    fake.py_opt_sa_search = lambda *a, **k: {}  # type: ignore[attr-defined]
    fake.py_opt_extract_pareto = lambda *a, **k: {}  # type: ignore[attr-defined]
    had = sys.modules.get("sc_neurocore_engine")
    sys.modules["sc_neurocore_engine"] = fake
    try:
        # Restore the module's original class identities on exit; a bare reload-to-restore
        # leaks fresh classes that sibling tests' by-value imports fail isinstance on.
        with preserve_module_identity(mod):
            reloaded = importlib.reload(mod)
            assert reloaded._HAS_RUST is True
    finally:
        if had is not None:
            sys.modules["sc_neurocore_engine"] = had
        else:
            sys.modules.pop("sc_neurocore_engine", None)
