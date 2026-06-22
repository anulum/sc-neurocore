# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SC Optimizer Tests

"""Tests for the stochastic optimizer module."""

from __future__ import annotations

import unittest

from sc_neurocore.optimizer.sc_optimizer import (
    SCOptimizer,
    HardwareBudget,
    LayerProfile,
    LayerConfig,
)


class TestResourceEstimation(unittest.TestCase):
    def setUp(self):
        self.opt = SCOptimizer(HardwareBudget(max_luts=100000, max_power_mw=1000.0))

    def test_deterministic_mode_baseline(self):
        luts, power, accuracy, latency = self.opt._estimate_resources(
            100, 1, "None", "Deterministic"
        )
        self.assertEqual(accuracy, 1.0)
        self.assertGreater(luts, 0)
        self.assertGreater(power, 0)

    def test_sc_lfsr_cheaper_than_deterministic(self):
        det_luts, _, _, _ = self.opt._estimate_resources(100, 256, "None", "Deterministic")
        sc_luts, _, _, _ = self.opt._estimate_resources(100, 256, "LFSR", "SC")
        self.assertLess(sc_luts, det_luts)

    def test_sobol_more_accurate_than_lfsr(self):
        _, _, acc_lfsr, _ = self.opt._estimate_resources(100, 256, "LFSR", "SC")
        _, _, acc_sobol, _ = self.opt._estimate_resources(100, 256, "Sobol", "SC")
        self.assertGreater(acc_sobol, acc_lfsr)

    def test_longer_bitstream_more_accurate(self):
        _, _, acc_short, _ = self.opt._estimate_resources(100, 64, "LFSR", "SC")
        _, _, acc_long, _ = self.opt._estimate_resources(100, 1024, "LFSR", "SC")
        self.assertGreater(acc_long, acc_short)

    def test_accuracy_clamped(self):
        _, _, acc, _ = self.opt._estimate_resources(100, 4, "None", "SC")
        self.assertGreaterEqual(acc, 0.1)
        self.assertLessEqual(acc, 1.0)


class TestCandidateGeneration(unittest.TestCase):
    def setUp(self):
        self.opt = SCOptimizer(HardwareBudget(max_luts=100000, max_power_mw=1000.0))

    def test_generates_candidates(self):
        layer = LayerProfile(id="L0", mac_count=100)
        candidates = self.opt._generate_candidates(layer)
        self.assertGreater(len(candidates), 0)
        modes = {c.mode for c in candidates}
        self.assertIn("SC", modes)
        self.assertIn("Deterministic", modes)

    def test_deterministic_candidate_exists(self):
        layer = LayerProfile(id="L0", mac_count=50)
        candidates = self.opt._generate_candidates(layer)
        det = [c for c in candidates if c.mode == "Deterministic"]
        self.assertEqual(len(det), 1)


class TestOptimization(unittest.TestCase):
    def test_single_layer(self):
        opt = SCOptimizer(HardwareBudget(max_luts=100000, max_power_mw=1000.0))
        network = [LayerProfile(id="L0", mac_count=100)]
        report = opt.optimize(network)
        self.assertIsNotNone(report)
        self.assertIn("L0", report.config)
        self.assertIsInstance(report.config["L0"], LayerConfig)

    def test_multi_layer(self):
        opt = SCOptimizer(HardwareBudget(max_luts=1000000, max_power_mw=5000.0))
        network = [
            LayerProfile(id=f"L{i}", mac_count=500, is_critical_path=(i == 0)) for i in range(10)
        ]
        report = opt.optimize(network)
        self.assertIsNotNone(report)
        self.assertEqual(len(report.config), 10)

    def test_critical_path_gets_priority(self):
        opt = SCOptimizer(HardwareBudget(max_luts=1000000, max_power_mw=50000.0))
        network = [
            LayerProfile(id="critical", mac_count=100, is_critical_path=True),
            LayerProfile(id="normal", mac_count=100, is_critical_path=False),
        ]
        report = opt.optimize(network)
        self.assertIsNotNone(report)
        self.assertGreaterEqual(
            report.config["critical"].accuracy_score,
            report.config["normal"].accuracy_score,
        )

    def test_tiny_budget_returns_none(self):
        opt = SCOptimizer(HardwareBudget(max_luts=1, max_power_mw=0.001))
        network = [LayerProfile(id="L0", mac_count=10000)]
        result = opt.optimize(network)
        self.assertIsNone(result)

    def test_resource_budget_respected(self):
        budget = HardwareBudget(max_luts=50000, max_power_mw=500.0)
        opt = SCOptimizer(budget)
        network = [LayerProfile(id=f"L{i}", mac_count=200) for i in range(5)]
        report = opt.optimize(network)
        if report:
            self.assertLessEqual(report.total_luts, budget.max_luts)
            self.assertLessEqual(report.total_power_mw, budget.max_power_mw)


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


def _fake_sa_result(n, *, feasible=True, with_pareto=True):
    return {
        "feasible": feasible,
        "layer_luts": [100] * n,
        "layer_power": [1.0] * n,
        "layer_accuracy": [0.9] * n,
        "pareto_luts": [100, 200] if with_pareto else [],
        "pareto_power": [1.0, 2.0] if with_pareto else [],
        "pareto_score": [0.9, 0.95] if with_pareto else [],
    }


def _install_fake_rust(monkeypatch, sa_result):
    import sc_neurocore.optimizer.sc_optimizer as mod

    monkeypatch.setattr(mod, "_HAS_RUST", True)
    monkeypatch.setattr(mod, "py_opt_sa_search", lambda *a, **k: sa_result, raising=False)
    monkeypatch.setattr(
        mod,
        "py_opt_extract_pareto",
        lambda luts, power, score: {"luts": luts, "power": power, "score": score},
        raising=False,
    )


def test_annealing_dispatches_to_rust_backend(monkeypatch):
    network = [LayerProfile(id="L0", mac_count=100), LayerProfile(id="L1", mac_count=200)]
    _install_fake_rust(monkeypatch, _fake_sa_result(len(network)))
    opt = SCOptimizer(HardwareBudget(max_luts=10**6, max_power_mw=10**4))
    report = opt.optimize_annealing(network)
    assert report is not None
    assert len(report.pareto_frontier) == 2


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

    fake = types.ModuleType("sc_neurocore_engine")
    fake.py_opt_sa_search = lambda *a, **k: {}  # type: ignore[attr-defined]
    fake.py_opt_extract_pareto = lambda *a, **k: {}  # type: ignore[attr-defined]
    had = sys.modules.get("sc_neurocore_engine")
    sys.modules["sc_neurocore_engine"] = fake
    try:
        reloaded = importlib.reload(mod)
        assert reloaded._HAS_RUST is True
    finally:
        if had is not None:
            sys.modules["sc_neurocore_engine"] = had
        else:
            sys.modules.pop("sc_neurocore_engine", None)
        importlib.reload(mod)


if __name__ == "__main__":
    unittest.main()
