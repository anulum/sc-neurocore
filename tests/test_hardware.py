# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for hardware/ HAL

from __future__ import annotations

import time

import numpy as np
import pytest

from sc_neurocore.hardware import (
    DeviceFamily,
    DeviceSpec,
    DEVICE_CATALOG,
    get_device,
    ResourceEstimate,
    ResourceEstimator,
    HardwareConstraints,
    ConstraintChecker,
    Mapper,
    DeploymentPackage,
    Deployer,
)


# ---------------------------------------------------------------------------
# Device Catalog
# ---------------------------------------------------------------------------


class TestDeviceCatalog:
    def test_all_families_have_specs(self):
        for family in DeviceFamily:
            assert family in DEVICE_CATALOG, f"{family.name} missing from catalog"

    @pytest.mark.parametrize("family", list(DeviceFamily))
    def test_device_spec_valid(self, family):
        spec = DEVICE_CATALOG[family]
        assert spec.cores > 0
        assert spec.neurons_per_core > 0
        assert spec.precision_bits > 0
        assert spec.tick_ns > 0
        assert spec.power_per_core_mw >= 0

    def test_get_device_by_enum(self):
        spec = get_device(DeviceFamily.LOIHI)
        assert spec.family == DeviceFamily.LOIHI

    def test_get_device_by_string(self):
        spec = get_device("loihi2")
        assert spec.family == DeviceFamily.LOIHI2

    def test_get_device_unknown_raises(self):
        with pytest.raises((ValueError, KeyError)):
            get_device("nonexistent")

    def test_loihi_specs_match_datasheet(self):
        loihi = get_device(DeviceFamily.LOIHI)
        assert loihi.cores == 128
        assert loihi.neurons_per_core == 1024
        assert loihi.weight_bits == 9

    def test_spinnaker_specs(self):
        spin = get_device(DeviceFamily.SPINNAKER)
        assert spin.cores == 18
        assert spin.supports_learning is True


# ---------------------------------------------------------------------------
# Resource Estimator
# ---------------------------------------------------------------------------


class TestResourceEstimator:
    def _make_adj(self, n, density=0.1, seed=42):
        rng = np.random.default_rng(seed)
        adj = (rng.random((n, n)) < density).astype(float)
        np.fill_diagonal(adj, 0)
        return adj

    def test_estimate_small_network(self):
        adj = self._make_adj(10, density=0.5)
        estimator = ResourceEstimator()
        result = estimator.estimate(adj, get_device(DeviceFamily.LOIHI))
        assert isinstance(result, ResourceEstimate)
        assert result.cores_needed >= 1
        assert result.neurons_mapped == 10
        assert result.fits is True

    def test_estimate_large_network(self):
        adj = self._make_adj(1000, density=0.01)
        estimator = ResourceEstimator()
        result = estimator.estimate(adj, get_device(DeviceFamily.LOIHI))
        assert result.neurons_mapped == 1000
        assert result.power_mw > 0

    def test_fits_small(self):
        adj = self._make_adj(10)
        estimator = ResourceEstimator()
        assert estimator.fits(adj, get_device(DeviceFamily.LOIHI)) is True

    def test_compare_devices(self):
        adj = self._make_adj(100, density=0.05)
        estimator = ResourceEstimator()
        devices = [
            get_device(f) for f in [DeviceFamily.LOIHI, DeviceFamily.SPINNAKER, DeviceFamily.AKIDA]
        ]
        results = estimator.compare(adj, devices)
        assert len(results) == 3
        assert all(isinstance(r, ResourceEstimate) for r in results)

    def test_utilization_range(self):
        adj = self._make_adj(50)
        estimator = ResourceEstimator()
        result = estimator.estimate(adj, get_device(DeviceFamily.LOIHI))
        assert 0 < result.utilization_pct <= 100

    def test_latency_positive(self):
        adj = self._make_adj(10)
        estimator = ResourceEstimator()
        result = estimator.estimate(adj, get_device(DeviceFamily.LOIHI))
        assert result.latency_us > 0


# ---------------------------------------------------------------------------
# Constraint Checker
# ---------------------------------------------------------------------------


class TestConstraintChecker:
    def test_no_violations_small_network(self):
        adj = np.zeros((5, 5))
        adj[0, 1] = 1.0
        adj[1, 2] = 1.0
        checker = ConstraintChecker()
        constraints = HardwareConstraints(max_fan_in=256, max_fan_out=4096)
        violations = checker.check(adj, constraints)
        assert len(violations) == 0

    def test_fan_in_violation(self):
        n = 50
        adj = np.zeros((n, n))
        adj[:, 0] = 1.0  # neuron 0 has fan-in = 49
        adj[0, 0] = 0
        checker = ConstraintChecker()
        constraints = HardwareConstraints(max_fan_in=10)
        violations = checker.check(adj, constraints)
        fan_in_violations = [v for v in violations if v.constraint == "fan_in"]
        assert len(fan_in_violations) >= 1
        assert fan_in_violations[0].neuron_id == 0

    def test_fan_out_violation(self):
        n = 50
        adj = np.zeros((n, n))
        adj[0, :] = 1.0
        adj[0, 0] = 0
        checker = ConstraintChecker()
        constraints = HardwareConstraints(max_fan_out=10)
        violations = checker.check(adj, constraints)
        fan_out_violations = [v for v in violations if v.constraint == "fan_out"]
        assert len(fan_out_violations) >= 1

    def test_delay_violation(self):
        adj = np.array([[0, 1], [0, 0]], dtype=float)
        delays = np.array([[0, 100], [0, 0]], dtype=float)
        checker = ConstraintChecker()
        constraints = HardwareConstraints(max_delay_ticks=63)
        violations = checker.check(adj, constraints, delays=delays)
        delay_v = [v for v in violations if v.constraint == "delay"]
        assert len(delay_v) >= 1

    def test_from_device_constraints(self):
        constraints = HardwareConstraints.from_device(get_device(DeviceFamily.LOIHI))
        assert constraints.max_fan_in == 4096
        assert constraints.weight_bits == 9

    def test_auto_fix_resolves_violations(self):
        n = 50
        adj = np.zeros((n, n))
        adj[:, 0] = 1.0
        adj[0, 0] = 0
        checker = ConstraintChecker()
        constraints = HardwareConstraints(max_fan_in=10)
        violations_before = checker.check(adj, constraints)
        assert len(violations_before) > 0

        fixed = checker.auto_fix(adj, constraints)
        violations_after = checker.check(fixed, constraints)
        fan_in_after = [v for v in violations_after if v.constraint == "fan_in"]
        assert len(fan_in_after) == 0


# ---------------------------------------------------------------------------
# Mapper
# ---------------------------------------------------------------------------


class TestMapper:
    def _make_adj(self, n=20, density=0.1, seed=42):
        rng = np.random.default_rng(seed)
        adj = (rng.random((n, n)) < density).astype(float)
        np.fill_diagonal(adj, 0)
        return adj

    def test_greedy_no_collisions(self):
        adj = self._make_adj()
        mapper = Mapper()
        placements = mapper.map_greedy(adj, get_device(DeviceFamily.LOIHI))
        neuron_ids = [p.neuron_id for p in placements]
        assert len(set(neuron_ids)) == len(neuron_ids)

    def test_balanced_all_placed(self):
        adj = self._make_adj()
        mapper = Mapper()
        placements = mapper.map_balanced(adj, get_device(DeviceFamily.LOIHI))
        assert len(placements) == 20

    def test_locality_clusters_neighbors(self):
        n = 20
        adj = np.zeros((n, n))
        # Create two clusters: 0-9 and 10-19
        for i in range(9):
            adj[i, i + 1] = 1.0
            adj[i + 1, i] = 1.0
        for i in range(10, 19):
            adj[i, i + 1] = 1.0
            adj[i + 1, i] = 1.0
        mapper = Mapper()
        # Use FPGA with small cores to force splitting
        device = DeviceSpec(
            family=DeviceFamily.FPGA_GENERIC,
            cores=10,
            neurons_per_core=10,
            synapses_per_core=1000,
            axons_per_core=100,
            tick_ns=100,
            precision_bits=16,
            supports_learning=True,
            power_per_core_mw=1.0,
        )
        placements = mapper.map_locality(adj, device)
        assert len(placements) == n
        # Check cluster 0 neurons are mostly on same core
        cluster0_cores = {placements[i].core_id for i in range(10)}
        assert len(cluster0_cores) <= 2  # should be 1 or 2 cores

    def test_greedy_core_ids_valid(self):
        adj = self._make_adj(100)
        device = get_device(DeviceFamily.LOIHI)
        mapper = Mapper()
        placements = mapper.map_greedy(adj, device)
        for p in placements:
            assert p.core_id >= 0
            assert p.local_id >= 0


# ---------------------------------------------------------------------------
# Deployer
# ---------------------------------------------------------------------------


class TestDeployer:
    def _pipeline(self, n=20, family=DeviceFamily.LOIHI):
        rng = np.random.default_rng(42)
        adj = (rng.random((n, n)) > 0.8).astype(float) * rng.random((n, n))
        np.fill_diagonal(adj, 0)
        device = get_device(family)
        mapper = Mapper()
        placements = mapper.map_greedy(adj, device)
        deployer = Deployer()
        return deployer, adj, device, placements

    def test_package_creates_blob(self):
        deployer, adj, device, placements = self._pipeline()
        pkg = deployer.package(adj, device, placements)
        assert isinstance(pkg, DeploymentPackage)
        assert len(pkg.config_blob) > 0

    def test_validate_good_package(self):
        deployer, adj, device, placements = self._pipeline()
        pkg = deployer.package(adj, device, placements)
        assert deployer.validate(pkg) is True

    def test_validate_empty_blob_fails(self):
        deployer, adj, device, placements = self._pipeline()
        pkg = deployer.package(adj, device, placements)
        pkg.config_blob = b""
        assert deployer.validate(pkg) is False

    def test_summary_contains_info(self):
        deployer, adj, device, placements = self._pipeline()
        pkg = deployer.package(adj, device, placements)
        summary = deployer.summary(pkg)
        assert "Neurons" in summary
        assert "Synapses" in summary
        assert "LOIHI" in summary

    def test_metadata_populated(self):
        deployer, adj, device, placements = self._pipeline()
        pkg = deployer.package(adj, device, placements)
        assert pkg.metadata["n_neurons"] == 20
        assert pkg.metadata["device_family"] == "LOIHI"

    def test_config_blob_starts_with_magic(self):
        deployer, adj, device, placements = self._pipeline()
        pkg = deployer.package(adj, device, placements)
        assert pkg.config_blob[:4] == b"SCNC"

    def test_full_pipeline_roundtrip(self):
        deployer, adj, device, placements = self._pipeline(n=50)
        pkg = deployer.package(adj, device, placements)
        assert deployer.validate(pkg)
        summary = deployer.summary(pkg)
        assert "50" in summary


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


class TestHardwareBenchmark:
    def test_map_10k_neurons(self):
        """Map 10,000 neurons to Loihi in < 2 seconds."""
        rng = np.random.default_rng(42)
        adj = (rng.random((10_000, 10_000)) > 0.999).astype(float)
        np.fill_diagonal(adj, 0)
        mapper = Mapper()
        device = get_device(DeviceFamily.LOIHI)
        t0 = time.perf_counter()
        placements = mapper.map_greedy(adj, device)
        elapsed = time.perf_counter() - t0
        assert len(placements) == 10_000
        assert elapsed < 2.0, f"10k mapping took {elapsed:.2f}s"

    def test_estimate_10k_neurons(self):
        """Resource estimation for 10k neurons in < 1 second."""
        rng = np.random.default_rng(42)
        adj = (rng.random((10_000, 10_000)) > 0.999).astype(float)
        np.fill_diagonal(adj, 0)
        estimator = ResourceEstimator()
        t0 = time.perf_counter()
        result = estimator.estimate(adj, get_device(DeviceFamily.LOIHI))
        elapsed = time.perf_counter() - t0
        assert result.neurons_mapped == 10_000
        assert elapsed < 1.0, f"10k estimation took {elapsed:.2f}s"

    def test_constraint_check_1k_neurons(self):
        """Constraint check for 1000 neurons in < 0.5 seconds."""
        rng = np.random.default_rng(42)
        adj = (rng.random((1000, 1000)) > 0.99).astype(float)
        np.fill_diagonal(adj, 0)
        checker = ConstraintChecker()
        constraints = HardwareConstraints.from_device(get_device(DeviceFamily.LOIHI))
        t0 = time.perf_counter()
        violations = checker.check(adj, constraints)
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.5, f"1k constraint check took {elapsed:.2f}s"
