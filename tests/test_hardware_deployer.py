# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDeployer from former test_hardware.py

"""Focused suite: TestDeployer from former test_hardware.py."""

from __future__ import annotations

from tests.hardware_support import *  # noqa: F403


class TestDeployer:
    """Package, validate and summarise deployments from a mapped random network."""

    def _pipeline(self, n=20, family=DeviceFamily.LOIHI):
        """Return a deployer, a random adjacency, its device and greedy placements."""
        rng = np.random.default_rng(42)
        adj = (rng.random((n, n)) > 0.8).astype(float) * rng.random((n, n))
        np.fill_diagonal(adj, 0)
        device = get_device(family)
        mapper = Mapper()
        placements = mapper.map_greedy(adj, device)
        deployer = Deployer()
        return deployer, adj, device, placements

    def test_package_creates_blob(self):
        """Packaging produces a non-empty configuration blob."""
        deployer, adj, device, placements = self._pipeline()
        pkg = deployer.package(adj, device, placements)
        assert isinstance(pkg, DeploymentPackage)
        assert len(pkg.config_blob) > 0

    def test_validate_good_package(self):
        """A freshly built package validates."""
        deployer, adj, device, placements = self._pipeline()
        pkg = deployer.package(adj, device, placements)
        assert deployer.validate(pkg) is True

    def test_validate_empty_blob_fails(self):
        """An empty configuration blob fails validation."""
        deployer, adj, device, placements = self._pipeline()
        pkg = deployer.package(adj, device, placements)
        pkg.config_blob = b""
        assert deployer.validate(pkg) is False

    def test_validate_rejects_duplicate_neuron_ids(self):
        """A neuron placed twice fails validation."""
        deployer, adj, device, placements = self._pipeline()
        pkg = deployer.package(adj, device, placements)
        dup = placements[0]
        pkg.placements = [dup, dup, *placements[2:]]
        assert deployer.validate(pkg) is False

    def test_validate_rejects_core_id_out_of_range(self):
        """A core id at the device's core count fails validation."""
        deployer, adj, device, placements = self._pipeline()
        pkg = deployer.package(adj, device, placements)
        bad = placements[0]
        bad.core_id = device.cores
        pkg.placements = [bad, *placements[1:]]
        assert deployer.validate(pkg) is False

    def test_validate_rejects_local_id_out_of_range(self):
        """A local id at the device's neurons per core fails validation."""
        deployer, adj, device, placements = self._pipeline()
        pkg = deployer.package(adj, device, placements)
        bad = placements[0]
        bad.local_id = device.neurons_per_core
        pkg.placements = [bad, *placements[1:]]
        assert deployer.validate(pkg) is False

    def test_summary_contains_info(self):
        """The summary names the neurons, synapses and device family."""
        deployer, adj, device, placements = self._pipeline()
        pkg = deployer.package(adj, device, placements)
        summary = deployer.summary(pkg)
        assert "Neurons" in summary
        assert "Synapses" in summary
        assert "LOIHI" in summary

    def test_metadata_populated(self):
        """The metadata records the neuron count and the device family."""
        deployer, adj, device, placements = self._pipeline()
        pkg = deployer.package(adj, device, placements)
        assert pkg.metadata["n_neurons"] == 20
        assert pkg.metadata["device_family"] == "LOIHI"

    def test_config_blob_starts_with_magic(self):
        """The blob starts with the SCNC magic."""
        deployer, adj, device, placements = self._pipeline()
        pkg = deployer.package(adj, device, placements)
        assert pkg.config_blob[:4] == b"SCNC"

    def test_full_pipeline_roundtrip(self):
        """A 50-neuron network packages, validates and summarises."""
        deployer, adj, device, placements = self._pipeline(n=50)
        pkg = deployer.package(adj, device, placements)
        assert deployer.validate(pkg)
        summary = deployer.summary(pkg)
        assert "50" in summary

    def test_package_uses_explicit_weights_override(self):
        """Explicit weights are coded and counted instead of the adjacency values."""
        deployer, adj, device, placements = self._pipeline(n=8)
        dense_weights = np.ones_like(adj, dtype=np.float32)
        np.fill_diagonal(dense_weights, 0.0)
        pkg = deployer.package(adj, device, placements, weights=dense_weights)
        # Config blob should encode non-zero synapses from explicit weights matrix,
        # and the metadata counts the synapses the blob codes.
        assert len(pkg.config_blob) > 16
        assert pkg.metadata["n_synapses"] == int(np.count_nonzero(dense_weights))

    def test_package_handles_all_zero_weights_scaling_branch(self):
        """All-zero weights still package and validate."""
        deployer, adj, device, placements = self._pipeline(n=8)
        zero_weights = np.zeros_like(adj, dtype=np.float32)
        pkg = deployer.package(adj, device, placements, weights=zero_weights)
        assert pkg.config_blob[:4] == b"SCNC"
        # Still valid package, even when all encoded weights are zero.
        assert deployer.validate(pkg) is True

    def test_package_marks_fits_false_when_core_usage_exceeds_device_capacity(self):
        """A package that needs more cores than the device has is marked as not fitting."""
        deployer, adj, _, _ = self._pipeline(n=20)
        tiny_device = DeviceSpec(
            family=DeviceFamily.FPGA_GENERIC,
            cores=1,
            neurons_per_core=1,
            synapses_per_core=1_000_000,
            axons_per_core=1_000_000,
            tick_ns=100.0,
            precision_bits=8,
            supports_learning=True,
            power_per_core_mw=1.0,
        )
        mapper = Mapper()
        placements = mapper.map_greedy(adj, tiny_device)
        pkg = deployer.package(adj, tiny_device, placements)
        assert pkg.metadata["n_cores_used"] > tiny_device.cores
        assert pkg.metadata["fits"] is False
