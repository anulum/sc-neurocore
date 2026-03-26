# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for bio module (DNA, GRN, neuromodulation)

import numpy as np

from sc_neurocore.bio import DNAEncoder, GeneticRegulatoryLayer, NeuromodulatorSystem


class TestDNAEncoder:
    def test_encode_basic(self):
        enc = DNAEncoder(mutation_rate=0.0)
        bits = np.array([0, 0, 0, 1, 1, 0, 1, 1], dtype=np.uint8)
        dna = enc.encode(bits)
        assert dna == "ACGT"

    def test_decode_lossless(self):
        enc = DNAEncoder(mutation_rate=0.0)
        bits = np.array([1, 0, 0, 1, 1, 1, 0, 0], dtype=np.uint8)
        dna = enc.encode(bits)
        recovered = enc.decode(dna)
        np.testing.assert_array_equal(bits, recovered)

    def test_odd_length_padded(self):
        enc = DNAEncoder(mutation_rate=0.0)
        bits = np.array([1, 0, 1], dtype=np.uint8)
        dna = enc.encode(bits)
        assert len(dna) == 2

    def test_roundtrip_even(self):
        np.random.seed(42)
        enc = DNAEncoder(mutation_rate=0.0)
        bits = np.random.randint(0, 2, 20).astype(np.uint8)
        dna = enc.encode(bits)
        recovered = enc.decode(dna)
        np.testing.assert_array_equal(bits, recovered)


class TestGeneticRegulatoryLayer:
    def test_initial_protein_zero(self):
        grn = GeneticRegulatoryLayer(n_neurons=10)
        np.testing.assert_array_equal(grn.protein_levels, 0.0)

    def test_spike_increases_protein(self):
        grn = GeneticRegulatoryLayer(n_neurons=5, production_rate=0.1)
        spikes = np.array([1, 0, 1, 0, 1], dtype=np.float64)
        grn.step(spikes)
        assert grn.protein_levels[0] > 0
        assert grn.protein_levels[1] == 0

    def test_decay_without_spikes(self):
        grn = GeneticRegulatoryLayer(n_neurons=3, production_rate=0.1, decay_rate=0.05)
        grn.protein_levels = np.array([1.0, 1.0, 1.0])
        grn.step(np.zeros(3))
        assert np.all(grn.protein_levels < 1.0)

    def test_protein_bounded(self):
        grn = GeneticRegulatoryLayer(n_neurons=2, production_rate=1.0)
        for _ in range(1000):
            grn.step(np.ones(2))
        assert np.all(grn.protein_levels <= 10.0)

    def test_modulators(self):
        grn = GeneticRegulatoryLayer(n_neurons=3)
        grn.protein_levels = np.array([0.5, 1.0, 0.0])
        m = grn.get_threshold_modulators()
        np.testing.assert_array_equal(m, grn.protein_levels)


class TestNeuromodulatorSystem:
    def test_defaults(self):
        nm = NeuromodulatorSystem()
        assert nm.da_level == 0.5
        assert nm.ht_level == 0.5
        assert nm.ne_level == 0.1

    def test_reward_boosts_dopamine(self):
        nm = NeuromodulatorSystem(da_level=0.5)
        nm.update_levels(reward=1.0, stress=0.0)
        assert nm.da_level > 0.5

    def test_stress_boosts_ne(self):
        nm = NeuromodulatorSystem(ne_level=0.1)
        nm.update_levels(reward=0.0, stress=1.0)
        assert nm.ne_level > 0.1

    def test_stress_drops_serotonin(self):
        nm = NeuromodulatorSystem(ht_level=0.5)
        nm.update_levels(reward=0.0, stress=1.0)
        assert nm.ht_level < 0.5

    def test_serotonin_bounded(self):
        nm = NeuromodulatorSystem()
        for _ in range(100):
            nm.update_levels(reward=0.0, stress=1.0)
        assert nm.ht_level >= 0.1

    def test_modulate_neuron(self):
        nm = NeuromodulatorSystem(da_level=0.8, ht_level=0.5, ne_level=0.3)
        params = {"v_threshold": 1.0, "noise_std": 0.1}
        mod = nm.modulate_neuron(params)
        assert mod["v_threshold"] < 1.0
        assert "noise_std" in mod
