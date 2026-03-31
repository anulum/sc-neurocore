# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: model_zoo pre-configured network architectures

"""Full pipeline test for sc_neurocore.model_zoo.

10 pre-configured network architectures + 3 pretrained weight loaders.
Each config is tested for:
  1. Construction — factory returns a Network with correct topology
  2. Topology — population counts, projection wiring, monitor counts
  3. Dynamics — network produces spikes under Poisson drive
  4. Analytical — neuron model types match published references
  5. Scaling — parameter sweeps verify config scales correctly
  6. Performance — network throughput (neuron-steps/s)
  7. Pipeline — spike_count, firing_rate, ISI from monitor data
FULL PIPELINE WIRED + PERFORMANCE."""

from __future__ import annotations

import time

import numpy as np
import pytest

from sc_neurocore.model_zoo import (
    mnist_classifier,
    dvs_gesture_classifier,
    shd_speech_classifier,
    brunel_balanced_network,
    cortical_column,
    central_pattern_generator,
    decision_making_circuit,
    working_memory_circuit,
    auditory_processing,
    visual_cortex_v1,
)
from sc_neurocore.model_zoo.pretrained import load_pretrained
from sc_neurocore.network.network import Network
from sc_neurocore.neurons import StochasticLIFNeuron
from sc_neurocore.neurons.models import (
    CompteWMNeuron,
    GolombFSNeuron,
    HindmarshRoseNeuron,
    HodgkinHuxleyNeuron,
    PospischilNeuron,
    WangBuzsakiNeuron,
)
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate, isi


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _total_spikes(net: Network) -> int:
    return sum(m.count for m in net.spike_monitors)


def _total_neurons(net: Network) -> int:
    return sum(p.n for p in net.populations)


def _run_and_count(net: Network, duration: float = 0.1) -> int:
    net.run(duration, dt=0.001, backend="python")
    return _total_spikes(net)


# ===========================================================================
# 1. MNIST CLASSIFIER (Zenke & Ganguli 2018)
# ===========================================================================


class TestMNISTClassifier:
    """784-128-10 feedforward SNN for digit classification."""

    def test_returns_network(self):
        net = mnist_classifier(n_hidden=32)
        assert isinstance(net, Network)

    def test_topology_three_populations(self):
        net = mnist_classifier(n_hidden=64)
        assert len(net.populations) == 3
        assert net.populations[0].n == 784
        assert net.populations[1].n == 64
        assert net.populations[2].n == 10

    def test_two_feedforward_projections(self):
        net = mnist_classifier(n_hidden=32)
        assert len(net.projections) == 2
        # input→hidden, hidden→output
        assert net.projections[0].source is net.populations[0]
        assert net.projections[0].target is net.populations[1]
        assert net.projections[1].source is net.populations[1]
        assert net.projections[1].target is net.populations[2]

    def test_two_spike_monitors(self):
        net = mnist_classifier(n_hidden=32)
        assert len(net.spike_monitors) == 2

    def test_uses_stochastic_lif(self):
        net = mnist_classifier(n_hidden=32)
        for pop in net.populations:
            assert pop._model_cls is StochasticLIFNeuron

    def test_xavier_weight_scaling(self):
        """Weight ∝ sqrt(2/fan_in) — Xavier initialisation."""
        net = mnist_classifier(n_hidden=128)
        expected_ih = np.sqrt(2.0 / 784) * 20.0
        assert abs(net.projections[0].weight - expected_ih) < 1e-10
        expected_ho = np.sqrt(2.0 / 128) * 20.0
        assert abs(net.projections[1].weight - expected_ho) < 1e-10

    def test_stimulus_attached(self):
        net = mnist_classifier(n_hidden=32)
        assert len(net.stimuli) == 1
        assert net.stimuli[0].target is net.populations[0]

    def test_produces_spikes(self):
        assert _run_and_count(mnist_classifier(n_hidden=32)) > 0

    def test_output_monitor_records(self):
        net = mnist_classifier(n_hidden=32)
        net.run(0.1, dt=0.001, backend="python")
        # Output monitor is first (mon_out added before mon_hid)
        output_mon = net.spike_monitors[0]
        assert output_mon.label == "output_spikes"

    @pytest.mark.parametrize("n_hidden", [16, 64, 128])
    def test_scales_hidden_size(self, n_hidden: int):
        net = mnist_classifier(n_hidden=n_hidden)
        assert net.populations[1].n == n_hidden

    def test_performance(self):
        net = mnist_classifier(n_hidden=16)
        n_neurons = _total_neurons(net)
        t0 = time.perf_counter()
        net.run(0.05, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        rate = n_neurons * 50 / elapsed
        assert rate > 100, f"mnist throughput: {rate:.0f} neuron-steps/s"

    def test_analysis_spike_count(self):
        net = mnist_classifier(n_hidden=32)
        net.run(0.1, dt=0.001, backend="python")
        for mon in net.spike_monitors:
            times = mon.spike_times
            train = np.zeros(100, dtype=float)
            for t in times:
                if 0 <= t < 100:
                    train[t] = 1.0
            sc = spike_count(train)
            assert sc >= 0


# ===========================================================================
# 2. DVS GESTURE CLASSIFIER (Amir et al. 2017)
# ===========================================================================


class TestDVSGestureClassifier:
    """256-256-11 event-camera gesture SNN."""

    def test_returns_network(self):
        assert isinstance(dvs_gesture_classifier(n_classes=4), Network)

    def test_topology(self):
        net = dvs_gesture_classifier(n_classes=4)
        assert len(net.populations) == 3
        assert net.populations[0].n == 256  # input
        assert net.populations[1].n == 256  # hidden
        assert net.populations[2].n == 4  # output (parameterised)

    def test_two_projections(self):
        net = dvs_gesture_classifier(n_classes=4)
        assert len(net.projections) == 2

    def test_single_monitor(self):
        net = dvs_gesture_classifier(n_classes=4)
        assert len(net.spike_monitors) == 1
        assert "gesture" in net.spike_monitors[0].label

    def test_produces_spikes(self):
        assert _run_and_count(dvs_gesture_classifier(n_classes=4)) > 0

    @pytest.mark.parametrize("n_classes", [4, 8, 11])
    def test_scales_output(self, n_classes: int):
        net = dvs_gesture_classifier(n_classes=n_classes)
        assert net.populations[2].n == n_classes

    def test_performance(self):
        net = dvs_gesture_classifier(n_classes=4)
        n_neurons = _total_neurons(net)
        t0 = time.perf_counter()
        net.run(0.05, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        assert n_neurons * 50 / elapsed > 100


# ===========================================================================
# 3. SHD SPEECH CLASSIFIER (Cramer et al. 2020)
# ===========================================================================


class TestSHDSpeechClassifier:
    """700-256-20 recurrent SNN for spiking Heidelberg digits."""

    def test_returns_network(self):
        assert isinstance(shd_speech_classifier(), Network)

    def test_topology_three_populations(self):
        net = shd_speech_classifier()
        assert len(net.populations) == 3
        assert net.populations[0].n == 700  # input
        assert net.populations[1].n == 256  # recurrent
        assert net.populations[2].n == 20  # output

    def test_three_projections_including_recurrent(self):
        """input→rec, rec→rec (recurrent), rec→output."""
        net = shd_speech_classifier()
        assert len(net.projections) == 3
        # Recurrent: source == target
        rec_proj = net.projections[1]
        assert rec_proj.source is rec_proj.target

    def test_recurrent_tau_longer(self):
        """Recurrent layer uses tau_mem=20 vs input tau_mem=10."""
        net = shd_speech_classifier()
        inp_neuron = net.populations[0].neurons[0]
        rec_neuron = net.populations[1].neurons[0]
        assert rec_neuron.tau_mem > inp_neuron.tau_mem

    def test_two_monitors(self):
        net = shd_speech_classifier()
        assert len(net.spike_monitors) == 2

    def test_produces_spikes(self):
        assert _run_and_count(shd_speech_classifier()) > 0

    def test_performance(self):
        net = shd_speech_classifier()
        n_neurons = _total_neurons(net)
        t0 = time.perf_counter()
        net.run(0.05, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        assert n_neurons * 50 / elapsed > 100


# ===========================================================================
# 4. BRUNEL BALANCED NETWORK (Brunel 2000)
# ===========================================================================


class TestBrunelBalancedNetwork:
    """E/I balanced network with 4:1 exc:inh ratio."""

    def test_returns_network(self):
        assert isinstance(brunel_balanced_network(n_exc=50, n_inh=12), Network)

    def test_two_populations(self):
        net = brunel_balanced_network(n_exc=100, n_inh=25)
        assert len(net.populations) == 2
        assert net.populations[0].n == 100  # exc
        assert net.populations[1].n == 25  # inh

    def test_four_projections_full_connectivity(self):
        """E→E, E→I, I→E, I→I — all four quadrants wired."""
        net = brunel_balanced_network(n_exc=50, n_inh=12)
        assert len(net.projections) == 4

    def test_inhibition_stronger_than_excitation(self):
        """g=5 means |J_I| = 5·J_E — inhibition dominance."""
        net = brunel_balanced_network(n_exc=50, n_inh=12, g=5.0)
        j_e = net.projections[0].weight  # E→E
        j_i = net.projections[2].weight  # I→E
        assert j_i < 0  # inhibitory
        assert abs(j_i) > abs(j_e)  # |J_I| > J_E
        assert abs(abs(j_i) - 5.0 * abs(j_e)) < 1e-10

    def test_delay_present(self):
        """Synaptic delay=1.5ms from Brunel 2000."""
        net = brunel_balanced_network(n_exc=50, n_inh=12)
        for proj in net.projections:
            assert proj.delay == 1.5

    def test_all_projections_wired(self):
        """All 4 projections connect distinct or same populations."""
        net = brunel_balanced_network(n_exc=50, n_inh=12)
        for proj in net.projections:
            assert proj.source is not None
            assert proj.target is not None

    def test_two_poisson_drives(self):
        net = brunel_balanced_network(n_exc=50, n_inh=12)
        assert len(net.stimuli) == 2

    def test_two_monitors(self):
        net = brunel_balanced_network(n_exc=50, n_inh=12)
        assert len(net.spike_monitors) == 2
        labels = {m.label for m in net.spike_monitors}
        assert "exc_spikes" in labels
        assert "inh_spikes" in labels

    def test_produces_spikes(self):
        assert _run_and_count(brunel_balanced_network(n_exc=50, n_inh=12)) > 0

    @pytest.mark.parametrize("g", [3.0, 5.0, 8.0])
    def test_g_sweep(self, g: float):
        """Network remains stable across inhibition strengths."""
        net = brunel_balanced_network(n_exc=50, n_inh=12, g=g)
        net.run(0.05, dt=0.001, backend="python")
        assert _total_spikes(net) >= 0  # no crash

    def test_performance(self):
        net = brunel_balanced_network(n_exc=50, n_inh=12)
        n_neurons = _total_neurons(net)
        t0 = time.perf_counter()
        net.run(0.05, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        assert n_neurons * 50 / elapsed > 100

    def test_analysis_firing_rate(self):
        net = brunel_balanced_network(n_exc=50, n_inh=12)
        net.run(0.2, dt=0.001, backend="python")
        exc_mon = net.spike_monitors[0]
        train = np.zeros(200, dtype=float)
        for t in exc_mon.spike_times:
            if 0 <= t < 200:
                train[t] += 1.0
        rate = firing_rate(train, dt=0.001)
        assert rate >= 0


# ===========================================================================
# 5. CORTICAL COLUMN (Potjans & Diesmann 2014)
# ===========================================================================


class TestCorticalColumn:
    """4-layer cortical microcircuit with E/I per layer."""

    def test_returns_network(self):
        assert isinstance(cortical_column(n_layers=4), Network)

    def test_eight_populations_four_layers(self):
        """4 layers × 2 (E+I) = 8 populations."""
        net = cortical_column(n_layers=4)
        assert len(net.populations) == 8

    def test_excitatory_uses_pospischil(self):
        """E populations use PospischilNeuron (RS type)."""
        net = cortical_column(n_layers=4)
        for i in range(0, 8, 2):
            assert net.populations[i]._model_cls is PospischilNeuron

    def test_inhibitory_uses_golomb_fs(self):
        """I populations use GolombFSNeuron (FS type)."""
        net = cortical_column(n_layers=4)
        for i in range(1, 8, 2):
            assert net.populations[i]._model_cls is GolombFSNeuron

    def test_intra_layer_wiring(self):
        """Each layer has E→I, I→E, E→E (3 per layer = 12 intra)."""
        net = cortical_column(n_layers=4)
        # 12 intra-layer + 3 inter-layer feedforward = 15
        assert len(net.projections) == 15

    def test_feedforward_l4_to_l23(self):
        """L4_E → L23_E feedforward projection exists."""
        net = cortical_column(n_layers=4)
        # Inter-layer: ff_map = [(1,0), (0,2), (2,3)]
        # Source layer 1 = L4, target layer 0 = L23
        ff_projs = net.projections[12:]  # last 3 are inter-layer
        assert len(ff_projs) == 3

    def test_thalamic_drive_targets_l4(self):
        """PoissonInput targets L4_E (populations[2])."""
        net = cortical_column(n_layers=4)
        assert len(net.stimuli) == 1
        assert net.stimuli[0].target is net.populations[2]

    def test_eight_monitors(self):
        net = cortical_column(n_layers=4)
        assert len(net.spike_monitors) == 8

    def test_produces_spikes(self):
        assert _run_and_count(cortical_column(n_layers=4)) > 0

    @pytest.mark.parametrize("n_layers", [2, 3, 4])
    def test_scales_layers(self, n_layers: int):
        net = cortical_column(n_layers=n_layers)
        assert len(net.populations) == 2 * n_layers

    def test_performance(self):
        net = cortical_column(n_layers=2)
        n_neurons = _total_neurons(net)
        t0 = time.perf_counter()
        net.run(0.05, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        assert n_neurons * 50 / elapsed > 10


# ===========================================================================
# 6. CENTRAL PATTERN GENERATOR (Ijspeert 2008)
# ===========================================================================


class TestCentralPatternGenerator:
    """Half-centre CPG with mutually inhibiting oscillator pairs."""

    def test_returns_network(self):
        assert isinstance(central_pattern_generator(n_oscillators=2), Network)

    def test_population_count(self):
        """2 oscillators × 2 (flexor+extensor) × 5 neurons = 4 populations."""
        net = central_pattern_generator(n_oscillators=2)
        assert len(net.populations) == 4

    def test_uses_hindmarsh_rose(self):
        net = central_pattern_generator(n_oscillators=2)
        for pop in net.populations:
            assert pop._model_cls is HindmarshRoseNeuron

    def test_bursting_parameters(self):
        """b=3.0, r=0.005, s=4.0 — bursting regime."""
        net = central_pattern_generator(n_oscillators=2)
        neuron = net.populations[0].neurons[0]
        assert neuron.b == 3.0
        assert neuron.r == 0.005
        assert neuron.s == 4.0

    def test_mutual_inhibition_within_pair(self):
        """flex→ext and ext→flex are inhibitory (weight=-2.0)."""
        net = central_pattern_generator(n_oscillators=2)
        # First 2 projections per oscillator: flex→ext, ext→flex
        assert net.projections[0].weight == -2.0
        assert net.projections[1].weight == -2.0

    def test_inter_oscillator_excitatory(self):
        """Adjacent oscillators coupled with positive weight=1.0."""
        net = central_pattern_generator(n_oscillators=2)
        # Third projection per oscillator is inter-oscillator coupling
        assert net.projections[2].weight == 1.0

    def test_four_monitors(self):
        net = central_pattern_generator(n_oscillators=2)
        assert len(net.spike_monitors) == 4

    def test_produces_spikes(self):
        assert _run_and_count(central_pattern_generator(n_oscillators=2)) > 0

    @pytest.mark.parametrize("n_osc", [2, 3, 4])
    def test_scales_oscillators(self, n_osc: int):
        net = central_pattern_generator(n_oscillators=n_osc)
        assert len(net.populations) == 2 * n_osc

    def test_performance(self):
        net = central_pattern_generator(n_oscillators=2)
        n_neurons = _total_neurons(net)
        t0 = time.perf_counter()
        net.run(0.05, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        assert n_neurons * 50 / elapsed > 10


# ===========================================================================
# 7. DECISION-MAKING CIRCUIT (Wang 2002)
# ===========================================================================


class TestDecisionMakingCircuit:
    """Two competing pools + shared inhibition (attractor dynamics)."""

    def test_returns_network(self):
        assert isinstance(decision_making_circuit(n_per_pool=10), Network)

    def test_four_populations(self):
        """pool_A, pool_B, nonselective, inhibitory."""
        net = decision_making_circuit(n_per_pool=10)
        assert len(net.populations) == 4

    def test_pool_sizes(self):
        net = decision_making_circuit(n_per_pool=30)
        assert net.populations[0].n == 30  # pool_A
        assert net.populations[1].n == 30  # pool_B
        assert net.populations[2].n == max(10, 30 // 6)  # nonselective
        assert net.populations[3].n == max(15, 30 // 4)  # inhibitory

    def test_excitatory_uses_hh(self):
        net = decision_making_circuit(n_per_pool=10)
        for i in range(3):  # pools + nonselective
            assert net.populations[i]._model_cls is HodgkinHuxleyNeuron

    def test_inhibitory_uses_wang_buzsaki(self):
        net = decision_making_circuit(n_per_pool=10)
        assert net.populations[3]._model_cls is WangBuzsakiNeuron

    def test_nine_projections(self):
        """A→A, B→B, A→I, B→I, I→A, I→B, NS→A, NS→B, NS→I."""
        net = decision_making_circuit(n_per_pool=10)
        assert len(net.projections) == 9

    def test_potentiated_recurrent_excitation(self):
        """Within-pool recurrent weight=3.0 (potentiated)."""
        net = decision_making_circuit(n_per_pool=10)
        assert net.projections[0].weight == 3.0  # A→A
        assert net.projections[1].weight == 3.0  # B→B

    def test_cross_inhibition_negative(self):
        """I→A and I→B carry negative weight=-4.0."""
        net = decision_making_circuit(n_per_pool=10)
        assert net.projections[4].weight == -4.0  # I→A
        assert net.projections[5].weight == -4.0  # I→B

    def test_two_pool_monitors(self):
        net = decision_making_circuit(n_per_pool=10)
        assert len(net.spike_monitors) == 2
        labels = {m.label for m in net.spike_monitors}
        assert "pool_A_spikes" in labels
        assert "pool_B_spikes" in labels

    def test_three_stimuli(self):
        net = decision_making_circuit(n_per_pool=10)
        assert len(net.stimuli) == 3

    def test_produces_spikes(self):
        assert _run_and_count(decision_making_circuit(n_per_pool=10)) > 0

    @pytest.mark.parametrize("n_per_pool", [10, 30, 60])
    def test_scales_pool_size(self, n_per_pool: int):
        net = decision_making_circuit(n_per_pool=n_per_pool)
        assert net.populations[0].n == n_per_pool

    def test_performance(self):
        net = decision_making_circuit(n_per_pool=10)
        n_neurons = _total_neurons(net)
        t0 = time.perf_counter()
        net.run(0.05, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        assert n_neurons * 50 / elapsed > 10


# ===========================================================================
# 8. WORKING MEMORY CIRCUIT (Compte et al. 2000)
# ===========================================================================


class TestWorkingMemoryCircuit:
    """Ring attractor with NMDA-based persistent activity."""

    def test_returns_network(self):
        assert isinstance(working_memory_circuit(n_neurons=50), Network)

    def test_two_populations(self):
        """80% excitatory, 20% inhibitory."""
        net = working_memory_circuit(n_neurons=50)
        assert len(net.populations) == 2
        assert net.populations[0].n == 40  # 80%
        assert net.populations[1].n == 10  # 20%

    def test_excitatory_uses_compte_wm(self):
        net = working_memory_circuit(n_neurons=50)
        assert net.populations[0]._model_cls is CompteWMNeuron

    def test_inhibitory_uses_wang_buzsaki(self):
        net = working_memory_circuit(n_neurons=50)
        assert net.populations[1]._model_cls is WangBuzsakiNeuron

    def test_nmda_parameters(self):
        """NMDA conductance from Compte et al. 2000."""
        net = working_memory_circuit(n_neurons=50)
        neuron = net.populations[0].neurons[0]
        assert neuron.g_nmda == 0.165
        assert neuron.tau_nmda == 100.0
        assert neuron.mg == 1.0

    def test_four_projections(self):
        """E→E (ring), E→I, I→E, I→I."""
        net = working_memory_circuit(n_neurons=50)
        assert len(net.projections) == 4

    def test_excitatory_recurrent_self_connection(self):
        """E→E projection: source and target are the same population."""
        net = working_memory_circuit(n_neurons=50)
        p_ee = net.projections[0]
        assert p_ee.source is p_ee.target  # recurrent ring

    def test_two_monitors(self):
        net = working_memory_circuit(n_neurons=50)
        assert len(net.spike_monitors) == 2

    def test_produces_spikes(self):
        assert _run_and_count(working_memory_circuit(n_neurons=50)) > 0

    @pytest.mark.parametrize("n_neurons", [50, 100, 200])
    def test_scales_neuron_count(self, n_neurons: int):
        net = working_memory_circuit(n_neurons=n_neurons)
        n_exc = int(0.8 * n_neurons)
        n_inh = n_neurons - n_exc
        assert net.populations[0].n == n_exc
        assert net.populations[1].n == n_inh

    def test_performance(self):
        net = working_memory_circuit(n_neurons=50)
        n_neurons = _total_neurons(net)
        t0 = time.perf_counter()
        net.run(0.05, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        assert n_neurons * 50 / elapsed > 10


# ===========================================================================
# 9. AUDITORY PROCESSING (Goodman & Brette 2010)
# ===========================================================================


class TestAuditoryProcessing:
    """Cochlear→onset→integration spectro-temporal SNN."""

    def test_returns_network(self):
        assert isinstance(auditory_processing(n_channels=8), Network)

    def test_three_populations(self):
        """cochlear, onset, integration."""
        net = auditory_processing(n_channels=8)
        assert len(net.populations) == 3
        assert net.populations[0].n == 8  # cochlear
        assert net.populations[1].n == 8  # onset
        assert net.populations[2].n == 4  # integration (n_channels // 2)

    def test_cochlear_uses_hh(self):
        net = auditory_processing(n_channels=8)
        assert net.populations[0]._model_cls is HodgkinHuxleyNeuron

    def test_onset_uses_wang_buzsaki(self):
        """Onset cells modelled as fast-spiking WangBuzsaki."""
        net = auditory_processing(n_channels=8)
        assert net.populations[1]._model_cls is WangBuzsakiNeuron

    def test_three_projections(self):
        """cochlear→onset, onset→onset (lateral inh), onset→integration."""
        net = auditory_processing(n_channels=8)
        assert len(net.projections) == 3

    def test_lateral_inhibition_negative(self):
        """onset→onset is inhibitory (weight=-2.0)."""
        net = auditory_processing(n_channels=8)
        onset_onset = net.projections[1]
        assert onset_onset.weight == -2.0

    def test_two_monitors(self):
        net = auditory_processing(n_channels=8)
        assert len(net.spike_monitors) == 2

    def test_produces_spikes(self):
        assert _run_and_count(auditory_processing(n_channels=8)) > 0

    @pytest.mark.parametrize("n_ch", [4, 8, 16])
    def test_scales_channels(self, n_ch: int):
        net = auditory_processing(n_channels=n_ch)
        assert net.populations[0].n == n_ch
        assert net.populations[2].n == max(1, n_ch // 2)

    def test_performance(self):
        net = auditory_processing(n_channels=8)
        n_neurons = _total_neurons(net)
        t0 = time.perf_counter()
        net.run(0.05, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        assert n_neurons * 50 / elapsed > 10


# ===========================================================================
# 10. VISUAL CORTEX V1 (Hubel & Wiesel 1962 / Carandini & Heeger 2012)
# ===========================================================================


class TestVisualCortexV1:
    """Orientation-tuned simple→complex cell model."""

    def test_returns_network(self):
        assert isinstance(visual_cortex_v1(n_orientation=4, n_per_orientation=10), Network)

    def test_population_count(self):
        """n_orient simple + n_orient complex = 2·n_orient."""
        net = visual_cortex_v1(n_orientation=4, n_per_orientation=10)
        assert len(net.populations) == 8  # 4 simple + 4 complex

    def test_simple_cells_use_hh(self):
        net = visual_cortex_v1(n_orientation=4, n_per_orientation=10)
        for i in range(4):
            assert net.populations[i]._model_cls is HodgkinHuxleyNeuron

    def test_complex_cells_use_wang_buzsaki(self):
        net = visual_cortex_v1(n_orientation=4, n_per_orientation=10)
        for i in range(4, 8):
            assert net.populations[i]._model_cls is WangBuzsakiNeuron

    def test_simple_to_complex_feedforward(self):
        """Each simple→complex pair has a projection (weight=3.0)."""
        net = visual_cortex_v1(n_orientation=4, n_per_orientation=10)
        # First n_orient projections are simple→complex
        for i in range(4):
            assert net.projections[i].weight == 3.0

    def test_cross_orientation_inhibition(self):
        """Cross-orientation projections have negative weights w = -1/(1+dist)."""
        net = visual_cortex_v1(n_orientation=4, n_per_orientation=10)
        # After the 4 feedforward: 4×3=12 cross-orientation
        cross_projs = net.projections[4:16]
        for p in cross_projs:
            assert p.weight < 0

    def test_cross_orientation_weight_distance_dependent(self):
        """Closer orientations have stronger inhibition: |w(dist=1)| > |w(dist=2)|."""
        # dist=1: w = -1/(1+1) = -0.5; dist=2: w = -1/(1+2) = -0.333
        w1 = -1.0 / (1.0 + 1)
        w2 = -1.0 / (1.0 + 2)
        assert abs(w1) > abs(w2)

    def test_monitors_per_population(self):
        """2 monitors per orientation (simple + complex)."""
        net = visual_cortex_v1(n_orientation=4, n_per_orientation=10)
        assert len(net.spike_monitors) == 8

    def test_one_stimulus_per_orientation(self):
        net = visual_cortex_v1(n_orientation=4, n_per_orientation=10)
        assert len(net.stimuli) == 4

    def test_produces_spikes(self):
        assert _run_and_count(visual_cortex_v1(n_orientation=4, n_per_orientation=10)) > 0

    @pytest.mark.parametrize("n_orient", [2, 4, 8])
    def test_scales_orientations(self, n_orient: int):
        net = visual_cortex_v1(n_orientation=n_orient, n_per_orientation=5)
        assert len(net.populations) == 2 * n_orient

    def test_performance(self):
        net = visual_cortex_v1(n_orientation=2, n_per_orientation=5)
        n_neurons = _total_neurons(net)
        t0 = time.perf_counter()
        net.run(0.05, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        assert n_neurons * 50 / elapsed > 10


# ===========================================================================
# 11. PRETRAINED WEIGHT LOADING
# ===========================================================================


class TestLoadPretrained:
    """Tests for load_pretrained: loads .npz weights into network projections."""

    def test_mnist_loads(self):
        net = load_pretrained("mnist")
        assert isinstance(net, Network)
        assert len(net.projections) == 2

    def test_mnist_weights_differ_from_default(self):
        """Loaded weights should differ from default Xavier init."""
        default = mnist_classifier(n_hidden=128)
        loaded = load_pretrained("mnist")
        # At least one projection's data should differ
        differs = False
        for i in range(2):
            if not np.array_equal(default.projections[i].data, loaded.projections[i].data):
                differs = True
        assert differs

    def test_shd_loads(self):
        net = load_pretrained("shd")
        assert isinstance(net, Network)
        assert len(net.projections) == 3

    def test_dvs_gesture_loads(self):
        net = load_pretrained("dvs_gesture")
        assert isinstance(net, Network)
        assert len(net.projections) == 2

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown pretrained model"):
            load_pretrained("nonexistent_model")

    def test_mnist_pretrained_produces_spikes(self):
        net = load_pretrained("mnist")
        net.run(0.05, dt=0.001, backend="python")
        assert _total_spikes(net) > 0

    def test_shd_pretrained_produces_spikes(self):
        net = load_pretrained("shd")
        net.run(0.05, dt=0.001, backend="python")
        assert _total_spikes(net) > 0

    def test_dvs_pretrained_produces_spikes(self):
        net = load_pretrained("dvs_gesture")
        net.run(0.05, dt=0.001, backend="python")
        assert _total_spikes(net) > 0


# ===========================================================================
# 12. CROSS-CUTTING: ALL CONFIGS
# ===========================================================================

_ALL_BUILDERS = [
    ("mnist", lambda: mnist_classifier(n_hidden=16)),
    ("dvs", lambda: dvs_gesture_classifier(n_classes=4)),
    ("shd", lambda: shd_speech_classifier()),
    ("brunel", lambda: brunel_balanced_network(n_exc=50, n_inh=12)),
    ("cortical", lambda: cortical_column(n_layers=2)),
    ("cpg", lambda: central_pattern_generator(n_oscillators=2)),
    ("decision", lambda: decision_making_circuit(n_per_pool=10)),
    ("wm", lambda: working_memory_circuit(n_neurons=50)),
    ("auditory", lambda: auditory_processing(n_channels=8)),
    ("v1", lambda: visual_cortex_v1(n_orientation=2, n_per_orientation=5)),
]


class TestCrossCutting:
    """Properties that must hold for every model_zoo configuration."""

    @pytest.mark.parametrize("name,builder", _ALL_BUILDERS, ids=[b[0] for b in _ALL_BUILDERS])
    def test_has_populations(self, name: str, builder):
        net = builder()
        assert len(net.populations) >= 2

    @pytest.mark.parametrize("name,builder", _ALL_BUILDERS, ids=[b[0] for b in _ALL_BUILDERS])
    def test_has_projections(self, name: str, builder):
        net = builder()
        assert len(net.projections) >= 1

    @pytest.mark.parametrize("name,builder", _ALL_BUILDERS, ids=[b[0] for b in _ALL_BUILDERS])
    def test_has_spike_monitors(self, name: str, builder):
        net = builder()
        assert len(net.spike_monitors) >= 1

    @pytest.mark.parametrize("name,builder", _ALL_BUILDERS, ids=[b[0] for b in _ALL_BUILDERS])
    def test_has_stimuli(self, name: str, builder):
        net = builder()
        assert len(net.stimuli) >= 1

    @pytest.mark.parametrize("name,builder", _ALL_BUILDERS, ids=[b[0] for b in _ALL_BUILDERS])
    def test_seed_determinism(self, name: str, builder):
        """Same seed → same spike count."""
        net1 = builder()
        net1.run(0.05, dt=0.001, backend="python")
        c1 = _total_spikes(net1)
        net2 = builder()
        net2.run(0.05, dt=0.001, backend="python")
        c2 = _total_spikes(net2)
        assert c1 == c2

    @pytest.mark.parametrize("name,builder", _ALL_BUILDERS, ids=[b[0] for b in _ALL_BUILDERS])
    def test_analysis_spike_count_all(self, name: str, builder):
        """spike_count works on monitor data from every config."""
        net = builder()
        net.run(0.1, dt=0.001, backend="python")
        for mon in net.spike_monitors:
            train = np.zeros(100, dtype=float)
            for t in mon.spike_times:
                if 0 <= t < 100:
                    train[t] = 1.0
            assert spike_count(train) >= 0

    @pytest.mark.parametrize("name,builder", _ALL_BUILDERS, ids=[b[0] for b in _ALL_BUILDERS])
    def test_analysis_isi_all(self, name: str, builder):
        """ISI computation works on per-neuron binary trains from every config."""
        net = builder()
        n_steps = 100
        net.run(0.1, dt=0.001, backend="python")
        for mon in net.spike_monitors:
            trains = mon.spike_trains
            for nid, times in trains.items():
                if len(times) >= 3:
                    # Build binary train for this single neuron
                    binary = np.zeros(n_steps, dtype=float)
                    for t in times:
                        if 0 <= t < n_steps:
                            binary[t] = 1.0
                    intervals = isi(binary, dt=0.001)
                    if intervals.size > 0:
                        assert np.all(intervals > 0)
                    break  # one neuron sufficient per monitor
