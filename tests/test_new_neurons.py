# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations


class TestAdExNeuron:
    def test_fires_with_input(self):
        from sc_neurocore.neurons.adex import AdExNeuron

        n = AdExNeuron()
        spikes = sum(n.step(500.0) for _ in range(2000))
        assert spikes > 0

    def test_adaptation(self):
        from sc_neurocore.neurons.adex import AdExNeuron

        n = AdExNeuron()
        for _ in range(1000):
            n.step(400.0)
        assert n.w > 0, "adaptation variable must grow"

    def test_reset(self):
        from sc_neurocore.neurons.adex import AdExNeuron

        n = AdExNeuron()
        for _ in range(100):
            n.step(500.0)
        n.reset()
        assert abs(n.v - n.v_rest) < 1e-10
        assert abs(n.w) < 1e-10


class TestExpIFNeuron:
    def test_fires(self):
        from sc_neurocore.neurons.adex import ExpIFNeuron

        n = ExpIFNeuron()
        spikes = sum(n.step(500.0) for _ in range(2000))
        assert spikes > 0

    def test_no_fire_without_input(self):
        from sc_neurocore.neurons.adex import ExpIFNeuron

        n = ExpIFNeuron()
        spikes = sum(n.step(0.0) for _ in range(500))
        assert spikes == 0


class TestLapicqueNeuron:
    def test_fires(self):
        from sc_neurocore.neurons.adex import LapicqueNeuron

        n = LapicqueNeuron()
        spikes = sum(n.step(5.0) for _ in range(200))
        assert spikes > 0

    def test_reset(self):
        from sc_neurocore.neurons.adex import LapicqueNeuron

        n = LapicqueNeuron()
        for _ in range(50):
            n.step(5.0)
        n.reset()
        assert abs(n.v) < 1e-10


class TestAlphaNeuron:
    def test_fires_with_excitation(self):
        from sc_neurocore.neurons.adex import AlphaNeuron

        n = AlphaNeuron()
        spikes = sum(n.step(5.0) for _ in range(200))
        assert spikes > 0

    def test_inhibition_blocks(self):
        from sc_neurocore.neurons.adex import AlphaNeuron

        n = AlphaNeuron()
        spikes = sum(n.step(2.0, 10.0) for _ in range(200))
        assert spikes == 0, "strong inhibition should block firing"


torch = pytest.importorskip("torch")


class TestTrainingCells:
    def test_expif_cell(self):
        from sc_neurocore.training.snn_modules import ExpIFCell

        cell = ExpIFCell()
        v = torch.zeros(4)
        spike, v = cell(torch.ones(4) * 2.0, v)
        assert spike.shape == (4,)

    def test_adex_cell(self):
        from sc_neurocore.training.snn_modules import AdExCell

        cell = AdExCell()
        v = torch.zeros(4)
        w = torch.zeros(4)
        spike, v, w = cell(torch.ones(4) * 2.0, v, w)
        assert spike.shape == (4,)

    def test_lapicque_cell(self):
        from sc_neurocore.training.snn_modules import LapicqueCell

        cell = LapicqueCell()
        v = torch.zeros(4)
        spike, v = cell(torch.ones(4) * 5.0, v)
        assert spike.shape == (4,)

    def test_alpha_cell(self):
        from sc_neurocore.training.snn_modules import AlphaCell

        cell = AlphaCell()
        v = torch.zeros(4)
        i_exc = torch.zeros(4)
        i_inh = torch.zeros(4)
        spike, i_exc, i_inh, v = cell(torch.ones(4), torch.zeros(4), i_exc, i_inh, v)
        assert spike.shape == (4,)

    def test_second_order_lif(self):
        from sc_neurocore.training.snn_modules import SecondOrderLIFCell

        cell = SecondOrderLIFCell()
        v = torch.zeros(4)
        a = torch.zeros(4)
        spike, a, v = cell(torch.ones(4) * 2.0, a, v)
        assert spike.shape == (4,)


class TestLavaBridge:
    def test_export_weights(self):
        import numpy as np
        from sc_neurocore.integrations.lava_bridge import export_weights_loihi

        w = np.array([[0.0, 0.5, 1.0], [0.25, 0.75, 0.5]])
        loihi_w = export_weights_loihi(w, weight_bits=8)
        assert loihi_w.dtype == np.int32
        assert loihi_w.shape == (2, 3)
        assert loihi_w[0, 0] == -127  # 0.0 → -1.0 → -127
        assert loihi_w[0, 2] == 127  # 1.0 → +1.0 → +127

    def test_converter(self):
        from sc_neurocore.integrations.lava_bridge import SCtoLavaConverter, LoihiNetworkConfig

        converter = SCtoLavaConverter(weight_bits=8)

        class FakeLayer:
            weights = [[0.5, 0.3], [0.7, 0.1], [0.9, 0.4]]

        config = converter.convert_dense_layer(FakeLayer())
        assert isinstance(config, LoihiNetworkConfig)
        assert config.n_inputs == 2
        assert config.n_outputs == 3

    def test_threshold_conversion(self):
        from sc_neurocore.integrations.lava_bridge import loihi_threshold_from_sc

        t = loihi_threshold_from_sc(1.0, weight_bits=8)
        assert t == 127
