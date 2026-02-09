"""Tests for surrogate gradient engine (Python bridge)."""

import numpy as np

from sc_neurocore_engine import DifferentiableDenseLayer, SurrogateLif


class TestSurrogateLif:
    def test_forward_backward_cycle(self):
        lif = SurrogateLif(surrogate="fast_sigmoid", k=25.0)
        _spike, _v = lif.forward(leak_k=20, gain_k=256, i_t=128)
        grad = lif.backward(1.0)
        assert isinstance(grad, float)
        assert grad != 0.0

    def test_clear_trace(self):
        lif = SurrogateLif(surrogate="arctan", k=10.0)
        for _ in range(10):
            lif.forward(20, 256, 128)
        assert lif.trace_len() == 10
        lif.clear_trace()
        assert lif.trace_len() == 0


class TestDifferentiableDenseLayer:
    def test_train_step(self):
        layer = DifferentiableDenseLayer(
            n_inputs=8,
            n_neurons=4,
            length=1024,
            surrogate="fast_sigmoid",
            k=25.0,
        )
        out1 = np.array(layer.forward([0.5] * 8))

        grad_in, grad_w = layer.backward([1.0] * 4)
        assert grad_in.shape == (8,)
        assert grad_w.shape == (4, 8)
        layer.update_weights(grad_w, lr=0.01)

        out2 = np.array(layer.forward([0.5] * 8))
        assert not np.allclose(out1, out2)
