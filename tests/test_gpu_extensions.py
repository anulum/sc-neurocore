"""
Phase 4 tests: GPU conv2d, batch LIF, torch bridge.

Tests are skipped if CuPy/PyTorch are not installed.
"""

import numpy as np
import pytest

from sc_neurocore.accel.gpu_backend import gpu_batch_lif_step
from sc_neurocore.accel.gpu_conv2d import gpu_conv2d_forward


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


class TestGPUConv2D:
    """Test the gpu_conv2d_forward function (runs on NumPy when no GPU)."""

    def test_gpu_conv2d_shape(self):
        np.random.seed(42)
        img = np.random.random((3, 8, 8))
        kernels = np.random.random((4, 3, 3, 3))
        out = gpu_conv2d_forward(img, kernels, stride=1, padding=1)
        assert out.shape == (4, 8, 8)

    def test_gpu_conv2d_no_padding(self):
        np.random.seed(42)
        img = np.random.random((1, 6, 6))
        kernels = np.random.random((2, 1, 3, 3))
        out = gpu_conv2d_forward(img, kernels, stride=1, padding=0)
        assert out.shape == (2, 4, 4)


class TestGPUBatchLIF:
    """Test vectorized batch LIF step on CPU (falls back from GPU)."""

    def test_batch_lif_step(self):
        N = 5
        currents = np.array([0.08, 0.06, 0.10, 0.04, 0.09])
        v = np.zeros(N)
        v_rest = np.zeros(N)
        v_reset = np.zeros(N)
        v_threshold = np.ones(N)
        dt_over_tau = np.full(N, 1.0 / 20.0)
        resistance_dt = np.ones(N)

        spikes, v_new = gpu_batch_lif_step(
            currents, v, v_rest, v_reset, v_threshold, dt_over_tau, resistance_dt
        )
        assert spikes.shape == (N,)
        assert v_new.shape == (N,)
        assert spikes.dtype == np.uint8


class TestTorchBridge:
    """Test PyTorch bridge — skip if torch not installed."""

    def test_import_guard(self):
        """Module should be importable even without torch."""
        from sc_neurocore.accel.torch_bridge import HAS_TORCH, SCDenseLayerTorch
        if not HAS_TORCH:
            with pytest.raises(ImportError):
                SCDenseLayerTorch(4, 3)

    @pytest.mark.skipif(
        not _torch_available(), reason="PyTorch not installed"
    )
    def test_forward_pass(self):
        import torch
        from sc_neurocore.accel.torch_bridge import SCDenseLayerTorch

        layer = SCDenseLayerTorch(n_inputs=4, n_neurons=3, length=128)
        x = torch.rand(4)
        out = layer(x)
        assert out.shape == (3,)
