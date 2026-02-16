"""
PyTorch tensor bridge for SC-NeuroCore.

Provides NumPy <-> Torch conversion helpers and an ``nn.Module`` wrapper
for ``VectorizedSCLayer`` so they can be plugged into standard PyTorch
training loops.

All imports are guarded — the module is importable even without PyTorch.
"""

from __future__ import annotations

import numpy as np

try:
    import torch  # pragma: no cover
    import torch.nn as nn  # pragma: no cover

    HAS_TORCH = True  # pragma: no cover
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    HAS_TORCH = False


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def numpy_to_torch(arr: np.ndarray, dtype=None, device: str = "cpu"):  # pragma: no cover
    """Convert a NumPy array to a PyTorch tensor."""
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for numpy_to_torch()")
    t = torch.from_numpy(np.ascontiguousarray(arr))
    if dtype is not None:
        t = t.to(dtype)
    return t.to(device)


def torch_to_numpy(tensor) -> np.ndarray:  # pragma: no cover
    """Convert a PyTorch tensor to a NumPy array (detached, on CPU)."""
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for torch_to_numpy()")
    return tensor.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# nn.Module wrapper
# ---------------------------------------------------------------------------

if HAS_TORCH:  # pragma: no cover

    class SCDenseLayerTorch(nn.Module):  # pragma: no cover
        """
        ``nn.Module`` wrapper around ``VectorizedSCLayer``.

        Forward pass converts tensor -> NumPy -> SC forward -> tensor.
        SC layers are non-differentiable by nature, but the wrapper lets
        them sit inside a PyTorch pipeline.
        """

        def __init__(self, n_inputs: int, n_neurons: int, length: int = 1024):
            super().__init__()
            from ..layers.vectorized_layer import VectorizedSCLayer

            self._sc_layer = VectorizedSCLayer(
                n_inputs=n_inputs, n_neurons=n_neurons, length=length, use_gpu=False
            )
            self.register_buffer(
                "sc_weights", torch.from_numpy(self._sc_layer.weights.copy())
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            squeeze = x.dim() == 1
            if squeeze:
                x = x.unsqueeze(0)

            results = []
            for i in range(x.shape[0]):
                out_np = self._sc_layer.forward(torch_to_numpy(x[i]))
                results.append(torch.from_numpy(out_np).to(x.device))

            out = torch.stack(results)
            return out.squeeze(0) if squeeze else out

else:

    class SCDenseLayerTorch:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is required for SCDenseLayerTorch. "
                "Install it with: pip install torch"
            )
