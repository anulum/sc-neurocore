"""SC-NeuroCore Engine v3.0 — Drop-in replacement for v2 hot paths."""

try:
    from sc_neurocore_engine.sc_neurocore_engine import (
        __version__,
        simd_tier,
        pack_bitstream,
        unpack_bitstream,
        popcount,
        Lfsr16,
        BitstreamEncoder,
        FixedPointLif,
    )
except ImportError as exc:
    raise ImportError(
        "sc_neurocore_engine native module not found. "
        "Build with: cd engine && maturin develop --release"
    ) from exc

from .layers import VectorizedSCLayer
from .neurons import FixedPointLIFNeuron
from .grad import SurrogateLif, DifferentiableDenseLayer
from .attention import StochasticAttention
from .graphs import StochasticGraphLayer
from .scpn import KuramotoSolver

__all__ = [
    "__version__",
    "simd_tier",
    "pack_bitstream",
    "unpack_bitstream",
    "popcount",
    "Lfsr16",
    "BitstreamEncoder",
    "FixedPointLif",
    "VectorizedSCLayer",
    "FixedPointLIFNeuron",
    "SurrogateLif",
    "DifferentiableDenseLayer",
    "StochasticAttention",
    "StochasticGraphLayer",
    "KuramotoSolver",
]
