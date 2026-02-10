"""SC-NeuroCore Engine v3.6 — Drop-in replacement for v2 hot paths."""

try:
    from sc_neurocore_engine.sc_neurocore_engine import (
        __version__,
        simd_tier,
        set_num_threads,
        pack_bitstream,
        unpack_bitstream,
        popcount,
        pack_bitstream_numpy,
        popcount_numpy,
        unpack_bitstream_numpy,
        batch_lif_run,
        batch_lif_run_multi,
        batch_lif_run_varying,
        batch_encode,
        batch_encode_numpy,
        Lfsr16,
        BitstreamEncoder,
        FixedPointLif,
        DenseLayer,
        SCPNMetrics,
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
from .ir import ScGraph, ScGraphBuilder, parse_ir

__all__ = [
    "__version__",
    "simd_tier",
    "set_num_threads",
    "pack_bitstream",
    "unpack_bitstream",
    "popcount",
    "pack_bitstream_numpy",
    "popcount_numpy",
    "unpack_bitstream_numpy",
    "batch_lif_run",
    "batch_lif_run_multi",
    "batch_lif_run_varying",
    "batch_encode",
    "batch_encode_numpy",
    "Lfsr16",
    "BitstreamEncoder",
    "FixedPointLif",
    "DenseLayer",
    "SCPNMetrics",
    "VectorizedSCLayer",
    "FixedPointLIFNeuron",
    "SurrogateLif",
    "DifferentiableDenseLayer",
    "StochasticAttention",
    "StochasticGraphLayer",
    "KuramotoSolver",
    "ScGraph",
    "ScGraphBuilder",
    "parse_ir",
]
