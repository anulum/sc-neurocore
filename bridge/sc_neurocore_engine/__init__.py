# SPDX-License-Identifier: AGPL-3.0-or-later
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li

"""SC-NeuroCore Engine — Drop-in replacement for v2 hot paths."""

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
        StdpSynapse,
        SCPNMetrics,
        BitStreamTensor,
        BrunelNetwork,
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
from .hdc import HDCVector
from .petri_net import PetriNetEngine

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
    "StdpSynapse",
    "SCPNMetrics",
    "BitStreamTensor",
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
    "HDCVector",
    "PetriNetEngine",
    "BrunelNetwork",
]
