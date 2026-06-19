# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Compiler intelligence facade

"""Advanced compiler features facade.

Re-exports intelligence capabilities from modular sub-packages.
"""

from __future__ import annotations

from .auto_quantization import (
    QuantSweepResult,
    auto_quantisation_sweep,
    format_quantisation_report,
)
from .bitstream_flow import (
    generate_oss_makefile,
)
from .compilation_cache import (
    CompilationCache,
)
from .debug_probes import (
    DebugProbeSpec,
    insert_debug_probes,
)
from .dispatch_planner import (
    DispatchPlan,
    plan_heterogeneous_dispatch,
)
from .dvs_bridge import (
    generate_dvs_aer_bridge,
)
from .hls_export import (
    generate_hls_cpp,
)
from .learning_export import (
    OnChipLearningParams,
    export_learning_config,
    generate_learning_params,
)
from .mxfp_encoding import (
    FP8_E4M3,
    MXFP4,
    MXFP6,
    MXFP8_E4M3,
    MXFP8_E5M2,
    MXFPConfig,
    mxfp_decode_block,
    mxfp_encode_block,
)
from .network_optimizer import (
    TopologyPlan,
    optimize_network_topology,
)
from .nir_import import (
    NIRGraph,
    import_nir_graph,
)
from .posit_arithmetic import (
    POSIT8_0,
    POSIT8_1,
    POSIT16_1,
    POSIT16_2,
    PositConfig,
    posit_decode,
    posit_encode,
)
from .reconfig_planner import (
    ReconfigPartition,
    plan_partial_reconfiguration,
)
from .target_recommender import (
    TargetRecommendation,
    recommend_target,
)
from .tcl_gen import (
    generate_tcl_project,
)
from .timescale_partitioner import (
    TimescalePartition,
    partition_timescales,
)
from .vhdl_emitter import (
    verilog_to_vhdl_wrapper,
)
from .weight_noise import (
    WeightNoiseProfile,
    create_noise_profile,
    inject_weight_noise,
)
from .weight_rom import (
    generate_weight_rom,
)

__all__ = [
    "CompilationCache",
    "DebugProbeSpec",
    "DispatchPlan",
    "FP8_E4M3",
    "MXFPConfig",
    "MXFP4",
    "MXFP6",
    "MXFP8_E4M3",
    "MXFP8_E5M2",
    "NIRGraph",
    "OnChipLearningParams",
    "POSIT8_0",
    "POSIT8_1",
    "POSIT16_1",
    "POSIT16_2",
    "PositConfig",
    "QuantSweepResult",
    "ReconfigPartition",
    "TargetRecommendation",
    "TimescalePartition",
    "TopologyPlan",
    "WeightNoiseProfile",
    "auto_quantisation_sweep",
    "create_noise_profile",
    "export_learning_config",
    "format_quantisation_report",
    "generate_dvs_aer_bridge",
    "generate_hls_cpp",
    "generate_learning_params",
    "generate_oss_makefile",
    "generate_tcl_project",
    "generate_weight_rom",
    "import_nir_graph",
    "inject_weight_noise",
    "insert_debug_probes",
    "mxfp_decode_block",
    "mxfp_encode_block",
    "optimize_network_topology",
    "partition_timescales",
    "plan_heterogeneous_dispatch",
    "plan_partial_reconfiguration",
    "posit_decode",
    "posit_encode",
    "recommend_target",
    "verilog_to_vhdl_wrapper",
]
