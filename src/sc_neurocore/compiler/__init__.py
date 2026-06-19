# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Compiler Package Init

"""Public compiler package exports for equation, IR, MLIR, and precision tooling."""

from .adaptive_precision import (
    LayerPrecision,
    SynapsePrecision,
    analyze_sensitivity,
    assign_lengths,
    assign_synapse_precisions,
    auto_tune_synapse_precisions,
    precision_plan_manifest,
    write_precision_formal_evidence_bundle,
)
from .equation_compiler import Q88, compile_to_verilog, equation_to_fpga
from .ir_type_checker import (
    IREdge,
    IRNode,
    IRTypeError,
    SignalType,
    check_ir_types,
)
from .live_control import (
    MMIOUpdateSpec,
    ParameterBankSpec,
    TrapSpec,
)
from .mlir_emitter import MLIRBundle, MLIREmitter, generate_mlir_bundle
from .pipeline import CompilerPipeline
from .quantizer import (
    Q8_8,
    Q16_16,
    CompiledBlockFloatingDense,
    CompiledMixedDense,
    PrecisionEnvelopeReport,
    PrecisionTrapReport,
    QFormat,
    QFormatMixed,
    compile_dense_block_floating,
    compile_dense_mixed_precision,
    dequantize,
    dequantize_weights,
    q_weights_to_sc_probabilities,
    quantization_error,
    quantize_weights,
)

__all__ = [
    "compile_to_verilog",
    "equation_to_fpga",
    "Q88",
    "CompilerPipeline",
    "MLIRBundle",
    "MLIREmitter",
    "generate_mlir_bundle",
    "CompiledBlockFloatingDense",
    "CompiledMixedDense",
    "PrecisionEnvelopeReport",
    "PrecisionTrapReport",
    "QFormat",
    "QFormatMixed",
    "Q8_8",
    "Q16_16",
    "compile_dense_block_floating",
    "compile_dense_mixed_precision",
    "dequantize",
    "quantize_weights",
    "dequantize_weights",
    "q_weights_to_sc_probabilities",
    "quantization_error",
    "check_ir_types",
    "IRNode",
    "IREdge",
    "IRTypeError",
    "SignalType",
    "LayerPrecision",
    "SynapsePrecision",
    "analyze_sensitivity",
    "assign_lengths",
    "assign_synapse_precisions",
    "auto_tune_synapse_precisions",
    "precision_plan_manifest",
    "write_precision_formal_evidence_bundle",
    "MMIOUpdateSpec",
    "ParameterBankSpec",
    "TrapSpec",
]
