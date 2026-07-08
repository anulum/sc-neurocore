# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — sc_neurocore.quantum -- Tier: research (experimental /

"""sc_neurocore.quantum -- Tier: research (experimental / research)."""

__tier__ = "research"

from .hybrid import QuantumStochasticLayer
from .hybrid_pipeline import HybridQuantumClassicalPipeline
from .noise_models import HeronR2NoiseModel, HeronR2NoiseParams
from .param_shift import ParameterShiftOptimizer, parameter_shift_gradient
from .sc_quantum_compiler import (
    QuantumGate,
    SCQuantumCircuit,
    compile_sc_layer,
    compile_sc_multiply,
    prob_to_ry_angle,
    ry_gate,
    sc_prob_to_statevector,
    statevector_to_prob,
)

__all__ = [
    "QuantumStochasticLayer",
    "HeronR2NoiseModel",
    "HeronR2NoiseParams",
    "ParameterShiftOptimizer",
    "parameter_shift_gradient",
    "HybridQuantumClassicalPipeline",
    "QuantumGate",
    "SCQuantumCircuit",
    "compile_sc_layer",
    "compile_sc_multiply",
    "prob_to_ry_angle",
    "ry_gate",
    "sc_prob_to_statevector",
    "statevector_to_prob",
]
