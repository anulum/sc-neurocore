# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — accel.julia package init

"""Julia acceleration namespace.

This tree is mixed quality by design:

- a small subset is wired from maintained Python loaders and covered by tests
- a much larger subset consists of research mirrors or source transcripts

Only Julia files explicitly loaded by maintained Python code should be treated
as authoritative execution paths.
"""

AUTHORITATIVE_JULIA_ENTRYPOINTS: tuple[str, ...] = (
    "_native/learning_bridge.jl",
    "chiplet/kl_refine.jl",
    "fault_injection/fault_injection.jl",
    "neurons/rk4_neurons.jl",
    "world_model/predictive_model.jl",
)

NON_AUTHORITATIVE_JULIA_MIRROR_GLOBS: tuple[str, ...] = (
    "studio/*.jl",
    "analysis/*.jl",
    "analysis_spike_stats/*.jl",
    "edge/*.jl",
    "model_zoo/*.jl",
)

__all__ = [
    "AUTHORITATIVE_JULIA_ENTRYPOINTS",
    "NON_AUTHORITATIVE_JULIA_MIRROR_GLOBS",
]
