# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio federation verbs

"""The SC-NeuroCore studio's verbs, expressed on the locked platform contract.

SC-NeuroCore is the neuromorphic stochastic-computing + FPGA laboratory of the
ecosystem, so its verbs are the stages of its own pipeline (encode → simulate →
analyse → compile → synthesise) plus its measurement surfaces. Six are drawn from
the shared :data:`scpn_studio_platform.verbs.CORE_VERBS` spine (``simulate``,
``analyse``, ``benchmark``, ``validate``, ``compile``, ``synthesise``); two are
domain-distinctive (``encode`` — float → stochastic bitstream; ``deploy`` — load a
synthesised bitstream onto FPGA silicon).

``deploy`` is the verb that carries SC-NeuroCore's distinctive contribution to the
contract: it produces ``hardware-validated`` evidence on the ``fpga`` substrate —
a bit-exact co-simulation match between the Q8.8 fixed-point reference and the
synthesised RTL, proven on a Zynq XC7Z020. That is the silicon-grade tier of the
evidence ladder, distinct from a ``measured`` software benchmark.
"""

from __future__ import annotations

from scpn_studio_platform.verbs import (
    Fidelity,
    SafetyTier,
    SideEffect,
    Timing,
    TimingClass,
    Verb,
)

STUDIO_ID = "sc-neurocore"
"""The studio identifier this vertical implements (also the federation name)."""

# ── evidence schema names this studio emits (studio.*.v1) ──────────────
BITSTREAM_ENCODING_SCHEMA = "studio.bitstream-encoding.v1"
SC_INFERENCE_SCHEMA = "studio.sc-inference.v1"
SPIKE_ANALYSIS_SCHEMA = "studio.spike-analysis.v1"
BACKEND_BENCHMARK_SCHEMA = "studio.backend-benchmark.v1"
COSIM_PARITY_SCHEMA = "studio.cosim-parity.v1"
RTL_COMPILATION_SCHEMA = "studio.rtl-compilation.v1"
FPGA_DEPLOYMENT_SCHEMA = "studio.fpga-deployment.v1"


# ── core-spine verbs ───────────────────────────────────────────────────
SIMULATE = Verb(
    name="simulate",
    safety_tier=SafetyTier.RESEARCH,
    side_effect=SideEffect.SIMULATED,
    timing=Timing(TimingClass.BATCH),
    fidelity=Fidelity.FIRST_PRINCIPLES,
    produces=(SC_INFERENCE_SCHEMA,),
    backends=("numpy", "rust"),
)
"""Run a stochastic-computing spiking network forward pass."""

ANALYSE = Verb(
    name="analyse",
    safety_tier=SafetyTier.RESEARCH,
    side_effect=SideEffect.READ_ONLY,
    timing=Timing(TimingClass.BATCH),
    produces=(SPIKE_ANALYSIS_SCHEMA,),
    backends=("numpy", "rust"),
)
"""Extract spike-train statistics (rate, correlation, information, decoding)."""

BENCHMARK = Verb(
    name="benchmark",
    safety_tier=SafetyTier.RESEARCH,
    side_effect=SideEffect.SIMULATED,
    timing=Timing(TimingClass.BATCH),
    produces=(BACKEND_BENCHMARK_SCHEMA,),
    backends=("numpy", "rust"),
)
"""Measure per-backend SC inference throughput against the NumPy floor."""

VALIDATE = Verb(
    name="validate",
    safety_tier=SafetyTier.RESEARCH,
    side_effect=SideEffect.READ_ONLY,
    timing=Timing(TimingClass.BATCH),
    produces=(COSIM_PARITY_SCHEMA,),
    backends=("python",),
)
"""Check a fixed-point reference against the synthesised RTL by co-simulation."""

COMPILE = Verb(
    name="compile",
    safety_tier=SafetyTier.RESEARCH,
    side_effect=SideEffect.READ_ONLY,
    timing=Timing(TimingClass.BATCH),
    produces=(RTL_COMPILATION_SCHEMA,),
    backends=("python",),
)
"""Lower a neuron/network model to synthesisable Verilog RTL (SC-NIR → SV)."""

SYNTHESISE = Verb(
    name="synthesise",
    safety_tier=SafetyTier.RESEARCH,
    side_effect=SideEffect.SIMULATED,
    timing=Timing(TimingClass.BATCH),
    produces=(FPGA_DEPLOYMENT_SCHEMA,),
    backends=("python",),
)
"""Synthesise RTL to an FPGA/ASIC target and record resource/timing closure."""


# ── domain-distinctive verbs ───────────────────────────────────────────
ENCODE = Verb(
    name="encode",
    safety_tier=SafetyTier.RESEARCH,
    side_effect=SideEffect.READ_ONLY,
    timing=Timing(TimingClass.INTERACTIVE),
    produces=(BITSTREAM_ENCODING_SCHEMA,),
    backends=("numpy", "rust"),
)
"""Encode float values to stochastic bitstreams (LFSR / Sobol / Halton)."""

DEPLOY = Verb(
    name="deploy",
    safety_tier=SafetyTier.RESEARCH,
    side_effect=SideEffect.LIVE_HARDWARE,
    timing=Timing(TimingClass.BATCH),
    produces=(FPGA_DEPLOYMENT_SCHEMA,),
    backends=("python",),
)
"""Load a synthesised bitstream onto FPGA silicon (PYNQ-Z2) and record bit-exactness."""


NEUROCORE_VERBS: tuple[Verb, ...] = (
    ENCODE,
    SIMULATE,
    ANALYSE,
    BENCHMARK,
    VALIDATE,
    COMPILE,
    SYNTHESISE,
    DEPLOY,
)
"""Every verb the SC-NeuroCore studio advertises, in manifest order."""


def core_verbs() -> tuple[Verb, ...]:
    """Return the SC-NeuroCore verbs drawn from the shared core spine.

    Returns
    -------
    tuple[Verb, ...]
        The subset of :data:`NEUROCORE_VERBS` whose names are in the platform
        :data:`scpn_studio_platform.verbs.CORE_VERBS`.
    """
    return tuple(verb for verb in NEUROCORE_VERBS if verb.is_core)


def domain_verbs() -> tuple[Verb, ...]:
    """Return the SC-NeuroCore verbs distinctive to a neuromorphic studio.

    Returns
    -------
    tuple[Verb, ...]
        The subset of :data:`NEUROCORE_VERBS` not present in the platform core
        spine (``encode``, ``deploy``).
    """
    return tuple(verb for verb in NEUROCORE_VERBS if not verb.is_core)


def evidence_schemas() -> tuple[str, ...]:
    """Return the distinct evidence schemas every SC-NeuroCore verb emits.

    Returns
    -------
    tuple[str, ...]
        The sorted, de-duplicated union of each verb's ``produces`` schemas.
    """
    schemas = {schema for verb in NEUROCORE_VERBS for schema in verb.produces}
    return tuple(sorted(schemas))
