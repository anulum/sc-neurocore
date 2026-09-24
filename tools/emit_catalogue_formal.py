#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Emit SymbiYosys jobs for declared dual-axis perfect catalogue models

"""Emit schema→RTL + formal wrappers + ``.sby`` for declared dual-axis perfect models.

Models whose descriptors declare ``is_perfect`` (science S5 + silicon ≥ target
H) are enrolled; enrolment follows the declaration and does not verify it. Each
generated job checks bounded safety properties through public ports only —
reset values and, where configured, event silence or a spike reset packet — on
the committed equation-compiler RTL (Q8.8 by default, with explicit per-schema
overrides), without hierarchical ``uut.*`` probes so ``default_nettype none``
stays clean. Curated jobs keep their hand-written harness.

``inventory.json`` states for every job what is checked, under which
assumptions, to which depth, and what the job does not establish: equivalence
with the model or the bit-true kernel, behaviour beyond the depth, and, for a
property guarded by an event, whether the event was shown reachable.

Usage
-----
From the SC-NEUROCORE repo root::

    .venv/bin/python tools/emit_catalogue_formal.py
    .venv/bin/python tools/emit_catalogue_formal.py --run-sby

Outputs land under ``hdl/formal/catalogue/``.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil

# Only the PATH-resolved SymbiYosys executable is invoked.
import subprocess  # nosec B404
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sc_neurocore.neurons.equation_builder import EquationNeuron

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "hdl" / "formal" / "catalogue"
DESC_DIR = ROOT / "src" / "sc_neurocore" / "neurons" / "model_descriptors"

# Schema stem used by UniversalNeuron.from_schema for each perfect class.
CLASS_TO_SCHEMA: dict[str, str] = {
    "AdaptiveThresholdIFNeuron": "adaptive_threshold_if",
    "AlphaNeuron": "alpha",
    "SCClippedLogisticBurstingMapNeuron": "sc_clipped_logistic_bursting_map",
    "ConnorStevensNeuron": "connor_stevens",
    "CourageNekorkinMapNeuron": "courage_nekorkin_map",
    "DPINeuron": "dpi_neuron",
    "ErmentroutKopellMapNeuron": "ermentrout_kopell_map_neuron",
    "ErmentroutKopellPopulation": "ermentrout_kopell_pop",
    "EscapeRateNeuron": "escape_rate",
    "FitzHughNagumoNeuron": "fitzhugh_nagumo",
    "FitzHughRinzelNeuron": "fitzhugh_rinzel",
    "HindmarshRoseNeuron": "hindmarsh_rose",
    "HodgkinHuxleyNeuron": "hodgkin_huxley",
    "IntegerQIFNeuron": "iqif",
    "Izhikevich2007Neuron": "izhikevich2007",
    "JansenRitUnit": "jansen_rit",
    "McCullochPittsNeuron": "mcculloch_pitts",
    "MedvedevMapNeuron": "medvedev_map",
    "MihalasNieburNeuron": "mihalas_niebur",
    "MorrisLecarNeuron": "morris_lecar",
    "PernarowskiNeuron": "pernarowski",
    "RulkovMapNeuron": "rulkov_map",
    "SCResettingWilsonHRNeuron": "sc_resetting_wilson_hr",
    "TermanWangOscillator": "terman_wang",
    "WilsonHRNeuron": "wilson_hr",
    "WongWangUnit": "wong_wang",
    "PoissonNeuron": "poisson",
    "ResonateAndFireNeuron": "resonate_fire",
    "SigmoidRateNeuron": "sigmoid_rate",
    "ThresholdLinearRateNeuron": "threshold_linear_rate",
    "WilsonCowanUnit": "wilson_cowan",
    "WangBuzsakiNeuron": "wang_buzsaki",
}

# Count-neutral SC identities with dedicated generated formal jobs. They are
# emitted alongside, but not counted as, source-literature S5 models.
RETAINED_SC_CLASS_TO_SCHEMA: dict[str, str] = {
    "SCInclusivePerfectIntegratorNeuron": "sc_perfect_integrator",
    "SCSymmetricQuadraticIFNeuron": "sc_symmetric_quadratic_if",
    "SCFourStateGLIFNeuron": "sc_four_state_glif",
    "SCScaledResetAdaptiveIFNeuron": "sc_scaled_reset_adaptive_if",
    "SCClippedRationalRecoveryMapNeuron": "sc_clipped_rational_recovery_map",
    "SCUpwardCrossingRulkovMapNeuron": "sc_upward_crossing_rulkov_map",
}

# Perfect models whose committed formal lane is intentionally curated rather
# than regenerated through ``UniversalNeuron.to_verilog``.  These designs carry
# model-specific fixed-point recurrences, network structure, or proof contracts
# that the generic equation compiler does not represent at the same fidelity.
# Keeping the mapping explicit prevents the inventory gate from overwriting
# higher-grade RTL merely to make every artefact look generator-produced.
CURATED_CLASS_TO_MODULE: dict[str, str] = {
    "AdExNeuron": "sc_adex",
    "AiharaMapNeuron": "sc_aihara_map",
    "AmariNeuralField": "sc_amari_field",
    "BrunelWangNeuron": "sc_brunel_wang",
    "CazellesMapNeuron": "sc_cazelles_map",
    "COBALIFNeuron": "sc_cobalifneuron",
    "ChialvoMapNeuron": "sc_chialvo_map",
    "CompteWMNeuron": "sc_compte_wm",
    "EnergyLIFNeuron": "energy_lif",
    "ExpIFNeuron": "sc_exponential_if",
    "GLIFNeuron": "sc_glif",
    "IbarzTanakaMapNeuron": "sc_ibarz_tanaka_rulkov_map",
    "LapicqueNeuron": "sc_lapicque_1907",
    "PerfectIntegratorNeuron": "sc_perfect_integrator_naud_gerstner_2012",
    "QuadraticIFNeuron": "sc_quadratic_if_latham_2000",
    "ThetaNeuron": "sc_theta",
    "SCInclusivePerfectIntegratorNeuron": "sc_perfect_integrator",
    "SCSymmetricQuadraticIFNeuron": "sc_quadratic_if",
    "MATNeuron": "sc_mat",
    "McKeanNeuron": "mckean",
    "NagumoSatoMapNeuron": "sc_nagumo_sato_map",
    "NMDANeuron": "sc_nmda_autapse",
    "NonResettingLIFNeuron": "sc_non_resetting_lif",
    "SigmaDeltaNeuron": "sc_sigma_delta",
}

CURATED_CLASS_TO_SCHEMA: dict[str, str] = {
    "AdExNeuron": "adex",
    "AiharaMapNeuron": "aihara_map",
    "AmariNeuralField": "amari_neural_field",
    "BrunelWangNeuron": "brunel_wang",
    "CazellesMapNeuron": "cazelles_map",
    "ChialvoMapNeuron": "chialvo_map",
    "COBALIFNeuron": "coba_lif",
    "CompteWMNeuron": "compte_wm",
    "EnergyLIFNeuron": "energy_lif",
    "ExpIFNeuron": "exp_if",
    "IbarzTanakaMapNeuron": "ibarz_tanaka_map",
    "LapicqueNeuron": "lapicque",
    "MATNeuron": "mat",
    "NagumoSatoMapNeuron": "nagumo_sato_map",
    "NonResettingLIFNeuron": "non_resetting_lif",
    "PerfectIntegratorNeuron": "perfect_integrator",
    "QuadraticIFNeuron": "quadratic_if",
    "SigmaDeltaNeuron": "sigma_delta",
    "ThetaNeuron": "theta",
    "SCSymmetricQuadraticIFNeuron": "sc_symmetric_quadratic_if",
}

# Other committed curated jobs cover retained SC variants or dedicated
# subsystem representatives.  They are valid formal evidence, but are not part
# of the one-job-per-perfect-model emitter count.
CURATED_FORMAL_MODULES: frozenset[str] = frozenset(
    {
        "benda_herz",
        "energy_lif",
        "mckean",
        "sc_adaptive_threshold_map",
        "sc_aihara_map",
        "sc_amari_field",
        "sc_brunel_wang",
        "sc_compte_wm",
        "sc_compte_wm_ring16",
        "sc_glif",
        "sc_lapicque",
        "sc_lapicque_1907",
        "sc_mat",
        "sc_nagumo_sato_map",
        "sc_nmda_autapse",
        "sc_non_resetting_adaptive_lif",
        "sc_non_resetting_lif",
        "sc_perfect_integrator_naud_gerstner_2012",
        "sc_quadratic_if_latham_2000",
        "sc_normalized_energy_lif",
        "sc_resetting_mat",
        "sc_resetting_wilson_hr",
        "sc_sigma_delta",
        "sc_sigma_delta_accumulator",
        "sc_stochastic_rate_adaptation",
        "sc_triangular_mckean",
        "sc_wb_nmda_magnesium_block",
    }
)

# BMC depth: small for huge LUT models; deeper for compact IF cores.
DEPTH_BY_SCHEMA: dict[str, int] = {
    "adaptive_threshold_if": 4,
    "aihara_map": 6,
    "alpha": 4,
    "amari_neural_field": 12,
    "brunel_wang": 4,
    "cazelles_map": 4,
    "sc_clipped_logistic_bursting_map": 4,
    "chialvo_map": 4,
    "coba_lif": 8,
    "connor_stevens": 4,
    "compte_wm": 4,
    "courage_nekorkin_map": 4,
    "dpi_neuron": 8,
    "ermentrout_kopell_map_neuron": 4,
    "ermentrout_kopell_pop": 4,
    "energy_lif": 2,
    "escape_rate": 4,
    "exp_if": 4,
    "hodgkin_huxley": 4,
    "ibarz_tanaka_map": 4,
    "iqif": 4,
    "jansen_rit": 4,
    "lapicque": 20,
    "mat": 12,
    "mcculloch_pitts": 4,
    "morris_lecar": 4,
    "nagumo_sato_map": 12,
    "non_resetting_lif": 12,
    "fitzhugh_nagumo": 4,
    "fitzhugh_rinzel": 4,
    "hindmarsh_rose": 4,
    "mckean": 4,
    "medvedev_map": 4,
    "mihalas_niebur": 2,
    "pernarowski": 4,
    "poisson": 4,
    "rulkov_map": 4,
    "sc_upward_crossing_rulkov_map": 4,
    "sc_four_state_glif": 6,
    "sc_scaled_reset_adaptive_if": 2,
    "sc_clipped_rational_recovery_map": 4,
    "sc_resetting_wilson_hr": 4,
    "resonate_fire": 4,
    "sigmoid_rate": 4,
    "sigma_delta": 12,
    "theta": 110,
    "terman_wang": 4,
    "threshold_linear_rate": 4,
    "wilson_hr": 4,
    "wilson_cowan": 4,
    "wong_wang": 4,
    "adex": 6,
    "glif": 6,
    "wang_buzsaki": 4,
}

# Heavy multi-state / transcendental cores: prove bounded public spike-port
# safety at tiny BMC depth. Event-silent schemas add their explicit zero-output
# invariant without claiming equation equivalence.
MINIMAL_SAFETY_SCHEMAS: frozenset[str] = frozenset(
    {
        "adaptive_threshold_if",
        "alpha",
        "cazelles_map",
        "sc_clipped_logistic_bursting_map",
        "chialvo_map",
        "courage_nekorkin_map",
        "dpi_neuron",
        "ermentrout_kopell_map_neuron",
        "ermentrout_kopell_pop",
        "escape_rate",
        "exp_if",
        "fitzhugh_nagumo",
        "fitzhugh_rinzel",
        "hindmarsh_rose",
        "mckean",
        "mcculloch_pitts",
        "medvedev_map",
        "mihalas_niebur",
        "morris_lecar",
        "pernarowski",
        "poisson",
        "rulkov_map",
        "sc_upward_crossing_rulkov_map",
        "sc_scaled_reset_adaptive_if",
        "sc_clipped_rational_recovery_map",
        "resonate_fire",
        "sigmoid_rate",
        "terman_wang",
        "threshold_linear_rate",
        "wilson_cowan",
        "connor_stevens",
        "hodgkin_huxley",
        "ibarz_tanaka_map",
        "jansen_rit",
        "wong_wang",
        "wang_buzsaki",
    }
)

# Continuous-rate models with a public spike port that must remain silent.
EVENT_SILENT_SCHEMAS: frozenset[str] = frozenset(
    {
        "ermentrout_kopell_pop",
        "sigmoid_rate",
        "threshold_linear_rate",
        "wilson_cowan",
    }
)

# Wilson-Cowan's generated Q32.32 RTL contains many exponential LUTs feeding
# public E/I outputs that are outside this bounded spike-port safety claim.
# Flattening lets Yosys prune those unobserved cones before the SMT handoff.
FLATTEN_FORMAL_SCHEMAS: frozenset[str] = frozenset({"wilson_cowan"})

# Jobs the SMT handoff sends to cvc5 instead of z3. IQIF's saturating Q32.0
# update, compared at 64 bits, passes in seconds under cvc5 1.1.2 and z3 4.16
# but runs for over ten minutes under z3 4.8.12, the Ubuntu 24.04 package CI
# installs.
CVC5_FORMAL_SCHEMAS: frozenset[str] = frozenset({"dpi_neuron", "iqif", "terman_wang"})

# Width overrides are additive: every pre-existing catalogue job retains Q8.8.
# Medvedev needs Q16.16 because its calibrated d=2271.19 cannot fit Q8.8.
# Ibarz-Tanaka needs Q16.16 because its source mu=0.001 rounds to zero in Q8.8.
# DPI needs Q16.16 to preserve its coupled-current event-count envelope; Q8.8
# rounds the 0.01 initial/reference currents too aggressively.
# ExpIF needs Q32.32 to preserve the enrolled source-exponential spike counts;
# its active Q16.16 trace does not satisfy the declared event contract.
# COBA LIF needs Q24.24 to preserve its four-stage RK4 event schedule and
# four-state co-simulation envelope; Q16.16 adds a refractory residue step.
# IQIF uses Q32.0 to retain the pinned signed-integer recurrence and its Q0.3
# arithmetic shift without introducing a fractional rescale. McCulloch-Pitts
# uses the same Q32.0 carrier for the non-negative excitatory-afferent count;
# -1 is the sole absolute-inhibition sentinel.
# Wong-Wang, Jansen-Rit, MPR, and resonate-and-fire stay on their enrolled
# Q32.32 co-simulation carriers: Q8.8 cannot represent their sub-unit timesteps.
# Their catalogue jobs are bounded public spike-port safety only and do not
# claim formal equivalence or H4.
DEFAULT_PRECISION = (16, 8)
PRECISION_BY_SCHEMA: dict[str, tuple[int, int]] = {
    "adaptive_threshold_if": (64, 32),
    "aihara_map": (32, 24),
    "alpha": (64, 32),
    "amari_neural_field": (32, 16),
    "brunel_wang": (32, 16),
    "coba_lif": (48, 24),
    "connor_stevens": (32, 16),
    "compte_wm": (32, 16),
    "courage_nekorkin_map": (64, 32),
    "dpi_neuron": (32, 16),
    "energy_lif": (64, 32),
    "ermentrout_kopell_pop": (64, 32),
    "escape_rate": (48, 24),
    "exp_if": (64, 32),
    "fitzhugh_nagumo": (32, 16),
    "hodgkin_huxley": (32, 16),
    "ibarz_tanaka_map": (32, 16),
    "iqif": (32, 0),
    "jansen_rit": (64, 32),
    "lapicque": (64, 32),
    "mat": (64, 32),
    "mcculloch_pitts": (32, 0),
    "medvedev_map": (32, 16),
    "mihalas_niebur": (64, 32),
    "morris_lecar": (32, 16),
    "nagumo_sato_map": (32, 16),
    "non_resetting_lif": (64, 32),
    "poisson": (48, 24),
    "quadratic_if": (32, 16),
    "resonate_fire": (64, 32),
    "rulkov_map": (32, 16),
    "sc_upward_crossing_rulkov_map": (32, 16),
    "sc_four_state_glif": (32, 16),
    "sc_scaled_reset_adaptive_if": (32, 16),
    "sc_clipped_rational_recovery_map": (64, 32),
    "sigmoid_rate": (64, 32),
    "sigma_delta": (64, 32),
    "threshold_linear_rate": (32, 16),
    # Ermentrout-Kopell phase and fixed-circle envelope use Q16.16.
    "theta": (32, 16),
    "wilson_cowan": (64, 32),
    "wong_wang": (64, 32),
    "wang_buzsaki": (32, 16),
}

# Transcendental Q16.16 designs can make an unconstrained-current BMC spend
# most of its time solving input cones unrelated to the stated reset property.
# Pin only those jobs to their enrolled receipt current and an initial reset;
# the DUT remains the exact committed compiler lowering.
FORMAL_FIXED_CURRENT_BY_SCHEMA: dict[str, float] = {
    "connor_stevens": 100.0,
    "hodgkin_huxley": 15.0,
    "mihalas_niebur": 0.002,
    "morris_lecar": 100.0,
    "dpi_neuron": 500.0,
    "sc_scaled_reset_adaptive_if": 3.0,
}

# Public-port post-event words for deterministic bounded protocols. These are
# safety properties of the committed fixed-point recurrence, not a claim of
# real-number or Python-to-RTL formal equivalence.
FORMAL_SPIKE_STATE_BY_SCHEMA: dict[str, dict[str, float]] = {
    "dpi_neuron": {
        "i_mem_out": 0.01,
        "refractory_time_out": 2.0,
    },
}
FORMAL_POST_SPIKE_STATE_BY_SCHEMA: dict[str, dict[str, float]] = {
    "dpi_neuron": {
        "i_mem_out": 0.01,
        "refractory_time_out": 1.9,
    },
}

# Schema display names are user-facing and may contain punctuation or
# diacritics.  Pin an ASCII HDL identifier where sanitising the display name
# would otherwise make a faithful schema impossible to commit as RTL.
MODULE_NAME_BY_SCHEMA: dict[str, str] = {
    "sc_four_state_glif": "sc_four_state_glif",
    "sc_scaled_reset_adaptive_if": "sc_scaled_reset_adaptive_if",
    "sc_clipped_rational_recovery_map": "sc_clipped_rational_recovery_map",
    "sc_upward_crossing_rulkov_map": "sc_upward_crossing_rulkov_map",
    "sc_resetting_wilson_hr": "sc_resetting_wilson_hr",
    "wang_buzsaki": "sc_wang_buzsaki",
}


@dataclass(frozen=True)
class EmitResult:
    """One catalogue formal job emission."""

    schema: str
    class_name: str
    module: str
    state_port: str | None
    rtl_path: Path
    formal_path: Path
    sby_path: Path
    depth: int
    data_width: int
    fraction: int
    origin: str = "generated"
    solver: str = "z3"
    properties: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    reachability_asserted: bool = False


def _perfect_class_names() -> list[str]:
    sys.path.insert(0, str(ROOT / "src"))
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib

    from sc_neurocore.neurons.descriptor_tiers import is_perfect
    from sc_neurocore.neurons.model_descriptor import parse_model_descriptor

    mapped_classes = set(CLASS_TO_SCHEMA) | set(CURATED_CLASS_TO_SCHEMA)
    names: list[str] = []
    for path in sorted(DESC_DIR.glob("*.toml")):
        desc = parse_model_descriptor(tomllib.loads(path.read_text(encoding="utf-8")))
        if is_perfect(desc) and desc.class_name in mapped_classes:
            names.append(desc.class_name)
    return names


@dataclass(frozen=True)
class ModulePorts:
    """Parsed equation-compiler module surface."""

    name: str
    primary_state: str | None
    signed_outputs: tuple[str, ...]
    bit_outputs: tuple[str, ...]
    has_current_input: bool


def _parse_module_ports(rtl: str) -> ModulePorts:
    """Return module name and output ports from generated RTL."""
    mod_match = re.search(r"(?m)^\s*module\s+(\w+)", rtl)
    if not mod_match:
        raise ValueError("generated RTL has no module declaration")
    module = mod_match.group(1)
    signed_outs = tuple(re.findall(r"output\s+reg\s+signed\s+\[[^\]]+\]\s+(\w+)", rtl))
    # Do not use a bare ``output reg (\w+)`` — it would capture the keyword
    # ``signed`` from ``output reg signed [15:0] …``.
    bit_outs = tuple(re.findall(r"output\s+reg\s+(?!signed\b)(\w+)", rtl))
    bit_outs = tuple(b for b in bit_outs if b not in signed_outs)
    primary = signed_outs[0] if signed_outs else None
    for preferred in ("v_out", "i_mem_out", "theta_out", "w_out", "u_out"):
        if preferred in signed_outs:
            primary = preferred
            break
    has_i = bool(re.search(r"input\s+wire\s+signed\s+\[[^\]]+\]\s+I_t", rtl))
    return ModulePorts(
        name=module,
        primary_state=primary,
        signed_outputs=signed_outs,
        bit_outputs=bit_outs,
        has_current_input=has_i,
    )


def _spdx_header(title: str) -> str:
    return (
        "// SPDX-License-Identifier: AGPL-3.0-or-later\n"
        "// Commercial license available\n"
        "// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.\n"
        "// © Code 2020–2026 Miroslav Šotek. All rights reserved.\n"
        "// ORCID: 0009-0009-3560-0851\n"
        "// Contact: www.anulum.li | protoscience@anulum.li\n"
        f"// SC-NeuroCore — {title}\n"
    )


def _formal_wrapper(
    ports: ModulePorts,
    *,
    minimal: bool,
    event_silent: bool = False,
    data_width: int = 16,
    fixed_current_word: int | None = None,
    spike_state_words: dict[str, int] | None = None,
    post_spike_state_words: dict[str, int] | None = None,
    reset_state_words: dict[str, int] | None = None,
    evidence_label: str = "dual-axis perfect model",
) -> str:
    """Build a port-only formal harness (no hierarchical probes).

    A non-minimal harness asserts that while reset is held every public state
    port carries its encoded initial value (``reset_state_words``). It does not
    assert that a signed ``data_width``-bit port lies in the signed
    ``data_width``-bit range: such a check cannot fail.
    """
    module = ports.name
    state_port = ports.primary_state
    wire_decls: list[str] = []
    for bit in ports.bit_outputs:
        wire_decls.append(f"    wire {bit};")
    for signed in ports.signed_outputs:
        wire_decls.append(f"    wire signed [{data_width - 1}:0] {signed};")
    connections = [
        ".clk(clk)",
        ".rst_n(rst_n)",
    ]
    if ports.has_current_input:
        connections.append(".I_t(I_t)")
    for bit in ports.bit_outputs:
        connections.append(f".{bit}({bit})")
    for signed in ports.signed_outputs:
        connections.append(f".{signed}({signed})")
    conn_block = ",\n        ".join(connections)
    wires = "\n".join(wire_decls)
    if minimal:
        bounded_protocol = (
            f"""
    // Bounded receipt protocol: initialise through reset and hold the exact
    // enrolled fixed-point drive while checking the public reset property.
    reg protocol_started = 1'b0;
    always @(posedge clk) begin
        if (!protocol_started)
            assume (!rst_n);
        else
            assume (rst_n);
        assume ($signed(I_t) == {data_width}'sd{fixed_current_word});
        protocol_started <= 1'b1;
    end

"""
            if fixed_current_word is not None and ports.has_current_input
            else ""
        )
        event_silence = (
            """
    reg past_valid = 1'b0;
    always @(posedge clk) begin
        past_valid <= 1'b1;
        if (past_valid && rst_n)
            assert (spike_out == 1'b0);
    end

"""
            if event_silent
            else ""
        )
        spike_state = ""
        if spike_state_words is not None:
            spike_assertions = "\n".join(
                f"            assert ($signed({port}) == {data_width}'sd{word});"
                for port, word in spike_state_words.items()
            )
            post_spike_assertions = ""
            if post_spike_state_words is not None:
                post_spike_lines = "\n".join(
                    f"            assert ($signed({port}) == {data_width}'sd{word});"
                    for port, word in post_spike_state_words.items()
                )
                post_spike_assertions = f"""
        if (spike_past_valid && rst_n && $past(spike_out)) begin
            assert (spike_out == 1'b0);
{post_spike_lines}
        end"""
            spike_state = f"""
    // The fixed-current protocol reaches a real event within this BMC depth.
    // Bind its reset packet and the next refractory sample through public ports.
    reg spike_past_valid = 1'b0;
    always @(posedge clk) begin
        spike_past_valid <= 1'b1;
        if (spike_past_valid && rst_n && spike_out) begin
{spike_assertions}
        end
{post_spike_assertions}
    end

"""
        formal_body = f"""
`ifdef FORMAL
{bounded_protocol}    // Minimal safety: async reset clears the spike flag.
    always @(*) begin
        if (!rst_n)
            assert (spike_out == 1'b0);
    end
{event_silence}{spike_state}`endif
"""
    else:
        if state_port is None:
            raise ValueError(f"{module}: non-minimal formal job requires a signed state output")
        reset_values = "\n".join(
            f"            assert ($signed({port}) == {_signed_literal(word, data_width)});"
            for port, word in (reset_state_words or {}).items()
        )
        formal_body = f"""
`ifdef FORMAL
    // Reset values: while reset is held the spike flag is clear and every public
    // state port carries its encoded initial value, which need not be zero.
    always @(*) begin
        if (!rst_n) begin
            assert (spike_out == 1'b0);
{reset_values}
        end
    end
`endif
"""
    return f"""{_spdx_header(f"Catalogue formal harness for {module}")}
`default_nettype none

// Formal wrapper for equation-compiler RTL of a {evidence_label}.
// Properties use only public ports so default_nettype none stays clean.
module {module}_formal (
    input wire clk,
    input wire rst_n,
    input wire signed [{data_width - 1}:0] I_t
);

{wires}

    {module} uut (
        {conn_block}
    );
{formal_body}
endmodule
"""


def _signed_literal(word: int, data_width: int) -> str:
    """Write a signed word as a Verilog literal (``-W'sdN`` for a negative one)."""
    return f"-{data_width}'sd{-word}" if word < 0 else f"{data_width}'sd{word}"


def _reset_state_words(
    equation_neuron: EquationNeuron, ports: ModulePorts, data_width: int, fraction: int
) -> dict[str, int]:
    """Return the encoded initial word of every public state port.

    Each ``<var>_out`` port resets to ``round(initial * 2**fraction)`` wrapped to
    the word, exactly as the equation compiler encodes the initial state.
    """
    from sc_neurocore.compiler.c_fixed_emitter import signed_q
    from sc_neurocore.compiler.verilog_compiler_config import Q88
    from sc_neurocore.hdl_gen._ident import sanitize_ident

    q = Q88(data_width=data_width, fraction=fraction)
    words: dict[str, int] = {}
    for variable in equation_neuron.equations:
        port = f"{sanitize_ident(variable, context='state variable')}_out"
        if port in ports.signed_outputs:
            words[port] = signed_q(q, float(equation_neuron.initial_state.get(variable, 0.0)))
    return words


def _sby_script(
    module: str,
    depth: int,
    *,
    flatten: bool = False,
    evidence_label: str = "dual-axis perfect model",
    solver: str = "z3",
) -> str:
    prep = f"prep -top {module}_formal" + (" -flatten" if flatten else "")
    sby_evidence_label = evidence_label[:1].upper() + evidence_label[1:]
    return (
        f"# SymbiYosys job for catalogue model {module}\n"
        f"# {sby_evidence_label} formal (BMC)\n"
        "\n"
        "[options]\n"
        "mode bmc\n"
        f"depth {depth}\n"
        "\n"
        "[engines]\n"
        f"smtbmc {solver}\n"
        "\n"
        "[script]\n"
        f"read -formal {module}_formal.v\n"
        f"read -formal {module}.v\n"
        f"{prep}\n"
        "\n"
        "[files]\n"
        f"{module}_formal.v\n"
        f"{module}.v\n"
    )


def _emit_schema(
    class_name: str,
    schema: str,
    *,
    evidence_label: str = "dual-axis perfect model",
) -> EmitResult:
    """Emit RTL, a formal wrapper, and an SBY job for one schema."""
    sys.path.insert(0, str(ROOT / "src"))
    from sc_neurocore.neurons.universal_dsl import UniversalNeuron

    neuron = UniversalNeuron.from_schema(schema)
    data_width, fraction = PRECISION_BY_SCHEMA.get(schema, DEFAULT_PRECISION)
    rtl = neuron.to_verilog(
        module_name=MODULE_NAME_BY_SCHEMA.get(schema),
        data_width=data_width,
        fraction=fraction,
    )
    ports = _parse_module_ports(rtl)
    module = ports.name
    depth = DEPTH_BY_SCHEMA.get(schema, 20)
    if "spike_out" not in ports.bit_outputs:
        raise ValueError(f"{module}: expected spike_out bit output, got {ports.bit_outputs}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rtl_path = OUT_DIR / f"{module}.v"
    formal_path = OUT_DIR / f"{module}_formal.v"
    sby_path = OUT_DIR / f"{module}.sby"

    rtl_path.write_text(rtl if rtl.endswith("\n") else rtl + "\n", encoding="utf-8")
    minimal = schema in MINIMAL_SAFETY_SCHEMAS
    reset_words = (
        None
        if minimal
        else _reset_state_words(neuron.to_equation_neuron(), ports, data_width, fraction)
    )
    properties, assumptions = _generated_claims(schema, minimal=minimal, ports=ports)
    solver = "cvc5" if schema in CVC5_FORMAL_SCHEMAS else "z3"
    formal_path.write_text(
        _formal_wrapper(
            ports,
            minimal=minimal,
            event_silent=schema in EVENT_SILENT_SCHEMAS,
            data_width=data_width,
            fixed_current_word=(
                round(FORMAL_FIXED_CURRENT_BY_SCHEMA[schema] * (1 << fraction))
                if schema in FORMAL_FIXED_CURRENT_BY_SCHEMA
                else None
            ),
            spike_state_words=(
                {
                    port: round(value * (1 << fraction))
                    for port, value in FORMAL_SPIKE_STATE_BY_SCHEMA[schema].items()
                }
                if schema in FORMAL_SPIKE_STATE_BY_SCHEMA
                else None
            ),
            post_spike_state_words=(
                {
                    port: round(value * (1 << fraction))
                    for port, value in FORMAL_POST_SPIKE_STATE_BY_SCHEMA[schema].items()
                }
                if schema in FORMAL_POST_SPIKE_STATE_BY_SCHEMA
                else None
            ),
            reset_state_words=reset_words,
            evidence_label=evidence_label,
        ),
        encoding="utf-8",
    )
    sby_path.write_text(
        _sby_script(
            module,
            depth,
            flatten=schema in FLATTEN_FORMAL_SCHEMAS,
            evidence_label=evidence_label,
            solver=solver,
        ),
        encoding="utf-8",
    )

    return EmitResult(
        schema=schema,
        class_name=class_name,
        module=module,
        state_port=ports.primary_state,
        rtl_path=rtl_path,
        formal_path=formal_path,
        sby_path=sby_path,
        depth=depth,
        data_width=data_width,
        fraction=fraction,
        solver=solver,
        properties=properties,
        assumptions=assumptions,
    )


def _generated_claims(
    schema: str, *, minimal: bool, ports: ModulePorts
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the properties a generated harness asserts and what it assumes."""
    properties = ["reset clears spike_out"]
    if not minimal:
        properties.append("reset holds every public state port at its encoded initial value")
    if minimal and schema in EVENT_SILENT_SCHEMAS:
        properties.append("no spike after reset")
    if minimal and schema in FORMAL_SPIKE_STATE_BY_SCHEMA:
        properties.append("public state words on every spike (guarded by the spike)")
    if minimal and schema in FORMAL_POST_SPIKE_STATE_BY_SCHEMA:
        properties.append("spike clear and state words on the cycle after a spike")
    fixed = minimal and schema in FORMAL_FIXED_CURRENT_BY_SCHEMA and ports.has_current_input
    assumptions = (
        (
            "reset on the first cycle only",
            f"input held at {FORMAL_FIXED_CURRENT_BY_SCHEMA[schema]!r}",
        )
        if fixed
        else ("reset and input unconstrained",)
    )
    return tuple(properties), assumptions


def _curated_schema(class_name: str, schema: str, module: str) -> EmitResult:
    """Validate and inventory a curated job without overwriting its stronger RTL."""
    rtl_path = OUT_DIR / f"{module}.v"
    formal_path = OUT_DIR / f"{module}_formal.v"
    sby_path = OUT_DIR / f"{module}.sby"
    for path in (rtl_path, formal_path, sby_path):
        if not path.is_file():
            raise FileNotFoundError(f"curated formal artefact is missing: {path}")
    ports = _parse_module_ports(rtl_path.read_text(encoding="utf-8"))
    if ports.name != module or (ports.primary_state is None and not ports.bit_outputs):
        raise ValueError(f"curated module {module} has an invalid public-port contract")
    data_width, fraction = PRECISION_BY_SCHEMA.get(schema, DEFAULT_PRECISION)
    harness = formal_path.read_text(encoding="utf-8")
    solver_match = re.search(r"smtbmc\s+(\w+)", sby_path.read_text(encoding="utf-8"))
    assertions = len(re.findall(r"\bassert\s*\(", harness))
    assumptions = len(re.findall(r"\bassume\s*\(", harness))
    return EmitResult(
        schema=schema,
        class_name=class_name,
        module=module,
        state_port=ports.primary_state,
        rtl_path=rtl_path,
        formal_path=formal_path,
        sby_path=sby_path,
        depth=DEPTH_BY_SCHEMA.get(schema, 20),
        data_width=data_width,
        fraction=fraction,
        origin="curated",
        solver=solver_match.group(1) if solver_match else "z3",
        properties=(f"curated harness: {assertions} assertions",),
        assumptions=(f"curated harness: {assumptions} assumptions",),
        reachability_asserted="assert (seen_spike)" in harness,
    )


def _emit_or_inventory(
    class_name: str,
    schema: str,
    *,
    evidence_label: str = "dual-axis perfect model",
) -> EmitResult:
    module = CURATED_CLASS_TO_MODULE.get(class_name)
    if module is not None:
        return _curated_schema(class_name, schema, module)
    return _emit_schema(class_name, schema, evidence_label=evidence_label)


def emit_one(class_name: str) -> EmitResult:
    """Emit RTL + formal wrapper + sby for one perfect class."""
    if class_name in CLASS_TO_SCHEMA:
        schema = CLASS_TO_SCHEMA[class_name]
    else:
        schema = CURATED_CLASS_TO_SCHEMA[class_name]
    return _emit_or_inventory(class_name, schema)


def emit_all() -> list[EmitResult]:
    """Emit formal jobs for every dual-axis perfect catalogue model."""
    results: list[EmitResult] = []
    for class_name in _perfect_class_names():
        results.append(emit_one(class_name))
    retained_results = [
        _emit_schema(
            "SCResettingWilsonHRNeuron",
            "sc_resetting_wilson_hr",
            evidence_label="retained SC project model",
        )
    ]
    for class_name, schema in RETAINED_SC_CLASS_TO_SCHEMA.items():
        retained_results.append(
            _emit_or_inventory(class_name, schema, evidence_label="retained SC project model")
        )
    inventory = OUT_DIR / "INVENTORY.md"
    lines = [
        "# Catalogue formal inventory (declared dual-axis perfect models)",
        "",
        "Generated by `tools/emit_catalogue_formal.py`. Each job is a SymbiYosys BMC",
        "harness over equation-compiler or explicitly curated RTL for a model whose",
        "descriptor declares science S5 + silicon H≥target; enrolment does not verify",
        "that declaration. Every job is a bounded safety check, not an equivalence",
        "proof: `inventory.json` lists what each job asserts, assumes and leaves open.",
        "",
        f"Jobs: **{len(results)}**",
        "",
        "| Class | Schema | Module | State port | Q format | Depth |",
        "| --- | --- | --- | --- | --- | ---: |",
    ]
    for row in results:
        state_port = f"`{row.state_port}`" if row.state_port is not None else "—"
        lines.append(
            f"| {row.class_name} | {row.schema} | `{row.module}` | "
            f"{state_port} | Q{row.data_width - row.fraction}.{row.fraction} | "
            f"{row.depth} |"
        )
    lines.extend(
        [
            "",
            "## Retained count-neutral SC project jobs",
            "",
            "These jobs are emitted and maintained by the same tool but are not included in the",
            "source-literature perfect-model count above.",
            "",
            "| Class | Schema | Module | State port | Q format | Depth |",
            "| --- | --- | --- | --- | --- | ---: |",
        ]
    )
    for row in retained_results:
        state_port = f"`{row.state_port}`" if row.state_port is not None else "—"
        lines.append(
            f"| {row.class_name} | {row.schema} | `{row.module}` | "
            f"{state_port} | Q{row.data_width - row.fraction}.{row.fraction} | "
            f"{row.depth} |"
        )
    inventory.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT_DIR / "inventory.json").write_text(
        json.dumps(
            {
                "schema_version": FORMAL_INVENTORY_SCHEMA_VERSION,
                "enrolment": (
                    "models whose descriptor declares science S5 and the terminal silicon "
                    "tier; the declaration is not verified by enrolment"
                ),
                "jobs": [_inventory_entry(row) for row in results],
                "retained_jobs": [_inventory_entry(row) for row in retained_results],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return results


FORMAL_INVENTORY_SCHEMA_VERSION = "sc-neurocore.formal-inventory.v1"


def _inventory_entry(row: EmitResult) -> dict[str, object]:
    """State what one job checks, under which assumptions, and what it leaves open."""
    not_established = [
        "equivalence with the model or with the bit-true kernel",
        f"any behaviour after {row.depth} cycles",
    ]
    if row.origin == "curated":
        not_established.append(
            "that the RTL is the current compiler's output: a curated job checks its "
            "committed RTL file, which this tool does not regenerate"
        )
    if not row.reachability_asserted:
        not_established.append(
            "reachability of an event: a property guarded by a spike holds vacuously "
            "if no spike occurs within the depth"
        )
    return {
        "class": row.class_name,
        "profile": row.schema,
        "module": row.module,
        "origin": row.origin,
        "q_format": f"Q{row.data_width - row.fraction}.{row.fraction}",
        "mode": "bmc",
        "depth": row.depth,
        "solver": row.solver,
        "claim": "bounded safety",
        "properties": list(row.properties),
        "assumptions": list(row.assumptions),
        "reachability_asserted": row.reachability_asserted,
        "not_established": not_established,
    }


def run_sby(results: list[EmitResult], *, timeout_s: int = 120) -> dict[str, str]:
    """Run each ``.sby`` from ``catalogue/``; return module → verdict string."""
    sby = shutil.which("sby")
    if sby is None:
        return {r.module: "SKIP (sby not on PATH)" for r in results}
    verdicts: dict[str, str] = {}
    for row in results:
        try:
            # The argument vector contains a fixed executable and generated basename only.
            proc = subprocess.run(  # nosec B603
                [sby, "-f", row.sby_path.name],
                cwd=OUT_DIR,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            verdicts[row.module] = "TIMEOUT"
            continue
        text = (proc.stdout or "") + (proc.stderr or "")
        if "DONE (PASS" in text:
            verdicts[row.module] = "PASS"
        elif "DONE (FAIL" in text:
            verdicts[row.module] = "FAIL"
        elif "DONE (ERROR" in text or "ERROR" in text:
            verdicts[row.module] = "ERROR"
        else:
            verdicts[row.module] = f"UNKNOWN rc={proc.returncode}"
    return verdicts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-sby",
        action="store_true",
        help="Execute each generated SymbiYosys job (requires sby on PATH)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Per-job sby timeout in seconds (default 120)",
    )
    args = parser.parse_args(argv)
    results = emit_all()
    print(f"Emitted {len(results)} catalogue formal jobs under {OUT_DIR}")
    for row in results:
        print(f"  {row.class_name:28} -> {row.sby_path.name} (depth={row.depth})")
    if args.run_sby:
        verdicts = run_sby(results, timeout_s=args.timeout)
        print("SymbiYosys verdicts:")
        for module, verdict in sorted(verdicts.items()):
            print(f"  {module:32} {verdict}")
        fails = [m for m, v in verdicts.items() if v not in {"PASS", "SKIP (sby not on PATH)"}]
        return 1 if fails else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
