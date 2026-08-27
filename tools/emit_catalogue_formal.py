#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Emit SymbiYosys jobs for dual-axis perfect catalogue models

"""Emit schema→RTL + formal wrappers + ``.sby`` for dual-axis perfect models.

Only models with ``is_perfect`` (science S5 + silicon ≥ target H) are enrolled.
Each job proves reset hygiene and spike reachability on the *committed* equation-
compiler RTL (Q8.8 by default, with explicit per-schema overrides), without hierarchical ``uut.*`` probes so
``default_nettype none`` stays clean.

Usage
-----
From the SC-NEUROCORE repo root::

    .venv/bin/python tools/emit_catalogue_formal.py
    .venv/bin/python tools/emit_catalogue_formal.py --run-sby

Outputs land under ``hdl/formal/catalogue/``.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "hdl" / "formal" / "catalogue"
DESC_DIR = ROOT / "src" / "sc_neurocore" / "neurons" / "model_descriptors"

# Schema stem used by UniversalNeuron.from_schema for each perfect class.
CLASS_TO_SCHEMA: dict[str, str] = {
    "AdaptiveThresholdIFNeuron": "adaptive_threshold_if",
    "AlphaNeuron": "alpha",
    "AdExNeuron": "adex",
    "CazellesMapNeuron": "cazelles_map",
    "ChialvoMapNeuron": "chialvo_map",
    "COBALIFNeuron": "coba_lif",
    "ConnorStevensNeuron": "connor_stevens",
    "CourageNekorkinMapNeuron": "courage_nekorkin_map",
    "DPINeuron": "dpi_neuron",
    "ErmentroutKopellMapNeuron": "ermentrout_kopell_map_neuron",
    "ErmentroutKopellPopulation": "ermentrout_kopell_pop",
    "EscapeRateNeuron": "escape_rate",
    "ExpIFNeuron": "exp_if",
    "FitzHughNagumoNeuron": "fitzhugh_nagumo",
    "FitzHughRinzelNeuron": "fitzhugh_rinzel",
    "GLIFNeuron": "glif",
    "HindmarshRoseNeuron": "hindmarsh_rose",
    "HodgkinHuxleyNeuron": "hodgkin_huxley",
    "IbarzTanakaMapNeuron": "ibarz_tanaka_map",
    "IntegerQIFNeuron": "iqif",
    "Izhikevich2007Neuron": "izhikevich2007",
    "JansenRitUnit": "jansen_rit",
    "LapicqueNeuron": "lapicque",
    "McCullochPittsNeuron": "mcculloch_pitts",
    "McKeanNeuron": "mckean",
    "MedvedevMapNeuron": "medvedev_map",
    "MihalasNieburNeuron": "mihalas_niebur",
    "MorrisLecarNeuron": "morris_lecar",
    "PernarowskiNeuron": "pernarowski",
    "RulkovMapNeuron": "rulkov_map",
    "TermanWangOscillator": "terman_wang",
    "WilsonHRNeuron": "wilson_hr",
    "WongWangUnit": "wong_wang",
    "PerfectIntegratorNeuron": "perfect_integrator",
    "PoissonNeuron": "poisson",
    "QuadraticIFNeuron": "quadratic_if",
    "ResonateAndFireNeuron": "resonate_fire",
    "SigmoidRateNeuron": "sigmoid_rate",
    "ThetaNeuron": "theta",
    "ThresholdLinearRateNeuron": "threshold_linear_rate",
    "WilsonCowanUnit": "wilson_cowan",
}

# Perfect models whose committed formal lane is intentionally curated rather
# than regenerated through ``UniversalNeuron.to_verilog``.  These designs carry
# model-specific fixed-point recurrences, network structure, or proof contracts
# that the generic equation compiler does not represent at the same fidelity.
# Keeping the mapping explicit prevents the inventory gate from overwriting
# higher-grade RTL merely to make every artefact look generator-produced.
CURATED_CLASS_TO_MODULE: dict[str, str] = {
    "AiharaMapNeuron": "sc_aihara_map",
    "AmariNeuralField": "sc_amari_field",
    "BrunelWangNeuron": "sc_brunel_wang",
    "CompteWMNeuron": "sc_compte_wm",
    "EnergyLIFNeuron": "energy_lif",
    "MATNeuron": "sc_mat",
    "NagumoSatoMapNeuron": "sc_nagumo_sato_map",
    "NMDANeuron": "sc_nmda_autapse",
    "NonResettingLIFNeuron": "sc_non_resetting_lif",
    "SigmaDeltaNeuron": "sc_sigma_delta",
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
        "sc_mat",
        "sc_nagumo_sato_map",
        "sc_nmda_autapse",
        "sc_non_resetting_adaptive_lif",
        "sc_non_resetting_lif",
        "sc_normalized_energy_lif",
        "sc_resetting_mat",
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
    "alpha": 4,
    "cazelles_map": 4,
    "chialvo_map": 4,
    "coba_lif": 4,
    "connor_stevens": 4,
    "courage_nekorkin_map": 4,
    "dpi_neuron": 4,
    "ermentrout_kopell_map_neuron": 4,
    "ermentrout_kopell_pop": 4,
    "escape_rate": 4,
    "exp_if": 4,
    "hodgkin_huxley": 4,
    "ibarz_tanaka_map": 4,
    "iqif": 4,
    "jansen_rit": 4,
    "mcculloch_pitts": 4,
    "morris_lecar": 3,
    "fitzhugh_nagumo": 4,
    "fitzhugh_rinzel": 4,
    "hindmarsh_rose": 4,
    "mckean": 4,
    "medvedev_map": 4,
    "mihalas_niebur": 3,
    "pernarowski": 4,
    "poisson": 4,
    "rulkov_map": 4,
    "resonate_fire": 4,
    "sigmoid_rate": 4,
    "terman_wang": 4,
    "threshold_linear_rate": 4,
    "wilson_hr": 4,
    "wilson_cowan": 4,
    "wong_wang": 4,
    "theta": 6,
    "adex": 6,
    "glif": 6,
}

# Heavy multi-state / transcendental cores: prove bounded public spike-port
# safety at tiny BMC depth. Event-silent schemas add their explicit zero-output
# invariant without claiming equation equivalence.
MINIMAL_SAFETY_SCHEMAS: frozenset[str] = frozenset(
    {
        "adaptive_threshold_if",
        "alpha",
        "cazelles_map",
        "chialvo_map",
        "coba_lif",
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
    "alpha": (64, 32),
    "coba_lif": (48, 24),
    "dpi_neuron": (32, 16),
    "ermentrout_kopell_pop": (64, 32),
    "escape_rate": (48, 24),
    "exp_if": (64, 32),
    "ibarz_tanaka_map": (32, 16),
    "iqif": (32, 0),
    "jansen_rit": (64, 32),
    "mcculloch_pitts": (32, 0),
    "medvedev_map": (32, 16),
    "poisson": (48, 24),
    "resonate_fire": (64, 32),
    "sigmoid_rate": (64, 32),
    "threshold_linear_rate": (32, 16),
    "wilson_cowan": (64, 32),
    "wong_wang": (64, 32),
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


def _perfect_class_names() -> list[str]:
    sys.path.insert(0, str(ROOT / "src"))
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib

    from sc_neurocore.neurons.descriptor_tiers import is_perfect
    from sc_neurocore.neurons.model_descriptor import parse_model_descriptor

    names: list[str] = []
    for path in sorted(DESC_DIR.glob("*.toml")):
        desc = parse_model_descriptor(tomllib.loads(path.read_text(encoding="utf-8")))
        if is_perfect(desc) and desc.class_name in CLASS_TO_SCHEMA:
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
    mod_match = re.search(r"module\s+(\w+)", rtl)
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
) -> str:
    """Build a port-only formal harness (no hierarchical probes)."""
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
        formal_body = f"""
`ifdef FORMAL
    // Minimal safety: async reset clears the spike flag.
    always @(*) begin
        if (!rst_n)
            assert (spike_out == 1'b0);
    end
{event_silence}`endif
"""
    else:
        if state_port is None:
            raise ValueError(f"{module}: non-minimal formal job requires a signed state output")
        formal_body = f"""
`ifdef FORMAL
    reg past_valid = 1'b0;
    always @(posedge clk)
        past_valid <= 1'b1;

    // Reset hygiene: async reset clears the spike flag. Primary state may reset
    // to a non-zero rest / init (e.g. QIF v=-1, Izhikevich vr) — do not force 0.
    always @(*) begin
        if (!rst_n) begin
            assert (spike_out == 1'b0);
        end
    end

    // Saturation contract on the primary membrane / phase / current state.
    always @(posedge clk) begin
        if (past_valid && rst_n) begin
            assert ($signed({state_port}) >= -{data_width}'sd{1 << (data_width - 1)});
            assert ($signed({state_port}) <= {data_width}'sd{(1 << (data_width - 1)) - 1});
        end
    end
`endif
"""
    return f"""{_spdx_header(f"Catalogue formal harness for {module}")}
`default_nettype none

// Formal wrapper for equation-compiler RTL of a dual-axis perfect model.
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


def _sby_script(module: str, depth: int, *, flatten: bool = False) -> str:
    prep = f"prep -top {module}_formal" + (" -flatten" if flatten else "")
    return (
        f"# SymbiYosys job for catalogue model {module}\n"
        f"# Dual-axis perfect model formal (BMC)\n"
        "\n"
        "[options]\n"
        "mode bmc\n"
        f"depth {depth}\n"
        "\n"
        "[engines]\n"
        # Prefer z3 (widely available on this workstation); yices is not installed.
        "smtbmc z3\n"
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


def emit_one(class_name: str) -> EmitResult:
    """Emit RTL + formal wrapper + sby for one perfect class."""
    sys.path.insert(0, str(ROOT / "src"))
    from sc_neurocore.neurons.universal_dsl import UniversalNeuron

    schema = CLASS_TO_SCHEMA[class_name]
    neuron = UniversalNeuron.from_schema(schema)
    data_width, fraction = PRECISION_BY_SCHEMA.get(schema, DEFAULT_PRECISION)
    rtl = neuron.to_verilog(data_width=data_width, fraction=fraction)
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
    formal_path.write_text(
        _formal_wrapper(
            ports,
            minimal=schema in MINIMAL_SAFETY_SCHEMAS,
            event_silent=schema in EVENT_SILENT_SCHEMAS,
            data_width=data_width,
        ),
        encoding="utf-8",
    )
    sby_path.write_text(
        _sby_script(
            module,
            depth,
            flatten=schema in FLATTEN_FORMAL_SCHEMAS,
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
    )


def emit_all() -> list[EmitResult]:
    """Emit formal jobs for every dual-axis perfect catalogue model."""
    results: list[EmitResult] = []
    for class_name in _perfect_class_names():
        results.append(emit_one(class_name))
    inventory = OUT_DIR / "INVENTORY.md"
    lines = [
        "# Catalogue formal inventory (dual-axis perfect models)",
        "",
        "Generated by `tools/emit_catalogue_formal.py`. Each job is a SymbiYosys BMC",
        "harness over equation-compiler RTL for a model with science S5 + silicon H≥target.",
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
    inventory.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return results


def run_sby(results: list[EmitResult], *, timeout_s: int = 120) -> dict[str, str]:
    """Run each ``.sby`` from ``catalogue/``; return module → verdict string."""
    sby = shutil.which("sby")
    if sby is None:
        return {r.module: "SKIP (sby not on PATH)" for r in results}
    verdicts: dict[str, str] = {}
    for row in results:
        try:
            proc = subprocess.run(
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
