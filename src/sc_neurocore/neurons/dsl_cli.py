# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — CLI entry point for UniversalNeuron DSL

"""Command-line interface for the UniversalNeuron DSL.

Subcommands
-----------
list
    List all 18 bundled model schemas with state variables and DOIs.

validate [model]
    Validate all schemas (or a specific one) against the JSON meta-schema.

info <model>
    Display model metadata, equations, parameters, threshold, and reset rules.

compile <model> [-p {q88,q412,q1616}] [-o output.v]
    Compile a model schema to synthesizable Verilog RTL.  The ``-p`` flag
    selects the fixed-point precision mode (default: ``q88``).

simulate <model> [-n steps] [-I current]
    Run the Python simulation for N steps with constant input current.

precision <model>
    Analyse how each parameter and the integration timestep encode
    across Q8.8, Q4.12, and Q16.16, with overflow/underflow warnings
    and a recommended precision mode.

Usage::

    python -m sc_neurocore.neurons list
    python -m sc_neurocore.neurons compile lif -p q1616 -o lif_hd.v
    python -m sc_neurocore.neurons precision lif
    python -m sc_neurocore.neurons simulate lif -n 200 -I 50.0
"""

from __future__ import annotations

import argparse
import sys

from sc_neurocore.neurons.schema_validator import validate_all_bundled, validate_schema
from sc_neurocore.neurons.universal_dsl import UniversalNeuron, list_bundled_schemas


def cmd_list(args: argparse.Namespace) -> None:
    """List all bundled schemas."""
    schemas = list_bundled_schemas()
    print(f"Bundled model schemas: {len(schemas)}")
    for name in schemas:
        n = UniversalNeuron.from_schema(name)
        vars_ = ", ".join(n.list_state_variables())
        doi = n.doi or "—"
        print(f"  {name:20s}  state=[{vars_}]  doi={doi}")


def cmd_validate(args: argparse.Namespace) -> None:
    """Validate schemas."""
    if args.model:
        errors = validate_schema(args.model)
        real_errors = [e for e in errors if e.level == "error"]
        warnings = [e for e in errors if e.level == "warning"]
        for e in errors:
            print(f"  {e}")
        if real_errors:
            print(f"\n✗ {args.model}: {len(real_errors)} error(s)")
            sys.exit(1)
        print(f"\n✓ {args.model}: OK ({len(warnings)} warning(s))")
    else:
        results = validate_all_bundled()
        total_errors = 0
        total_warnings = 0
        for name, errors in sorted(results.items()):
            real_errors = [e for e in errors if e.level == "error"]
            warnings = [e for e in errors if e.level == "warning"]
            status = "✓" if not real_errors else "✗"
            print(f"  {status} {name:20s}  {len(real_errors)} error(s), {len(warnings)} warning(s)")
            for e in errors:
                if e.level == "error":
                    print(f"      {e}")
            total_errors += len(real_errors)
            total_warnings += len(warnings)
        print(
            f"\nTotal: {len(results)} schemas, {total_errors} error(s), {total_warnings} warning(s)"
        )
        if total_errors > 0:
            sys.exit(1)


def cmd_info(args: argparse.Namespace) -> None:
    """Show model information."""
    n = UniversalNeuron.from_schema(args.model)
    schema = n.schema
    meta = schema.get("metadata", {})

    print(f"Model: {n.name}")
    print(f"Author: {meta.get('author', '—')}")
    print(f"Year: {meta.get('year', '—')}")
    print(f"DOI: {meta.get('doi', '—')}")
    print(f"Description: {meta.get('description', '—')}")
    print()
    print(f"State variables: {n.list_state_variables()}")
    print(f"Parameters: {n.list_parameters()}")
    print()
    print("ODEs:")
    for var, eq in n.list_equations().items():
        print(f"  d{var}/dt = {eq}")

    if schema.get("threshold"):
        print(f"\nThreshold: {schema['threshold'].get('condition', '—')}")
    if schema.get("reset"):
        print("Reset:")
        for var, expr in schema["reset"].items():
            print(f"  {var} ← {expr}")
    if n.extensions:
        print(f"\nExtensions: {n.extensions}")


# ── Precision mode registry ─────────────────────────────────────────────
# Each entry: cli_name → (data_width, fraction, display_name, description)
# Ordered by data_width then fraction for display consistency.
PRECISION_MODES: dict[str, tuple[int, int, str, str]] = {
    # 8-bit tier
    "q17": (8, 7, "Q1.7", "8-bit ultra-compact (Loihi/TrueNorth-class)"),
    # 16-bit tier
    "q88": (16, 8, "Q8.8", "16-bit default (mV-scale models)"),
    "q412": (16, 12, "Q4.12", "16-bit high precision (normalised dynamics)"),
    "q115": (16, 15, "Q1.15", "16-bit DSP fractional (ARM CMSIS standard)"),
    # 18-bit tier (DSP48-native)
    "q99": (18, 9, "Q9.9", "18-bit DSP48-native (zero-waste Xilinx/Lattice)"),
    # 24-bit tier
    "q1212": (24, 12, "Q12.12", "24-bit audio-grade (Loihi-2 native)"),
    # 27-bit tier (Intel Stratix DSP-native)
    "q1413": (27, 13, "Q14.13", "27-bit Stratix-native (Intel 27×27 DSP)"),
    # 32-bit tier
    "q2012": (32, 12, "Q20.12", "32-bit network-level (10K synapse accumulation)"),
    "q1616": (32, 16, "Q16.16", "32-bit gold standard"),
    "q824": (32, 24, "Q8.24", "32-bit ultra-precision (EP training)"),
    # 36-bit tier (DSP48E2-native)
    "q1818": (36, 18, "Q18.18", "36-bit DSP48E2-native (UltraScale)"),
}


def cmd_compile(args: argparse.Namespace) -> None:
    """Compile model to Verilog with optional hardware target profile."""
    from sc_neurocore.compiler.platforms import get_profile

    n = UniversalNeuron.from_schema(args.model)

    # Determine precision: --target overrides --precision defaults
    if args.target:
        profile = get_profile(args.target)
        dw = profile.data_width
        frac = profile.fraction
        display = f"{profile.q_format_label} ({profile.name})"
        overflow = args.overflow or profile.overflow
        rounding = args.rounding or profile.rounding
    else:
        dw, frac, display, _desc = PRECISION_MODES[args.precision]
        overflow = args.overflow or "saturate"
        rounding = args.rounding or "truncate"

    try:
        verilog = n.to_verilog(
            module_name=args.module_name,
            data_width=dw,
            fraction=frac,
            overflow=overflow,
            rounding=rounding,
        )
    except Exception as e:
        print(f"Compilation error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.output:
        with open(args.output, "w") as f:
            f.write(verilog)
        print(
            f"Written: {args.output} ({len(verilog)} bytes, {display}, "
            f"overflow={overflow}, rounding={rounding})"
        )
    else:
        print(verilog)


def cmd_simulate(args: argparse.Namespace) -> None:
    """Simulate model for N steps."""
    n = UniversalNeuron.from_schema(args.model)
    spikes = 0
    for step in range(args.steps):
        fired = n.step(I=args.current)
        if fired:
            spikes += 1
            if not args.quiet:
                print(f"  SPIKE at step {step}")

    print(f"\nModel: {n.name}")
    print(f"Steps: {args.steps}, Current: {args.current}")
    print(f"Spikes: {spikes}")
    print(f"Final state: {n.state}")


def cmd_precision(args: argparse.Namespace) -> None:
    """Show precision diagnostics for a model at every supported format.

    Analyses all 9 precision modes, checking parameter range,
    dt encoding, and providing a recommended mode.
    """
    from sc_neurocore.compiler.equation_compiler import Q88

    n = UniversalNeuron.from_schema(args.model)
    schema = n.schema
    dt = schema.get("integration", {}).get("dt", 0.001)
    params = {
        **schema.get("parameters", {}),
        **schema.get("state", {}),
    }

    # Build analysis modes from registry
    modes = [
        (display, Q88(data_width=dw, fraction=frac))
        for _cli, (dw, frac, display, _desc) in PRECISION_MODES.items()
    ]

    print(f"Precision analysis for: {n.name}")
    print(f"{'=' * 72}")

    compatible: list[tuple[str, int, int]] = []  # (display, dw, frac)

    for mode_name, q in modes:
        print(f"\n{mode_name} ({q.data_width}-bit, {q.fraction} frac):")
        report = q.precision_report(dt, params)
        print(report)

        # Check compatibility: dt must not underflow AND all params in range
        dt_raw = int(round(dt * (1 << q.fraction)))
        range_ok = all(q.min_value <= v <= q.max_value for v in params.values())
        if dt_raw > 0 and range_ok:
            compatible.append((mode_name, q.data_width, q.fraction))

    # Recommendation engine
    print(f"\n{'=' * 72}")
    print(f"Compatible modes: {', '.join(c[0] for c in compatible) or 'NONE'}")

    if not compatible:
        print("Recommendation: Use custom (data_width, fraction) via API")
    else:
        # Prefer smallest data_width that works, then most fraction bits
        best = min(compatible, key=lambda c: (c[1], -c[2]))
        print(f"Recommendation: {best[0]} (smallest compatible format)")

        # Also suggest the most precise compatible
        precise = max(compatible, key=lambda c: c[2])
        if precise[0] != best[0]:
            print(f"  For max precision: {precise[0]}")


def cmd_platforms(args: argparse.Namespace) -> None:
    """List all available hardware target profiles."""
    from sc_neurocore.compiler.platforms import HardwareProfile, list_profiles

    profiles = list_profiles()

    # Group by platform_class
    classes: dict[str, list[HardwareProfile]] = {
        "fpga": [],
        "neuromorphic": [],
        "accelerator": [],
        "dsp": [],
        "asic": [],
        "emerging": [],
        "simulation": [],
    }
    for p in profiles:
        classes.setdefault(p.platform_class, []).append(p)

    for cls_name, cls_profiles in classes.items():
        if not cls_profiles:
            continue
        print(f"\n{cls_name.upper()} ({len(cls_profiles)} targets):")
        print(
            f"  {'Name':18s} {'Vendor':12s} {'Family':18s} {'Format':10s} "
            f"{'Bits':>4s} {'OVF':>8s} {'RND':>10s}  Notes"
        )
        print(f"  {'-' * 110}")
        for p in cls_profiles:
            print(
                f"  {p.name:18s} {p.vendor:12s} {p.family:18s} "
                f"{p.q_format_label:10s} {p.data_width:4d} "
                f"{p.overflow:>8s} {p.rounding:>10s}  {p.notes[:50]}"
            )

    print(f"\nTotal: {len(profiles)} hardware profiles")
    print("Usage: python -m sc_neurocore.neurons compile lif --target loihi2")


def main() -> None:
    """Parse CLI arguments and dispatch to the requested subcommand."""
    parser = argparse.ArgumentParser(
        prog="sc_neurocore.neurons.dsl_cli",
        description="SC-NeuroCore UniversalNeuron DSL command-line interface",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # list
    sub.add_parser("list", help="List all bundled model schemas")

    # validate
    p_val = sub.add_parser("validate", help="Validate model schemas")
    p_val.add_argument("model", nargs="?", help="Specific model to validate (default: all)")

    # info
    p_info = sub.add_parser("info", help="Show model information")
    p_info.add_argument("model", help="Model name (e.g. 'lif')")

    # compile
    p_comp = sub.add_parser("compile", help="Compile model to Verilog RTL")
    p_comp.add_argument("model", help="Model name")
    p_comp.add_argument("--output", "-o", help="Output file (default: stdout)")
    p_comp.add_argument("--module-name", help="Verilog module name (default: auto)")
    p_comp.add_argument(
        "--precision",
        "-p",
        default="q88",
        choices=list(PRECISION_MODES.keys()),
        help="Fixed-point format (default: q88). Options: "
        + ", ".join(f"{k} ({v[2]})" for k, v in PRECISION_MODES.items()),
    )
    p_comp.add_argument(
        "--target",
        "-t",
        default=None,
        help="Hardware target profile (e.g. 'loihi2', 'artix7'). "
        "Overrides --precision with optimal settings for the target.",
    )
    p_comp.add_argument(
        "--overflow",
        default=None,
        choices=["saturate", "wrap", "trap"],
        help="Overflow mode (default: saturate, or from target profile)",
    )
    p_comp.add_argument(
        "--rounding",
        default=None,
        choices=["truncate", "nearest", "bankers", "stochastic"],
        help="Rounding mode (default: truncate, or from target profile)",
    )

    # simulate
    p_sim = sub.add_parser("simulate", help="Simulate model for N steps")
    p_sim.add_argument("model", help="Model name")
    p_sim.add_argument("--steps", "-n", type=int, default=100, help="Number of steps")
    p_sim.add_argument("--current", "-I", type=float, default=10.0, help="Input current")
    p_sim.add_argument("--quiet", "-q", action="store_true", help="Suppress per-spike output")

    # precision
    p_prec = sub.add_parser("precision", help="Show precision diagnostics for a model")
    p_prec.add_argument("model", help="Model name")

    # platforms
    sub.add_parser("platforms", help="List all available hardware target profiles")

    args = parser.parse_args()

    commands = {
        "list": cmd_list,
        "validate": cmd_validate,
        "info": cmd_info,
        "compile": cmd_compile,
        "simulate": cmd_simulate,
        "precision": cmd_precision,
        "platforms": cmd_platforms,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
