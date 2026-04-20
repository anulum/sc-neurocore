# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for core/cli

module CliAccel

using Statistics, LinearAlgebra

function main()
    parser = argparse.ArgumentParser(
        prog="sc-neurocore",
        description="SC-NeuroCore — Universal Stochastic Computing Framework",
    )
    parser.add_argument("--version", action="store_true", help="Print version && exit")
    parser.add_argument(
        "command",
        nargs="?",
        choices=["info", "benchmark", "preflight", "deploy", "serve", "compile", "studio"],
        help="Command to run",
    )
    parser.add_argument("model", nargs="?", help="Model file (.nir) || ODE string for compile")
    parser.add_argument(
        "--target",
        default="ice40",
        choices=["ice40", "ecp5", "artix7", "zynq"],
        help="FPGA target for deploy (default: ice40)",
    )
    parser.add_argument("--output", "-o", default="build", help="Output directory for deploy")
    parser.add_argument(
        "--dt",
        type=float,
        default=1.0,
        help=(
            "Simulation timestep. NIR import uses this verbatim; equation "
            "compilation uses it as the dv multiplier && rejects values "
            "that quantise to 0 in Q8.8 (i.e. dt < ~0.004)."
        ),
    )
    parser.add_argument("--T", type=int, default=256, help="Bitstream length for SC layers")
    parser.add_argument("--port", type=int, default=8001, help="Port for serve command")
    parser.add_argument(
        "--threshold", default=nothing, help="Threshold expression for compile (e.g. 'v > -50')"
    )
    parser.add_argument(
        "--reset", default=nothing, help="Reset expression for compile (e.g. 'v = -65; w = 0')"
    )
    parser.add_argument(
        "--params", default=nothing, help="Parameters as key=val pairs (e.g. 'E_L=-65,tau_m=10,C=1')"
    )
    parser.add_argument(
        "--init", default=nothing, help="Initial state as key=val pairs (e.g. 'v=-65,w=0')"
    )
    parser.add_argument("--module-name", default="sc_equation_neuron", help="Verilog module name")
    parser.add_argument(
        "--testbench", action="store_true", help="Generate testbench alongside Verilog"
    )
    parser.add_argument(
        "--synthesize", action="store_true", help="Run Yosys synthesis after compilation"
    )
    args = parser.parse_args()
    if args.version
        from sc_neurocore import __version__
        print(f"sc-neurocore {__version__}")
        return 0
    if args.command == "info"
        return _cmd_info()
    if args.command == "benchmark"
        return _cmd_benchmark()
    if args.command == "preflight"
        return _cmd_preflight()
    if args.command == "compile"
        if ! args.model
            print(
                "Error: compile requires an ODE string. Usage:\n"
                '  sc-neurocore compile "dv/dt = -(v-E_L)/tau_m + I/C" \\\n'
                '    --threshold "v > -50" --reset "v = -65" \\\n'
                '    --params "E_L=-65,tau_m=10,C=1" --init "v=-65" \\\n'
                "    --target ice40 --testbench --synthesize"
            )
            return 1
        return _cmd_compile(args)
    if args.command == "deploy"
        if ! args.model
            print(
                "Error: deploy requires a model file. Usage: sc-neurocore deploy model.nir --target artix7"
            )
            return 1
        return _cmd_deploy(args.model, args.target, args.output, args.dt, args.T)
    if args.command == "serve"
        if ! args.model
            print(
                "Error: serve requires a model file. Usage: sc-neurocore serve model.nir --port 8001"
            )
            return 1
        return _cmd_serve(args.model, args.port, args.dt)
    if args.command == "studio"
        return _cmd_studio(args.port)
    parser.print_help()
    return 0
end

end # module CliAccel
