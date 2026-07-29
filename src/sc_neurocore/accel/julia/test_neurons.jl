# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia neuron parity-suite aggregator

using Printf

# Each entry owns its model-specific inputs, equations, state validation, and
# output contract. Do not replace this registry with a universal `simulate`
# call: the Julia neuron surfaces are intentionally heterogeneous.
const NEURON_PARITY_SUITES = (
    "adex_parity_test.jl",
    "connor_stevens_parity_test.jl",
    "dpi_neuron_parity_test.jl",
    "expif_parity_test.jl",
    "hodgkin_huxley_parity_test.jl",
    "lapicque_parity_test.jl",
    "morris_lecar_parity_test.jl",
    "perfect_integrator_parity_test.jl",
    "quadratic_if_parity_test.jl",
    "theta_parity_test.jl",
    "wang_buzsaki_parity_test.jl",
)

function test_neuron_parity_suites(suite_dir::String = @__DIR__)
    passed = 0
    failed = 0

    println("=" ^ 70)
    println("SC-NeuroCore authoritative Julia neuron parity suites")
    println("=" ^ 70)
    println("Testing $(length(NEURON_PARITY_SUITES)) model-specific suites...")
    println()

    for suite in NEURON_PARITY_SUITES
        path = joinpath(suite_dir, suite)
        if !isfile(path)
            failed += 1
            @printf("  FAIL %-44s missing\n", suite)
            continue
        end

        process = run(ignorestatus(`$(Base.julia_cmd()) --startup-file=no $path`))
        if success(process)
            passed += 1
            @printf("  PASS %-44s\n", suite)
        else
            failed += 1
            @printf("  FAIL %-44s exit=%d\n", suite, process.exitcode)
        end
    end

    println()
    println("=" ^ 70)
    println("RESULTS: $passed passed, $failed failed out of $(length(NEURON_PARITY_SUITES))")
    println("=" ^ 70)
    return passed, failed
end

if abspath(PROGRAM_FILE) == @__FILE__
    _, failed = test_neuron_parity_suites()
    exit(failed == 0 ? 0 : 1)
end
