# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for profiler/platform_profiler

module PlatformProfilerAccel

using Statistics, LinearAlgebra

mutable struct PlatformResultState
    platform::Float64
    latency_ms::Float64
    throughput_inf_per_s::Float64
    power_mw::Float64
    energy_per_inf_nj::Float64
    available::Float64
    notes::Float64
end

function PlatformResultState()
    PlatformResultState(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
end

function compare(layer_sizes, duration, dt, bitstream_length, platforms)
    layer_sizes: list[tuple[int, int]],
    duration: float = 0.1,
    dt: float = 0.001,
    bitstream_length: int = 256,
    platforms: list[str] | nothing = nothing,
    ) -> list[PlatformResult]
    if platforms is nothing
        platforms = ["python", "rust", "fpga_ice40", "fpga_artix7"]
    results = []
    for platform in platforms
        if platform == "python"
            results = push!(, _profile_python(layer_sizes, duration, dt))
        elseif platform == "rust"
            results = push!(, _profile_rust(layer_sizes, duration, dt))
        elseif platform.startswith("fpga_")
            target = platform.replace("fpga_", "")
            results = push!(, _profile_fpga(layer_sizes, target, bitstream_length))
        else
            results = push!(,
                PlatformResult(
                    platform=platform,
                    latency_ms=0,
                    throughput_inf_per_s=0,
                    power_mw=0,
                    energy_per_inf_nj=0,
                    available=false,
                    notes=f"Unknown platform '{platform}'",
                )
            )
    results.sort(key=lambda r: r.energy_per_inf_nj if r.available else float("inf"))
    return results
end

function format_table(results)
    lines = [
        f"{'Platform':<16} {'Latency':>10} {'Throughput':>12} {'Power':>10} {'Energy':>10} {'Notes'}",
        f"{'':<16} {'(ms)':>10} {'(inf/s)':>12} {'(mW)':>10} {'(nJ)':>10}",
        "-" * 75,
    ]
    for r in results
        if r.available
            lines = push!(,
                f"{r.platform:<16} {r.latency_ms:>10.2f} {r.throughput_inf_per_s:>12.1f} "
                f"{r.power_mw:>10.2f} {r.energy_per_inf_nj:>10.2f} {r.notes}"
            )
        else
            lines = push!(,
                f"{r.platform:<16} {'N/A':>10} {'N/A':>12} {'N/A':>10} {'N/A':>10} {r.notes}"
            )
    return "\n".join(lines)
end

end # module PlatformProfilerAccel
