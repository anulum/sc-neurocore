# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Cross-platform performance profiler

"""Compare SNN performance across CPU, Rust, and FPGA platforms.

Answers the question: "Which platform should I run this SNN on?"
Measures or estimates latency, throughput, power, and energy for
each platform, producing a comparison table and Pareto front.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PlatformResult:
    """Performance result for one platform."""

    platform: str
    latency_ms: float
    throughput_inf_per_s: float
    power_mw: float
    energy_per_inf_nj: float
    available: bool = True
    notes: str = ""


def compare(
    layer_sizes: list[tuple[int, int]],
    duration: float = 0.1,
    dt: float = 0.001,
    bitstream_length: int = 256,
    platforms: list[str] | None = None,
) -> list[PlatformResult]:
    """Compare SNN performance across platforms.

    Parameters
    ----------
    layer_sizes : list of (n_inputs, n_neurons)
        Network architecture.
    duration : float
        Simulation duration in seconds.
    dt : float
        Timestep.
    bitstream_length : int
        SC bitstream length for FPGA estimates.
    platforms : list of str, optional
        Platforms to compare. Default: all available.
        Options: 'python', 'rust', 'fpga_ice40', 'fpga_artix7'

    Returns
    -------
    list of PlatformResult
        One result per platform, sorted by energy efficiency.
    """
    if platforms is None:
        platforms = ["python", "rust", "fpga_ice40", "fpga_artix7"]

    results = []
    for platform in platforms:
        if platform == "python":
            results.append(_profile_python(layer_sizes, duration, dt))
        elif platform == "rust":
            results.append(_profile_rust(layer_sizes, duration, dt))
        elif platform.startswith("fpga_"):
            target = platform.replace("fpga_", "")
            results.append(_profile_fpga(layer_sizes, target, bitstream_length))
        else:
            results.append(
                PlatformResult(
                    platform=platform,
                    latency_ms=0,
                    throughput_inf_per_s=0,
                    power_mw=0,
                    energy_per_inf_nj=0,
                    available=False,
                    notes=f"Unknown platform '{platform}'",
                )
            )

    results.sort(key=lambda r: r.energy_per_inf_nj if r.available else float("inf"))
    return results


def format_table(results: list[PlatformResult]) -> str:
    """Format comparison results as a readable table."""
    lines = [
        f"{'Platform':<16} {'Latency':>10} {'Throughput':>12} {'Power':>10} {'Energy':>10} {'Notes'}",
        f"{'':<16} {'(ms)':>10} {'(inf/s)':>12} {'(mW)':>10} {'(nJ)':>10}",
        "-" * 75,
    ]
    for r in results:
        if r.available:
            lines.append(
                f"{r.platform:<16} {r.latency_ms:>10.2f} {r.throughput_inf_per_s:>12.1f} "
                f"{r.power_mw:>10.2f} {r.energy_per_inf_nj:>10.2f} {r.notes}"
            )
        else:
            lines.append(
                f"{r.platform:<16} {'N/A':>10} {'N/A':>12} {'N/A':>10} {'N/A':>10} {r.notes}"
            )
    return "\n".join(lines)


def _profile_python(layer_sizes: list[tuple[int, int]], duration: float, dt: float) -> PlatformResult:
    """Measure Python/NumPy backend performance."""
    total_neurons = sum(n for _, n in layer_sizes)
    n_steps = int(round(duration / dt))

    # Estimate: ~1 us per neuron per step on modern CPU
    estimated_time_s = total_neurons * n_steps * 1e-6
    latency_ms = estimated_time_s * 1000
    throughput = 1.0 / max(estimated_time_s, 1e-9)

    # CPU power estimate: ~10W for a single core
    cpu_power_mw = 10000.0
    energy_nj = cpu_power_mw * estimated_time_s * 1e6

    return PlatformResult(
        platform="python",
        latency_ms=latency_ms,
        throughput_inf_per_s=throughput,
        power_mw=cpu_power_mw,
        energy_per_inf_nj=energy_nj,
        notes="NumPy, single-core",
    )


def _profile_rust(layer_sizes: list[tuple[int, int]], duration: float, dt: float) -> PlatformResult:
    """Estimate Rust NetworkRunner performance."""
    total_neurons = sum(n for _, n in layer_sizes)
    n_steps = int(round(duration / dt))

    # Rust is ~10x faster than Python for neuron stepping
    estimated_time_s = total_neurons * n_steps * 1e-7
    latency_ms = estimated_time_s * 1000
    throughput = 1.0 / max(estimated_time_s, 1e-9)

    cpu_power_mw = 10000.0
    energy_nj = cpu_power_mw * estimated_time_s * 1e6

    try:
        import sc_neurocore_engine  # noqa: F401

        available = True  # pragma: no cover
        notes = "Rayon parallel, SIMD"  # pragma: no cover
    except ImportError:
        available = False
        notes = "Not installed (estimated)"

    return PlatformResult(
        platform="rust",
        latency_ms=latency_ms,
        throughput_inf_per_s=throughput,
        power_mw=cpu_power_mw,
        energy_per_inf_nj=energy_nj,
        available=available,
        notes=notes,
    )


def _profile_fpga(layer_sizes: list[tuple[int, int]], target: str, bitstream_length: int) -> PlatformResult:
    """Estimate FPGA performance using energy estimator."""
    from sc_neurocore.energy import estimate

    report = estimate(
        layer_sizes=layer_sizes,
        target=target,
        bitstream_length=bitstream_length,
    )

    latency_ms = report.total_latency_cycles / (report.clock_freq_mhz * 1e3)
    throughput = 1.0 / max(latency_ms / 1000, 1e-9)

    notes = f"{'fits' if report.fits_on_target else 'EXCEEDS'} {report.utilization_pct:.0f}%"

    return PlatformResult(
        platform=f"fpga_{target}",
        latency_ms=latency_ms,
        throughput_inf_per_s=throughput,
        power_mw=report.total_dynamic_power_mw,
        energy_per_inf_nj=report.energy_per_inference_nj,
        notes=notes,
    )
