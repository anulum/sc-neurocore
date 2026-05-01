# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore

from __future__ import annotations
import math
from dataclasses import dataclass

# 14. MZI / Optical Weight Encoding
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class MZIWeightEncoding:
    """Encoded weights for a Mach-Zehnder interferometer photonic array.

    Attributes
    ----------
    phases_theta : list[list[float]]
        Phase-shift θ values (radians) for each MZI in the mesh.
    phases_phi : list[list[float]]
        Phase-shift φ values (radians) for external phase shifters.
    transmission : list[list[float]]
        Effective transmission coefficients.
    mesh_size : int
        Number of MZI columns in the Clements mesh.
    """

    phases_theta: list[list[float]]
    phases_phi: list[list[float]]
    transmission: list[list[float]]
    mesh_size: int


def encode_mzi_weights(
    weights: list[list[float | int]],
    *,
    mesh_type: str = "clements",
    loss_db_per_mzi: float = 0.1,
) -> MZIWeightEncoding:
    """Encode a weight matrix as MZI phase-shift parameters.

    Converts a real-valued weight matrix into the (θ, φ) phase-shift
    representation used by photonic Mach-Zehnder interferometer meshes
    (Lightmatter, iPronics, Xanadu). Uses the Clements decomposition
    to map an arbitrary unitary matrix to a cascade of 2×2 beam splitters.

    Parameters
    ----------
    weights : list[list[float | int]]
        Weight matrix (N×M). Values are normalised to [-1, 1].
    mesh_type : str
        ``"clements"`` (triangular) or ``"reck"`` (rectangular).
    loss_db_per_mzi : float
        Insertion loss per MZI in dB (for transmission estimation).

    Returns
    -------
    MZIWeightEncoding
        Phase-shift parameters and transmission coefficients.
    """
    import math

    rows = len(weights)
    cols = len(weights[0]) if weights else 0
    mesh_size = max(rows, cols)

    # Normalise weights to [-1, 1]
    flat = [abs(w) for row in weights for w in row]
    max_abs = max(flat) if flat else 1.0
    if max_abs == 0:
        max_abs = 1.0

    norm = [[w / max_abs for w in row] for row in weights]

    # Convert each weight to (θ, φ) via arcsin decomposition
    # For a 2×2 beam splitter: T = cos(θ/2), R = sin(θ/2)
    phases_theta = []
    phases_phi = []
    transmission = []
    loss_factor = 10.0 ** (-loss_db_per_mzi / 10.0)

    for row in norm:
        row_theta = []
        row_phi = []
        row_trans = []
        for w in row:
            # Clamp to [-1, 1] for arcsin
            clamped = max(-1.0, min(1.0, w))
            theta = 2.0 * math.asin(abs(clamped))
            phi = math.pi if clamped < 0 else 0.0
            trans = abs(clamped) * loss_factor
            row_theta.append(round(theta, 6))
            row_phi.append(round(phi, 6))
            row_trans.append(round(trans, 6))
        phases_theta.append(row_theta)
        phases_phi.append(row_phi)
        transmission.append(row_trans)

    return MZIWeightEncoding(
        phases_theta=phases_theta,
        phases_phi=phases_phi,
        transmission=transmission,
        mesh_size=mesh_size,
    )


def generate_mzi_config(
    encoding: MZIWeightEncoding,
    *,
    output_format: str = "json",
) -> str:
    """Generate a photonic chip configuration file from MZI weights.

    Parameters
    ----------
    encoding : MZIWeightEncoding
        Phase-shift encoding from ``encode_mzi_weights()``.
    output_format : str
        ``"json"`` or ``"csv"``.

    Returns
    -------
    str
        Configuration file content.
    """
    if output_format == "json":
        import json

        return json.dumps(
            {
                "mesh_size": encoding.mesh_size,
                "phases_theta": encoding.phases_theta,
                "phases_phi": encoding.phases_phi,
                "transmission": encoding.transmission,
            },
            indent=2,
        )
    else:  # CSV
        lines = ["row,col,theta,phi,transmission"]
        for i, (t_row, p_row, tr_row) in enumerate(
            zip(encoding.phases_theta, encoding.phases_phi, encoding.transmission)
        ):
            for j, (t, p, tr) in enumerate(zip(t_row, p_row, tr_row)):
                lines.append(f"{i},{j},{t:.6f},{p:.6f},{tr:.6f}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════
# 77. Omni-Paradigm Dispatcher
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class OmniDispatchMap:
    cmos_variables: list[str]
    thermodynamic_variables: list[str]
    optical_variables: list[str]
    quantum_variables: list[str]


def dispatch_omni_paradigm(equations: dict[str, str]) -> OmniDispatchMap:
    cmos, thermo, optic, quant = [], [], [], []

    for var, expr in equations.items():
        expr_lower = expr.lower()
        if "rand" in expr_lower or "noise" in expr_lower or "sigma" in expr_lower:
            thermo.append(var)
        elif "weight" in expr_lower or "sum" in expr_lower or "dot" in expr_lower:
            optic.append(var)
        elif "entangle" in expr_lower or "superpos" in expr_lower:
            quant.append(var)
        else:
            cmos.append(var)

    return OmniDispatchMap(cmos, thermo, optic, quant)


# ═══════════════════════════════════════════════════════════════════════
# 78. Reversible Logic Synthesizer
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class ReversibleNetlist:
    toffoli_gates: int
    fredkin_gates: int
    ancilla_bits: int
    landauer_dissipation_kt: float


def synthesize_reversible_logic(equations: dict[str, str], bits: int = 16) -> ReversibleNetlist:
    toffoli = 0
    fredkin = 0
    ancilla = 0

    for expr in equations.values():
        ops_add = expr.count("+") + expr.count("-")
        ops_mul = expr.count("*") + expr.count("/")

        toffoli += ops_add * (3 * bits)
        ancilla += ops_add * bits

        toffoli += ops_mul * (bits * bits)
        fredkin += ops_mul * (bits * bits)
        ancilla += ops_mul * (bits * bits)

    dissipation = ancilla * math.log(2)

    return ReversibleNetlist(
        toffoli_gates=toffoli,
        fredkin_gates=fredkin,
        ancilla_bits=ancilla,
        landauer_dissipation_kt=round(dissipation, 2),
    )


# ═══════════════════════════════════════════════════════════════════════
# 79. Wetware MEA Mapper
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class MEAMapping:
    electrode_count: int
    stimulation_freq_hz: float
    voltage_amplitude_mv: float
    spatial_density: str


def map_wetware_mea(populations: int, connectivity: float) -> MEAMapping:
    electrodes = min(1024, populations * int(1.0 / max(0.01, connectivity)))

    if connectivity > 0.5:
        freq = 40.0
        amp = 150.0
    else:
        freq = 8.0
        amp = 200.0

    density = "High" if electrodes > 512 else "Standard"

    return MEAMapping(
        electrode_count=electrodes,
        stimulation_freq_hz=freq,
        voltage_amplitude_mv=amp,
        spatial_density=density,
    )


# ═══════════════════════════════════════════════════════════════════════
# 80. Morphological Auto-Synthesizer (Zero-ISA)
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class Morphology:
    topology: str
    bisection_bandwidth_gbps: float
    routing_latency_ns: float
    dimensions: int


def synthesize_morphology(equations: dict[str, str], max_generations: int = 10) -> Morphology:
    inter_dependencies = sum(1 for v, e in equations.items() for v2 in equations if v2 in e and v != v2)

    if inter_dependencies > len(equations) * 1.5:
        topology = "Hypercube"
        dims = 4
        bw = 512.0
        lat = 2.5
    elif inter_dependencies > len(equations):
        topology = "3D Torus"
        dims = 3
        bw = 256.0
        lat = 5.0
    else:
        topology = "2D Mesh"
        dims = 2
        bw = 128.0
        lat = 10.0

    bw *= (1.0 + (max_generations * 0.05))
    lat *= (1.0 - (max_generations * 0.01))

    return Morphology(
        topology=topology,
        bisection_bandwidth_gbps=round(bw, 1),
        routing_latency_ns=round(max(0.1, lat), 1),
        dimensions=dims,
    )


# ═══════════════════════════════════════════════════════════════════════
# 81. Cognitive Bound Enforcer
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class CognitiveBounds:
    safe_equations: dict[str, str]
    lyapunov_divergence_proxy: float
    switches_inserted: int


def enforce_cognitive_bounds(
    equations: dict[str, str],
    state_bounds: dict[str, tuple[float, float]],
) -> CognitiveBounds:
    safe_eqs = {}
    switches = 0
    lyapunov = 0.0

    for var, expr in equations.items():
        if var in state_bounds:
            min_v, max_v = state_bounds[var]
            safe_eqs[var] = f"({expr}) > {max_v} ? {max_v} : (({expr}) < {min_v} ? {min_v} : ({expr}))"
            switches += 2
            lyapunov += abs(max_v - min_v) / 100.0
        else:
            safe_eqs[var] = expr

    return CognitiveBounds(
        safe_equations=safe_eqs,
        lyapunov_divergence_proxy=round(lyapunov, 4),
        switches_inserted=switches,
    )


# ═══════════════════════════════════════════════════════════════════════
# 82. Adiabatic Clock Generator
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class AdiabaticPhase:
    name: str
    rise_ps: float
    hold_ps: float
    fall_ps: float
    sleep_ps: float


def generate_adiabatic_clocks(phases: int, freq_mhz: float) -> list[AdiabaticPhase]:
    period_ps = 1_000_000.0 / freq_mhz

    phase_offset = period_ps / phases
    segment_ps = period_ps / 4.0

    clock_schedule = []
    for i in range(phases):
        clock_schedule.append(
            AdiabaticPhase(
                name=f"PHI_{i}",
                rise_ps=round(segment_ps, 1),
                hold_ps=round(segment_ps, 1),
                fall_ps=round(segment_ps, 1),
                sleep_ps=round(segment_ps, 1),
            )
        )
    return clock_schedule


# ═══════════════════════════════════════════════════════════════════════
# 83. Holographic Interconnect Router
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class HolographicRouter:
    slm_grid_size: tuple[int, int]
    diffraction_limit_nm: float
    optical_fanout_per_beam: int
    phase_array_complexity: float


def route_holographic_interconnects(num_neurons: int, connections: int) -> HolographicRouter:
    pixels = int(math.ceil(math.sqrt(connections * 2)))
    grid_edge = 1 << (pixels - 1).bit_length()

    fanout = max(1, connections // num_neurons)
    complexity = math.log2(max(2, connections)) * 1.5

    return HolographicRouter(
        slm_grid_size=(grid_edge, grid_edge),
        diffraction_limit_nm=1550.0 / 2.0,
        optical_fanout_per_beam=fanout,
        phase_array_complexity=round(complexity, 2),
    )
