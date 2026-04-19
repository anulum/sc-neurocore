# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Photonic Emitter & Compiler

"""Photonic netlist emitter, SC-to-optical compiler, and 1D FDTD co-simulation.

Converts SC bitstreams into optical pulse trains (phase/amplitude modulation)
targeting Mach-Zehnder interferometers and microring resonators.  Includes a
minimal finite-difference time-domain solver for waveguide propagation
verification.
"""

from __future__ import annotations

import math
import textwrap
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from sc_neurocore_engine import py_ph_analyze_crosstalk
    import sc_neurocore_engine as _sne

    _HAS_RUST_PH = all(
        hasattr(_sne, _name) for _name in ("py_ph_route_waveguides", "py_ph_analyze_power_budget")
    )
    del _sne
except ImportError:
    _HAS_RUST_PH = False


# ── Existing Photonic Emitter (preserved) ────────────────────────────


class PhotonicEmitter:
    """Industrial-grade Photonic Emitter.
    Uses topological sorting to ensure optical ports are defined before being coupled.
    """

    def __init__(self, target_pdk: str = "generic_si_photonics"):
        self.target_pdk = target_pdk

    def _topological_sort(self, nodes: List[Any]) -> List[Any]:
        in_degree = {n.id: 0 for n in nodes}
        node_map = {n.id: n for n in nodes}
        adj = {n.id: [] for n in nodes}
        output_to_id = {n.output: n.id for n in nodes}

        for n in nodes:
            for inp in n.inputs:
                if inp in output_to_id:
                    adj[output_to_id[inp]].append(n.id)
                    in_degree[n.id] += 1

        queue = [n_id for n_id, deg in in_degree.items() if deg == 0]
        sorted_nodes = []
        while queue:
            curr = queue.pop(0)
            sorted_nodes.append(node_map[curr])
            for neighbor in adj[curr]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        return sorted_nodes

    def emit_lumerical_netlist(self, ir_graph: Any) -> str:
        sorted_nodes = self._topological_sort(ir_graph.nodes)
        netlist = ["# SC-NeuroCore Photonic Design", f"# PDK: {self.target_pdk}", ""]

        established_ports = set()

        for node in sorted_nodes:
            if node.type == "SC_AND":
                netlist.append(f"ADD MZI_MODULATOR {node.id}")
                netlist.append(f"CONNECT {node.id}:in1 {node.inputs[0]}")
                netlist.append(f"CONNECT {node.id}:in2 {node.inputs[1]}")
                netlist.append(f"SET {node.id}:phase_pi 3.14159")
            elif node.type == "LIF_MEMBRANE":
                netlist.append(f"ADD RESONANT_CAVITY {node.id}")
                netlist.append(f"CONNECT {node.id}:input {node.inputs[0]}")
                netlist.append(f"SET {node.id}:Q_factor 15000")

            established_ports.add(node.output)

        return "\n".join(netlist)


# ── Optical Modulation Types ─────────────────────────────────────────


class OpticalModulation(Enum):
    """Optical modulation scheme."""

    PHASE = "phase"
    AMPLITUDE = "amplitude"
    HYBRID = "hybrid"


@dataclass
class PhotonicTarget:
    """Hardware target specification for a photonic backend."""

    name: str
    wavelength_nm: float = 1550.0
    modulation: OpticalModulation = OpticalModulation.PHASE
    modulator_type: str = "MZI"
    q_factor: float = 15000.0
    insertion_loss_db: float = 0.5
    thermo_optic_coeff: float = 1.86e-4  # dn/dT for silicon (K⁻¹)

    @classmethod
    def lightmatter(cls) -> PhotonicTarget:
        return cls("Lightmatter", 1550.0, OpticalModulation.PHASE, "MZI", 20000.0, 0.3)

    @classmethod
    def silicon_photonics(cls) -> PhotonicTarget:
        return cls("SiPh-Generic", 1310.0, OpticalModulation.AMPLITUDE, "Microring", 12000.0, 0.8)

    @classmethod
    def two_d_waveguide(cls) -> PhotonicTarget:
        return cls("2D-Material", 850.0, OpticalModulation.HYBRID, "MZI", 5000.0, 1.2)


# ── Bitstream-to-Optical Converter ───────────────────────────────────


@dataclass
class OpticalPulse:
    """Single optical pulse with phase and amplitude."""

    phase: float  # radians
    amplitude: float  # normalised [0, 1]
    wavelength_nm: float
    duration_ps: float  # picoseconds


class BitstreamToOptical:
    """Converts SC bitstreams into optical pulse trains."""

    def __init__(self, target: PhotonicTarget):
        self.target = target

    def convert(
        self,
        bitstream: np.ndarray,
        pulse_duration_ps: float = 10.0,
    ) -> List[OpticalPulse]:
        """Map a boolean SC bitstream to an optical pulse train.

        Phase modulation: bit=1 → phase=0, bit=0 → phase=π.
        Amplitude modulation: bit=1 → amp=1, bit=0 → amp=0.
        Hybrid: both phase and amplitude encoding.
        """
        pulses = []
        for bit in bitstream:
            b = int(bit) & 1
            if self.target.modulation == OpticalModulation.PHASE:
                phase = 0.0 if b else math.pi
                amplitude = 1.0
            elif self.target.modulation == OpticalModulation.AMPLITUDE:
                phase = 0.0
                amplitude = float(b)
            else:
                phase = 0.0 if b else math.pi / 2
                amplitude = 0.8 + 0.2 * float(b)
            pulses.append(
                OpticalPulse(
                    phase=phase,
                    amplitude=amplitude,
                    wavelength_nm=self.target.wavelength_nm,
                    duration_ps=pulse_duration_ps,
                )
            )
        return pulses

    def to_phase_array(self, bitstream: np.ndarray) -> np.ndarray:
        """Vectorised phase extraction."""
        bs = bitstream.astype(np.float64)
        if self.target.modulation == OpticalModulation.PHASE:
            return np.where(bs > 0.5, 0.0, math.pi)
        elif self.target.modulation == OpticalModulation.AMPLITUDE:
            return np.zeros_like(bs)
        else:
            return np.where(bs > 0.5, 0.0, math.pi / 2)

    def to_amplitude_array(self, bitstream: np.ndarray) -> np.ndarray:
        """Vectorised amplitude extraction."""
        bs = bitstream.astype(np.float64)
        if self.target.modulation == OpticalModulation.PHASE:
            return np.ones_like(bs)
        elif self.target.modulation == OpticalModulation.AMPLITUDE:
            return bs
        else:
            return 0.8 + 0.2 * bs

    def optical_power_profile(
        self,
        bitstream: np.ndarray,
        input_power_mw: float = 1.0,
    ) -> np.ndarray:
        """Compute output power profile accounting for insertion loss."""
        amplitudes = self.to_amplitude_array(bitstream)
        loss_linear = 10.0 ** (-self.target.insertion_loss_db / 10.0)
        return amplitudes * amplitudes * input_power_mw * loss_linear


# ── FDTD Co-Simulation ──────────────────────────────────────────────


class FDTDSolver:
    """1D finite-difference time-domain solver for waveguide co-simulation.

    Solves Maxwell's equations on a 1D grid with Yee discretisation.
    Suitable for verifying pulse propagation, dispersion, and loss in
    simple waveguide geometries.
    """

    def __init__(
        self,
        grid_size: int = 1000,
        dx_um: float = 0.01,
        dt_factor: float = 0.5,
        refractive_index: float = 3.48,
    ):
        self.grid_size = grid_size
        self.dx = dx_um * 1e-6  # metres
        self.c0 = 3e8
        self.n = refractive_index
        self.v = self.c0 / self.n
        self.dt = dt_factor * self.dx / self.c0
        self.ez = np.zeros(grid_size, dtype=np.float64)
        self.hy = np.zeros(grid_size, dtype=np.float64)
        self._loss_per_metre = 0.0

    def set_loss(self, loss_db_per_cm: float) -> None:
        """Set propagation loss."""
        self._loss_per_metre = loss_db_per_cm * 100.0

    def inject_pulse(
        self,
        position: int,
        wavelength_nm: float = 1550.0,
        amplitude: float = 1.0,
        phase: float = 0.0,
    ) -> None:
        """Inject a Gaussian-envelope optical pulse at a grid position."""
        freq = self.c0 / (wavelength_nm * 1e-9)
        sigma = 20
        for i in range(max(0, position - 3 * sigma), min(self.grid_size, position + 3 * sigma)):
            r = (i - position) / sigma
            envelope = amplitude * math.exp(-0.5 * r * r)
            self.ez[i] = envelope * math.cos(2 * math.pi * freq * 0 + phase)

    def step(self, n_steps: int = 1) -> None:
        """Advance the simulation by *n_steps* timesteps."""
        coeff_e = self.dt / (self.dx * self.n**2 * 8.854e-12)
        coeff_h = self.dt / (self.dx * 4 * math.pi * 1e-7)

        if self._loss_per_metre > 0:
            alpha = self._loss_per_metre * np.log(10) / 20.0
            loss_factor = math.exp(-alpha * self.dx)
        else:
            loss_factor = 1.0

        for _ in range(n_steps):
            self.hy[:-1] += coeff_h * (self.ez[1:] - self.ez[:-1])
            self.ez[1:] += coeff_e * (self.hy[1:] - self.hy[:-1])
            if loss_factor < 1.0:
                self.ez *= loss_factor

    def field_energy(self) -> float:
        """Total electromagnetic energy in the grid."""
        return float(np.sum(self.ez**2) + np.sum(self.hy**2))

    def snapshot(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return a copy of the current E and H fields."""
        return self.ez.copy(), self.hy.copy()


# ── Photonic Compiler ────────────────────────────────────────────────


@dataclass
class CompilationResult:
    """Result of a photonic compilation pass."""

    target: str
    num_modulators: int
    optical_power_mean_mw: float
    phase_coverage_rad: float
    netlist: str
    fdtd_energy: float = 0.0


class PhotonicCompiler:
    """End-to-end compiler: SC IR → optical mapping → netlist → co-sim."""

    def __init__(self, target: Optional[PhotonicTarget] = None):
        self.target = target or PhotonicTarget.silicon_photonics()
        self.converter = BitstreamToOptical(self.target)
        self.emitter = PhotonicEmitter(self.target.name)

    def compile_bitstream(
        self,
        bitstream: np.ndarray,
        run_fdtd: bool = False,
        fdtd_steps: int = 100,
    ) -> CompilationResult:
        """Compile a single SC bitstream to a photonic deployment."""
        phases = self.converter.to_phase_array(bitstream)
        power = self.converter.optical_power_profile(bitstream)

        mzi_count = int(np.sum(np.abs(np.diff(phases)) > 0.01))

        netlist_lines = [
            "# SC-NeuroCore Photonic Compilation",
            f"# Target: {self.target.name}",
            f"# Wavelength: {self.target.wavelength_nm} nm",
            f"# Modulation: {self.target.modulation.value}",
            "",
            f"SET global:wavelength {self.target.wavelength_nm}e-9",
            f"SET global:q_factor {self.target.q_factor}",
            "",
        ]
        for i, (ph, amp) in enumerate(zip(phases, self.converter.to_amplitude_array(bitstream))):
            if self.target.modulator_type == "MZI":
                netlist_lines.append(f"ADD MZI mod_{i}")
                netlist_lines.append(f"SET mod_{i}:phase {ph:.6f}")
                netlist_lines.append(f"SET mod_{i}:amplitude {amp:.6f}")
            else:
                netlist_lines.append(f"ADD MICRORING ring_{i}")
                netlist_lines.append(f"SET ring_{i}:coupling {amp:.6f}")
                netlist_lines.append(f"SET ring_{i}:detuning {ph:.6f}")

        netlist = "\n".join(netlist_lines)

        fdtd_energy = 0.0
        if run_fdtd:
            solver = FDTDSolver(grid_size=500, refractive_index=self.target.wavelength_nm / 450.0)
            solver.inject_pulse(50, self.target.wavelength_nm, amplitude=float(np.mean(power)))
            solver.step(fdtd_steps)
            fdtd_energy = solver.field_energy()

        return CompilationResult(
            target=self.target.name,
            num_modulators=max(1, mzi_count),
            optical_power_mean_mw=float(np.mean(power)),
            phase_coverage_rad=float(np.max(phases) - np.min(phases)),
            netlist=netlist,
            fdtd_energy=fdtd_energy,
        )

    def generate_mzi_verilog(self, bit_width: int = 16) -> str:
        """Generate SystemVerilog for an MZI modulator."""
        bw = bit_width
        return textwrap.dedent(f"""\
            // SPDX-License-Identifier: AGPL-3.0-or-later
            // Commercial license available
            // © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
            // © Code 2020–2026 Miroslav Šotek. All rights reserved.
            // ORCID: 0009-0009-3560-0851
            // Contact: www.anulum.li | protoscience@anulum.li
            // SC-NeuroCore — Mach-Zehnder Interferometer Modulator

            module sc_photonic_mzi #(
                parameter BW = {bw}
            )(
                input  logic clk,
                input  logic rst_n,
                input  logic [{bw - 1}:0] i_bitstream,
                input  logic signed [{bw - 1}:0] i_phase_q8_8,
                output logic [{bw - 1}:0] o_optical_out,
                output logic o_valid
            );

                localparam signed [{bw - 1}:0] PI_Q8_8 = {bw}'sd804;

                logic signed [{bw - 1}:0] phase_reg;
                logic [{bw - 1}:0] arm_a, arm_b;

                always_ff @(posedge clk or negedge rst_n) begin
                    if (!rst_n) begin
                        phase_reg <= '0;
                        o_optical_out <= '0;
                        o_valid <= 1'b0;
                    end else begin
                        phase_reg <= i_phase_q8_8;
                        arm_a <= i_bitstream;
                        arm_b <= (phase_reg > (PI_Q8_8 >>> 1)) ? ~i_bitstream : i_bitstream;
                        o_optical_out <= arm_a ^ arm_b;
                        o_valid <= 1'b1;
                    end
                end

            endmodule
        """)

    def generate_microring_verilog(self, bit_width: int = 16) -> str:
        """Generate SystemVerilog for a microring resonator."""
        bw = bit_width
        return textwrap.dedent(f"""\
            // SPDX-License-Identifier: AGPL-3.0-or-later
            // Commercial license available
            // © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
            // © Code 2020–2026 Miroslav Šotek. All rights reserved.
            // ORCID: 0009-0009-3560-0851
            // Contact: www.anulum.li | protoscience@anulum.li
            // SC-NeuroCore — Microring Resonator Modulator

            module sc_photonic_microring #(
                parameter BW = {bw},
                parameter Q_FACTOR = 15000
            )(
                input  logic clk,
                input  logic rst_n,
                input  logic [{bw - 1}:0] i_bitstream,
                input  logic [{bw - 1}:0] i_coupling,
                output logic [{bw - 1}:0] o_through,
                output logic [{bw - 1}:0] o_drop,
                output logic o_resonant
            );

                logic [{bw - 1}:0] coupling_reg;
                logic [{bw - 1}:0] accumulator;

                always_ff @(posedge clk or negedge rst_n) begin
                    if (!rst_n) begin
                        coupling_reg <= '0;
                        accumulator <= '0;
                        o_through <= '0;
                        o_drop <= '0;
                        o_resonant <= 1'b0;
                    end else begin
                        coupling_reg <= i_coupling;
                        o_through <= i_bitstream & ~coupling_reg;
                        o_drop    <= i_bitstream & coupling_reg;
                        accumulator <= accumulator + {{({bw}-1){{1'b0}}}}, (|o_drop)}};
                        o_resonant  <= (accumulator > ({bw}'d{2 ** (bw - 2)}));
                    end
                end

            endmodule
        """)


# ── 2D FDTD Solver ──────────────────────────────────────────────────


class FDTD2DSolver:
    """2D finite-difference time-domain solver (TE mode).

    Yee grid with optional PML absorbing boundaries.
    Solves for Ez, Hx, Hy on a 2D cross-section of the photonic chip.
    """

    def __init__(
        self,
        nx: int = 200,
        ny: int = 100,
        dx_um: float = 0.01,
        dy_um: float = 0.01,
        dt_factor: float = 0.5,
        pml_layers: int = 10,
    ):
        self.nx = nx
        self.ny = ny
        self.dx = dx_um * 1e-6
        self.dy = dy_um * 1e-6
        self.c0 = 3e8
        ds_min = min(self.dx, self.dy)
        self.dt = dt_factor * ds_min / (self.c0 * math.sqrt(2))
        self.pml_layers = pml_layers

        # Fields: Ez (nx, ny), Hx (nx, ny), Hy (nx, ny)
        self.ez = np.zeros((nx, ny), dtype=np.float64)
        self.hx = np.zeros((nx, ny), dtype=np.float64)
        self.hy = np.zeros((nx, ny), dtype=np.float64)

        # Material map: refractive index per cell
        self.n_map = np.ones((nx, ny), dtype=np.float64)

        # PML conductivity profiles
        self._sigma_x = np.zeros((nx, ny), dtype=np.float64)
        self._sigma_y = np.zeros((nx, ny), dtype=np.float64)
        self._build_pml()

    def _build_pml(self) -> None:
        """Construct PML damping mask (exponential taper)."""
        self._damping = np.ones((self.nx, self.ny), dtype=np.float64)
        p = self.pml_layers
        for i in range(p):
            strength = 1.0 - 0.8 * ((p - i) / p) ** 2
            self._damping[i, :] = np.minimum(self._damping[i, :], strength)
            self._damping[self.nx - 1 - i, :] = np.minimum(
                self._damping[self.nx - 1 - i, :], strength
            )
            self._damping[:, i] = np.minimum(self._damping[:, i], strength)
            self._damping[:, self.ny - 1 - i] = np.minimum(
                self._damping[:, self.ny - 1 - i], strength
            )

    def set_waveguide(
        self,
        y_center: int,
        width_cells: int,
        refractive_index: float = 3.48,
        x_start: int = 0,
        x_end: Optional[int] = None,
    ) -> None:
        """Define a horizontal waveguide stripe."""
        x_end = x_end or self.nx
        y_lo = max(0, y_center - width_cells // 2)
        y_hi = min(self.ny, y_center + width_cells // 2)
        self.n_map[x_start:x_end, y_lo:y_hi] = refractive_index

    def inject_source(
        self,
        x: int,
        y: int,
        wavelength_nm: float = 1550.0,
        amplitude: float = 1.0,
        sigma_cells: int = 10,
    ) -> None:
        """Inject a 2D Gaussian source."""
        freq = self.c0 / (wavelength_nm * 1e-9)
        for ix in range(max(0, x - 3 * sigma_cells), min(self.nx, x + 3 * sigma_cells)):
            for iy in range(max(0, y - 3 * sigma_cells), min(self.ny, y + 3 * sigma_cells)):
                dx_r = (ix - x) / sigma_cells
                dy_r = (iy - y) / sigma_cells
                envelope = amplitude * math.exp(-0.5 * (dx_r**2 + dy_r**2))
                self.ez[ix, iy] = envelope * math.cos(2 * math.pi * freq * 0)

    def step(self, n_steps: int = 1) -> None:
        """Advance TE simulation by n_steps."""
        eps0 = 8.854e-12
        mu0 = 4 * math.pi * 1e-7

        eps_map = eps0 * self.n_map**2
        coeff_ez = self.dt / eps_map

        coeff_hx = self.dt / (mu0 * self.dx)
        coeff_hy = self.dt / (mu0 * self.dy)

        for _ in range(n_steps):
            # Update Hx: dHx/dt = -1/mu0 * dEz/dy
            self.hx[:, :-1] -= coeff_hx * (self.ez[:, 1:] - self.ez[:, :-1])

            # Update Hy: dHy/dt = 1/mu0 * dEz/dx
            self.hy[:-1, :] += coeff_hy * (self.ez[1:, :] - self.ez[:-1, :])

            # Update Ez: dEz/dt = 1/eps * (dHy/dx - dHx/dy)
            self.ez[1:, :] += coeff_ez[1:, :] * (self.hy[1:, :] - self.hy[:-1, :])
            self.ez[:, 1:] -= coeff_ez[:, 1:] * (self.hx[:, 1:] - self.hx[:, :-1])

            # PML damping
            self.ez *= self._damping
            self.hx *= self._damping
            self.hy *= self._damping

    def field_energy(self) -> float:
        """Total EM energy in the grid."""
        return float(np.sum(self.ez**2) + np.sum(self.hx**2) + np.sum(self.hy**2))

    def field_at_point(self, x: int, y: int) -> float:
        """Read Ez at a grid point."""
        return float(self.ez[x, y])

    def cross_section(self, x: int) -> np.ndarray:
        """Return Ez cross-section at a given x position."""
        return self.ez[x, :].copy()

    def snapshot(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return copies of all field components."""
        return self.ez.copy(), self.hx.copy(), self.hy.copy()


# ── Meep Co-Simulation Adapter ──────────────────────────────────────


class MeepAdapter:
    """Optional adapter for the Meep FDTD solver.

    Provides a thin wrapper that translates SC-NeuroCore photonic
    configurations into Meep simulation objects. Meep is an optional
    dependency; all methods gracefully degrade if not installed.
    """

    @staticmethod
    def is_available() -> bool:
        """Check if Meep is installed."""
        try:
            import meep  # noqa: F401

            return True
        except ImportError:
            return False

    @staticmethod
    def build_waveguide_geometry(
        target: PhotonicTarget,
        waveguide_width_um: float = 0.5,
        length_um: float = 10.0,
        substrate_index: float = 1.45,
    ) -> Dict[str, Any]:
        """Build a Meep geometry dict (usable even without Meep).

        Returns a serialisable description of the simulation setup.
        """
        core_index = 3.48 if target.wavelength_nm > 1000 else 2.0
        wavelength_um = target.wavelength_nm / 1000.0
        freq = 1.0 / wavelength_um  # Meep normalised frequency

        return {
            "cell_size": [length_um, 3.0 * waveguide_width_um, 0],
            "resolution": 20,
            "sources": [
                {
                    "type": "ContinuousSource"
                    if target.modulation == OpticalModulation.PHASE
                    else "GaussianSource",
                    "frequency": freq,
                    "center": [-length_um / 2 + 0.5, 0, 0],
                    "size": [0, waveguide_width_um, 0],
                }
            ],
            "geometry": [
                {
                    "type": "Block",
                    "material_index": core_index,
                    "center": [0, 0, 0],
                    "size": [length_um, waveguide_width_um, "Infinity"],
                },
            ],
            "substrate_index": substrate_index,
            "pml_layers": 1.0,
            "wavelength_nm": target.wavelength_nm,
            "modulation": target.modulation.value,
        }

    @staticmethod
    def run_simulation(geometry: Dict[str, Any], run_time: float = 50.0) -> Dict[str, Any]:
        """Run a Meep simulation or return a mock result if unavailable."""
        if not MeepAdapter.is_available():
            # Mock result for testing without Meep
            return {
                "transmission": 0.85,
                "reflection": 0.02,
                "field_decay": 1e-4,
                "run_time": run_time,
                "mock": True,
                "wavelength_nm": geometry.get("wavelength_nm", 1550.0),
            }

        import meep as mp

        cell_size = geometry["cell_size"]
        resolution = geometry["resolution"]
        src_spec = geometry["sources"][0]
        geo_spec = geometry["geometry"][0]

        cell = mp.Vector3(*cell_size)
        freq = src_spec["frequency"]

        sources = [
            mp.Source(
                mp.ContinuousSource(frequency=freq),
                component=mp.Ez,
                center=mp.Vector3(*src_spec["center"]),
                size=mp.Vector3(*src_spec["size"]),
            )
        ]

        mat = mp.Medium(index=geo_spec["material_index"])
        geo_objs = [
            mp.Block(
                size=mp.Vector3(geo_spec["size"][0], geo_spec["size"][1]),
                center=mp.Vector3(*geo_spec["center"]),
                material=mat,
            )
        ]

        sim = mp.Simulation(
            cell_size=cell,
            resolution=resolution,
            sources=sources,
            geometry=geo_objs,
            boundary_layers=[mp.PML(geometry["pml_layers"])],
        )

        flux_region = mp.FluxRegion(
            center=mp.Vector3(cell_size[0] / 2 - 1, 0),
            size=mp.Vector3(0, cell_size[1]),
        )
        trans = sim.add_flux(freq, 0, 1, flux_region)

        sim.run(until=run_time)

        flux_data = mp.get_fluxes(trans)
        return {
            "transmission": float(flux_data[0]) if flux_data else 0.0,
            "reflection": 0.0,
            "field_decay": 0.0,
            "run_time": run_time,
            "mock": False,
            "wavelength_nm": geometry.get("wavelength_nm", 1550.0),
        }


# ── Waveguide Crosstalk Model ───────────────────────────────────────


@dataclass
class WaveguidePair:
    """A pair of adjacent waveguides for crosstalk analysis."""

    waveguide_width_nm: float = 450.0
    gap_nm: float = 200.0
    coupling_length_um: float = 10.0
    core_index: float = 3.48
    cladding_index: float = 1.45
    wavelength_nm: float = 1550.0

    @property
    def effective_index_diff(self) -> float:
        """Approximate effective index difference between even/odd modes."""
        # Exponential evanescent decay model
        decay_length_nm = self.wavelength_nm / (
            2 * math.pi * math.sqrt(self.core_index**2 - self.cladding_index**2)
        )
        return 0.1 * math.exp(-self.gap_nm / decay_length_nm)

    @property
    def coupling_coefficient(self) -> float:
        """Coupling coefficient κ (per micrometre)."""
        dn = self.effective_index_diff
        return math.pi * dn / (self.wavelength_nm * 1e-3)

    @property
    def coupling_ratio(self) -> float:
        """Power coupling ratio at the end of the coupling length."""
        kl = self.coupling_coefficient * self.coupling_length_um
        return math.sin(kl) ** 2

    @property
    def isolation_db(self) -> float:
        """Isolation in dB (lower coupling = better isolation)."""
        ratio = self.coupling_ratio
        if ratio < 1e-15:
            return 300.0
        return -10.0 * math.log10(max(ratio, 1e-30))


class CrosstalkModel:
    """Models evanescent crosstalk between adjacent waveguides.

    Uses coupled-mode theory to compute power transfer between
    parallel waveguide runs on a photonic chip.
    """

    def __init__(self):
        self.pairs: List[WaveguidePair] = []

    def add_pair(self, pair: WaveguidePair) -> None:
        self.pairs.append(pair)

    def transfer_matrix(self, pair: WaveguidePair) -> np.ndarray:
        """2×2 transfer matrix for a directional coupler."""
        kl = pair.coupling_coefficient * pair.coupling_length_um
        c = math.cos(kl)
        s = math.sin(kl)
        return np.array([[c, 1j * s], [1j * s, c]])

    def compute_crosstalk(
        self, pair: WaveguidePair, input_power: Tuple[float, float] = (1.0, 0.0)
    ) -> Tuple[float, float]:
        """Compute output power on both waveguides."""
        t = self.transfer_matrix(pair)
        inp = np.array(input_power, dtype=complex)
        out = t @ inp
        return float(np.abs(out[0]) ** 2), float(np.abs(out[1]) ** 2)

    def worst_case_isolation(self) -> float:
        """Worst-case isolation across all pairs (minimum dB)."""
        if not self.pairs:
            return float("inf")
        return min(p.isolation_db for p in self.pairs)

    def analyze_bank(
        self, waveguides: int, gap_nm: float, coupling_length_um: float
    ) -> Dict[str, Any]:
        """Analyze a bank of parallel waveguides.

        Delegates to Rust engine when available for O(N²) acceleration.
        """
        if _HAS_RUST_PH and waveguides > 10:
            channel_ids = list(range(waveguides - 1))
            wavelengths = [1550.0] * (waveguides - 1)
            bandwidths = [0.8] * (waveguides - 1)
            powers = [1.0] * (waveguides - 1)
            result = py_ph_analyze_crosstalk(
                channel_ids,
                wavelengths,
                bandwidths,
                powers,
            )
            return {
                "num_waveguides": waveguides,
                "num_pairs": waveguides - 1,
                "gap_nm": gap_nm,
                "worst_isolation_db": result.get("min_isolation_db", float("inf")),
                "mean_coupling_ratio": result.get("mean_coupling", 0.0),
                "max_coupling_ratio": result.get("max_coupling", 0.0),
                "crosstalk_safe": result.get("min_isolation_db", 0.0) > 20.0,
                "backend": "rust",
            }

        pairs = []
        for i in range(waveguides - 1):
            pairs.append(WaveguidePair(gap_nm=gap_nm, coupling_length_um=coupling_length_um))
        self.pairs = pairs

        isolations = [p.isolation_db for p in pairs]
        couplings = [p.coupling_ratio for p in pairs]

        return {
            "num_waveguides": waveguides,
            "num_pairs": len(pairs),
            "gap_nm": gap_nm,
            "worst_isolation_db": min(isolations) if isolations else float("inf"),
            "mean_coupling_ratio": float(np.mean(couplings)) if couplings else 0.0,
            "max_coupling_ratio": max(couplings) if couplings else 0.0,
            "crosstalk_safe": all(iso > 20.0 for iso in isolations),
        }
