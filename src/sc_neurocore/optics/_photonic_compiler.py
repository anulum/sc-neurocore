# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Photonic compiler and layout export

"""Compile optical bitstreams and export deterministic HDL or GDS artefacts."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from ._photonic_conversion import BitstreamToOptical, _normalise_bitstream
from ._photonic_emitter import PhotonicEmitter
from ._photonic_fdtd import FDTDSolver, _require_count
from ._photonic_types import PhotonicTarget, _require_non_negative, _require_positive


@dataclass
class CompilationResult:
    """Result of a photonic compilation pass."""

    target: str
    num_modulators: int
    optical_power_mean_mw: float
    phase_coverage_rad: float
    netlist: str
    fdtd_energy: float = 0.0

    def __post_init__(self) -> None:
        """Validate compilation metadata before export."""
        if not isinstance(self.target, str) or not self.target.strip():
            raise ValueError("target must be a non-empty string")
        _require_count(self.num_modulators, "num_modulators")
        _require_non_negative(self.optical_power_mean_mw, "optical_power_mean_mw")
        _require_non_negative(self.phase_coverage_rad, "phase_coverage_rad")
        _require_non_negative(self.fdtd_energy, "fdtd_energy")
        if not isinstance(self.netlist, str):
            raise TypeError("netlist must be a string")

    def to_gdsii(
        self,
        filename: str,
        mzi_length_um: float = 10.0,
        pitch_um: float = 100.0,
    ) -> Dict[str, Any]:
        """Export the compiled MZI cascade to GDSII through gdsfactory.

        The layout is a linear cascade at ``pitch_um`` spacing. An identifying
        header and a bounded copy of the logical netlist are stored on TEXT
        layer 63/0. The optional ``gdsfactory`` dependency is loaded only when
        this method is called.
        """
        if self.num_modulators <= 0:
            raise NotImplementedError(
                "to_gdsii() requires num_modulators > 0; the compiler produced an "
                "empty layout (check the input bitstream and compiler target)."
            )
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError("filename must be a non-empty string")
        _require_positive(mzi_length_um, "mzi_length_um")
        _require_positive(pitch_um, "pitch_um")
        try:
            import gdsfactory as gf
        except ImportError as exc:
            raise ImportError(
                "gdsfactory is not installed. Run `pip install gdsfactory` or "
                "`pip install 'sc-neurocore[optics]'` to enable GDSII export."
            ) from exc

        try:
            gf.get_active_pdk()
        except (ValueError, AttributeError):
            try:
                gf.gpdk.PDK.activate()
            except AttributeError:
                from gdsfactory.gpdk import get_generic_pdk

                get_generic_pdk().activate()

        kdb_cell = gf.kcl.create_cell(f"SC_NeuroCore_Target_{self.target}", allow_duplicate=True)
        component = gf.Component(kdb_cell=kdb_cell)
        component.add_label(
            text=f"sc_neurocore:{self.target} N={self.num_modulators}",
            position=(0.0, 10.0),
            layer=(63, 0),
        )
        if self.netlist:
            component.add_label(
                text=self.netlist[: min(200, len(self.netlist))],
                position=(0.0, -10.0),
                layer=(63, 0),
            )

        mzi_cell = gf.components.mzi(length_x=mzi_length_um)
        x = 0.0
        for _ in range(self.num_modulators):
            reference = component.add_ref(mzi_cell)
            reference.x = x
            x += pitch_um

        component.write_gds(filename)
        return {
            "filename": filename,
            "n_modulators": self.num_modulators,
            "mzi_length_um": mzi_length_um,
            "pitch_um": pitch_um,
            "total_length_um": x,
            "target": self.target,
        }


class PhotonicCompiler:
    """Compile an SC bitstream into optical mapping, netlist, and co-simulation."""

    def __init__(self, target: Optional[PhotonicTarget] = None):
        if target is not None and not isinstance(target, PhotonicTarget):
            raise TypeError("target must be a PhotonicTarget or None")
        self.target = target or PhotonicTarget.silicon_photonics()
        self.converter = BitstreamToOptical(self.target)
        self.emitter = PhotonicEmitter(self.target.name)

    def compile_bitstream(
        self,
        bitstream: np.ndarray[Any, Any],
        run_fdtd: bool = False,
        fdtd_steps: int = 100,
    ) -> CompilationResult:
        """Compile one non-empty binary SC bitstream to a photonic deployment."""
        if bitstream is None:
            raise ValueError("Input bitstream cannot be empty.")
        if not isinstance(run_fdtd, bool):
            raise TypeError("run_fdtd must be a Boolean")
        _require_count(fdtd_steps, "fdtd_steps")
        normalised = _normalise_bitstream(bitstream)
        if normalised.size == 0:
            raise ValueError("Input bitstream cannot be empty.")

        phases = self.converter.to_phase_array(normalised)
        power = self.converter.optical_power_profile(normalised)
        amplitudes = self.converter.to_amplitude_array(normalised)
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
        for index, (phase, amplitude) in enumerate(zip(phases, amplitudes)):
            if self.target.modulator_type == "MZI":
                netlist_lines.append(f"ADD MZI mod_{index}")
                netlist_lines.append(f"SET mod_{index}:phase {phase:.6f}")
                netlist_lines.append(f"SET mod_{index}:amplitude {amplitude:.6f}")
            else:
                netlist_lines.append(f"ADD MICRORING ring_{index}")
                netlist_lines.append(f"SET ring_{index}:coupling {amplitude:.6f}")
                netlist_lines.append(f"SET ring_{index}:detuning {phase:.6f}")

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
            netlist="\n".join(netlist_lines),
            fdtd_energy=fdtd_energy,
        )

    def generate_mzi_verilog(self, bit_width: int = 16) -> str:
        """Generate SystemVerilog for an MZI modulator."""
        _require_count(bit_width, "bit_width", minimum=2)
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
        _require_count(bit_width, "bit_width", minimum=2)
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


__all__ = ["CompilationResult", "PhotonicCompiler"]
