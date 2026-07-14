# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — CMOS hardware-profile registration facade

"""Register CMOS, FPGA, ASIC, MCU, and conventional accelerator profiles."""

from __future__ import annotations

from ._cmos_accelerator_profiles import (
    _register_ai_accelerator_profiles,
    _register_edge_ai_accelerator_profiles,
    _register_riscv_ai_accelerator_profiles,
    _register_vision_sensor_profiles,
)
from ._cmos_architecture_profiles import (
    _register_cgra_profiles,
    _register_emerging_compute_profiles,
    _register_stacked_3d_profiles,
)
from ._cmos_fpga_profiles import (
    _register_additional_fpga_profiles,
    _register_embedded_fpga_profiles,
    _register_ice40_fpga_profiles,
    _register_intel_fpga_profiles,
    _register_lattice_fpga_profiles,
    _register_other_fpga_profiles,
    _register_radiation_hardened_fpga_profiles,
    _register_xilinx_fpga_profiles,
)
from ._cmos_processor_profiles import (
    _register_dsp_profiles,
    _register_edge_mcu_profiles,
)
from ._cmos_reference_profiles import (
    _register_asic_profiles,
    _register_simulation_profiles,
)
from .registry import HardwareProfile as HardwareProfile

_PROFILE_REGISTRARS = (
    _register_xilinx_fpga_profiles,
    _register_intel_fpga_profiles,
    _register_lattice_fpga_profiles,
    _register_other_fpga_profiles,
    _register_ice40_fpga_profiles,
    _register_asic_profiles,
    _register_simulation_profiles,
    _register_additional_fpga_profiles,
    _register_ai_accelerator_profiles,
    _register_dsp_profiles,
    _register_emerging_compute_profiles,
    _register_radiation_hardened_fpga_profiles,
    _register_edge_ai_accelerator_profiles,
    _register_embedded_fpga_profiles,
    _register_vision_sensor_profiles,
    _register_cgra_profiles,
    _register_stacked_3d_profiles,
    _register_edge_mcu_profiles,
    _register_riscv_ai_accelerator_profiles,
)

for _register_profile_group in _PROFILE_REGISTRARS:
    _register_profile_group()
