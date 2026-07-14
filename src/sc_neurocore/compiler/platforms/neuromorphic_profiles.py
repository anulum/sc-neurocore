# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — neuromorphic hardware-profile registration facade

"""Register event-driven, emerging, and specialised hardware profiles."""

from __future__ import annotations

from ._event_driven_hardware_profiles import (
    _register_additional_neuromorphic_profiles,
    _register_neuromorphic_chip_profiles,
    _register_recent_neuromorphic_profiles,
)
from ._heterogeneous_accelerator_profiles import (
    _register_ai_accelerator_profiles,
    _register_automotive_edge_profiles,
    _register_chiplet_accelerator_profiles,
)
from ._memory_compute_profiles import (
    _register_ferroelectric_profiles,
    _register_processing_in_memory_profiles,
    _register_rram_profiles,
    _register_spintronic_profiles,
    _register_sram_cim_profiles,
)
from ._physical_compute_profiles import (
    _register_analog_mixed_signal_profiles,
    _register_cryogenic_cmos_profiles,
    _register_emerging_compute_profiles,
    _register_molecular_profiles,
    _register_photonic_compute_profiles,
)
from ._programmable_aerospace_profiles import (
    _register_additional_fpga_profiles,
    _register_aerospace_profiles,
)
from .registry import HardwareProfile as HardwareProfile

_PROFILE_REGISTRARS = (
    _register_neuromorphic_chip_profiles,
    _register_recent_neuromorphic_profiles,
    _register_additional_fpga_profiles,
    _register_ai_accelerator_profiles,
    _register_emerging_compute_profiles,
    _register_photonic_compute_profiles,
    _register_chiplet_accelerator_profiles,
    _register_processing_in_memory_profiles,
    _register_additional_neuromorphic_profiles,
    _register_aerospace_profiles,
    _register_automotive_edge_profiles,
    _register_spintronic_profiles,
    _register_ferroelectric_profiles,
    _register_analog_mixed_signal_profiles,
    _register_rram_profiles,
    _register_sram_cim_profiles,
    _register_cryogenic_cmos_profiles,
    _register_molecular_profiles,
)

for _register_profile_group in _PROFILE_REGISTRARS:
    _register_profile_group()
