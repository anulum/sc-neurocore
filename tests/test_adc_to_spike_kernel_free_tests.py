# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Module-level tests from former test_adc_to_spike_kernel.py

"""Module-level tests from former test_adc_to_spike_kernel.py."""

from __future__ import annotations

from tests.adc_to_spike_kernel_support import *  # noqa: F403


def test_sensors_package_exports_adc_to_spike_kernel_surface() -> None:
    """The sensors package facade exposes the ADC-to-spike kernel surface."""
    expected_names = {
        "ADCSpikeWindowConfig",
        "ADCSpikeWindowResult",
        "adc_to_spike_windows",
        "adc_to_spike_windows_q",
        "available_backends",
        "quantise_adc",
    }

    assert expected_names <= set(sensors.__all__)
    assert sensors.ADCSpikeWindowConfig is ADCSpikeWindowConfig
    assert sensors.adc_to_spike_windows is adc_to_spike_windows
    assert sensors.adc_to_spike_windows_q is adc_to_spike_windows_q
