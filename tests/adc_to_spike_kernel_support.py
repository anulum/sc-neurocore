# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_adc_to_spike_kernel.py

from __future__ import annotations

"""Algorithm, validation and dispatch tests for the ADC-to-spike encoder.

Cross-language bit-exact parity and golden-reference parity are exercised in
``tests/test_adc_to_spike_kernel_parity.py``.
"""
from collections.abc import Callable
import numpy as np
import numpy.testing as npt
import numpy.typing as nptyping
import pytest
import sc_neurocore.sensors as sensors
from sc_neurocore.sensors import adc_to_spike_kernel as kernel
from sc_neurocore.sensors.adc_to_spike_kernel import (
    ADCSpikeWindowConfig,
    ADCSpikeWindowResult,
    adc_to_spike_windows,
    adc_to_spike_windows_q,
    available_backends,
    quantise_adc,
)

__all__ = ['Callable', 'np', 'npt', 'nptyping', 'pytest', 'sensors', 'kernel', 'ADCSpikeWindowConfig', 'ADCSpikeWindowResult', 'adc_to_spike_windows', 'adc_to_spike_windows_q', 'available_backends', 'quantise_adc']
