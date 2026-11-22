# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — The GPU waveform encoder refuses a software adapter with the reason

"""Hold the GPU waveform encoder's refusal of a software adapter through the public encoder.

The engine accepts only a physical GPU adapter. Asked for Mesa's software Vulkan adapter
(llvmpipe, through ``SC_GPU_ADAPTER``), it refuses with the adapter's name. Which refusal a
host gives depends on what it has, and each case is asserted exactly:

- an engine without the GPU encoder refuses ``backend="gpu"`` for the missing entry point;
- an engine with it, on a host without the software adapter, finds no such adapter;
- an engine with it, on a host with the software adapter, refuses it as not physical.

The CI job with the software adapter sets ``SC_NEUROCORE_REQUIRE_SOFTWARE_GPU_REFUSAL=1``, so
there the third case is required, not merely allowed.
"""

from __future__ import annotations

import glob
import os

import pytest

from sc_neurocore.spike_codec import probe_codec
from sc_neurocore.spike_codec.waveform_stream import LosslessStreamEncoder

#: The places a Vulkan loader reads its driver manifests from, and the software driver's name.
ICD_GLOBS = ("/usr/share/vulkan/icd.d/lvp_icd*.json", "/etc/vulkan/icd.d/lvp_icd*.json")


def _software_adapter_installed() -> bool:
    """Return whether Mesa's software Vulkan driver is installed where the loader reads it."""
    named = os.environ.get("VK_ICD_FILENAMES", "") + os.environ.get("VK_DRIVER_FILES", "")
    return "lvp_icd" in named or any(glob.glob(pattern) for pattern in ICD_GLOBS)


def test_a_software_adapter_is_refused_with_the_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    """Selecting llvmpipe gives the refusal this host's engine and drivers call for.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Sets ``SC_GPU_ADAPTER`` for the encoder's construction.
    """
    monkeypatch.setenv("SC_GPU_ADAPTER", "llvmpipe")
    encoder_exported = probe_codec("lossless_stream").encoder_only_backends == ("gpu",)
    required = os.environ.get("SC_NEUROCORE_REQUIRE_SOFTWARE_GPU_REFUSAL") == "1"
    if required:
        assert encoder_exported, "the engine was not built with its gpu feature"
        assert _software_adapter_installed(), "Mesa's software Vulkan driver is not installed"
    if not encoder_exported:
        reason = "WaveformLosslessGpuEncoder entry point"
    elif not _software_adapter_installed():
        reason = "no GPU adapter available"
    else:
        reason = "llvmpipe.* is not a physical GPU adapter"
    with pytest.raises(ValueError, match=reason):
        LosslessStreamEncoder(3, 10, 3, "gpu", gpu_max_samples=12)
