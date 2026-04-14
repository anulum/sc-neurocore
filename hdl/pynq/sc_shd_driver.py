# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SHD FPGA Driver for PYNQ-Z2
"""SC-NeuroCore SHD FPGA driver for PYNQ-Z2.

Provides a Python interface to the sc_shd_top hardware accelerator
via memory-mapped AXI-Lite registers.

Usage:
    from pynq import Overlay
    from sc_shd_driver import SHDAccelerator

    overlay = Overlay("sc_shd.bit")
    accel = SHDAccelerator(overlay.sc_shd_axi_wrapper_0)

    # Load scales (from trained model checkpoint)
    accel.set_scales(scale_l1, scale_l2, scale_l3)

    # Run inference on a spike raster (T x 140 binary matrix)
    predicted_class = accel.classify(spike_raster)
"""

import numpy as np


# Register offsets (byte addresses)
_CTRL = 0x00
_T_ORIG = 0x04
_SCALE_L1 = 0x08
_SCALE_L2 = 0x0C
_SCALE_L3 = 0x10
_SPIKE_IN_0 = 0x14
_SPIKE_IN_1 = 0x18
_SPIKE_IN_2 = 0x1C
_SPIKE_IN_3 = 0x20
_SPIKE_IN_4 = 0x24
_SPIKE_COMMIT = 0x28
_OUT_V_BASE = 0x40


class SHDAccelerator:
    """Hardware accelerator for SHD speech digit classification.

    Wraps the AXI-Lite register interface to sc_shd_top, providing
    a high-level API for FPGA inference.
    """

    def __init__(self, mmio):
        """Initialise with a PYNQ MMIO object (from overlay IP).

        Parameters
        ----------
        mmio : pynq.MMIO or pynq.DefaultIP
            Memory-mapped IO for the sc_shd_axi_wrapper IP block.
            From overlay: ``overlay.sc_shd_axi_wrapper_0.mmio``
            or directly: ``overlay.sc_shd_axi_wrapper_0``
        """
        if hasattr(mmio, "mmio"):
            self._mmio = mmio.mmio
        else:
            self._mmio = mmio

    def set_scales(self, scale_l1: int, scale_l2: int, scale_l3: int) -> None:
        """Set Q16.16 per-layer scales (from training checkpoint).

        Parameters
        ----------
        scale_l1, scale_l2, scale_l3 : int
            Signed 32-bit Q16.16 fixed-point scale factors.
        """
        self._mmio.write(_SCALE_L1, scale_l1 & 0xFFFFFFFF)
        self._mmio.write(_SCALE_L2, scale_l2 & 0xFFFFFFFF)
        self._mmio.write(_SCALE_L3, scale_l3 & 0xFFFFFFFF)

    def classify(self, spike_raster: np.ndarray) -> int:
        """Run inference on a spike raster and return predicted class.

        Parameters
        ----------
        spike_raster : np.ndarray
            Binary array of shape (T, 140) where T is the number of
            timesteps and 140 is the number of input channels.

        Returns
        -------
        int
            Predicted class index (0-19) = argmax of output voltages.
        """
        t_orig = spike_raster.shape[0]
        self._mmio.write(_T_ORIG, t_orig)

        # Start inference
        self._mmio.write(_CTRL, 1)

        # Stream spike vectors cycle by cycle
        t_total = t_orig + 30  # Pipeline padding (see sc_shd_top.v)
        for t in range(t_total):
            if t < t_orig:
                row = spike_raster[t]
            else:
                row = np.zeros(140, dtype=np.uint8)

            # Pack 140 bits into 5 x 32-bit words
            bits = 0
            for i in range(140):
                if row[i]:
                    bits |= 1 << i

            self._mmio.write(_SPIKE_IN_0, bits & 0xFFFFFFFF)
            self._mmio.write(_SPIKE_IN_1, (bits >> 32) & 0xFFFFFFFF)
            self._mmio.write(_SPIKE_IN_2, (bits >> 64) & 0xFFFFFFFF)
            self._mmio.write(_SPIKE_IN_3, (bits >> 96) & 0xFFFFFFFF)
            self._mmio.write(_SPIKE_IN_4, (bits >> 128) & 0xFFF)
            self._mmio.write(_SPIKE_COMMIT, 1)  # Advance one cycle

        # Wait for done
        while True:
            status = self._mmio.read(_CTRL)
            if status & 0x4:  # done bit
                break

        # Read output voltages
        return self.read_output_argmax()

    def read_output_argmax(self) -> int:
        """Read the 20 output voltage sums and return argmax."""
        voltages = self.read_output_voltages()
        return int(np.argmax(voltages))

    def read_output_voltages(self) -> np.ndarray:
        """Read all 20 class output voltages as signed int32 array."""
        out = np.zeros(20, dtype=np.int32)
        for i in range(20):
            raw = self._mmio.read(_OUT_V_BASE + i * 4)
            # Convert unsigned to signed int32
            if raw >= 0x80000000:
                raw -= 0x100000000
            out[i] = raw
        return out

    @property
    def is_running(self) -> bool:
        """Check if inference is currently in progress."""
        return bool(self._mmio.read(_CTRL) & 0x2)

    @property
    def is_done(self) -> bool:
        """Check if inference has completed."""
        return bool(self._mmio.read(_CTRL) & 0x4)
