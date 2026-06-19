# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Raised when physical hardware is required but missing

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class RealityHardwareError(ImportError):
    """Raised when physical hardware is required but missing."""

    pass


class SC_NeuroCore_Driver:
    """
    Primary driver for the sc-neurocore FPGA overlay on PYNQ-Z2.

    This driver enforces 'Reality Checks'. It will NOT run on standard x86 CPUs
    unless explicitly in 'EMULATION' mode.
    """

    def __init__(
        self,
        bitstream_path: str = "sc_neurocore.bit",
        mode: str = "HARDWARE",
        seed: int = 42,
    ) -> None:
        """Construct a driver in HARDWARE or EMULATION mode.

        Parameters
        ----------
        bitstream_path : str
            Path to the ``.bit`` file for HARDWARE mode.
        mode : str
            ``'HARDWARE'`` or ``'EMULATION'``.
        seed : int
            Per-instance RNG seed used by EMULATION ``run_step`` so
            successive calls are deterministic given the same seed.
            Two drivers built with the same seed produce identical
            output sequences.
        """
        self.mode = mode
        self.overlay = None
        self.dma = None
        self.bitstream_path = bitstream_path
        self._rng = np.random.default_rng(seed)

        if self.mode == "HARDWARE":
            self._connect_to_fpga()
        elif self.mode == "EMULATION":
            logger.warning(
                "Running in EMULATION mode. Results may not reflect quantum stochasticity."
            )
        else:
            raise ValueError("Invalid mode. Use 'HARDWARE' or 'EMULATION'.")

    def _connect_to_fpga(self) -> None:
        """
        Attempts to load the PYNQ libraries and flash the bitstream.
        """
        try:
            from pynq import Overlay, allocate  # type: ignore[import-not-found]  # noqa: F401

            if not os.path.exists(self.bitstream_path):
                # Look in standard install location if not local
                fallback_path = f"/usr/local/lib/pynq/overlays/sc_neurocore/{self.bitstream_path}"
                if os.path.exists(fallback_path):
                    self.bitstream_path = fallback_path
                else:
                    raise FileNotFoundError(f"Bitstream not found at {self.bitstream_path}")

            logger.info(f"Loading bitstream: {self.bitstream_path}")
            self.overlay = Overlay(self.bitstream_path)

            # Check for specific IP blocks to verify it's the right bitstream
            if not hasattr(self.overlay, "scpn_layer_1_0"):
                from sc_neurocore.exceptions import SCHardwareError

                raise SCHardwareError("Loaded bitstream does not contain SCPN Layer 1 IP.")

            logger.info("FPGA Overlay loaded successfully.")

        except ImportError:
            logger.error("PYNQ library not found.")
            raise RealityHardwareError(
                "CRITICAL: PYNQ library missing. This code must run on a Xilinx Zynq SoC (PYNQ-Z2/Z1). "
                "If you are on x86, set mode='EMULATION'."
            )
        except (FileNotFoundError, OSError, RuntimeError) as e:
            logger.error(f"FPGA Connection Failed: {e}")
            raise RealityHardwareError(f"Hardware initialization failed: {e}")

    def write_layer_params(self, layer_id: int, params: dict[str, float]) -> None:
        """
        Writes parameters to a specific layer's AXI-Lite registers.
        """
        if self.mode == "EMULATION":
            logger.debug(f"Emulating write to Layer {layer_id}: {params}")
            return

        # Hardware implementation
        layer_ip = getattr(self.overlay, f"scpn_layer_{layer_id}_0", None)
        if not layer_ip:
            raise ValueError(f"Layer {layer_id} not found in hardware.")

        # Example register map (offset 0x10 = gain, 0x14 = threshold)
        if "gain" in params:
            layer_ip.write(0x10, int(params["gain"] * 65536))  # Fixed point
        if "threshold" in params:
            layer_ip.write(0x14, int(params["threshold"] * 65536))

    def run_step(self, input_vector: object) -> np.ndarray[Any, Any]:
        """
        Executes one integration step on the FPGA.

        EMULATION mode returns a 16-element pseudo-random vector from
        the per-instance RNG seeded in ``__init__``. Two drivers built
        with the same seed produce identical sequences. HARDWARE mode
        is not yet implemented (DMA transfer requires PYNQ overlay).
        """
        if self.mode == "EMULATION":
            # Deterministic mock — uses per-instance RNG, not global numpy.
            return self._rng.random(16)

        raise NotImplementedError(
            "HARDWARE DMA transfer requires PYNQ overlay. Use mode='EMULATION' for development."
        )


if __name__ == "__main__":
    # Test strict mode
    try:
        driver = SC_NeuroCore_Driver(mode="HARDWARE")
        print("Hardware connected.")
    except RealityHardwareError as e:
        print(f"\n[STRICT CHECK PASSED]: Driver correctly failed on non-FPGA host.\nError: {e}")
    except (OSError, RuntimeError) as e:
        print(f"An unexpected error occurred: {e}")
