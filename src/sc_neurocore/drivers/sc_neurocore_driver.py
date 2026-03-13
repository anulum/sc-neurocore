# SPDX-License-Identifier: AGPL-3.0-or-later
import os
import logging
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

    def __init__(self, bitstream_path="sc_neurocore.bit", mode="HARDWARE"):  # type: ignore
        self.mode = mode
        self.overlay = None
        self.dma = None
        self.bitstream_path = bitstream_path

        if self.mode == "HARDWARE":
            self._connect_to_fpga()  # type: ignore
        elif self.mode == "EMULATION":
            logger.warning(
                "Running in EMULATION mode. Results may not reflect quantum stochasticity."
            )
        else:
            raise ValueError("Invalid mode. Use 'HARDWARE' or 'EMULATION'.")

    def _connect_to_fpga(self):  # type: ignore
        """
        Attempts to load the PYNQ libraries and flash the bitstream.
        """
        try:
            from pynq import Overlay, allocate  # type: ignore  # noqa: F401

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

    def write_layer_params(self, layer_id, params):  # type: ignore
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

    def run_step(self, input_vector):  # type: ignore
        """
        Executes one integration step on the FPGA.
        """
        if self.mode == "EMULATION":
            # Simple mock function
            return np.random.rand(16)

        raise NotImplementedError(
            "HARDWARE DMA transfer requires PYNQ overlay. " "Use mode='EMULATION' for development."
        )


if __name__ == "__main__":
    # Test strict mode
    try:
        driver = SC_NeuroCore_Driver(mode="HARDWARE")  # type: ignore
        print("Hardware connected.")
    except RealityHardwareError as e:
        print(f"\n[STRICT CHECK PASSED]: Driver correctly failed on non-FPGA host.\nError: {e}")
    except (OSError, RuntimeError) as e:
        print(f"An unexpected error occurred: {e}")
