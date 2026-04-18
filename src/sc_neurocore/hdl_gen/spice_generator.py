# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Generates SPICE netlists for Memristive Crossbars

from typing import Any
import logging
import numpy as np

logger = logging.getLogger(__name__)


class SpiceGenerator:
    """
    Generates SPICE netlists for Memristive Crossbars.
    """

    @staticmethod
    def generate_crossbar(weights: np.ndarray[Any, Any], filename: str) -> None:
        """
        weights: (Rows, Cols) - Conductance values [0, 1] mapped to [G_off, G_on].
        """
        rows, cols = weights.shape
        g_on = 100e-6  # 100 uS (10 kOhm)
        g_off = 1e-6  # 1 uS (1 MOhm)

        netlist = f"* Memristor Crossbar {rows}x{cols}\n"
        netlist += ".PARAM VDD=1.0\n\n"

        # Inputs
        for r in range(rows):
            netlist += f"Vin_{r} in_{r} 0 DC 0.0\n"

        # Memristors
        for r in range(rows):
            for c in range(cols):
                w = weights[r, c]
                g = g_off + w * (g_on - g_off)
                r_val = 1.0 / g
                netlist += f"R_{r}_{c} in_{r} out_{c} {r_val:.2f}\n"

        # Outputs (current sensing ideally, here just nodes)
        # Add load resistors
        for c in range(cols):
            netlist += f"Rload_{c} out_{c} 0 1k\n"

        netlist += "\n.END\n"

        with open(filename, "w") as f:
            f.write(netlist)
        logger.info("SPICE Netlist saved to %s", filename)
