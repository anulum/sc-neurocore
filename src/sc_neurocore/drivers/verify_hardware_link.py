# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Verify Hardware Link

import sys
import logging
from pathlib import Path

from sc_neurocore.drivers.sc_neurocore_driver import SC_NeuroCore_Driver, RealityHardwareError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SCPN_Diagnostic")


def verify_link() -> None:
    print("=" * 60)
    print("SCPN HARDWARE LINK DIAGNOSTIC TOOL")
    print("=" * 60)

    print("\n[1/3] Checking FPGA Subsystem (Sector B)...")
    try:
        driver = SC_NeuroCore_Driver(mode="HARDWARE")
        print(">> SUCCESS: PYNQ-Z2 Detected. Bitstream loaded.")
    except RealityHardwareError:
        print(">> FAILURE: PYNQ Hardware not found. (Expected if on x86 Dev Workstation)")
        print(">> NOTE: This implies we are in 'Simulation Mode'.")
    except (OSError, RuntimeError) as e:
        print(f">> ERROR: Unexpected failure: {e}")

    print("\n[2/3] Checking Genomic Interface (Layer 6)...")
    try:
        # Import dynamically to avoid crashing if deps are missing
        sys.path.append(
            str(
                Path(__file__).resolve().parent
                / "../../../SCPN-CODEBASE/HolonomicAtlas/src/interfaces"
            )
        )
        from scpn_evo2_real_interface import Evo2RealInterface

        evo = Evo2RealInterface()
        evo.connect()  # Will fail if no server
    except ImportError:
        print(">> FAILURE: Interface module not found in path.")
    except (OSError, ConnectionError, RuntimeError) as e:
        print(f">> WARNING: Evo 2 Server unreachable ({e}).")

    print("\n[3/3] Checking Robotics Link (Layer 12)...")
    try:
        from scpn_opentrions_verify import OpentronsVerifier

        ot2 = OpentronsVerifier()
        if ot2.ping():
            print(">> SUCCESS: Opentrons OT-2 Online.")
        else:
            print(">> FAILURE: Robot offline.")
    except ImportError:
        print(">> FAILURE: Interface module not found.")
    except (OSError, RuntimeError) as e:
        print(f">> ERROR: {e}")

    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    verify_link()
