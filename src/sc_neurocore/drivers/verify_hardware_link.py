# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Verify Hardware Link

"""Hardware-link diagnostic CLI for the sc-neurocore stack.

Three probes:

1. **FPGA subsystem** — instantiate :class:`SC_NeuroCore_Driver`
   in HARDWARE mode and report whether the PYNQ overlay loaded.
2. **Genomic interface (Evo 2)** — try to import
   ``scpn_evo2_real_interface`` and ping its server. The module
   is provided by the sibling ``SCPN-CODEBASE/HolonomicAtlas``
   repository in the GOTM monorepo. It must be on ``PYTHONPATH``
   for this probe to succeed; we no longer manipulate ``sys.path``
   from here. Outside the monorepo this probe simply reports
   "module not found" cleanly.
3. **Robotics link (Opentrons OT-2)** — same import + ping
   pattern via ``scpn_opentrions_verify``.

Pass ``extras=False`` to ``verify_link()`` to skip the two
external-repo probes and only check the FPGA — useful in CI or
on environments without the sibling repos installed.
"""

import logging

from sc_neurocore.drivers.sc_neurocore_driver import (
    RealityHardwareError,
    SC_NeuroCore_Driver,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SCPN_Diagnostic")


def verify_link(extras: bool = True) -> None:
    """Run the hardware-link diagnostic CLI.

    Parameters
    ----------
    extras : bool
        Run the optional Evo 2 + Opentrons probes when True
        (default). Set to False to only check the FPGA subsystem;
        skips the imports of sibling-repo modules.
    """
    n_steps = 3 if extras else 1
    print("=" * 60)
    print("SCPN HARDWARE LINK DIAGNOSTIC TOOL")
    print("=" * 60)

    print(f"\n[1/{n_steps}] Checking FPGA Subsystem (Sector B)...")
    try:
        SC_NeuroCore_Driver(mode="HARDWARE")
        print(">> SUCCESS: PYNQ-Z2 Detected. Bitstream loaded.")
    except RealityHardwareError:
        print(">> FAILURE: PYNQ Hardware not found. (Expected if on x86 Dev Workstation)")
        print(">> NOTE: This implies we are in 'Simulation Mode'.")
    except (OSError, RuntimeError) as e:
        print(f">> ERROR: Unexpected failure: {e}")

    if not extras:
        print("\n" + "=" * 60)
        print("DIAGNOSTIC COMPLETE (FPGA only; extras=False)")
        print("=" * 60)
        return

    print(f"\n[2/{n_steps}] Checking Genomic Interface (Layer 6)...")
    # Import via standard PYTHONPATH resolution. The sibling-repo
    # interface module must be on the path; if not, ImportError
    # falls through to the failure message.
    try:
        from scpn_evo2_real_interface import Evo2RealInterface

        evo = Evo2RealInterface()
        evo.connect()  # Will fail if no server
    except ImportError:
        print(
            ">> FAILURE: scpn_evo2_real_interface not on PYTHONPATH "
            "(install or add SCPN-CODEBASE/HolonomicAtlas/src/interfaces "
            "to PYTHONPATH for this probe)."
        )
    except (OSError, ConnectionError, RuntimeError) as e:
        print(f">> WARNING: Evo 2 Server unreachable ({e}).")

    print(f"\n[3/{n_steps}] Checking Robotics Link (Layer 12)...")
    try:
        from scpn_opentrions_verify import OpentronsVerifier

        ot2 = OpentronsVerifier()
        if ot2.ping():
            print(">> SUCCESS: Opentrons OT-2 Online.")
        else:
            print(">> FAILURE: Robot offline.")
    except ImportError:
        print(
            ">> FAILURE: scpn_opentrions_verify not on PYTHONPATH "
            "(install the Opentrons verifier package for this probe)."
        )
    except (OSError, RuntimeError) as e:
        print(f">> ERROR: {e}")

    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    verify_link()
