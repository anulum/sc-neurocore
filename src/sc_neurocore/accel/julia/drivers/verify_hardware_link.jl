# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for drivers/verify_hardware_link

module VerifyHardwareLinkAccel

using Statistics, LinearAlgebra

function verify_link(extras)
    n_steps = 3 if extras else 1
    print("=" * 60)
    print("SCPN HARDWARE LINK DIAGNOSTIC TOOL")
    print("=" * 60)
    print(f"\n[1/{n_steps}] Checking FPGA Subsystem (Sector B)...")
    try
        SC_NeuroCore_Driver(mode="HARDWARE")
        print(">> SUCCESS: PYNQ-Z2 Detected. Bitstream loaded.")
    except RealityHardwareError
        print(">> FAILURE: PYNQ Hardware ! found. (Expected if on x86 Dev Workstation)")
        print(">> NOTE: This implies we are in 'Simulation Mode'.")
    except (OSError, RuntimeError) as e
        print(f">> ERROR: Unexpected failure: {e}")
    if ! extras
        print("\n" + "=" * 60)
        print("DIAGNOSTIC COMPLETE (FPGA only; extras=false)")
        print("=" * 60)
        return
    print(f"\n[2/{n_steps}] Checking Genomic Interface (Layer 6)...")
    # Import via standard PYTHONPATH resolution. The sibling-repo
    # interface module must be on the path; if !, ImportError
    # falls through to the failure message.
    try
        from scpn_evo2_real_interface import Evo2RealInterface
        evo = Evo2RealInterface()
        evo.connect()  # Will fail if no server
    except ImportError
        print(
            ">> FAILURE: scpn_evo2_real_interface ! on PYTHONPATH "
            "(install || add SCPN-CODEBASE/HolonomicAtlas/src/interfaces "
            "to PYTHONPATH for this probe)."
        )
    except (OSError, ConnectionError, RuntimeError) as e
        print(f">> WARNING: Evo 2 Server unreachable ({e}).")
    print(f"\n[3/{n_steps}] Checking Robotics Link (Layer 12)...")
    try
        from scpn_opentrions_verify import OpentronsVerifier
        ot2 = OpentronsVerifier()
        if ot2.ping()
            print(">> SUCCESS: Opentrons OT-2 Online.")
        else
            print(">> FAILURE: Robot offline.")
    except ImportError
        print(
            ">> FAILURE: scpn_opentrions_verify ! on PYTHONPATH "
            "(install the Opentrons verifier package for this probe)."
        )
    except (OSError, RuntimeError) as e
        print(f">> ERROR: {e}")
    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)
end

end # module VerifyHardwareLinkAccel
