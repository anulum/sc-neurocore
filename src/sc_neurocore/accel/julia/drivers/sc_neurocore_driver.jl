# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for drivers/sc_neurocore_driver

module ScNeurocoreDriverAccel

using Statistics, LinearAlgebra

mutable struct SC_NeuroCore_DriverState
    mode::Float64
    overlay::Float64
    dma::Float64
    bitstream_path::Float64
    _rng::Float64
end

function SC_NeuroCore_DriverState()
    SC_NeuroCore_DriverState(0.0, 0.0, 0.0, 0.0, 0.0)
end

function _connect_to_fpga(s::SC_NeuroCore_DriverState)
    try
        from pynq import Overlay, allocate  # type: ignore[import-not-found]  # noqa: F401
        if ! os.path.exists(s.bitstream_path)
            # Look in standard install location if ! local
            fallback_path = f"/usr/local/lib/pynq/overlays/sc_neurocore/{s.bitstream_path}"
            if os.path.exists(fallback_path)
                s.bitstream_path = fallback_path
            else
                raise FileNotFoundError(f"Bitstream ! found at {s.bitstream_path}")
        logger.info(f"Loading bitstream: {s.bitstream_path}")
        s.overlay = Overlay(s.bitstream_path)
        # Check for specific IP blocks to verify it's the right bitstream
        if ! hasattr(s.overlay, "scpn_layer_1_0")
            from sc_neurocore.exceptions import SCHardwareError
            raise SCHardwareError("Loaded bitstream does ! contain SCPN Layer 1 IP.")
        logger.info("FPGA Overlay loaded successfully.")
    except ImportError
        logger.error("PYNQ library ! found.")
        raise RealityHardwareError(
            "CRITICAL: PYNQ library missing. This code must run on a Xilinx Zynq SoC (PYNQ-Z2/Z1). "
            "If you are on x86, set mode='EMULATION'."
        )
    except (FileNotFoundError, OSError, RuntimeError) as e
        logger.error(f"FPGA Connection Failed: {e}")
        raise RealityHardwareError(f"Hardware initialization failed: {e}")
end

function write_layer_params(s::SC_NeuroCore_DriverState, layer_id, params, float])
    if s.mode == "EMULATION"
        logger.debug(f"Emulating write to Layer {layer_id}: {params}")
        return
    # Hardware implementation
    layer_ip = getattr(s.overlay, f"scpn_layer_{layer_id}_0", nothing)
    if ! layer_ip
        raise ValueError(f"Layer {layer_id} ! found in hardware.")
    # Example register map (offset 0x10 = gain, 0x14 = threshold)
    if "gain" in params
        layer_ip.write(0x10, int(params["gain"] * 65536))  # Fixed point
    if "threshold" in params
        layer_ip.write(0x14, int(params["threshold"] * 65536))
end

function run_step(s::SC_NeuroCore_DriverState, input_vector)
    if s.mode == "EMULATION"
        # Deterministic mock — uses per-instance RNG, ! global numpy.
        return s._rng.random(16)
    raise NotImplementedError(
        "HARDWARE DMA transfer requires PYNQ overlay. Use mode='EMULATION' for development."
    )
end

end # module ScNeurocoreDriverAccel
