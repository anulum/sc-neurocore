# SPDX-License-Identifier: AGPL-3.0-or-later
"""Photonic computation, FDTD simulation, and SC compilation demonstration."""

import sys
import numpy as np
import matplotlib.pyplot as plt

# We want to use the native Rust accelerated functions for crosstalk
import sc_neurocore_engine

from sc_neurocore.optics import (
    PhotonicCompiler,
    PhotonicTarget,
    CrosstalkModel,
    WaveguidePair,
)
from sc_neurocore.neurons.models.arcane_neuron import ArcaneNeuron

def main():
    print("=== SC-NeuroCore: Photonic Compilation & Simulation Demo ===\n")

    # 1. SC Sequence Generation via ArcaneNeuron
    print("[1] Generating Stochastic Computing Bitstream...")
    neuron = ArcaneNeuron(theta=0.5, tau_fast=1.0)
    
    # Let's inject some current to produce an SC bitstream
    spikes_out = []
    current_trace = [1.2, 5.8, 0.0, 8.0, 0.0] * 40  # 200 timesteps of dynamic stimulus
    
    for I in current_trace:
        fired = neuron.step(I)
        spikes_out.append(float(fired))
        
    bitstream = np.array(spikes_out, dtype=np.float64)
    print(f"   → Got bitstream of length {len(bitstream)}, Mean Activity: {np.mean(bitstream):.3f}")
    
    # 2. Photonic Target Configuration
    print("\n[2] Configuring Photonic Compiler (Silicon Photonics)...")
    target = PhotonicTarget.silicon_photonics()
    
    # The compiler leverages the engine to map SC bitstreams -> specific modulator sequences
    compiler = PhotonicCompiler(target=target)
    
    print("   → Running Photonic Compilation + 1D FDTD Co-simulation...")
    # This invokes Yee discretisation + Maxwell's solver mapping pulses exactly
    res = compiler.compile_bitstream(bitstream, run_fdtd=True, fdtd_steps=200)
    
    print(f"      Target Rig         : {res.target}")
    print(f"      Modulators Reqs    : {res.num_modulators}")
    print(f"      Mean Optical Power : {res.optical_power_mean_mw:.4f} mW")
    print(f"      FDTD Grid Energy   : {res.fdtd_energy:.4f}")
    
    # Enable internal physics flag testing
    import sc_neurocore.optics as pc_optics
    print(f"      Rust Rust Engine Enabled? : {pc_optics.photonic_emitter._HAS_RUST_PH}")

    assert pc_optics.photonic_emitter._HAS_RUST_PH is True, "ERROR: Rust photonic functions failed to load natively via the bridge!"
    
    # 3. Crosstalk Modeling (invoking the newly verified Rust bindings directly!)
    print("\n[3] Waveguide Routing Crosstalk Analysis...")
    cx_model = CrosstalkModel()
    pair = WaveguidePair(
        waveguide_width_nm=400.0,
        gap_nm=180.0,
        coupling_length_um=15.0,
        core_index=3.48,
        cladding_index=1.45,
        wavelength_nm=1550.0
    )
    cx_model.add_pair(pair)
    
    # Calling the analyze_bank which delegates to Rust Engine internally when _HAS_RUST_PH is True
    bank_analysis = cx_model.analyze_bank(waveguides=8, gap_nm=180.0, coupling_length_um=15.0)
    print(f"   → Bank Analysis Matrix Waveguides: {bank_analysis['num_waveguides']}")
    print(f"   → Effective Signal-to-Leakage Ratio: {bank_analysis['worst_isolation_db']:.2f} dB")
    print(f"   → Mean Coupling Ratio: {bank_analysis['mean_coupling_ratio']:.4f}")
    print(f"   → Crosstalk Safe Bounds: {bank_analysis['crosstalk_safe']}")

    print("\n[4] Power Budget Report (Native Rust)...")
    sources = [0, 1, 2, 3]
    targets = [1, 2, 3, 4]
    losses = [2.5, 3.1, 1.2, 4.0] # dB path losses
    pb = sc_neurocore_engine.py_ph_analyze_power_budget(sources, targets, losses, 0.0, -20.0)
    print(f"   → Paths Analyzed: {pb['n_paths']}")
    print(f"   → Margins (dB): {[round(m, 2) for m in pb['margins_db']]}")
    print(f"   → All Paths Passed: {all(pb['passed'])}")

    print("\n[5] Finished Optical Netlist Mapping Preview:")
    for line in res.netlist.splitlines()[:6]:
        print(f"   {line}")
    print("   ...")
    
    print("\n[6] Exporting Physical Foundry Layout...")
    try:
        res.to_gdsii("demo_layout.gds")
    except ImportError as e:
        print(f"   → Layout export skipped: {e}")
        
    print("\n[7] Verifying Meep Integration...")
    from sc_neurocore.optics import MeepAdapter
    if MeepAdapter.is_available():
        print("   → Meep FDTD library loaded correctly for advanced 2D/3D physics.")
    else:
        print("   → Meep not available locally. Falling back to native SC-NeuroCore FDTD Solvers.")

    # Visualization
    t_axis = np.arange(len(bitstream))
    optical_phases = compiler.converter.to_phase_array(bitstream)
    optical_power = compiler.converter.optical_power_profile(bitstream)

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    axs[0].step(t_axis, bitstream, where="post", color="black", label="SC Bitstream (ArcaneNeuron)")
    axs[0].set_ylabel("Binary")
    axs[0].set_title("Stochastic Logic Source")
    axs[0].legend(loc="upper right")
    
    axs[1].step(t_axis, optical_phases, where="post", color="blue", label="Modulated Phase (rad)")
    axs[1].set_ylabel("Phase")
    axs[1].set_title(f"Target: {target.modulator_type}")
    axs[1].legend(loc="upper right")
    
    axs[2].plot(t_axis, optical_power, drawstyle="steps-post", color="red", label="Optical Power Envelope")
    axs[2].set_ylabel("Power (mW)")
    axs[2].set_xlabel("Time Step")
    axs[2].set_title("Photonic Pulse Envelope")
    axs[2].legend(loc="upper right")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
