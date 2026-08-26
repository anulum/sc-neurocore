# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for gamma_oscillation
#
# Accelerator kernel for the PING gamma oscillation circuit.
# Implements the step logic matching the conductance-based dynamics
# described in:
# Börgers, C. & Kopell, N. (2003). Synchronization in Networks of
# Excitatory and Inhibitory Neurons with Sparse, Random Connectivity.
# Neural Computation 15(3): 509-538.

from std.memory import UnsafePointer
from std.math import exp, sqrt

@export
def py_ping_step(
    n_excit: Int,
    n_inhib: Int,
    v_e_addr: Int,
    g_ampa_e_addr: Int,
    g_gaba_e_addr: Int,
    refrac_e_addr: Int,
    i_drive_e_addr: Int,
    xi_e_addr: Int,
    spikes_e_out_addr: Int,
    v_i_addr: Int,
    g_ampa_i_addr: Int,
    g_gaba_i_addr: Int,
    refrac_i_addr: Int,
    i_drive_i_addr: Int,
    xi_i_addr: Int,
    spikes_i_out_addr: Int,
    e_l: Float64,
    e_ampa: Float64,
    e_gaba: Float64,
    g_l: Float64,
    c_m: Float64,
    v_threshold: Float64,
    v_reset: Float64,
    t_refrac: Float64,
    tau_ampa: Float64,
    tau_gaba: Float64,
    sigma_e: Float64,
    sigma_i: Float64,
    dt: Float64,
    out_n_e_spikes_addr: Int,
    out_n_i_spikes_addr: Int
):
    var v_e = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=v_e_addr)
    var g_ampa_e = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=g_ampa_e_addr)
    var g_gaba_e = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=g_gaba_e_addr)
    var refrac_e = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=refrac_e_addr)
    var i_drive_e = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=i_drive_e_addr)
    var xi_e = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=xi_e_addr)
    var spikes_e_out = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=spikes_e_out_addr)

    var v_i = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=v_i_addr)
    var g_ampa_i = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=g_ampa_i_addr)
    var g_gaba_i = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=g_gaba_i_addr)
    var refrac_i = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=refrac_i_addr)
    var i_drive_i = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=i_drive_i_addr)
    var xi_i = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=xi_i_addr)
    var spikes_i_out = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=spikes_i_out_addr)

    var out_n_e_spikes = UnsafePointer[UInt32, MutAnyOrigin](unsafe_from_address=out_n_e_spikes_addr)
    var out_n_i_spikes = UnsafePointer[UInt32, MutAnyOrigin](unsafe_from_address=out_n_i_spikes_addr)

    var decay_ampa = exp(-dt / tau_ampa)
    var decay_gaba = exp(-dt / tau_gaba)
    var dt_over_cm = dt / c_m
    var sqrt_dt = sqrt(dt)

    for k in range(n_excit):
        g_ampa_e[k] *= decay_ampa
        g_gaba_e[k] *= decay_gaba
    for k in range(n_inhib):
        g_ampa_i[k] *= decay_ampa
        g_gaba_i[k] *= decay_gaba

    var n_e: UInt32 = 0
    for k in range(n_excit):
        var in_refrac = refrac_e[k] > 0.0
        var v_old = v_e[k]
        var i_leak = -g_l * (v_old - e_l)
        var i_ampa_cur = -g_ampa_e[k] * (v_old - e_ampa)
        var i_gaba_cur = -g_gaba_e[k] * (v_old - e_gaba)
        var i_total = i_leak + i_ampa_cur + i_gaba_cur + i_drive_e[k]
        var noise = sqrt_dt * sigma_e * xi_e[k]

        var v_new: Float64 = v_old + i_total * dt_over_cm + noise
        if in_refrac:
            v_new = v_reset

        var spk = (v_new >= v_threshold) and not in_refrac

        if spk:
            v_e[k] = v_reset
            refrac_e[k] = t_refrac
            spikes_e_out[k] = 1
            n_e += 1
        else:
            v_e[k] = v_new
            var new_refrac = refrac_e[k] - dt
            if new_refrac > 0.0:
                refrac_e[k] = new_refrac
            else:
                refrac_e[k] = 0.0
            spikes_e_out[k] = 0

    out_n_e_spikes[] = n_e

    var n_i: UInt32 = 0
    for k in range(n_inhib):
        var in_refrac = refrac_i[k] > 0.0
        var v_old = v_i[k]
        var i_leak = -g_l * (v_old - e_l)
        var i_ampa_cur = -g_ampa_i[k] * (v_old - e_ampa)
        var i_gaba_cur = -g_gaba_i[k] * (v_old - e_gaba)
        var i_total = i_leak + i_ampa_cur + i_gaba_cur + i_drive_i[k]
        var noise = sqrt_dt * sigma_i * xi_i[k]

        var v_new: Float64 = v_old + i_total * dt_over_cm + noise
        if in_refrac:
            v_new = v_reset

        var spk = (v_new >= v_threshold) and not in_refrac

        if spk:
            v_i[k] = v_reset
            refrac_i[k] = t_refrac
            spikes_i_out[k] = 1
            n_i += 1
        else:
            v_i[k] = v_new
            var new_refrac = refrac_i[k] - dt
            if new_refrac > 0.0:
                refrac_i[k] = new_refrac
            else:
                refrac_i[k] = 0.0
            spikes_i_out[k] = 0

    out_n_i_spikes[] = n_i
