# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for network/gamma_oscillation
#
# Accelerator kernel for the PING gamma oscillation circuit.
# Implements the step logic matching the conductance-based dynamics 
# described in:
# Börgers, C. & Kopell, N. (2003). Synchronization in Networks of 
# Excitatory and Inhibitory Neurons with Sparse, Random Connectivity. 
# Neural Computation 15(3): 509-538.

module GammaOscillationAccel

function py_ping_step(
    v_e_ptr, g_ampa_e_ptr, g_gaba_e_ptr, refrac_e_ptr, i_drive_e_ptr, xi_e_ptr, spikes_e_out_ptr,
    v_i_ptr, g_ampa_i_ptr, g_gaba_i_ptr, refrac_i_ptr, i_drive_i_ptr, xi_i_ptr, spikes_i_out_ptr,
    e_l::Float64, e_ampa::Float64, e_gaba::Float64, g_l::Float64, c_m::Float64,
    v_threshold::Float64, v_reset::Float64, t_refrac::Float64,
    tau_ampa::Float64, tau_gaba::Float64, sigma_e::Float64, sigma_i::Float64, dt::Float64
)
    decay_ampa = exp(-dt / tau_ampa)
    decay_gaba = exp(-dt / tau_gaba)
    dt_over_cm = dt / c_m
    sqrt_dt = sqrt(dt)

    n_excit = length(v_e_ptr)
    n_inhib = length(v_i_ptr)

    for k in 1:length(g_ampa_e_ptr)
        g_ampa_e_ptr[k] *= decay_ampa
    end
    for k in 1:length(g_ampa_i_ptr)
        g_ampa_i_ptr[k] *= decay_ampa
    end
    for k in 1:length(g_gaba_e_ptr)
        g_gaba_e_ptr[k] *= decay_gaba
    end
    for k in 1:length(g_gaba_i_ptr)
        g_gaba_i_ptr[k] *= decay_gaba
    end

    n_e::Int = 0
    for k in 1:n_excit
        in_refrac = refrac_e_ptr[k] > 0.0
        v_old = v_e_ptr[k]
        i_leak = -g_l * (v_old - e_l)
        i_ampa_cur = -g_ampa_e_ptr[k] * (v_old - e_ampa)
        i_gaba_cur = -g_gaba_e_ptr[k] * (v_old - e_gaba)
        i_total = i_leak + i_ampa_cur + i_gaba_cur + i_drive_e_ptr[k]
        noise = sqrt_dt * sigma_e * xi_e_ptr[k]

        v_new = in_refrac ? v_reset : v_old + i_total * dt_over_cm + noise
        spk = (v_new >= v_threshold) && !in_refrac

        v_e_ptr[k] = spk ? v_reset : v_new
        new_refrac = spk ? t_refrac : refrac_e_ptr[k] - dt
        refrac_e_ptr[k] = new_refrac > 0.0 ? new_refrac : 0.0
        spikes_e_out_ptr[k] = spk ? 1 : 0
        if spk
            n_e += 1
        end
    end

    n_i::Int = 0
    for k in 1:n_inhib
        in_refrac = refrac_i_ptr[k] > 0.0
        v_old = v_i_ptr[k]
        i_leak = -g_l * (v_old - e_l)
        i_ampa_cur = -g_ampa_i_ptr[k] * (v_old - e_ampa)
        i_gaba_cur = -g_gaba_i_ptr[k] * (v_old - e_gaba)
        i_total = i_leak + i_ampa_cur + i_gaba_cur + i_drive_i_ptr[k]
        noise = sqrt_dt * sigma_i * xi_i_ptr[k]

        v_new = in_refrac ? v_reset : v_old + i_total * dt_over_cm + noise
        spk = (v_new >= v_threshold) && !in_refrac

        v_i_ptr[k] = spk ? v_reset : v_new
        new_refrac = spk ? t_refrac : refrac_i_ptr[k] - dt
        refrac_i_ptr[k] = new_refrac > 0.0 ? new_refrac : 0.0
        spikes_i_out_ptr[k] = spk ? 1 : 0
        if spk
            n_i += 1
        end
    end

    return n_e, n_i
end

end # module GammaOscillationAccel
