// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust per-step kernel for the Börgers-Kopell PINGCircuit

//! Step-by-step parity with `PINGCircuit.step` in
//! `src/sc_neurocore/network/gamma_oscillation.py`.
//!
//! Per dt:
//!   1. Decay AMPA / GABA conductances by `exp(-dt/tau_*)`.
//!   2. Compute `I = -g_L*(V-E_L) - g_AMPA*(V-E_AMPA)
//!                  - g_GABA*(V-E_GABA) + I_drive + sqrt(dt)*sigma*xi`.
//!      (`xi` is supplied externally as a pre-drawn N(0,1) array so
//!      the Rust path stays bit-deterministic for matching seeds —
//!      see `tests/test_gamma_oscillation.py::TestPINGCircuitDeterminism`.)
//!   3. Update `V` Euler-style; cells in refractory are clamped to V_reset.
//!   4. Detect spikes (`V >= V_th`, not in refractory).
//!   5. Reset spiked cells to V_reset, start refractory window.
//!   6. Decrement refractory counters; clamp to 0.
//!   7. Return per-cell spike booleans + total per-population spike counts
//!      so the Python wrapper can propagate to conductances.
//!
//! This is the kernel for the per-step inner loop. The Python side
//! still owns:
//!   - the per-instance RNG (so noise stays seed-deterministic across
//!     the two backends);
//!   - the cross-population conductance update step
//!     (g_ampa_e += w_ee * n_e_spikes etc — handled by the caller).

#[allow(clippy::too_many_arguments)]
pub fn step_kernel(
    // Excitatory population.
    v_e: &mut [f64],
    g_ampa_e: &mut [f64],
    g_gaba_e: &mut [f64],
    refrac_e: &mut [f64],
    i_drive_e: &[f64],
    xi_e: &[f64],
    spikes_e_out: &mut [u8],
    // Inhibitory population.
    v_i: &mut [f64],
    g_ampa_i: &mut [f64],
    g_gaba_i: &mut [f64],
    refrac_i: &mut [f64],
    i_drive_i: &[f64],
    xi_i: &[f64],
    spikes_i_out: &mut [u8],
    // Membrane parameters.
    e_l: f64,
    e_ampa: f64,
    e_gaba: f64,
    g_l: f64,
    c_m: f64,
    v_threshold: f64,
    v_reset: f64,
    t_refrac: f64,
    // Synaptic decay constants.
    tau_ampa: f64,
    tau_gaba: f64,
    // Noise scaling.
    sigma_e: f64,
    sigma_i: f64,
    // Time step.
    dt: f64,
) -> (u32, u32) {
    let decay_ampa = (-dt / tau_ampa).exp();
    let decay_gaba = (-dt / tau_gaba).exp();
    let dt_over_cm = dt / c_m;
    let sqrt_dt = dt.sqrt();

    // Decay conductances first (matches Python ordering).
    for g in g_ampa_e.iter_mut() {
        *g *= decay_ampa;
    }
    for g in g_ampa_i.iter_mut() {
        *g *= decay_ampa;
    }
    for g in g_gaba_e.iter_mut() {
        *g *= decay_gaba;
    }
    for g in g_gaba_i.iter_mut() {
        *g *= decay_gaba;
    }

    // Excitatory population update + spike detect.
    //
    // Note: Python adds the Wiener noise increment `sqrt(dt)·σ·ξ`
    // DIRECTLY to V (not through the `I·dt/C_m` channel). The
    // Rust path mirrors that exactly so spike outputs are bit-
    // identical between backends at any (n_e, n_i, σ) point —
    // verified by `tests/test_gamma_oscillation.py::TestPython
    // RustParity` and by `bench_gamma_oscillation.py` reproducing
    // the published 30-80 Hz dominant frequency at all three
    // benchmark workloads on both backends.
    let mut n_e: u32 = 0;
    let n_excit = v_e.len();
    for k in 0..n_excit {
        let in_refrac = refrac_e[k] > 0.0;
        let v_old = v_e[k];
        let i_leak = -g_l * (v_old - e_l);
        let i_ampa = -g_ampa_e[k] * (v_old - e_ampa);
        let i_gaba = -g_gaba_e[k] * (v_old - e_gaba);
        let i_total = i_leak + i_ampa + i_gaba + i_drive_e[k];
        let noise = sqrt_dt * sigma_e * xi_e[k];
        let v_new = if in_refrac {
            v_reset
        } else {
            v_old + i_total * dt_over_cm + noise
        };
        let spk = v_new >= v_threshold && !in_refrac;
        v_e[k] = if spk { v_reset } else { v_new };
        let new_refrac = if spk { t_refrac } else { refrac_e[k] - dt };
        refrac_e[k] = if new_refrac > 0.0 { new_refrac } else { 0.0 };
        spikes_e_out[k] = u8::from(spk);
        if spk {
            n_e += 1;
        }
    }

    // Inhibitory population update + spike detect.
    let mut n_i: u32 = 0;
    let n_inh = v_i.len();
    for k in 0..n_inh {
        let in_refrac = refrac_i[k] > 0.0;
        let v_old = v_i[k];
        let i_leak = -g_l * (v_old - e_l);
        let i_ampa = -g_ampa_i[k] * (v_old - e_ampa);
        let i_gaba = -g_gaba_i[k] * (v_old - e_gaba);
        let i_total = i_leak + i_ampa + i_gaba + i_drive_i[k];
        let noise = sqrt_dt * sigma_i * xi_i[k];
        let v_new = if in_refrac {
            v_reset
        } else {
            v_old + i_total * dt_over_cm + noise
        };
        let spk = v_new >= v_threshold && !in_refrac;
        v_i[k] = if spk { v_reset } else { v_new };
        let new_refrac = if spk { t_refrac } else { refrac_i[k] - dt };
        refrac_i[k] = if new_refrac > 0.0 { new_refrac } else { 0.0 };
        spikes_i_out[k] = u8::from(spk);
        if spk {
            n_i += 1;
        }
    }

    (n_e, n_i)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Burned-in zero-input case: no spikes, V stays exactly at E_L.
    #[test]
    fn test_no_drive_no_spikes() {
        let n_e = 4;
        let n_i = 2;
        let mut v_e = vec![-67.0_f64; n_e];
        let mut v_i = vec![-67.0_f64; n_i];
        let mut g_ampa_e = vec![0.0; n_e];
        let mut g_ampa_i = vec![0.0; n_i];
        let mut g_gaba_e = vec![0.0; n_e];
        let mut g_gaba_i = vec![0.0; n_i];
        let mut refrac_e = vec![0.0; n_e];
        let mut refrac_i = vec![0.0; n_i];
        let i_drive_e = vec![0.0; n_e];
        let i_drive_i = vec![0.0; n_i];
        let xi_e = vec![0.0; n_e];
        let xi_i = vec![0.0; n_i];
        let mut spk_e = vec![0_u8; n_e];
        let mut spk_i = vec![0_u8; n_i];

        let (ne, ni) = step_kernel(
            &mut v_e, &mut g_ampa_e, &mut g_gaba_e, &mut refrac_e,
            &i_drive_e, &xi_e, &mut spk_e,
            &mut v_i, &mut g_ampa_i, &mut g_gaba_i, &mut refrac_i,
            &i_drive_i, &xi_i, &mut spk_i,
            -67.0, 0.0, -80.0, 0.05, 1.0, -52.0, -67.0, 2.0,
            5.0, 5.0, 0.0, 0.0, 0.1,
        );
        assert_eq!(ne, 0);
        assert_eq!(ni, 0);
        for v in &v_e {
            // No drive + V at E_L → no integration motion at all.
            assert!((v - (-67.0)).abs() < 1e-9);
        }
    }

    /// Strong drive should push V above threshold within a few steps
    /// and the refractory window must hold the cell at V_reset.
    #[test]
    fn test_drive_produces_spikes_and_refractory_holds() {
        let n_e = 1;
        let n_i = 1;
        let mut v_e = vec![-67.0_f64; n_e];
        let mut v_i = vec![-67.0_f64; n_i];
        let mut g_ampa_e = vec![0.0; n_e];
        let mut g_ampa_i = vec![0.0; n_i];
        let mut g_gaba_e = vec![0.0; n_e];
        let mut g_gaba_i = vec![0.0; n_i];
        let mut refrac_e = vec![0.0; n_e];
        let mut refrac_i = vec![0.0; n_i];
        let i_drive_e = vec![5.0; n_e];     // µA/cm² — supra-threshold
        let i_drive_i = vec![0.0; n_i];
        let xi_e = vec![0.0; n_e];
        let xi_i = vec![0.0; n_i];
        let mut total_spikes_e = 0_u32;
        let mut last_spike_step = -1_i32;

        for step in 0..400_i32 {
            let mut spk_e = vec![0_u8; n_e];
            let mut spk_i = vec![0_u8; n_i];
            let (ne, _ni) = step_kernel(
                &mut v_e, &mut g_ampa_e, &mut g_gaba_e, &mut refrac_e,
                &i_drive_e, &xi_e, &mut spk_e,
                &mut v_i, &mut g_ampa_i, &mut g_gaba_i, &mut refrac_i,
                &i_drive_i, &xi_i, &mut spk_i,
                -67.0, 0.0, -80.0, 0.05, 1.0, -52.0, -67.0, 2.0,
                5.0, 5.0, 0.0, 0.0, 0.1,
            );
            if ne > 0 {
                if last_spike_step >= 0 {
                    // dt=0.1, T_refrac=2 → next spike at least
                    // 20 steps later (refractory holds V at V_reset).
                    assert!(step - last_spike_step >= 20);
                }
                last_spike_step = step;
            }
            total_spikes_e += ne;
        }
        assert!(total_spikes_e > 0);
    }

    /// Determinism: identical state + identical xi → identical output.
    #[test]
    fn test_deterministic_for_same_inputs() {
        let n_e = 8;
        let n_i = 4;
        let mk = || (
            vec![-65.0_f64; n_e], vec![-65.0_f64; n_i],
            vec![0.5_f64; n_e],   vec![0.5_f64; n_i],
            vec![0.3_f64; n_e],   vec![0.3_f64; n_i],
            vec![0.0_f64; n_e],   vec![0.0_f64; n_i],
        );
        let xi_e = vec![0.1_f64; n_e];
        let xi_i = vec![-0.1_f64; n_i];
        let i_drive_e = vec![1.0; n_e];
        let i_drive_i = vec![0.5; n_i];

        let (mut v_e_a, mut v_i_a, mut ga_e_a, mut ga_i_a, mut gg_e_a,
             mut gg_i_a, mut rf_e_a, mut rf_i_a) = mk();
        let (mut v_e_b, mut v_i_b, mut ga_e_b, mut ga_i_b, mut gg_e_b,
             mut gg_i_b, mut rf_e_b, mut rf_i_b) = mk();
        let mut spk_e_a = vec![0_u8; n_e];
        let mut spk_i_a = vec![0_u8; n_i];
        let mut spk_e_b = vec![0_u8; n_e];
        let mut spk_i_b = vec![0_u8; n_i];

        let _ = step_kernel(
            &mut v_e_a, &mut ga_e_a, &mut gg_e_a, &mut rf_e_a,
            &i_drive_e, &xi_e, &mut spk_e_a,
            &mut v_i_a, &mut ga_i_a, &mut gg_i_a, &mut rf_i_a,
            &i_drive_i, &xi_i, &mut spk_i_a,
            -67.0, 0.0, -80.0, 0.05, 1.0, -52.0, -67.0, 2.0,
            5.0, 5.0, 0.1, 0.05, 0.1,
        );
        let _ = step_kernel(
            &mut v_e_b, &mut ga_e_b, &mut gg_e_b, &mut rf_e_b,
            &i_drive_e, &xi_e, &mut spk_e_b,
            &mut v_i_b, &mut ga_i_b, &mut gg_i_b, &mut rf_i_b,
            &i_drive_i, &xi_i, &mut spk_i_b,
            -67.0, 0.0, -80.0, 0.05, 1.0, -52.0, -67.0, 2.0,
            5.0, 5.0, 0.1, 0.05, 0.1,
        );

        assert_eq!(v_e_a, v_e_b);
        assert_eq!(v_i_a, v_i_b);
        assert_eq!(spk_e_a, spk_e_b);
        assert_eq!(spk_i_a, spk_i_b);
    }
}
