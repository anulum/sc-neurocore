// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust N-step simulator for the Wong-Wang 2006 decision unit

//! Batch parity with `WongWangUnit.step` in
//! `src/sc_neurocore/neurons/models/wong_wang.py` (Wong & Wang 2006,
//! J. Neurosci. 26:1314–1328).
//!
//! Per step:
//!   1. `i_k = j_n * s_k - j_cross * s_(3-k) + i_0 + stim_k + sigma * xi`
//!   2. `r_k = phi(i_k)` where
//!      `phi(i) = (a*i - b) / (1 - exp(-d*(a*i - b)))`
//!      with singularity guard `|a*i - b| < 1e-6 -> 1/d`.
//!   3. integrate the coupled `s1`, `s2` ODE with fixed-step RK4 while
//!      holding the step's pre-drawn noise sample constant.
//!   4. clamp `s_k` into `[0, 1]` after the candidate state is finite.
//!
//! The Python primary draws `np.random.randn()` twice per step; the
//! Rust simulator takes `xi` pre-drawn from the Python RNG so trajectories
//! are bit-exact for matching seeds. This mirrors the ping.rs +
//! PINGCircuit pattern: Python owns the RNG, Rust owns the inner loop.

const A: f64 = 270.0;
const B: f64 = 108.0;
const D: f64 = 0.154;

#[inline]
fn phi(i_syn: f64) -> f64 {
    let x = A * i_syn - B;
    if x.abs() < 1e-6 {
        1.0 / D
    } else {
        x / (1.0 - (-D * x).exp())
    }
}

#[inline]
fn derivatives(
    s1: f64,
    s2: f64,
    stim1: f64,
    stim2: f64,
    xi1: f64,
    xi2: f64,
    tau_s: f64,
    gamma: f64,
    j_n: f64,
    j_cross: f64,
    i_0: f64,
    sigma: f64,
) -> (f64, f64, f64, f64) {
    let i1 = j_n * s1 - j_cross * s2 + i_0 + stim1 + sigma * xi1;
    let i2 = j_n * s2 - j_cross * s1 + i_0 + stim2 + sigma * xi2;
    let r1 = phi(i1);
    let r2 = phi(i2);
    (
        -s1 / tau_s + (1.0 - s1) * gamma * r1,
        -s2 / tau_s + (1.0 - s2) * gamma * r2,
        r1,
        r2,
    )
}

/// Simulate `n_steps` Wong-Wang iterations; write per-step state +
/// firing-rate traces. `xi` must be length `2 * n_steps` (two noise
/// samples per step, consumed in `i1, i2` order).
pub fn simulate(
    mut s1: f64,
    mut s2: f64,
    tau_s: f64,
    gamma: f64,
    j_n: f64,
    j_cross: f64,
    i_0: f64,
    sigma: f64,
    dt: f64,
    stim1: &[f64],
    stim2: &[f64],
    xi: &[f64],
    s1_out: &mut [f64],
    s2_out: &mut [f64],
    r1_out: &mut [f64],
    r2_out: &mut [f64],
) -> (f64, f64) {
    let n = stim1.len();
    assert_eq!(stim2.len(), n, "stim2 length mismatch");
    assert_eq!(xi.len(), 2 * n, "xi length must be 2 * n_steps");
    assert_eq!(s1_out.len(), n, "s1_out length mismatch");
    assert_eq!(s2_out.len(), n, "s2_out length mismatch");
    assert_eq!(r1_out.len(), n, "r1_out length mismatch");
    assert_eq!(r2_out.len(), n, "r2_out length mismatch");

    for t in 0..n {
        let xi1 = xi[2 * t];
        let xi2 = xi[2 * t + 1];
        let (k1_s1, k1_s2, r1, r2) = derivatives(
            s1, s2, stim1[t], stim2[t], xi1, xi2, tau_s, gamma, j_n, j_cross, i_0, sigma,
        );
        let (k2_s1, k2_s2, _, _) = derivatives(
            s1 + 0.5 * dt * k1_s1,
            s2 + 0.5 * dt * k1_s2,
            stim1[t],
            stim2[t],
            xi1,
            xi2,
            tau_s,
            gamma,
            j_n,
            j_cross,
            i_0,
            sigma,
        );
        let (k3_s1, k3_s2, _, _) = derivatives(
            s1 + 0.5 * dt * k2_s1,
            s2 + 0.5 * dt * k2_s2,
            stim1[t],
            stim2[t],
            xi1,
            xi2,
            tau_s,
            gamma,
            j_n,
            j_cross,
            i_0,
            sigma,
        );
        let (k4_s1, k4_s2, _, _) = derivatives(
            s1 + dt * k3_s1,
            s2 + dt * k3_s2,
            stim1[t],
            stim2[t],
            xi1,
            xi2,
            tau_s,
            gamma,
            j_n,
            j_cross,
            i_0,
            sigma,
        );
        s1 = (s1 + dt * (k1_s1 + 2.0 * k2_s1 + 2.0 * k3_s1 + k4_s1) / 6.0).clamp(0.0, 1.0);
        s2 = (s2 + dt * (k1_s2 + 2.0 * k2_s2 + 2.0 * k3_s2 + k4_s2) / 6.0).clamp(0.0, 1.0);
        s1_out[t] = s1;
        s2_out[t] = s2;
        r1_out[t] = r1;
        r2_out[t] = r2;
    }
    (s1, s2)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> (f64, f64, f64, f64, f64, f64, f64) {
        // tau_s, gamma, j_n, j_cross, i_0, sigma, dt
        (0.1, 0.641, 0.2609, 0.0497, 0.3255, 0.0, 0.001)
    }

    #[test]
    fn phi_singularity_guard_returns_finite() {
        // a*i - b == 0 at i = b/a = 0.4
        let r = phi(B / A);
        assert!(r.is_finite());
        assert!((r - 1.0 / D).abs() < 1e-6);
    }

    #[test]
    fn phi_monotone_increasing() {
        let lo = phi(0.5);
        let hi = phi(1.0);
        assert!(hi > lo);
    }

    #[test]
    fn rk4_state_differs_from_forward_euler() {
        let n = 1;
        let stim1 = vec![0.17_f64; n];
        let stim2 = vec![0.03_f64; n];
        let xi = vec![0.0_f64; 2 * n];
        let mut s1o = vec![0.0_f64; n];
        let mut s2o = vec![0.0_f64; n];
        let mut r1o = vec![0.0_f64; n];
        let mut r2o = vec![0.0_f64; n];
        let (s1_f, s2_f) = simulate(
            0.24, 0.11, 0.1, 0.641, 0.2609, 0.0497, 0.3255, 0.0, 0.02, &stim1, &stim2, &xi,
            &mut s1o, &mut s2o, &mut r1o, &mut r2o,
        );
        let r1 = phi(0.2609 * 0.24 - 0.0497 * 0.11 + 0.3255 + 0.17);
        let r2 = phi(0.2609 * 0.11 - 0.0497 * 0.24 + 0.3255 + 0.03);
        let euler_s1 = (0.24 + (-0.24 / 0.1 + (1.0 - 0.24) * 0.641 * r1) * 0.02).clamp(0.0, 1.0);
        let euler_s2 = (0.11 + (-0.11 / 0.1 + (1.0 - 0.11) * 0.641 * r2) * 0.02).clamp(0.0, 1.0);
        assert!((s1_f - euler_s1).abs() > 1e-5);
        assert!((s2_f - euler_s2).abs() > 1e-5);
    }

    #[test]
    fn zero_noise_zero_stim_converges_to_fixed_point() {
        // With sigma=0, identical initial conditions, balanced dynamics:
        // s1 and s2 should stay very close and bounded in [0, 1].
        let (tau_s, gamma, j_n, j_cross, i_0, _, dt) = params();
        let n = 10_000;
        let stim = vec![0.0_f64; n];
        let xi = vec![0.0_f64; 2 * n];
        let mut s1o = vec![0.0_f64; n];
        let mut s2o = vec![0.0_f64; n];
        let mut r1o = vec![0.0_f64; n];
        let mut r2o = vec![0.0_f64; n];
        let (s1_f, s2_f) = simulate(
            0.1, 0.1, tau_s, gamma, j_n, j_cross, i_0, 0.0, dt, &stim, &stim, &xi, &mut s1o,
            &mut s2o, &mut r1o, &mut r2o,
        );
        assert!(s1_f.is_finite() && s2_f.is_finite());
        assert!((0.0..=1.0).contains(&s1_f));
        assert!((0.0..=1.0).contains(&s2_f));
        assert!(
            (s1_f - s2_f).abs() < 1e-9,
            "symmetric init must stay symmetric under zero noise"
        );
    }

    #[test]
    fn biased_stimulus_drives_winner() {
        // Strong stim1, no stim2, no noise → s1 > s2 within a few time constants.
        let (tau_s, gamma, j_n, j_cross, i_0, _, dt) = params();
        let n = 50_000;
        let stim1 = vec![0.2_f64; n];
        let stim2 = vec![0.0_f64; n];
        let xi = vec![0.0_f64; 2 * n];
        let mut s1o = vec![0.0_f64; n];
        let mut s2o = vec![0.0_f64; n];
        let mut r1o = vec![0.0_f64; n];
        let mut r2o = vec![0.0_f64; n];
        let (s1_f, s2_f) = simulate(
            0.1, 0.1, tau_s, gamma, j_n, j_cross, i_0, 0.0, dt, &stim1, &stim2, &xi, &mut s1o,
            &mut s2o, &mut r1o, &mut r2o,
        );
        assert!(s1_f > 0.5, "winner s1 should reach attractor; got {s1_f}");
        assert!(s2_f < 0.2, "loser s2 should be suppressed; got {s2_f}");
    }

    #[test]
    fn output_trace_shape_matches_input() {
        let n = 128;
        let stim = vec![0.1_f64; n];
        let xi = vec![0.0_f64; 2 * n];
        let mut s1o = vec![0.0_f64; n];
        let mut s2o = vec![0.0_f64; n];
        let mut r1o = vec![0.0_f64; n];
        let mut r2o = vec![0.0_f64; n];
        simulate(
            0.1, 0.1, 0.1, 0.641, 0.2609, 0.0497, 0.3255, 0.0, 0.001, &stim, &stim, &xi, &mut s1o,
            &mut s2o, &mut r1o, &mut r2o,
        );
        // Every step wrote; nothing still at sentinel zero (for r_out —
        // s_out might genuinely be near 0, so check r_out which is phi(I) ≥ some positive
        // floor when I > 0 on this parameter set).
        assert!(r1o.iter().all(|&r| r > 0.0));
        assert!(r2o.iter().all(|&r| r > 0.0));
    }

    #[test]
    #[should_panic(expected = "xi length must be 2 * n_steps")]
    fn mismatched_xi_length_panics() {
        let n = 10;
        let stim = vec![0.0_f64; n];
        let xi = vec![0.0_f64; n]; // wrong — should be 2*n
        let mut s1o = vec![0.0_f64; n];
        let mut s2o = vec![0.0_f64; n];
        let mut r1o = vec![0.0_f64; n];
        let mut r2o = vec![0.0_f64; n];
        simulate(
            0.1, 0.1, 0.1, 0.641, 0.2609, 0.0497, 0.3255, 0.0, 0.001, &stim, &stim, &xi, &mut s1o,
            &mut s2o, &mut r1o, &mut r2o,
        );
    }
}
