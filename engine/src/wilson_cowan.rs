// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust N-step simulator for the Wilson-Cowan 1972 E/I rate model

//! Batch parity with `WilsonCowanUnit.step` in
//! `src/sc_neurocore/neurons/models/wilson_cowan.py` (Wilson & Cowan
//! 1972, Biophys. J. 12:1–24).
//!
//! Per step:
//!   s_e = sigmoid(w_ee · E − w_ei · I + ext)
//!   s_i = sigmoid(w_ie · E − w_ii · I)
//!   E += (−E + s_e) · dt / τ_e
//!   I += (−I + s_i) · dt / τ_i
//!
//! where `sigmoid(x) = 1 / (1 + exp(−a·(x − θ)))`.
//!
//! The model is deterministic (no noise), so bit-exact parity with the
//! Python primary requires only matching arithmetic — no pre-drawn
//! RNG buffer needed.

#[inline]
fn sigmoid(a: f64, theta: f64, x: f64) -> f64 {
    1.0 / (1.0 + (-a * (x - theta)).exp())
}

/// Simulate `ext_input.len()` Wilson-Cowan iterations, writing per-step
/// `E` and `I` traces into caller-allocated buffers. Returns final
/// `(E, I)` for convenience.
#[allow(clippy::too_many_arguments)]
pub fn simulate(
    mut e: f64,
    mut i: f64,
    w_ee: f64,
    w_ei: f64,
    w_ie: f64,
    w_ii: f64,
    tau_e: f64,
    tau_i: f64,
    a: f64,
    theta: f64,
    dt: f64,
    ext_input: &[f64],
    e_out: &mut [f64],
    i_out: &mut [f64],
) -> (f64, f64) {
    let n = ext_input.len();
    assert_eq!(e_out.len(), n, "e_out length mismatch");
    assert_eq!(i_out.len(), n, "i_out length mismatch");

    for t in 0..n {
        let s_e = sigmoid(a, theta, w_ee * e - w_ei * i + ext_input[t]);
        let s_i = sigmoid(a, theta, w_ie * e - w_ii * i);
        e += (-e + s_e) / tau_e * dt;
        i += (-i + s_i) / tau_i * dt;
        e_out[t] = e;
        i_out[t] = i;
    }
    (e, i)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn defaults() -> (f64, f64, f64, f64, f64, f64, f64, f64, f64) {
        // w_ee, w_ei, w_ie, w_ii, tau_e, tau_i, a, theta, dt
        (10.0, 6.0, 10.0, 1.0, 1.0, 2.0, 1.2, 4.0, 0.1)
    }

    #[test]
    fn sigmoid_monotone_increasing() {
        let (_, _, _, _, _, _, a, theta, _) = defaults();
        let lo = sigmoid(a, theta, 0.0);
        let mid = sigmoid(a, theta, 4.0);
        let hi = sigmoid(a, theta, 10.0);
        assert!(lo < mid && mid < hi);
    }

    #[test]
    fn sigmoid_at_theta_equals_half() {
        let (_, _, _, _, _, _, a, theta, _) = defaults();
        let r = sigmoid(a, theta, theta);
        assert!((r - 0.5).abs() < 1e-12);
    }

    #[test]
    fn sigmoid_asymptotes() {
        let (_, _, _, _, _, _, a, theta, _) = defaults();
        assert!(sigmoid(a, theta, -1e6) < 1e-50);
        assert!((sigmoid(a, theta, 1e6) - 1.0).abs() < 1e-50);
    }

    #[test]
    fn quiescent_converges() {
        let (w_ee, w_ei, w_ie, w_ii, tau_e, tau_i, a, theta, dt) = defaults();
        let n = 20_000;
        let ext = vec![0.0_f64; n];
        let mut e_out = vec![0.0_f64; n];
        let mut i_out = vec![0.0_f64; n];
        let (e_f, i_f) = simulate(
            0.1, 0.05, w_ee, w_ei, w_ie, w_ii, tau_e, tau_i, a, theta, dt, &ext, &mut e_out,
            &mut i_out,
        );
        assert!(e_f.is_finite() && i_f.is_finite());
        assert!(e_f < 0.2, "quiescent E must stay low, got {e_f}");
        assert!(i_f < 0.2, "quiescent I must stay low, got {i_f}");
    }

    #[test]
    fn strong_drive_elevates_activity() {
        let (w_ee, w_ei, w_ie, w_ii, tau_e, tau_i, a, theta, dt) = defaults();
        let n = 10_000;
        let ext = vec![10.0_f64; n];
        let mut e_out = vec![0.0_f64; n];
        let mut i_out = vec![0.0_f64; n];
        let (e_f, _) = simulate(
            0.1, 0.05, w_ee, w_ei, w_ie, w_ii, tau_e, tau_i, a, theta, dt, &ext, &mut e_out,
            &mut i_out,
        );
        assert!(e_f > 0.3, "strong external drive must elevate E, got {e_f}");
    }

    #[test]
    fn output_trace_shape_matches_input() {
        let n = 64;
        let ext = vec![1.0_f64; n];
        let mut e_out = vec![f64::NAN; n];
        let mut i_out = vec![f64::NAN; n];
        simulate(
            0.1, 0.05, 10.0, 6.0, 10.0, 1.0, 1.0, 2.0, 1.2, 4.0, 0.1, &ext, &mut e_out, &mut i_out,
        );
        assert!(e_out.iter().all(|v| v.is_finite()));
        assert!(i_out.iter().all(|v| v.is_finite()));
    }

    #[test]
    #[should_panic(expected = "e_out length mismatch")]
    fn mismatched_e_out_panics() {
        let n = 10;
        let ext = vec![0.0_f64; n];
        let mut e_out = vec![0.0_f64; n + 1];
        let mut i_out = vec![0.0_f64; n];
        simulate(
            0.1, 0.05, 10.0, 6.0, 10.0, 1.0, 1.0, 2.0, 1.2, 4.0, 0.1, &ext, &mut e_out, &mut i_out,
        );
    }
}
