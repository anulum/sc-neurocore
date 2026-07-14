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
//!   dE/dt = (−E + sigmoid(w_ee · E − w_ei · I + ext)) / τ_e
//!   dI/dt = (−I + sigmoid(w_ie · E − w_ii · I)) / τ_i
//!   (E, I) advance through one fixed-step RK4 update.
//!
//! where `sigmoid(x) = logistic(a·(x − θ)) − logistic(−a·θ)`.
//!
//! The model is deterministic (no noise), so parity needs no pre-drawn RNG
//! buffer. Transcendental library implementations are compared under the
//! public bounded floating-point trajectory contract.

#[inline]
fn logistic(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let exp_z = z.exp();
        exp_z / (1.0 + exp_z)
    }
}

#[inline]
fn sigmoid(a: f64, theta: f64, x: f64) -> f64 {
    // Published Wilson-Cowan 1972 two-term form:
    //   S(x) = 1/(1+exp(-a(x-θ))) − 1/(1+exp(aθ))
    // Range is [-β, 1-β] where β = 1/(1+exp(aθ)).
    logistic(a * (x - theta)) - logistic(-a * theta)
}

#[inline]
fn finite_rate(value: f64, a: f64, theta: f64) -> bool {
    let baseline = logistic(-a * theta);
    value.is_finite() && value >= -baseline && value <= 1.0
}

#[expect(
    clippy::too_many_arguments,
    reason = "native parity surface passes the complete scientific configuration"
)]
fn valid_configuration(
    e: f64,
    i: f64,
    w_ee: f64,
    w_ei: f64,
    w_ie: f64,
    w_ii: f64,
    tau_e: f64,
    tau_i: f64,
    a: f64,
    theta: f64,
    dt: f64,
) -> bool {
    [w_ee, w_ei, w_ie, w_ii]
        .into_iter()
        .all(|value| value.is_finite() && value >= 0.0)
        && tau_e.is_finite()
        && tau_e > 0.0
        && tau_i.is_finite()
        && tau_i > 0.0
        && a.is_finite()
        && a > 0.0
        && theta.is_finite()
        && dt.is_finite()
        && dt > 0.0
        && finite_rate(e, a, theta)
        && finite_rate(i, a, theta)
}

#[inline]
fn derivatives(
    e: f64,
    i: f64,
    ext: f64,
    params: (f64, f64, f64, f64, f64, f64, f64, f64),
) -> (f64, f64) {
    let (w_ee, w_ei, w_ie, w_ii, tau_e, tau_i, a, theta) = params;
    let s_e = sigmoid(a, theta, w_ee * e - w_ei * i + ext);
    let s_i = sigmoid(a, theta, w_ie * e - w_ii * i);
    ((-e + s_e) / tau_e, (-i + s_i) / tau_i)
}

/// Simulate `ext_input.len()` Wilson-Cowan iterations, writing per-step
/// `E` and `I` traces into caller-allocated buffers. Returns final
/// `(E, I)` for convenience.
#[expect(
    clippy::too_many_arguments,
    reason = "Python extension parity surface passes canonical scalar parameters"
)]
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
) -> Result<(f64, f64), &'static str> {
    let n = ext_input.len();
    if e_out.len() != n {
        return Err("e_out length mismatch");
    }
    if i_out.len() != n {
        return Err("i_out length mismatch");
    }
    if !valid_configuration(e, i, w_ee, w_ei, w_ie, w_ii, tau_e, tau_i, a, theta, dt) {
        return Err("invalid Wilson-Cowan numerical configuration");
    }
    if !ext_input.iter().all(|value| value.is_finite()) {
        return Err("external input must be finite");
    }
    let params = (w_ee, w_ei, w_ie, w_ii, tau_e, tau_i, a, theta);
    let mut next_e_out = Vec::with_capacity(n);
    let mut next_i_out = Vec::with_capacity(n);

    for &ext in ext_input {
        let (k1_e, k1_i) = derivatives(e, i, ext, params);
        let (k2_e, k2_i) = derivatives(e + 0.5 * dt * k1_e, i + 0.5 * dt * k1_i, ext, params);
        let (k3_e, k3_i) = derivatives(e + 0.5 * dt * k2_e, i + 0.5 * dt * k2_i, ext, params);
        let (k4_e, k4_i) = derivatives(e + dt * k3_e, i + dt * k3_i, ext, params);
        if ![k1_e, k1_i, k2_e, k2_i, k3_e, k3_i, k4_e, k4_i]
            .into_iter()
            .all(f64::is_finite)
        {
            return Err("invalid Wilson-Cowan derivative");
        }
        let next_e = e + dt * (k1_e + 2.0 * k2_e + 2.0 * k3_e + k4_e) / 6.0;
        let next_i = i + dt * (k1_i + 2.0 * k2_i + 2.0 * k3_i + k4_i) / 6.0;
        if !finite_rate(next_e, a, theta) || !finite_rate(next_i, a, theta) {
            return Err("invalid Wilson-Cowan candidate state");
        }
        e = next_e;
        i = next_i;
        next_e_out.push(e);
        next_i_out.push(i);
    }
    e_out.copy_from_slice(&next_e_out);
    i_out.copy_from_slice(&next_i_out);
    Ok((e, i))
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
    fn sigmoid_at_zero_is_zero() {
        // Two-term form zeroes the baseline so S(0) = 0 exactly.
        let (_, _, _, _, _, _, a, theta, _) = defaults();
        assert!(sigmoid(a, theta, 0.0).abs() < 1e-12);
    }

    #[test]
    fn sigmoid_at_theta_equals_half_minus_baseline() {
        let (_, _, _, _, _, _, a, theta, _) = defaults();
        let baseline = 1.0 / (1.0 + (a * theta).exp());
        let r = sigmoid(a, theta, theta);
        assert!((r - (0.5 - baseline)).abs() < 1e-12);
    }

    #[test]
    fn sigmoid_asymptotes_respect_baseline() {
        // As x → +∞, S(x) → 1 − baseline. As x → −∞, S(x) → −baseline.
        let (_, _, _, _, _, _, a, theta, _) = defaults();
        let baseline = 1.0 / (1.0 + (a * theta).exp());
        assert!((sigmoid(a, theta, 1e6) - (1.0 - baseline)).abs() < 1e-50);
        assert!((sigmoid(a, theta, -1e6) - (-baseline)).abs() < 1e-50);
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
        )
        .unwrap();
        assert!(e_f.is_finite() && i_f.is_finite());
        assert!(e_f < 0.2, "quiescent E must stay low, got {e_f}");
        assert!(i_f < 0.2, "quiescent I must stay low, got {i_f}");
    }

    #[test]
    fn high_drive_elevates_activity() {
        let (w_ee, w_ei, w_ie, w_ii, tau_e, tau_i, a, theta, dt) = defaults();
        let n = 10_000;
        let ext = vec![10.0_f64; n];
        let mut e_out = vec![0.0_f64; n];
        let mut i_out = vec![0.0_f64; n];
        let (e_f, _) = simulate(
            0.1, 0.05, w_ee, w_ei, w_ie, w_ii, tau_e, tau_i, a, theta, dt, &ext, &mut e_out,
            &mut i_out,
        )
        .unwrap();
        assert!(e_f > 0.3, "high external drive must elevate E, got {e_f}");
    }

    #[test]
    fn rk4_step_matches_reference_and_separates_from_euler() {
        let mut e_out = vec![0.0_f64; 1];
        let mut i_out = vec![0.0_f64; 1];
        let ext = vec![3.0_f64; 1];
        simulate(
            0.24, 0.11, 10.0, 6.0, 10.0, 1.0, 1.0, 2.0, 1.2, 4.0, 0.35, &ext, &mut e_out,
            &mut i_out,
        )
        .unwrap();
        let euler_e = 0.40111014473980233_f64;
        let euler_i = 0.10924537850891547_f64;
        assert!((e_out[0] - 0.42143718680097664_f64).abs() < 1e-15);
        assert!((i_out[0] - 0.13798020053932203_f64).abs() < 1e-15);
        assert!((e_out[0] - euler_e).abs() > 1e-2);
        assert!((i_out[0] - euler_i).abs() > 1e-2);
    }

    #[test]
    fn output_trace_shape_matches_input() {
        let n = 64;
        let ext = vec![1.0_f64; n];
        let mut e_out = vec![f64::NAN; n];
        let mut i_out = vec![f64::NAN; n];
        simulate(
            0.1, 0.05, 10.0, 6.0, 10.0, 1.0, 1.0, 2.0, 1.2, 4.0, 0.1, &ext, &mut e_out, &mut i_out,
        )
        .unwrap();
        assert!(e_out.iter().all(|v| v.is_finite()));
        assert!(i_out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn mismatched_e_out_is_rejected() {
        let n = 10;
        let ext = vec![0.0_f64; n];
        let mut e_out = vec![0.0_f64; n + 1];
        let mut i_out = vec![0.0_f64; n];
        let error = simulate(
            0.1, 0.05, 10.0, 6.0, 10.0, 1.0, 1.0, 2.0, 1.2, 4.0, 0.1, &ext, &mut e_out, &mut i_out,
        )
        .unwrap_err();
        assert_eq!(error, "e_out length mismatch");
    }

    #[test]
    fn invalid_contract_preserves_caller_buffers() {
        let ext = vec![1.0_f64, f64::NAN, 1.0];
        let mut e_out = vec![-999.0_f64; ext.len()];
        let mut i_out = vec![-999.0_f64; ext.len()];
        let result = simulate(
            0.1, 0.05, 10.0, 6.0, 10.0, 1.0, 1.0, 2.0, 1.2, 4.0, 0.1, &ext, &mut e_out, &mut i_out,
        );
        assert!(result.is_err());
        assert_eq!(e_out, vec![-999.0; ext.len()]);
        assert_eq!(i_out, vec![-999.0; ext.len()]);
    }
}
