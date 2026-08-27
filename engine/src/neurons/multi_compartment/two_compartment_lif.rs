// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Two-compartment LIF neuron models

//! TC-LIF (Zhang et al. 2024) and the preserved SC exponential variant.

/// TC-LIF — Zhang et al. (2024) two-compartment spiking neuron.
///
/// Discrete map (paper Eqs. 10–12, exact ordering U_D → U_S → S):
///
/// ```text
/// U_D[t] = U_D[t-1] + beta1 * U_S[t-1] + I[t] - gamma * S[t-1]
/// U_S[t] = U_S[t-1] + beta2 * U_D[t]          - v_th  * S[t-1]
/// S[t]   = Theta(U_S[t] - v_th)
/// ```
///
/// One external input enters the dendrite; both compartments reset
/// softly through the delayed spike `S[t-1]`. `beta1 ∈ (-1, 0)` and
/// `beta2 ∈ (0, 1)` per the paper's sigmoid parametrisation; the
/// defaults are the published S-MNIST feedforward profile (Table 5) —
/// the paper has dataset-specific profiles and no universal default.
///
/// Zhang, Yang, Ma, Wu, Li & Tan, AAAI 38(15):16838, 2024.
#[derive(Clone, Debug)]
pub struct TwoCompartmentLIFNeuron {
    pub u_d: f64,
    pub u_s: f64,
    pub s_prev: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub gamma: f64,
    pub v_th: f64,
}

impl TwoCompartmentLIFNeuron {
    /// Construct the published S-MNIST feedforward profile.
    pub fn new() -> Self {
        Self {
            u_d: 0.0,
            u_s: 0.0,
            s_prev: 0.0,
            beta1: -0.5,
            beta2: 0.5,
            gamma: 0.5,
            v_th: 1.0,
        }
    }

    fn valid(&self) -> bool {
        [
            self.u_d,
            self.u_s,
            self.s_prev,
            self.beta1,
            self.beta2,
            self.gamma,
            self.v_th,
        ]
        .into_iter()
        .all(f64::is_finite)
            && (-1e6..=1e6).contains(&self.u_d)
            && (-1e6..=1e6).contains(&self.u_s)
            && (self.s_prev == 0.0 || self.s_prev == 1.0)
            && self.beta1 > -1.0
            && self.beta1 < 0.0
            && self.beta2 > 0.0
            && self.beta2 < 1.0
            && (0.0..=10.0).contains(&self.gamma)
            && self.v_th > 0.0
            && self.v_th <= 100.0
    }

    /// Advance one step after validating the input and configuration.
    ///
    /// Computes the paper map on candidates and commits only on success:
    /// a non-finite input, a configuration outside the public bounds, or
    /// a non-finite candidate returns `Err` with the pre-step state
    /// preserved exactly.
    pub fn try_step(&mut self, i_ext: f64) -> Result<i32, &'static str> {
        if !i_ext.is_finite() {
            return Err("i_ext must be finite");
        }
        if !self.valid() {
            return Err("TC-LIF state and parameters must satisfy the public bounds");
        }

        let u_d_candidate = self.u_d + self.beta1 * self.u_s + i_ext - self.gamma * self.s_prev;
        let u_s_candidate = self.u_s + self.beta2 * u_d_candidate - self.v_th * self.s_prev;
        if !(u_d_candidate.is_finite() && u_s_candidate.is_finite()) {
            return Err("TC-LIF candidate state became non-finite");
        }
        let spike = if u_s_candidate >= self.v_th { 1 } else { 0 };

        self.u_d = u_d_candidate;
        self.u_s = u_s_candidate;
        self.s_prev = spike as f64;
        Ok(spike)
    }

    /// Fail-closed wrapper for legacy callers: returns 0 on any rejected
    /// input without mutating state.
    pub fn step(&mut self, i_ext: f64) -> i32 {
        self.try_step(i_ext).unwrap_or(0)
    }

    /// Restore the dynamic state to zero, preserving parameters.
    pub fn reset(&mut self) {
        self.u_d = 0.0;
        self.u_s = 0.0;
        self.s_prev = 0.0;
    }
}

impl Default for TwoCompartmentLIFNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// SC exponential two-compartment LIF — preserved engine recurrence.
///
/// Historical production-engine model formerly published under the
/// `TwoCompartmentLIFNeuron` name, kept verbatim as a count-neutral SC
/// identity with its original two-current API and hard soma reset:
///
/// ```text
/// V_d[t] = exp(-dt/tau_d) * V_d[t-1] + I_dend[t]
/// V_s[t] = exp(-dt/tau_s) * V_s[t-1] + I_soma[t] + kappa * V_d[t]
/// spike when V_s >= theta; V_s -> V_reset, V_d unchanged
/// ```
///
/// It is structurally distinct from the Zhang et al. (2024) TC-LIF and
/// makes no publication-exact claim.
#[derive(Clone, Debug)]
pub struct SCExponentialTwoCompartmentLIF {
    pub v_s: f64,
    pub v_d: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub theta: f64,
    pub tau_s: f64,
    pub tau_d: f64,
    pub kappa: f64,
    pub dt: f64,
}

impl SCExponentialTwoCompartmentLIF {
    /// Construct the historical engine default configuration.
    pub fn new() -> Self {
        Self {
            v_s: 0.0,
            v_d: 0.0,
            v_rest: 0.0,
            v_reset: 0.0,
            theta: 1.0,
            tau_s: 2.0,
            tau_d: 20.0,
            kappa: 0.5,
            dt: 1.0,
        }
    }

    /// Advance one step with the historical recurrence, preserved
    /// verbatim from the pre-2026-08-27 engine implementation.
    pub fn step(&mut self, i_soma: f64, i_dend: f64) -> i32 {
        let alpha_s = (-self.dt / self.tau_s).exp();
        let alpha_d = (-self.dt / self.tau_d).exp();
        self.v_d = alpha_d * self.v_d + i_dend;
        self.v_s = alpha_s * self.v_s + i_soma + self.kappa * self.v_d;
        if self.v_s >= self.theta {
            self.v_s = self.v_reset;
            1
        } else {
            0
        }
    }

    /// Restore both compartments to the rest potential.
    pub fn reset(&mut self) {
        self.v_s = self.v_rest;
        self.v_d = self.v_rest;
    }
}

impl Default for SCExponentialTwoCompartmentLIF {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tc_lif_matches_independent_paper_oracle() {
        let (beta1, beta2, gamma, v_th) = (-0.5, 0.5, 0.5, 1.0);
        let mut oracle = (0.0_f64, 0.0_f64, 0.0_f64);
        let mut neuron = TwoCompartmentLIFNeuron::new();
        for index in 0..200 {
            let input = 0.3 + 0.2 * ((index % 7) as f64 - 3.0);
            let u_d = oracle.0 + beta1 * oracle.1 + input - gamma * oracle.2;
            let u_s = oracle.1 + beta2 * u_d - v_th * oracle.2;
            let spike = if u_s >= v_th { 1.0 } else { 0.0 };
            oracle = (u_d, u_s, spike);
            let got = neuron.try_step(input).expect("finite configured drive");
            assert_eq!(got as f64, spike);
            assert_eq!(neuron.u_d, u_d);
            assert_eq!(neuron.u_s, u_s);
            assert_eq!(neuron.s_prev, spike);
        }
    }

    #[test]
    fn tc_lif_delayed_soft_reset_uses_previous_spike() {
        let mut neuron = TwoCompartmentLIFNeuron::new();
        // Force a spike, then verify the NEXT step subtracts gamma and v_th.
        let spike = neuron.try_step(4.0).expect("finite drive");
        assert_eq!(spike, 1);
        let (u_d, u_s) = (neuron.u_d, neuron.u_s);
        neuron.try_step(0.0).expect("finite drive");
        let expected_u_d = u_d + neuron.beta1 * u_s + 0.0 - neuron.gamma;
        let expected_u_s = u_s + neuron.beta2 * expected_u_d - neuron.v_th;
        assert_eq!(neuron.u_d, expected_u_d);
        assert_eq!(neuron.u_s, expected_u_s);
    }

    #[test]
    fn tc_lif_beta_signs_are_enforced() {
        let mut neuron = TwoCompartmentLIFNeuron::new();
        neuron.beta1 = 0.5;
        assert!(neuron.try_step(0.0).is_err());
        let mut neuron = TwoCompartmentLIFNeuron::new();
        neuron.beta2 = -0.5;
        assert!(neuron.try_step(0.0).is_err());
    }

    #[test]
    fn tc_lif_invalid_input_is_rejected_atomically() {
        let mut neuron = TwoCompartmentLIFNeuron::new();
        let before = neuron.clone();
        assert!(neuron.try_step(f64::NAN).is_err());
        assert!(neuron.try_step(f64::INFINITY).is_err());
        assert_eq!(neuron.u_d, before.u_d);
        assert_eq!(neuron.u_s, before.u_s);
        assert_eq!(neuron.s_prev, before.s_prev);
    }

    #[test]
    fn tc_lif_fires_and_resets() {
        let mut neuron = TwoCompartmentLIFNeuron::new();
        let total: i32 = (0..100).map(|_| neuron.step(0.5)).sum();
        assert!(total > 0);
        neuron.reset();
        assert_eq!(neuron.u_d, 0.0);
        assert_eq!(neuron.u_s, 0.0);
        assert_eq!(neuron.s_prev, 0.0);
    }

    #[test]
    fn sc_exponential_reproduces_frozen_engine_anchors() {
        // Anchors captured from the pre-change built engine.
        let mut neuron = SCExponentialTwoCompartmentLIF::new();
        let spikes: Vec<i32> = (0..10).map(|_| neuron.step(0.5, 0.3)).collect();
        assert_eq!(spikes, vec![0, 1, 0, 1, 1, 1, 1, 1, 1, 1]);
        assert_eq!(neuron.v_s, 0.0);
        assert!((neuron.v_d - 2.420_328_258_950_689).abs() < 1e-15);

        let mut long_run = SCExponentialTwoCompartmentLIF::new();
        for _ in 0..50 {
            long_run.step(0.2, 0.1);
        }
        assert_eq!(long_run.v_s, 0.0);
        assert!((long_run.v_d - 1.882_108_201_469_840_3).abs() < 1e-15);
    }

    #[test]
    fn sc_exponential_reset_restores_rest() {
        let mut neuron = SCExponentialTwoCompartmentLIF::new();
        for _ in 0..20 {
            neuron.step(0.5, 0.3);
        }
        neuron.reset();
        assert_eq!(neuron.v_s, 0.0);
        assert_eq!(neuron.v_d, 0.0);
    }
}
