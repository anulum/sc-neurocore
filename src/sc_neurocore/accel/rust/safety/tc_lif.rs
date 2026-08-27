// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Standalone Rust safety mirror for TwoCompartmentLIFNeuron

/// Standalone safety mirror of the TC-LIF map (Zhang et al. 2024,
/// Eqs. 10–12), matching the Python reference and its atomic
/// fail-closed contract. Defaults are the published S-MNIST
/// feedforward profile.
#[derive(Debug, Clone)]
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

    /// Advance one step; `Err` preserves the pre-step state exactly for a
    /// non-finite input, an out-of-bounds configuration, or a non-finite
    /// candidate.
    pub fn step(&mut self, i_ext: f64) -> Result<i32, &'static str> {
        if !i_ext.is_finite() {
            return Err("i_ext must be finite");
        }
        if !validate_tc_lif(self) {
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

    /// Restore the dynamic state to zero, preserving parameters.
    pub fn reset(&mut self) {
        self.u_d = 0.0;
        self.u_s = 0.0;
        self.s_prev = 0.0;
    }
}

/// Return whether every state and configuration field is finite and
/// inside the public descriptor bounds.
pub fn validate_tc_lif(state: &TwoCompartmentLIFNeuron) -> bool {
    [
        state.u_d,
        state.u_s,
        state.s_prev,
        state.beta1,
        state.beta2,
        state.gamma,
        state.v_th,
    ]
    .into_iter()
    .all(f64::is_finite)
        && (-1e6..=1e6).contains(&state.u_d)
        && (-1e6..=1e6).contains(&state.u_s)
        && (state.s_prev == 0.0 || state.s_prev == 1.0)
        && state.beta1 > -1.0
        && state.beta1 < 0.0
        && state.beta2 > 0.0
        && state.beta2 < 1.0
        && (0.0..=10.0).contains(&state.gamma)
        && state.v_th > 0.0
        && state.v_th <= 100.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_independent_paper_oracle() {
        let (beta1, beta2, gamma, v_th) = (-0.5, 0.5, 0.5, 1.0);
        let mut oracle = (0.0_f64, 0.0_f64, 0.0_f64);
        let mut state = TwoCompartmentLIFNeuron::new();
        for index in 0..200 {
            let input = 0.3 + 0.2 * ((index % 7) as f64 - 3.0);
            let u_d = oracle.0 + beta1 * oracle.1 + input - gamma * oracle.2;
            let u_s = oracle.1 + beta2 * u_d - v_th * oracle.2;
            let spike = if u_s >= v_th { 1.0 } else { 0.0 };
            oracle = (u_d, u_s, spike);
            assert_eq!(state.step(input), Ok(spike as i32));
            assert_eq!(state.u_d, u_d);
            assert_eq!(state.u_s, u_s);
        }
    }

    #[test]
    fn invalid_drive_is_atomic() {
        let mut state = TwoCompartmentLIFNeuron::new();
        let before = state.clone();
        assert!(state.step(f64::NAN).is_err());
        assert!(state.step(f64::INFINITY).is_err());
        assert_eq!(state.u_d, before.u_d);
        assert_eq!(state.u_s, before.u_s);
    }

    #[test]
    fn invalid_configuration_is_atomic() {
        let mut state = TwoCompartmentLIFNeuron::new();
        state.beta1 = 0.5;
        let before = state.clone();
        assert!(state.step(0.0).is_err());
        assert_eq!(state.u_d, before.u_d);
        assert_eq!(state.beta1, before.beta1);
    }

    #[test]
    fn reset_preserves_parameters() {
        let mut state = TwoCompartmentLIFNeuron::new();
        state.gamma = 0.7;
        state.u_s = 0.4;
        state.reset();
        assert_eq!(state.u_s, 0.0);
        assert_eq!(state.gamma, 0.7);
    }
}
