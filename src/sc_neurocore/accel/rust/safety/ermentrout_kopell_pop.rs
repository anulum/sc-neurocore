// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Independent Rust safety mirror for the MPR mean field

// Dependency-free mirror after R=tau*r and t'=t/tau restore physical variables.

/// Complete macroscopic population state and numerical configuration.
#[derive(Debug, Clone)]
pub struct ErmentroutKopellPopulation {
    /// Population firing rate.
    pub r: f64,
    /// Mean membrane potential.
    pub v: f64,
    /// Positive membrane time scale.
    pub tau: f64,
    /// Non-negative Lorentzian excitability half-width.
    pub delta: f64,
    /// Centre of the neuronal excitability distribution.
    pub eta_bar: f64,
    /// Recurrent coupling strength.
    pub j: f64,
    /// Positive explicit-Euler step.
    pub dt: f64,
}

impl ErmentroutKopellPopulation {
    /// Construct the maintained phase-portrait parameter set.
    pub fn new() -> Self {
        Self {
            r: 0.1,
            v: -2.0,
            tau: 1.0,
            delta: 1.0,
            eta_bar: -5.0,
            j: 15.0,
            dt: 0.01,
        }
    }

    fn valid(&self) -> bool {
        [
            self.r,
            self.v,
            self.tau,
            self.delta,
            self.eta_bar,
            self.j,
            self.dt,
        ]
        .into_iter()
        .all(f64::is_finite)
            && self.r >= 0.0
            && self.tau > 0.0
            && self.delta >= 0.0
            && self.dt > 0.0
    }

    /// Apply one simultaneous Euler update, preserving state on rejection.
    pub fn step(&mut self, ext_input: f64) -> Result<f64, &'static str> {
        if !self.valid() || !ext_input.is_finite() {
            return Err("invalid MPR state, parameter, or input");
        }
        let scaled_rate = std::f64::consts::PI * self.tau * self.r;
        let dr = self.delta / (std::f64::consts::PI * self.tau * self.tau)
            + 2.0 * self.r * self.v / self.tau;
        let dv = (self.v * self.v + self.eta_bar + ext_input + self.j * self.tau * self.r
            - scaled_rate * scaled_rate)
            / self.tau;
        let next_r = self.r + self.dt * dr;
        let next_v = self.v + self.dt * dv;
        if !next_r.is_finite() || !next_v.is_finite() || next_r < 0.0 {
            return Err("invalid MPR candidate state");
        }
        self.r = next_r;
        self.v = next_v;
        Ok(self.r)
    }

    /// Restore both dynamic states while preserving all parameters.
    pub fn reset(&mut self) {
        self.r = 0.1;
        self.v = -2.0;
    }
}

impl Default for ErmentroutKopellPopulation {
    fn default() -> Self {
        Self::new()
    }
}

/// Return whether the complete state and parameter contract is valid.
pub fn validate_ermentrout_kopell_pop(state: &ErmentroutKopellPopulation) -> bool {
    state.valid()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_step_matches_source_equations() {
        let mut state = ErmentroutKopellPopulation {
            r: 0.2,
            v: -1.5,
            tau: 2.0,
            delta: 0.7,
            eta_bar: -3.0,
            j: 12.0,
            dt: 0.005,
        };
        let expected_r =
            0.2 + 0.005 * (0.7 / (std::f64::consts::PI * 4.0) + 2.0 * 0.2 * -1.5 / 2.0);
        let expected_v = -1.5
            + 0.005
                * ((-1.5_f64).powi(2) - 3.0 + 1.25 + 12.0 * 2.0 * 0.2
                    - (std::f64::consts::PI * 2.0 * 0.2).powi(2))
                / 2.0;
        state.step(1.25).unwrap();
        assert_eq!(state.r, expected_r);
        assert_eq!(state.v, expected_v);
    }

    #[test]
    fn invalid_input_does_not_mutate_state() {
        let mut state = ErmentroutKopellPopulation::new();
        let before = (state.r, state.v);
        assert!(state.step(f64::NAN).is_err());
        assert_eq!((state.r, state.v), before);
    }

    #[test]
    fn configured_trajectory_remains_finite() {
        let mut state = ErmentroutKopellPopulation::new();
        for index in 0..2_000 {
            let drive = 1.5 + 0.5 * ((index as f64) * 0.017).sin();
            state.step(drive).unwrap();
        }
        assert!(validate_ermentrout_kopell_pop(&state));
    }

    #[test]
    fn reset_preserves_parameters() {
        let mut state = ErmentroutKopellPopulation::new();
        state.tau = 2.0;
        state.delta = 0.7;
        state.j = 12.0;
        state.step(1.0).unwrap();
        state.reset();
        assert_eq!((state.r, state.v), (0.1, -2.0));
        assert_eq!((state.tau, state.delta, state.j), (2.0, 0.7, 12.0));
    }
}
