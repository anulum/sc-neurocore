// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety mirror for Wong-Wang 2006

const A: f64 = 270.0;
const B: f64 = 108.0;
const D: f64 = 0.154;

#[derive(Debug, Clone)]
pub struct WongWangUnit {
    pub s1: f64,
    pub s2: f64,
    pub noise1: f64,
    pub noise2: f64,
    pub tau_s: f64,
    pub tau_ampa: f64,
    pub gamma: f64,
    pub j_n: f64,
    pub j_cross: f64,
    pub i_0: f64,
    pub sigma: f64,
    pub dt: f64,
}

impl WongWangUnit {
    pub fn new() -> Self {
        Self {
            s1: 0.1,
            s2: 0.1,
            noise1: 0.0,
            noise2: 0.0,
            tau_s: 0.1,
            tau_ampa: 0.002,
            gamma: 0.641,
            j_n: 0.2609,
            j_cross: 0.0497,
            i_0: 0.3255,
            sigma: 0.02,
            dt: 0.0001,
        }
    }

    pub fn phi(i_syn: f64) -> Result<f64, &'static str> {
        if !i_syn.is_finite() {
            return Err("Wong-Wang synaptic current must be finite");
        }
        let x = A * i_syn - B;
        let scaled = -D * x;
        let response = if scaled > 700.0 {
            0.0
        } else if x.abs() < 1.0e-7 {
            1.0 / D
        } else {
            x / -scaled.exp_m1()
        };
        if !response.is_finite() {
            return Err("Wong-Wang transfer response must be finite");
        }
        Ok(response.max(0.0))
    }

    pub fn step(
        &mut self,
        stim1: f64,
        stim2: f64,
        xi1: f64,
        xi2: f64,
    ) -> Result<(f64, f64), &'static str> {
        if !validate_wong_wang(self) {
            return Err("invalid Wong-Wang runtime state");
        }
        if ![stim1, stim2, xi1, xi2]
            .iter()
            .all(|value| value.is_finite())
        {
            return Err("invalid Wong-Wang stimulus or Gaussian sample");
        }
        let current1 = self.j_n * self.s1 - self.j_cross * self.s2 + self.i_0 + stim1 + self.noise1;
        let current2 = self.j_n * self.s2 - self.j_cross * self.s1 + self.i_0 + stim2 + self.noise2;
        let rate1 = Self::phi(current1)?;
        let rate2 = Self::phi(current2)?;
        let ds1 = -self.s1 / self.tau_s + (1.0 - self.s1) * self.gamma * rate1;
        let ds2 = -self.s2 / self.tau_s + (1.0 - self.s2) * self.gamma * rate2;
        let noise_scale = (self.dt / self.tau_ampa).sqrt() * self.sigma;
        let candidate = (
            self.s1 + self.dt * ds1,
            self.s2 + self.dt * ds2,
            self.noise1 - (self.dt / self.tau_ampa) * self.noise1 + noise_scale * xi1,
            self.noise2 - (self.dt / self.tau_ampa) * self.noise2 + noise_scale * xi2,
        );
        if ![candidate.0, candidate.1, candidate.2, candidate.3]
            .iter()
            .all(|value| value.is_finite())
        {
            return Err("invalid Wong-Wang candidate state");
        }
        if !(0.0..=1.0).contains(&candidate.0) || !(0.0..=1.0).contains(&candidate.1) {
            return Err("Wong-Wang candidate gating state left [0, 1]");
        }
        self.s1 = candidate.0;
        self.s2 = candidate.1;
        self.noise1 = candidate.2;
        self.noise2 = candidate.3;
        Ok((rate1, rate2))
    }

    pub fn reset(&mut self) {
        self.s1 = 0.1;
        self.s2 = 0.1;
        self.noise1 = 0.0;
        self.noise2 = 0.0;
    }
}

impl Default for WongWangUnit {
    fn default() -> Self {
        Self::new()
    }
}

pub fn validate_wong_wang(state: &WongWangUnit) -> bool {
    finite_gate(state.s1)
        && finite_gate(state.s2)
        && state.noise1.is_finite()
        && state.noise2.is_finite()
        && state.tau_s.is_finite()
        && state.tau_s > 0.0
        && state.tau_ampa.is_finite()
        && state.tau_ampa > 0.0
        && state.gamma.is_finite()
        && state.gamma > 0.0
        && state.j_n.is_finite()
        && state.j_n >= 0.0
        && state.j_cross.is_finite()
        && state.j_cross >= 0.0
        && state.i_0.is_finite()
        && state.sigma.is_finite()
        && state.sigma >= 0.0
        && state.dt.is_finite()
        && state.dt > 0.0
}

fn finite_gate(value: f64) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_the_published_reduced_model() {
        let state = WongWangUnit::new();
        assert!(validate_wong_wang(&state));
        assert_eq!(state.tau_ampa, 0.002);
        assert_eq!(state.dt, 0.0001);
    }

    #[test]
    fn euler_and_ou_update_match_the_source_equations() {
        let mut state = WongWangUnit::new();
        state.s1 = 0.24;
        state.s2 = 0.11;
        state.noise1 = 0.01;
        state.noise2 = -0.02;
        let old = state.clone();
        let rates = state.step(0.17, 0.03, 0.5, -1.0).unwrap();
        let expected_s1 =
            old.s1 + old.dt * (-old.s1 / old.tau_s + (1.0 - old.s1) * old.gamma * rates.0);
        let expected_s2 =
            old.s2 + old.dt * (-old.s2 / old.tau_s + (1.0 - old.s2) * old.gamma * rates.1);
        let scale = (old.dt / old.tau_ampa).sqrt() * old.sigma;
        assert_eq!(state.s1, expected_s1);
        assert_eq!(state.s2, expected_s2);
        assert_eq!(
            state.noise1,
            old.noise1 - (old.dt / old.tau_ampa) * old.noise1 + scale * 0.5
        );
        assert_eq!(
            state.noise2,
            old.noise2 - (old.dt / old.tau_ampa) * old.noise2 - scale
        );
    }

    #[test]
    fn rejection_is_atomic() {
        let mut state = WongWangUnit::new();
        let before = state.clone();
        assert!(state.step(f64::NAN, 0.0, 0.0, 0.0).is_err());
        assert_eq!(state.s1, before.s1);
        assert_eq!(state.noise2, before.noise2);
    }

    #[test]
    fn reset_preserves_parameters() {
        let mut state = WongWangUnit::new();
        state.tau_s = 0.12;
        state.tau_ampa = 0.003;
        state.dt = 0.0002;
        state.s1 = 0.2;
        state.noise1 = 0.3;
        state.reset();
        assert_eq!(
            (state.s1, state.s2, state.noise1, state.noise2),
            (0.1, 0.1, 0.0, 0.0)
        );
        assert_eq!(
            (state.tau_s, state.tau_ampa, state.dt),
            (0.12, 0.003, 0.0002)
        );
    }
}
