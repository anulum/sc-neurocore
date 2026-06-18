// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for sfa

const V_MIN: f64 = -200.0;
const V_MAX: f64 = 100.0;
const G_MAX: f64 = 1.0e9;

#[derive(Debug, Clone)]
pub struct SFANeuron {
    pub v: f64,
    pub g_sfa: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub tau_m: f64,
    pub tau_sfa: f64,
    pub delta_g: f64,
    pub e_k: f64,
    pub resistance: f64,
    pub dt: f64,
}

impl SFANeuron {
    pub fn new() -> Self {
        Self {
            v: -70.0_f64,
            g_sfa: 0.0_f64,
            v_rest: -70.0_f64,
            v_reset: -70.0_f64,
            v_threshold: -50.0_f64,
            tau_m: 10.0_f64,
            tau_sfa: 200.0_f64,
            delta_g: 0.5_f64,
            e_k: -80.0_f64,
            resistance: 1.0_f64,
            dt: 1.0_f64,
        }
    }

    pub fn step(&mut self, i_ext: f64) -> i32 {
        if !i_ext.is_finite() || self.validate().is_err() {
            return -1;
        }
        let (v_candidate, g_candidate) = self.rk4_candidate(self.v, self.g_sfa, i_ext);
        if !v_candidate.is_finite() || !g_candidate.is_finite() {
            return -1;
        }
        if !(V_MIN..=V_MAX).contains(&v_candidate)
            || g_candidate < 0.0
            || g_candidate > G_MAX
        {
            return -1;
        }
        if v_candidate >= self.v_threshold {
            let g_after_spike = g_candidate + self.delta_g;
            if !g_after_spike.is_finite() || g_after_spike > G_MAX {
                return -1;
            }
            self.v = self.v_reset;
            self.g_sfa = g_after_spike;
            return 1;
        }
        self.v = v_candidate;
        self.g_sfa = g_candidate;
        0
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.g_sfa = 0.0_f64;
    }

    fn validate(&self) -> Result<(), &'static str> {
        for value in [self.v, self.v_rest, self.v_reset, self.v_threshold, self.e_k] {
            if !value.is_finite() {
                return Err("finite SFA voltage parameter invalid");
            }
        }
        if !(V_MIN..=V_MAX).contains(&self.v) || !(V_MIN..=V_MAX).contains(&self.v_reset) {
            return Err("SFA voltage outside safety envelope");
        }
        if !self.g_sfa.is_finite() || self.g_sfa < 0.0 || self.g_sfa > G_MAX {
            return Err("SFA adaptation conductance outside safety envelope");
        }
        for value in [self.tau_m, self.tau_sfa, self.resistance, self.dt] {
            if !value.is_finite() || value <= 0.0 {
                return Err("positive SFA parameter invalid");
            }
        }
        if !self.delta_g.is_finite() || self.delta_g < 0.0 || self.delta_g > G_MAX {
            return Err("SFA adaptation increment outside safety envelope");
        }
        Ok(())
    }

    fn derivatives(&self, v: f64, g_sfa: f64, i_ext: f64) -> (f64, f64) {
        let dv = (-(v - self.v_rest) - g_sfa * (v - self.e_k) + self.resistance * i_ext)
            / self.tau_m;
        (dv, -g_sfa / self.tau_sfa)
    }

    fn rk4_candidate(&self, v: f64, g_sfa: f64, i_ext: f64) -> (f64, f64) {
        let (k1v, k1g) = self.derivatives(v, g_sfa, i_ext);
        let (k2v, k2g) = self.derivatives(
            v + 0.5 * self.dt * k1v,
            g_sfa + 0.5 * self.dt * k1g,
            i_ext,
        );
        let (k3v, k3g) = self.derivatives(
            v + 0.5 * self.dt * k2v,
            g_sfa + 0.5 * self.dt * k2g,
            i_ext,
        );
        let (k4v, k4g) = self.derivatives(v + self.dt * k3v, g_sfa + self.dt * k3g, i_ext);
        (
            v + (self.dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v),
            g_sfa + (self.dt / 6.0) * (k1g + 2.0 * k2g + 2.0 * k3g + k4g),
        )
    }
}

pub fn validate_sfa(state: &SFANeuron) -> bool {
    state.validate().is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sfa_new() {
        let state = SFANeuron::new();
        assert!(state.v.is_finite());
        assert!(validate_sfa(&state));
    }

    #[test]
    fn test_sfa_step() {
        let mut state = SFANeuron::new();
        let spike = state.step(10.0);
        assert!(spike == 0 || spike == 1);
    }

    #[test]
    fn test_sfa_rk4_candidate_commit() {
        let mut state = SFANeuron::new();
        state.g_sfa = 0.25;
        let (v_candidate, g_candidate) = state.rk4_candidate(state.v, state.g_sfa, 10.0);
        assert_eq!(state.step(10.0), 0);
        assert!((state.v - v_candidate).abs() < 1.0e-12);
        assert!((state.g_sfa - g_candidate).abs() < 1.0e-12);
    }

    #[test]
    fn test_sfa_invalid_state_preserved() {
        let mut state = SFANeuron::new();
        state.g_sfa = -1.0;
        let before = (state.v, state.g_sfa);
        assert_eq!(state.step(10.0), -1);
        assert_eq!((state.v, state.g_sfa), before);
    }

    #[test]
    fn test_sfa_spike_adds_adaptation_candidate() {
        let mut state = SFANeuron::new();
        let (_, g_candidate) = state.rk4_candidate(state.v, state.g_sfa, 250.0);
        assert_eq!(state.step(250.0), 1);
        assert_eq!(state.v, state.v_reset);
        assert!((state.g_sfa - (g_candidate + state.delta_g)).abs() < 1.0e-12);
    }
}
