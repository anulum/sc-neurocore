// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for wilson_hr

#![allow(unused_variables, dead_code, non_snake_case)]

#[derive(Debug, Clone)]
pub struct WilsonHRNeuron {
    pub v: f64,
    pub r: f64,
    pub tau_r: f64,
    pub v_peak: f64,
    pub dt: f64,
}

impl WilsonHRNeuron {
    pub fn new() -> Self {
        Self {
            v: -0.7_f64,
            r: 0.1_f64,
            tau_r: 1.9_f64,
            v_peak: 0.4_f64,
            dt: 0.05_f64,
        }
    }

    pub fn step(&mut self, i_ext: f64) -> Result<i32, &'static str> {
        if !validate_wilson_hr(self) {
            return Err("invalid Wilson-HR runtime state");
        }
        if !i_ext.is_finite() {
            return Err("invalid Wilson-HR external current");
        }

        let poly = -(17.81 + 47.71 * self.v + 32.63 * self.v.powi(2)) * (self.v - 0.55);
        let syn = -26.0 * self.r * (self.v + 0.92);
        let dv = (poly + syn + i_ext) * self.dt;
        let dr = (-self.r + 1.35 * self.v + 1.03) / self.tau_r * self.dt;
        let next_v = self.v + dv;
        let next_r = self.r + dr;
        if !poly.is_finite()
            || !syn.is_finite()
            || !dv.is_finite()
            || !dr.is_finite()
            || !next_v.is_finite()
            || !next_r.is_finite()
        {
            return Err("invalid Wilson-HR candidate state");
        }

        self.v = next_v;
        self.r = next_r;
        if self.v >= self.v_peak {
            self.v = -0.7;
            return Ok(1);
        }
        Ok(0)
    }

    pub fn reset(&mut self) {
        // self.v = -0.7
        // self.r = 0.1
        self.v = -0.7_f64;
        self.r = 0.1_f64;
        self.tau_r = 1.9_f64;
        self.v_peak = 0.4_f64;
        self.dt = 0.05_f64;
    }
}

pub fn validate_wilson_hr(state: &WilsonHRNeuron) -> bool {
    state.v.is_finite()
        && state.r.is_finite()
        && state.tau_r.is_finite()
        && state.tau_r > 0.0
        && state.v_peak.is_finite()
        && state.dt.is_finite()
        && state.dt > 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wilson_hr_new() {
        let state = WilsonHRNeuron::new();
        assert!(state.v.is_finite());
        assert!(validate_wilson_hr(&state));
    }

    #[test]
    fn test_wilson_hr_step() {
        let mut state = WilsonHRNeuron::new();
        let spike = state.step(10.0).unwrap();
        assert!(spike == 0 || spike == 1);
    }

    #[test]
    fn test_wilson_hr_rejects_invalid_runtime_state() {
        let mut state = WilsonHRNeuron::new();
        state.r = f64::INFINITY;
        assert!(state.step(0.3).is_err());
    }
}
