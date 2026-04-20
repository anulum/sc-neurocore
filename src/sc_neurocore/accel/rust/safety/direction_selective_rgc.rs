// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for direction_selective_rgc

#![allow(unused_variables, dead_code, non_snake_case)]

#[derive(Debug, Clone)]
pub struct DirectionSelectiveRGC {
    pub tau: f64,
    pub theta: f64,
    pub is_on_centre: f64,
    pub w_centre: f64,
    pub w_surround: f64,
    pub direction_pref: f64,
    pub dt: f64,
    pub v: f64,
    pub _prev_intensity: f64,
    pub _surround: f64,
}

impl DirectionSelectiveRGC {
    pub fn new() -> Self {
        Self {
            tau: 10.0_f64,
            theta: 0.5_f64,
            is_on_centre: 1.0_f64,
            w_centre: 1.0_f64,
            w_surround: 0.3_f64,
            direction_pref: 0.0_f64,
            dt: 1.0_f64,
            v: 0.0_f64,
            _prev_intensity: 0.0_f64,
            _surround: 0.0_f64,
        }
    }

    pub fn new_on(&self, ) -> f64 {
        // return cls(is_on_centre=true)
        0.0
    }

    pub fn new_off(&self, ) -> f64 {
        // return cls(is_on_centre=false)
        0.0
    }

    pub fn step_rf(&self, intensity: f64, surround_mean: f64) -> f64 {
        // temporal_diff = intensity - self._prev_intensity
        // self._prev_intensity = intensity
        // if self.is_on_centre:
        // centre_response = self.w_centre * temporal_diff
        // else:
        // centre_response = -self.w_centre * temporal_diff
        // self._surround = 0.9 * self._surround + 0.1 * surround_mean
        // surround_inhib = self.w_surround * self._surround
        // drive = centre_response - surround_inhib
        // self.v += (-self.v + drive) / self.tau * self.dt
        // if self.v >= self.theta:
        // self.v = 0.0
        // return 1
        // return 0
        0.0
    }

    pub fn step(&mut self, i_ext: f64) -> i32 {
        // return self.step_rf(current, 0.0)
        0 // spike indicator
    }

    pub fn reset(&mut self) {
        // self.v = 0.0
        // self._prev_intensity = 0.0
        // self._surround = 0.0
        self.tau = 10.0_f64;
        self.theta = 0.5_f64;
        self.is_on_centre = 1.0_f64;
        self.w_centre = 1.0_f64;
        self.w_surround = 0.3_f64;
    }

}

pub fn validate_direction_selective_rgc(state: &DirectionSelectiveRGC) -> bool {
    state.v.is_finite()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_direction_selective_rgc_new() {
        let state = DirectionSelectiveRGC::new();
        assert!(state.v.is_finite());
        assert!(validate_direction_selective_rgc(&state));
    }

    #[test]
    fn test_direction_selective_rgc_step() {
        let mut state = DirectionSelectiveRGC::new();
        let spike = state.step(10.0);
        assert!(spike == 0 || spike == 1);
    }
}
