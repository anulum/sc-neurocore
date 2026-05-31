// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for direction_selective_rgc

#[derive(Debug, Clone, PartialEq)]
pub struct DirectionSelectiveRGC {
    pub tau: f64,
    pub theta: f64,
    pub is_on_centre: f64,
    pub w_centre: f64,
    pub w_surround: f64,
    pub direction_pref: f64,
    pub dt: f64,
    pub v: f64,
    pub prev_intensity: f64,
    pub surround: f64,
}

impl DirectionSelectiveRGC {
    pub fn new() -> Self {
        Self {
            tau: 10.0,
            theta: 0.5,
            is_on_centre: 1.0,
            w_centre: 1.0,
            w_surround: 0.3,
            direction_pref: 0.0,
            dt: 1.0,
            v: 0.0,
            prev_intensity: 0.0,
            surround: 0.0,
        }
    }

    pub fn new_on() -> Self {
        Self::new()
    }

    pub fn new_off() -> Self {
        let mut state = Self::new();
        state.is_on_centre = 0.0;
        state
    }

    fn valid_runtime(&self) -> bool {
        [
            self.tau,
            self.theta,
            self.is_on_centre,
            self.w_centre,
            self.w_surround,
            self.direction_pref,
            self.dt,
            self.v,
            self.prev_intensity,
            self.surround,
        ]
        .iter()
        .all(|x| x.is_finite())
            && self.tau > 0.0
            && self.theta > 0.0
            && self.dt > 0.0
            && self.w_centre >= 0.0
            && self.w_surround >= 0.0
            && self.prev_intensity >= 0.0
            && self.surround >= 0.0
            && (self.is_on_centre == 0.0 || self.is_on_centre == 1.0)
    }

    pub fn step_rf(&mut self, intensity: f64, surround_mean: f64) -> i32 {
        if !intensity.is_finite()
            || !surround_mean.is_finite()
            || intensity < 0.0
            || surround_mean < 0.0
            || !self.valid_runtime()
        {
            return 0;
        }
        let temporal_diff = intensity - self.prev_intensity;
        let mut centre_response = self.w_centre * temporal_diff;
        if self.is_on_centre == 0.0 {
            centre_response = -centre_response;
        }
        let next_surround = 0.9 * self.surround + 0.1 * surround_mean;
        let drive = centre_response - self.w_surround * next_surround;
        let decay = (-self.dt / self.tau).exp();
        let next_v = drive + (self.v - drive) * decay;
        if !next_surround.is_finite()
            || !drive.is_finite()
            || !decay.is_finite()
            || !next_v.is_finite()
            || next_surround < 0.0
        {
            return 0;
        }
        self.prev_intensity = intensity;
        self.surround = next_surround;
        if next_v >= self.theta {
            self.v = 0.0;
            1
        } else {
            self.v = next_v;
            0
        }
    }

    pub fn step(&mut self, i_ext: f64) -> i32 {
        self.step_rf(i_ext, 0.0)
    }

    pub fn reset(&mut self) {
        self.v = 0.0;
        self.prev_intensity = 0.0;
        self.surround = 0.0;
    }
}

pub fn validate_direction_selective_rgc(state: &DirectionSelectiveRGC) -> bool {
    state.valid_runtime()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exact_voltage(v: f64, drive: f64, tau: f64, dt: f64) -> f64 {
        drive + (v - drive) * (-dt / tau).exp()
    }

    #[test]
    fn test_direction_selective_rgc_new() {
        let state = DirectionSelectiveRGC::new();
        assert!(state.v.is_finite());
        assert!(validate_direction_selective_rgc(&state));
    }

    #[test]
    fn exact_membrane_relaxation() {
        let mut state = DirectionSelectiveRGC::new();
        state.tau = 7.0;
        state.theta = 100.0;
        state.dt = 1.25;
        state.w_centre = 1.4;
        state.w_surround = 0.2;
        state.v = 0.35;
        let expected_surround = 0.9 * state.surround + 0.1 * 0.5;
        let expected_drive =
            state.w_centre * (2.0 - state.prev_intensity) - state.w_surround * expected_surround;
        let expected_v = exact_voltage(state.v, expected_drive, state.tau, state.dt);
        assert_eq!(state.step_rf(2.0, 0.5), 0);
        assert!((state.v - expected_v).abs() < 1e-12);
        assert!((state.surround - expected_surround).abs() < 1e-12);
    }

    #[test]
    fn invalid_drive_preserves_state() {
        let mut state = DirectionSelectiveRGC::new();
        let before = state.clone();
        assert_eq!(state.step_rf(f64::NAN, 0.0), 0);
        assert_eq!(state, before);
    }

    #[test]
    fn corrupt_state_preserves_state() {
        let mut state = DirectionSelectiveRGC::new();
        state.surround = f64::INFINITY;
        let before = state.clone();
        assert_eq!(state.step_rf(1.0, 0.0), 0);
        assert_eq!(state, before);
    }
}
