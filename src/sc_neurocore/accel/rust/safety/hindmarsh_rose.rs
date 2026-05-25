// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for hindmarsh_rose

#![allow(unused_variables, dead_code, non_snake_case)]

#[derive(Debug, Clone)]
pub struct HindmarshRoseNeuron {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub b: f64,
    pub r: f64,
    pub s: f64,
    pub x_rest: f64,
    pub dt: f64,
    pub x_threshold: f64,
}

impl HindmarshRoseNeuron {
    pub fn new() -> Self {
        Self {
            x: -1.6_f64,
            y: -10.0_f64,
            z: 2.0_f64,
            b: 3.0_f64,
            r: 0.001_f64,
            s: 4.0_f64,
            x_rest: -1.6_f64,
            dt: 0.1_f64,
            x_threshold: 1.0_f64,
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        if !validate_hindmarsh_rose(self) || !current.is_finite() {
            self.x = f64::NAN;
            self.y = f64::NAN;
            self.z = f64::NAN;
            return 0;
        }
        let x_prev = self.x;
        let dx = self.y - self.x.powi(3) + self.b * self.x.powi(2) - self.z + current;
        let dy = 1.0 - 5.0 * self.x.powi(2) - self.y;
        let dz = self.r * (self.s * (self.x - self.x_rest) - self.z);
        self.x += dx * self.dt;
        self.y += dy * self.dt;
        self.z += dz * self.dt;
        if !validate_hindmarsh_rose(self) {
            self.x = f64::NAN;
            self.y = f64::NAN;
            self.z = f64::NAN;
            return 0;
        }
        if self.x >= self.x_threshold && x_prev < self.x_threshold {
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        // self.x = -1.6
        // self.y = -10.0
        // self.z = 2.0
        self.x = -1.6_f64;
        self.y = -10.0_f64;
        self.z = 2.0_f64;
        self.b = 3.0_f64;
        self.r = 0.001_f64;
    }
}

pub fn validate_hindmarsh_rose(state: &HindmarshRoseNeuron) -> bool {
    state.x.is_finite()
        && state.y.is_finite()
        && state.z.is_finite()
        && state.b.is_finite()
        && state.r.is_finite()
        && state.s.is_finite()
        && state.x_rest.is_finite()
        && state.dt.is_finite()
        && state.x_threshold.is_finite()
        && state.r > 0.0
        && state.s > 0.0
        && state.dt > 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hindmarsh_rose_new() {
        let state = HindmarshRoseNeuron::new();
        assert!(validate_hindmarsh_rose(&state));
    }

    #[test]
    fn test_hindmarsh_rose_step() {
        let mut state = HindmarshRoseNeuron::new();
        let x0 = state.x;
        let y0 = state.y;
        let z0 = state.z;
        let current = 3.0;
        let expected_x = x0 + (y0 - x0.powi(3) + state.b * x0.powi(2) - z0 + current) * state.dt;
        let expected_y = y0 + (1.0 - 5.0 * x0.powi(2) - y0) * state.dt;
        let expected_z = z0 + state.r * (state.s * (x0 - state.x_rest) - z0) * state.dt;

        let spike = state.step(current);

        assert!(spike == 0 || spike == 1);
        assert!((state.x - expected_x).abs() < 1e-12);
        assert!((state.y - expected_y).abs() < 1e-12);
        assert!((state.z - expected_z).abs() < 1e-12);
    }

    #[test]
    fn test_hindmarsh_rose_rejects_invalid_state() {
        let mut state = HindmarshRoseNeuron::new();
        state.dt = 0.0;
        assert_eq!(state.step(3.0), 0);
        assert!(state.x.is_nan());
    }
}
