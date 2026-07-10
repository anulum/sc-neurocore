// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for hindmarsh_rose

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
            return 0;
        }
        let x_prev = self.x;
        let (x0, y0, z0) = (self.x, self.y, self.z);
        let dt = self.dt;
        let Some(k1) = self.derivatives(x0, y0, z0, current) else {
            return 0;
        };
        let Some(k2) = self.derivatives(
            x0 + 0.5 * dt * k1.0,
            y0 + 0.5 * dt * k1.1,
            z0 + 0.5 * dt * k1.2,
            current,
        ) else {
            return 0;
        };
        let Some(k3) = self.derivatives(
            x0 + 0.5 * dt * k2.0,
            y0 + 0.5 * dt * k2.1,
            z0 + 0.5 * dt * k2.2,
            current,
        ) else {
            return 0;
        };
        let Some(k4) = self.derivatives(x0 + dt * k3.0, y0 + dt * k3.1, z0 + dt * k3.2, current)
        else {
            return 0;
        };
        let next_x = x0 + (dt / 6.0) * (k1.0 + 2.0 * k2.0 + 2.0 * k3.0 + k4.0);
        let next_y = y0 + (dt / 6.0) * (k1.1 + 2.0 * k2.1 + 2.0 * k3.1 + k4.1);
        let next_z = z0 + (dt / 6.0) * (k1.2 + 2.0 * k2.2 + 2.0 * k3.2 + k4.2);
        if !(next_x.is_finite() && next_y.is_finite() && next_z.is_finite()) {
            return 0;
        }
        self.x = next_x;
        self.y = next_y;
        self.z = next_z;
        if self.x >= self.x_threshold && x_prev < self.x_threshold {
            1
        } else {
            0
        }
    }

    fn derivatives(&self, x: f64, y: f64, z: f64, current: f64) -> Option<(f64, f64, f64)> {
        if !(x.is_finite() && y.is_finite() && z.is_finite() && current.is_finite()) {
            return None;
        }
        let derivative = (
            y - x.powi(3) + self.b * x.powi(2) - z + current,
            1.0 - 5.0 * x.powi(2) - y,
            self.r * (self.s * (x - self.x_rest) - z),
        );
        if derivative.0.is_finite() && derivative.1.is_finite() && derivative.2.is_finite() {
            Some(derivative)
        } else {
            None
        }
    }

    pub fn reset(&mut self) {
        self.x = -1.6_f64;
        self.y = -10.0_f64;
        self.z = 2.0_f64;
    }
}

impl Default for HindmarshRoseNeuron {
    fn default() -> Self {
        Self::new()
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
        let dt = state.dt;
        let k1 = state.derivatives(x0, y0, z0, current).unwrap();
        let k2 = state
            .derivatives(
                x0 + 0.5 * dt * k1.0,
                y0 + 0.5 * dt * k1.1,
                z0 + 0.5 * dt * k1.2,
                current,
            )
            .unwrap();
        let k3 = state
            .derivatives(
                x0 + 0.5 * dt * k2.0,
                y0 + 0.5 * dt * k2.1,
                z0 + 0.5 * dt * k2.2,
                current,
            )
            .unwrap();
        let k4 = state
            .derivatives(x0 + dt * k3.0, y0 + dt * k3.1, z0 + dt * k3.2, current)
            .unwrap();
        let expected_x = x0 + (dt / 6.0) * (k1.0 + 2.0 * k2.0 + 2.0 * k3.0 + k4.0);
        let expected_y = y0 + (dt / 6.0) * (k1.1 + 2.0 * k2.1 + 2.0 * k3.1 + k4.1);
        let expected_z = z0 + (dt / 6.0) * (k1.2 + 2.0 * k2.2 + 2.0 * k3.2 + k4.2);

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
        assert_eq!(state.x, -1.6);
    }

    #[test]
    fn test_hindmarsh_rose_rejects_non_finite_candidate_without_mutation() {
        let mut state = HindmarshRoseNeuron::new();
        state.x = 1.0e103;
        let before = (state.x, state.y, state.z);
        assert_eq!(state.step(3.0), 0);
        assert_eq!((state.x, state.y, state.z), before);
    }

    #[test]
    fn matches_python_golden_spike_count() {
        // Parity with models/hindmarsh_rose.py (RK4 integrator, default parameters). The
        // Hindmarsh-Rose right-hand side is polynomial (exact arithmetic), so the trajectory is
        // bit-for-bit across languages and the spike count is an exact observable, not a
        // "spike is 0 or 1" smoke check. Over 2000 macro steps: silent at zero drive, a 26-spike
        // burst train at I=3, and 52 at I=5. The Go, Julia, Mojo and Rust-engine backends reproduce
        // the same trajectory bit-for-bit via test_hindmarsh_rose_backends.py.
        for (current, want) in [(0.0_f64, 0_usize), (3.0, 26), (5.0, 52)] {
            let mut state = HindmarshRoseNeuron::new();
            let spikes = (0..2000).filter(|_| state.step(current) == 1).count();
            assert_eq!(spikes, want, "I={current}");
        }
    }
}
