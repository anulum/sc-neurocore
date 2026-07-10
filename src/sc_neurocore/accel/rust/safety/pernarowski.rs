// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for Pernarowski

#[derive(Debug, Clone)]
pub struct PernarowskiNeuron {
    pub v: f64,
    pub w: f64,
    pub z: f64,
    pub alpha: f64,
    pub beta: f64,
    pub eps1: f64,
    pub eps2: f64,
    pub gamma: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl PernarowskiNeuron {
    pub fn new() -> Self {
        Self {
            v: -1.0_f64,
            w: 0.0_f64,
            z: 0.0_f64,
            alpha: 0.1_f64,
            beta: 0.5_f64,
            eps1: 0.1_f64,
            eps2: 0.001_f64,
            gamma: 0.5_f64,
            dt: 0.1_f64,
            v_threshold: 0.5_f64,
        }
    }

    fn derivatives(&self, v: f64, w: f64, z: f64, current: f64) -> Option<(f64, f64, f64)> {
        if !(v.is_finite() && w.is_finite() && z.is_finite() && current.is_finite()) {
            return None;
        }
        let dv = v - v.powi(3) / 3.0 - w - z + current;
        let dw = self.eps1 * (v - self.gamma * w + self.alpha);
        let dz = self.eps2 * (self.beta * (v + 0.7) - z);
        if dv.is_finite() && dw.is_finite() && dz.is_finite() {
            Some((dv, dw, dz))
        } else {
            None
        }
    }

    fn rk4_candidate(&self, current: f64) -> Option<(f64, f64, f64)> {
        let dt = self.dt;
        let (k1v, k1w, k1z) = self.derivatives(self.v, self.w, self.z, current)?;
        let (k2v, k2w, k2z) = self.derivatives(
            self.v + 0.5 * dt * k1v,
            self.w + 0.5 * dt * k1w,
            self.z + 0.5 * dt * k1z,
            current,
        )?;
        let (k3v, k3w, k3z) = self.derivatives(
            self.v + 0.5 * dt * k2v,
            self.w + 0.5 * dt * k2w,
            self.z + 0.5 * dt * k2z,
            current,
        )?;
        let (k4v, k4w, k4z) = self.derivatives(
            self.v + dt * k3v,
            self.w + dt * k3w,
            self.z + dt * k3z,
            current,
        )?;
        let v = self.v + dt * (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0;
        let w = self.w + dt * (k1w + 2.0 * k2w + 2.0 * k3w + k4w) / 6.0;
        let z = self.z + dt * (k1z + 2.0 * k2z + 2.0 * k3z + k4z) / 6.0;
        if v.is_finite() && w.is_finite() && z.is_finite() {
            Some((v, w, z))
        } else {
            None
        }
    }

    pub fn step(&mut self, i_ext: f64) -> i32 {
        if !validate_pernarowski(self) || !i_ext.is_finite() {
            return 0;
        }

        let v_prev = self.v;
        let Some((v, w, z)) = self.rk4_candidate(i_ext) else {
            return 0;
        };
        self.v = v;
        self.w = w;
        self.z = z;
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        // Mirror models/pernarowski.py `reset`: restore only the state variables,
        // never the parameters (alpha/beta et al. are configuration, not state).
        self.v = -1.0_f64;
        self.w = 0.0_f64;
        self.z = 0.0_f64;
    }
}

impl Default for PernarowskiNeuron {
    fn default() -> Self {
        Self::new()
    }
}

pub fn validate_pernarowski(state: &PernarowskiNeuron) -> bool {
    state.v.is_finite()
        && state.w.is_finite()
        && state.z.is_finite()
        && state.alpha.is_finite()
        && state.beta.is_finite()
        && state.eps1.is_finite()
        && state.eps1 > 0.0
        && state.eps2.is_finite()
        && state.eps2 > 0.0
        && state.gamma.is_finite()
        && state.gamma > 0.0
        && state.dt.is_finite()
        && state.dt > 0.0
        && state.v_threshold.is_finite()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rk4_reference(n: &PernarowskiNeuron, current: f64) -> (f64, f64, f64) {
        let rhs = |v: f64, w: f64, z: f64| {
            (
                v - v.powi(3) / 3.0 - w - z + current,
                n.eps1 * (v - n.gamma * w + n.alpha),
                n.eps2 * (n.beta * (v + 0.7) - z),
            )
        };
        let dt = n.dt;
        let (k1v, k1w, k1z) = rhs(n.v, n.w, n.z);
        let (k2v, k2w, k2z) = rhs(
            n.v + 0.5 * dt * k1v,
            n.w + 0.5 * dt * k1w,
            n.z + 0.5 * dt * k1z,
        );
        let (k3v, k3w, k3z) = rhs(
            n.v + 0.5 * dt * k2v,
            n.w + 0.5 * dt * k2w,
            n.z + 0.5 * dt * k2z,
        );
        let (k4v, k4w, k4z) = rhs(n.v + dt * k3v, n.w + dt * k3w, n.z + dt * k3z);
        (
            n.v + dt * (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0,
            n.w + dt * (k1w + 2.0 * k2w + 2.0 * k3w + k4w) / 6.0,
            n.z + dt * (k1z + 2.0 * k2z + 2.0 * k3z + k4z) / 6.0,
        )
    }

    #[test]
    fn test_pernarowski_new() {
        let state = PernarowskiNeuron::new();
        assert!(state.v.is_finite());
        assert!(validate_pernarowski(&state));
    }

    #[test]
    fn test_pernarowski_step() {
        let mut state = PernarowskiNeuron::new();
        let spike = state.step(10.0);
        assert!(spike == 0 || spike == 1);
    }

    #[test]
    fn test_pernarowski_matches_rk4_candidate() {
        let mut state = PernarowskiNeuron::new();
        state.v = -0.8;
        state.w = 0.2;
        state.z = -0.1;
        let expected = rk4_reference(&state, 0.5);
        assert_eq!(state.step(0.5), 0);
        assert!((state.v - expected.0).abs() < 1e-14);
        assert!((state.w - expected.1).abs() < 1e-14);
        assert!((state.z - expected.2).abs() < 1e-14);
    }

    #[test]
    fn test_invalid_input_preserves_state() {
        let mut state = PernarowskiNeuron::new();
        let before = (state.v, state.w, state.z);
        assert_eq!(state.step(f64::NAN), 0);
        assert_eq!((state.v, state.w, state.z), before);
    }

    #[test]
    fn test_overflow_candidate_preserves_state() {
        let mut state = PernarowskiNeuron::new();
        state.v = 1.0e160;
        let before = (state.v, state.w, state.z);
        assert_eq!(state.step(0.5), 0);
        assert_eq!((state.v, state.w, state.z), before);
    }

    #[test]
    fn matches_python_golden_spike_count() {
        // Parity with models/pernarowski.py (RK4 integrator, default parameters). The
        // Pernarowski 1994 beta-cell burster is an *autonomous* slow-wave oscillator: the
        // intrinsic burst rhythm is driven by the slow z-adaptation, so external current does
        // not gate spiking (silent/single/train sweeps do not apply). The right-hand side is a
        // polynomial (a cubic; `v.powi(3)` matches the Python `v*v*v` bit-for-bit), so the
        // trajectory is exact across languages and the burst-spike count is an exact observable.
        // At the default resting start with zero drive the intrinsic rhythm produces 7 spikes
        // over 2000 macro steps, 17 over 5000, and 27 over 8000. The Go, Julia, Mojo and
        // Rust-engine backends reproduce the same trajectory bit-for-bit (max|Δ|=0) via
        // test_pernarowski_backends.py.
        for (n_steps, want) in [(2000_usize, 7_usize), (5000, 17), (8000, 27)] {
            let mut state = PernarowskiNeuron::new();
            let spikes = (0..n_steps).filter(|_| state.step(0.0) == 1).count();
            assert_eq!(spikes, want, "n_steps={n_steps}");
        }
    }
}
