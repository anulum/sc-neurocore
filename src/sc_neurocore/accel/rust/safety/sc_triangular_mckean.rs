// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for retained triangular SC oscillator

//! Dependency-free safety oracle for the count-neutral SC triangular recurrence.

/// Complete state and configuration for the retained project recurrence.
#[derive(Debug, Clone)]
pub struct SCTriangularMcKeanNeuron {
    pub v: f64,
    pub w: f64,
    pub a: f64,
    pub epsilon: f64,
    pub gamma: f64,
    pub dt: f64,
    pub v_peak: f64,
}

impl SCTriangularMcKeanNeuron {
    /// Construct the compatibility profile used by the project recurrence.
    pub fn new() -> Self {
        Self {
            v: 0.0,
            w: 0.0,
            a: 0.25,
            epsilon: 0.01,
            gamma: 0.5,
            dt: 0.1,
            v_peak: 0.8,
        }
    }

    /// Evaluate the retained three-branch voltage function.
    pub fn _f(&self, v: f64) -> f64 {
        let mid1 = self.a / 2.0;
        let mid2 = (1.0 + self.a) / 2.0;
        if v < mid1 {
            -v
        } else if v < mid2 {
            v - self.a
        } else {
            1.0 - v
        }
    }

    fn derivatives(&self, v: f64, w: f64, current: f64) -> Option<(f64, f64)> {
        if !(v.is_finite() && w.is_finite() && current.is_finite()) {
            return None;
        }
        let dv = self._f(v) - w + current;
        let dw = self.epsilon * (v - self.gamma * w);
        if dv.is_finite() && dw.is_finite() {
            Some((dv, dw))
        } else {
            None
        }
    }

    fn rk4_candidate(&self, current: f64) -> Option<(f64, f64)> {
        let dt = self.dt;
        let (k1v, k1w) = self.derivatives(self.v, self.w, current)?;
        let (k2v, k2w) =
            self.derivatives(self.v + 0.5 * dt * k1v, self.w + 0.5 * dt * k1w, current)?;
        let (k3v, k3w) =
            self.derivatives(self.v + 0.5 * dt * k2v, self.w + 0.5 * dt * k2w, current)?;
        let (k4v, k4w) = self.derivatives(self.v + dt * k3v, self.w + dt * k3w, current)?;
        let v = self.v + dt * (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0;
        let w = self.w + dt * (k1w + 2.0 * k2w + 2.0 * k3w + k4w) / 6.0;
        if v.is_finite() && w.is_finite() {
            Some((v, w))
        } else {
            None
        }
    }

    /// Advance atomically and report an upward `v_peak` crossing.
    pub fn step(&mut self, i_ext: f64) -> i32 {
        if !validate_sc_triangular_mckean(self) || !i_ext.is_finite() {
            return 0;
        }
        let v_prev = self.v;
        let Some((v, w)) = self.rk4_candidate(i_ext) else {
            return 0;
        };
        self.v = v;
        self.w = w;
        if self.v >= self.v_peak && v_prev < self.v_peak {
            1
        } else {
            0
        }
    }

    /// Restore the compatibility defaults.
    pub fn reset(&mut self) {
        self.v = 0.0;
        self.w = 0.0;
        self.a = 0.25;
        self.epsilon = 0.01;
        self.gamma = 0.5;
    }
}

impl Default for SCTriangularMcKeanNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Validate finite state and the enrolled parameter constraints.
pub fn validate_sc_triangular_mckean(state: &SCTriangularMcKeanNeuron) -> bool {
    state.v.is_finite()
        && state.w.is_finite()
        && state.a.is_finite()
        && state.a > 0.0
        && state.a < 1.0
        && state.epsilon.is_finite()
        && state.epsilon > 0.0
        && state.gamma.is_finite()
        && state.gamma > 0.0
        && state.dt.is_finite()
        && state.dt > 0.0
        && state.v_peak.is_finite()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rk4_reference(n: &SCTriangularMcKeanNeuron, current: f64) -> (f64, f64) {
        let rhs = |v: f64, w: f64| (n._f(v) - w + current, n.epsilon * (v - n.gamma * w));
        let dt = n.dt;
        let (k1v, k1w) = rhs(n.v, n.w);
        let (k2v, k2w) = rhs(n.v + 0.5 * dt * k1v, n.w + 0.5 * dt * k1w);
        let (k3v, k3w) = rhs(n.v + 0.5 * dt * k2v, n.w + 0.5 * dt * k2w);
        let (k4v, k4w) = rhs(n.v + dt * k3v, n.w + dt * k3w);
        (
            n.v + dt * (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0,
            n.w + dt * (k1w + 2.0 * k2w + 2.0 * k3w + k4w) / 6.0,
        )
    }

    #[test]
    fn test_sc_triangular_mckean_new() {
        let state = SCTriangularMcKeanNeuron::new();
        assert!(state.v.is_finite());
        assert!(validate_sc_triangular_mckean(&state));
    }

    #[test]
    fn test_sc_triangular_mckean_step() {
        let mut state = SCTriangularMcKeanNeuron::new();
        let spike = state.step(10.0);
        assert!(spike == 0 || spike == 1);
    }

    #[test]
    fn test_sc_triangular_mckean_matches_rk4_candidate() {
        let mut state = SCTriangularMcKeanNeuron::new();
        state.v = 0.2;
        state.w = -0.1;
        let expected = rk4_reference(&state, 0.5);
        assert_eq!(state.step(0.5), 0);
        assert!((state.v - expected.0).abs() < 1.0e-12);
        assert!((state.w - expected.1).abs() < 1.0e-12);
    }

    #[test]
    fn test_sc_triangular_mckean_invalid_current_preserves_state() {
        let mut state = SCTriangularMcKeanNeuron::new();
        let before = (state.v, state.w);
        assert_eq!(state.step(f64::NAN), 0);
        assert_eq!((state.v, state.w), before);
    }

    #[test]
    fn test_sc_triangular_mckean_overflow_candidate_preserves_state() {
        let mut state = SCTriangularMcKeanNeuron::new();
        state.v = 1.0e308;
        state.w = -1.7e308;
        let before = (state.v, state.w);
        assert_eq!(state.step(1.7e308), 0);
        assert_eq!((state.v, state.w), before);
    }

    #[test]
    fn matches_python_golden_spike_count() {
        // Parity with models/sc_triangular_mckean.py (candidate-first RK4, default parameters): the piecewise-linear
        // right-hand side is exact arithmetic, so the trajectory is bit-for-bit across languages and
        // the spike count is an exact observable (not a "spike is 0 or 1" smoke check). Over 20000
        // macro steps: silent at zero drive, a single upstroke at I=0.2 (sub-threshold approach to the
        // slow limit cycle), and a seven-spike relaxation train at I=0.5. The Go, Julia, Mojo and
        // Rust-engine backends reproduce the same trajectory bit-for-bit via test_sc_triangular_mckean_backends.py.
        for (current, want) in [(0.0_f64, 0_usize), (0.2, 1), (0.5, 7)] {
            let mut state = SCTriangularMcKeanNeuron::new();
            let spikes = (0..20000).filter(|_| state.step(current) == 1).count();
            assert_eq!(spikes, want, "I={current}");
        }
    }
}
