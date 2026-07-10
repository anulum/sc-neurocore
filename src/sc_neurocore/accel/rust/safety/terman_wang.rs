// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for Terman-Wang

#[derive(Debug, Clone)]
pub struct TermanWangOscillator {
    pub v: f64,
    pub w: f64,
    pub alpha: f64,
    pub beta: f64,
    pub epsilon: f64,
    pub rho: f64,
    pub dt: f64,
    pub v_peak: f64,
}

impl TermanWangOscillator {
    pub fn new() -> Self {
        Self {
            v: -1.5,
            w: -0.5,
            alpha: 3.0,
            beta: 0.2,
            epsilon: 0.02,
            rho: 0.0,
            dt: 0.05,
            v_peak: 1.5,
        }
    }

    fn derivatives(&self, v: f64, w: f64, current: f64) -> Option<(f64, f64)> {
        if !(v.is_finite() && w.is_finite() && current.is_finite()) {
            return None;
        }
        let f = 3.0 * v - v.powi(3) + 2.0;
        let g = self.alpha * (1.0 + (v / self.beta).tanh());
        let dv = f - w + current + self.rho;
        let dw = self.epsilon * (g - w);
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

    pub fn step(&mut self, i_ext: f64) -> Result<i32, &'static str> {
        if !validate_terman_wang(self) {
            return Err("invalid Terman-Wang runtime state");
        }
        if !i_ext.is_finite() {
            return Err("invalid Terman-Wang external current");
        }
        let v_prev = self.v;
        let Some((v, w)) = self.rk4_candidate(i_ext) else {
            return Err("non-finite Terman-Wang RK4 candidate");
        };
        self.v = v;
        self.w = w;
        Ok(if self.v >= self.v_peak && v_prev < self.v_peak {
            1
        } else {
            0
        })
    }

    pub fn reset(&mut self) {
        // Mirror models/terman_wang.py `reset`: restore only the state variables,
        // never the parameters (alpha/beta/epsilon are configuration, not state).
        self.v = -1.5;
        self.w = -0.5;
    }
}

impl Default for TermanWangOscillator {
    fn default() -> Self {
        Self::new()
    }
}

pub fn validate_terman_wang(state: &TermanWangOscillator) -> bool {
    state.v.is_finite()
        && state.w.is_finite()
        && state.alpha.is_finite()
        && state.beta.is_finite()
        && state.beta > 0.0
        && state.epsilon.is_finite()
        && state.epsilon > 0.0
        && state.rho.is_finite()
        && state.dt.is_finite()
        && state.dt > 0.0
        && state.v_peak.is_finite()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rk4_reference(n: &TermanWangOscillator, current: f64) -> (f64, f64) {
        let rhs = |v: f64, w: f64| {
            let f = 3.0 * v - v.powi(3) + 2.0;
            let g = n.alpha * (1.0 + (v / n.beta).tanh());
            (f - w + current + n.rho, n.epsilon * (g - w))
        };
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
    fn test_terman_wang_new() {
        let state = TermanWangOscillator::new();
        assert!(state.v.is_finite());
        assert!(validate_terman_wang(&state));
    }

    #[test]
    fn test_terman_wang_step() {
        let mut state = TermanWangOscillator::new();
        let spike = state.step(10.0).unwrap();
        assert!(spike == 0 || spike == 1);
    }

    #[test]
    fn test_terman_wang_matches_rk4_candidate() {
        let mut state = TermanWangOscillator::new();
        state.v = -1.2;
        state.w = -0.25;
        let expected = rk4_reference(&state, 1.0);
        assert_eq!(state.step(1.0).unwrap(), 0);
        assert!((state.v - expected.0).abs() < 1e-14);
        assert!((state.w - expected.1).abs() < 1e-14);
    }

    #[test]
    fn test_terman_wang_rejects_invalid_runtime_state() {
        let mut state = TermanWangOscillator::new();
        state.v = f64::INFINITY;
        let before = state.clone();
        assert!(state.step(1.0).is_err());
        assert_eq!(state.v, before.v);
        assert_eq!(state.w, before.w);
    }

    #[test]
    fn test_terman_wang_rejects_invalid_current_without_mutation() {
        let mut state = TermanWangOscillator::new();
        let before = state.clone();
        assert!(state.step(f64::NAN).is_err());
        assert_eq!(state.v, before.v);
        assert_eq!(state.w, before.w);
    }

    #[test]
    fn test_terman_wang_rejects_overflow_candidate_without_mutation() {
        let mut state = TermanWangOscillator::new();
        state.v = 1.0e308;
        let before = state.clone();
        assert!(state.step(1.0).is_err());
        assert_eq!(state.v, before.v);
        assert_eq!(state.w, before.w);
    }

    #[test]
    fn matches_python_golden_spike_count() {
        // Parity with models/terman_wang.py (RK4 integrator, default parameters). The Terman-Wang
        // 1995 relaxation oscillator has a cubic fast nullcline plus a `tanh` sigmoid recovery
        // gate; `tanh` is the one non-exact operation, so the trajectory is not bit-for-bit across
        // libms (on Linux the Rust engine shares Python's glibc `tanh` and IS bit-identical, but
        // the portable, declared observable is the spike count — a 2-D autonomous flow cannot be
        // chaotic, so a bounded per-step `tanh` gap cannot change the crossing count). The drive
        // gates the regime cleanly: silent at I=-1.0 (hyperpolarised, no oscillation), a single
        // upstroke at I=0.0, and a three-spike relaxation train at I=0.5, each over 8000 macro
        // steps. Verified python-vs-rust spike counts match with max|Δ|=0 on this host; the Go,
        // Julia and Mojo backends reproduce the same counts (ULP-bounded) via
        // test_terman_wang_backends.py.
        for (current, want) in [(-1.0_f64, 0_usize), (0.0, 1), (0.5, 3)] {
            let mut state = TermanWangOscillator::new();
            let spikes = (0..8000)
                .filter(|_| state.step(current).expect("finite step") == 1)
                .count();
            assert_eq!(spikes, want, "I={current}");
        }
    }
}
