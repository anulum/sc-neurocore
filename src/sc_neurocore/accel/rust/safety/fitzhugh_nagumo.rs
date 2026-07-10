// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for fitzhugh_nagumo

#[derive(Debug, Clone)]
pub struct FitzHughNagumoNeuron {
    pub v: f64,
    pub w: f64,
    pub a: f64,
    pub b: f64,
    pub epsilon: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl FitzHughNagumoNeuron {
    pub fn new() -> Self {
        Self {
            v: -1.0,
            w: -0.5,
            a: 0.7,
            b: 0.8,
            epsilon: 0.08,
            dt: 0.1,
            v_threshold: 1.0,
        }
    }

    pub fn step(&mut self, i_ext: f64) -> i32 {
        if !(i_ext.is_finite() && validate_fitzhugh_nagumo(self)) {
            return -1;
        }
        let v_prev = self.v;
        let Some((new_v, new_w)) = rk4_candidate(self, i_ext) else {
            return -1;
        };
        if !(new_v.is_finite() && new_w.is_finite()) {
            return -1;
        }
        self.v = new_v;
        self.w = new_w;
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = -1.0;
        self.w = -0.5;
        self.a = 0.7;
        self.b = 0.8;
        self.epsilon = 0.08;
        self.dt = 0.1;
        self.v_threshold = 1.0;
    }
}

impl Default for FitzHughNagumoNeuron {
    fn default() -> Self {
        Self::new()
    }
}

pub fn validate_fitzhugh_nagumo(state: &FitzHughNagumoNeuron) -> bool {
    state.v.is_finite()
        && state.w.is_finite()
        && state.a.is_finite()
        && state.b.is_finite()
        && state.epsilon.is_finite()
        && state.dt.is_finite()
        && state.v_threshold.is_finite()
        && state.b > 0.0
        && state.epsilon > 0.0
        && state.dt > 0.0
}

fn rhs(state: &FitzHughNagumoNeuron, v: f64, w: f64, i_ext: f64) -> Option<(f64, f64)> {
    if !(v.is_finite() && w.is_finite() && i_ext.is_finite()) {
        return None;
    }
    let dv = v - v.powi(3) / 3.0 - w + i_ext;
    let dw = state.epsilon * (v + state.a - state.b * w);
    if dv.is_finite() && dw.is_finite() {
        Some((dv, dw))
    } else {
        None
    }
}

fn rk4_candidate(state: &FitzHughNagumoNeuron, i_ext: f64) -> Option<(f64, f64)> {
    let (v0, w0, dt) = (state.v, state.w, state.dt);
    let (k1v, k1w) = rhs(state, v0, w0, i_ext)?;
    let (k2v, k2w) = rhs(state, v0 + 0.5 * dt * k1v, w0 + 0.5 * dt * k1w, i_ext)?;
    let (k3v, k3w) = rhs(state, v0 + 0.5 * dt * k2v, w0 + 0.5 * dt * k2w, i_ext)?;
    let (k4v, k4w) = rhs(state, v0 + dt * k3v, w0 + dt * k3w, i_ext)?;
    Some((
        v0 + dt * (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0,
        w0 + dt * (k1w + 2.0 * k2w + 2.0 * k3w + k4w) / 6.0,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fitzhugh_nagumo_new() {
        let state = FitzHughNagumoNeuron::new();
        assert!(state.v.is_finite());
        assert!(validate_fitzhugh_nagumo(&state));
    }

    #[test]
    fn test_fitzhugh_nagumo_rk4_step_updates_state() {
        let mut state = FitzHughNagumoNeuron::new();
        let before = state.clone();
        let spike = state.step(0.8);
        assert!(spike == 0 || spike == 1);
        assert_ne!(state.v, before.v);
        assert_ne!(state.w, before.w);
        assert!(validate_fitzhugh_nagumo(&state));
    }

    #[test]
    fn test_fitzhugh_nagumo_matches_reference_rk4_candidate() {
        let mut state = FitzHughNagumoNeuron::new();
        let (expected_v, expected_w) = rk4_candidate(&state, 0.8).unwrap();
        assert_eq!(state.step(0.8), 0);
        assert!((state.v - expected_v).abs() < 1.0e-15);
        assert!((state.w - expected_w).abs() < 1.0e-15);
    }

    #[test]
    fn test_fitzhugh_nagumo_rejects_invalid_without_mutation() {
        let mut state = FitzHughNagumoNeuron::new();
        let before = state.clone();
        assert_eq!(state.step(f64::NAN), -1);
        assert_eq!(state.v, before.v);
        assert_eq!(state.w, before.w);
    }

    #[test]
    fn test_fitzhugh_nagumo_overflow_candidate_preserves_state() {
        let mut state = FitzHughNagumoNeuron::new();
        state.v = 1.0e103;
        let before = state.clone();
        assert_eq!(state.step(0.0), -1);
        assert_eq!(state.v, before.v);
        assert_eq!(state.w, before.w);
    }

    #[test]
    fn matches_python_golden_spike_count() {
        // Parity with models/fitzhugh_nagumo.py: simulate(100, 10.0) fires once
        // and simulate(2000, 0.5) fires five times (a clean partial train on the
        // limit cycle). Looping step() is exactly the simulate() recurrence, so
        // this pins the kernel to the Python golden spike count rather than a
        // "spike is 0 or 1" smoke check.
        let mut single = FitzHughNagumoNeuron::new();
        let single_spikes = (0..100).filter(|_| single.step(10.0) == 1).count();
        assert_eq!(single_spikes, 1);

        let mut train = FitzHughNagumoNeuron::new();
        let train_spikes = (0..2000).filter(|_| train.step(0.5) == 1).count();
        assert_eq!(train_spikes, 5);
    }
}
