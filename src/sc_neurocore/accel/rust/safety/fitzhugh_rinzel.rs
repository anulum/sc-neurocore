// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for fitzhugh_rinzel

#![allow(dead_code)]

#[derive(Debug, Clone)]
pub struct FitzHughRinzelNeuron {
    pub v: f64,
    pub w: f64,
    pub y: f64,
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
    pub delta: f64,
    pub mu: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl FitzHughRinzelNeuron {
    pub fn new() -> Self {
        Self {
            v: -1.0,
            w: -0.5,
            y: 0.0,
            a: 0.7,
            b: 0.8,
            c: -0.775,
            d: 1.0,
            delta: 0.08,
            mu: 0.0001,
            dt: 0.1,
            v_threshold: 1.0,
        }
    }

    pub fn step(&mut self, i_ext: f64) -> i32 {
        if !validate_fitzhugh_rinzel(self) || !i_ext.is_finite() {
            return 0;
        }
        let v_prev = self.v;
        let Some((next_v, next_w, next_y)) = rk4_candidate(self, i_ext) else {
            return 0;
        };
        if !(next_v.is_finite() && next_w.is_finite() && next_y.is_finite()) {
            return 0;
        }
        self.v = next_v;
        self.w = next_w;
        self.y = next_y;
        i32::from(v_prev < self.v_threshold && self.v >= self.v_threshold)
    }

    pub fn reset(&mut self) {
        self.v = -1.0;
        self.w = -0.5;
        self.y = 0.0;
        self.a = 0.7;
        self.b = 0.8;
        self.c = -0.775;
        self.d = 1.0;
        self.delta = 0.08;
        self.mu = 0.0001;
        self.dt = 0.1;
        self.v_threshold = 1.0;
    }
}

pub fn validate_fitzhugh_rinzel(state: &FitzHughRinzelNeuron) -> bool {
    state.v.is_finite()
        && state.w.is_finite()
        && state.y.is_finite()
        && state.a.is_finite()
        && state.b.is_finite()
        && state.c.is_finite()
        && state.d.is_finite()
        && state.delta.is_finite()
        && state.mu.is_finite()
        && state.dt.is_finite()
        && state.v_threshold.is_finite()
        && state.b > 0.0
        && state.d > 0.0
        && state.delta > 0.0
        && state.mu > 0.0
        && state.dt > 0.0
}

fn derivatives(
    state: &FitzHughRinzelNeuron,
    v: f64,
    w: f64,
    y: f64,
    i_ext: f64,
) -> Option<(f64, f64, f64)> {
    if !(v.is_finite() && w.is_finite() && y.is_finite() && i_ext.is_finite()) {
        return None;
    }
    let dv = v - v.powi(3) / 3.0 - w + y + i_ext;
    let dw = state.delta * (state.a + v - state.b * w);
    let dy = state.mu * (state.c - v - state.d * y);
    if dv.is_finite() && dw.is_finite() && dy.is_finite() {
        Some((dv, dw, dy))
    } else {
        None
    }
}

fn rk4_candidate(state: &FitzHughRinzelNeuron, i_ext: f64) -> Option<(f64, f64, f64)> {
    let (v0, w0, y0, dt) = (state.v, state.w, state.y, state.dt);
    let (k1v, k1w, k1y) = derivatives(state, v0, w0, y0, i_ext)?;
    let (k2v, k2w, k2y) = derivatives(
        state,
        v0 + 0.5 * dt * k1v,
        w0 + 0.5 * dt * k1w,
        y0 + 0.5 * dt * k1y,
        i_ext,
    )?;
    let (k3v, k3w, k3y) = derivatives(
        state,
        v0 + 0.5 * dt * k2v,
        w0 + 0.5 * dt * k2w,
        y0 + 0.5 * dt * k2y,
        i_ext,
    )?;
    let (k4v, k4w, k4y) = derivatives(state, v0 + dt * k3v, w0 + dt * k3w, y0 + dt * k3y, i_ext)?;
    Some((
        v0 + dt * (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0,
        w0 + dt * (k1w + 2.0 * k2w + 2.0 * k3w + k4w) / 6.0,
        y0 + dt * (k1y + 2.0 * k2y + 2.0 * k3y + k4y) / 6.0,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fitzhugh_rinzel_new() {
        let state = FitzHughRinzelNeuron::new();
        assert!(state.v.is_finite());
        assert!(validate_fitzhugh_rinzel(&state));
    }

    #[test]
    fn test_fitzhugh_rinzel_step() {
        let mut state = FitzHughRinzelNeuron::new();
        let spike = state.step(10.0);
        assert!(spike == 0 || spike == 1);
    }

    #[test]
    fn test_fitzhugh_rinzel_matches_rk4_candidate() {
        let mut state = FitzHughRinzelNeuron::new();
        state.v = -1.0;
        state.w = 0.2;
        state.y = 0.1;
        let expected = rk4_candidate(&state, 0.5).unwrap();
        let spike = state.step(0.5);
        assert_eq!(spike, 0);
        assert!((state.v - expected.0).abs() < 1.0e-15);
        assert!((state.w - expected.1).abs() < 1.0e-15);
        assert!((state.y - expected.2).abs() < 1.0e-15);
    }

    #[test]
    fn test_fitzhugh_rinzel_invalid_current_preserves_state() {
        let mut state = FitzHughRinzelNeuron::new();
        let before = (state.v, state.w, state.y);
        assert_eq!(state.step(f64::NAN), 0);
        assert_eq!((state.v, state.w, state.y), before);
    }

    #[test]
    fn test_fitzhugh_rinzel_overflow_candidate_preserves_state() {
        let mut state = FitzHughRinzelNeuron::new();
        state.v = 1.0e155;
        let before = (state.v, state.w, state.y);
        assert_eq!(state.step(0.5), 0);
        assert_eq!((state.v, state.w, state.y), before);
    }
}
