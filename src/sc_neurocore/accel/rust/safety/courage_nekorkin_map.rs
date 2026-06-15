// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for courage_nekorkin_map

//! Standalone fail-closed mirror of the Courbage-Nekorkin-Vdovin (2007)
//! discontinuous two-dimensional spiking map (Chaos 17:043109; arXiv:0712.2097,
//! eqs. 3-5). `step` rejects non-finite input current or corrupted state and
//! preserves the previous state instead of mutating it, so a poisoned trajectory
//! never propagates. The arithmetic mirrors the engine struct operation for
//! operation, so the trace is bit-identical to the Python NumPy reference.

#![allow(dead_code)]

#[derive(Debug, Clone)]
pub struct CourageNekorkinMapNeuron {
    pub x: f64,
    pub y: f64,
    pub m0: f64,
    pub m1: f64,
    pub a: f64,
    pub d: f64,
    pub j: f64,
    pub beta: f64,
    pub eps: f64,
    pub x_threshold: f64,
}

impl CourageNekorkinMapNeuron {
    pub fn new() -> Self {
        Self {
            x: 0.0_f64,
            y: 0.0_f64,
            m0: 0.0864_f64,
            m1: 0.65_f64,
            a: 0.2_f64,
            d: 0.235_f64,
            j: 0.2_f64,
            beta: 0.085_f64,
            eps: 0.02_f64,
            x_threshold: 0.235_f64,
        }
    }

    /// Continuity breakpoints `(Jmin, Jmax)` of the piecewise-linear `F` (eq. 4).
    fn breakpoints(&self) -> (f64, f64) {
        let am1 = self.a * self.m1;
        let den = self.m0 + self.m1;
        (am1 / den, (self.m0 + am1) / den)
    }

    /// Piecewise-linear `F(x)` (Courbage et al. 2007, eq. 4).
    pub fn f(&self, x: f64) -> f64 {
        let (jmin, jmax) = self.breakpoints();
        if x <= jmin {
            -self.m0 * x
        } else if x < jmax {
            self.m1 * (x - self.a)
        } else {
            -self.m0 * (x - 1.0)
        }
    }

    /// Advance one step. Returns `1` on an upward `x_threshold` crossing, `0`
    /// otherwise, and `-1` (no mutation) on a fail-closed rejection.
    pub fn step(&mut self, current: f64) -> i32 {
        if !current.is_finite()
            || !self.x.is_finite()
            || !self.y.is_finite()
            || !(self.m0 + self.m1).is_finite()
            || (self.m0 + self.m1) == 0.0
        {
            return -1;
        }
        let x_prev = self.x;
        let am1 = self.a * self.m1;
        let den = self.m0 + self.m1;
        let jmin = am1 / den;
        let jmax = (self.m0 + am1) / den;
        let fx = if self.x <= jmin {
            -self.m0 * self.x
        } else if self.x < jmax {
            self.m1 * (self.x - self.a)
        } else {
            -self.m0 * (self.x - 1.0)
        };
        let h = if (self.x - self.d) >= 0.0 { 1.0 } else { 0.0 };
        let x_new = self.x + fx - self.y - self.beta * h + current;
        let y_new = self.y + self.eps * (self.x - self.j);
        if !x_new.is_finite() || !y_new.is_finite() {
            return -1;
        }
        self.x = x_new;
        self.y = y_new;
        if self.x >= self.x_threshold && x_prev < self.x_threshold {
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.x = 0.0_f64;
        self.y = 0.0_f64;
    }
}

impl Default for CourageNekorkinMapNeuron {
    fn default() -> Self {
        Self::new()
    }
}

pub fn validate_courage_nekorkin_map(state: &CourageNekorkinMapNeuron) -> bool {
    state.x.is_finite()
        && state.y.is_finite()
        && state.m0 > 0.0
        && state.m1 > 0.0
        && (state.m0 + state.m1) > 0.0
        && state.beta >= 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_courage_nekorkin_map_new() {
        let state = CourageNekorkinMapNeuron::new();
        assert!(validate_courage_nekorkin_map(&state));
    }

    #[test]
    fn test_breakpoints_default() {
        let s = CourageNekorkinMapNeuron::new();
        let (jmin, jmax) = s.breakpoints();
        assert!((jmin - 0.2 * 0.65 / (0.0864 + 0.65)).abs() < 1e-15);
        assert!((jmax - (0.0864 + 0.2 * 0.65) / (0.0864 + 0.65)).abs() < 1e-15);
        // Default discontinuity must sit strictly inside (Jmin, Jmax) — eq. 6.
        assert!(jmin < s.d && s.d < jmax);
    }

    #[test]
    fn test_step_returns_binary_in_regime() {
        let mut state = CourageNekorkinMapNeuron::new();
        let spike = state.step(0.0);
        assert!(spike == 0 || spike == 1);
    }

    #[test]
    fn test_step_fails_closed_on_non_finite_current() {
        let mut state = CourageNekorkinMapNeuron::new();
        let (x0, y0) = (state.x, state.y);
        assert_eq!(state.step(f64::NAN), -1);
        assert_eq!(state.x, x0);
        assert_eq!(state.y, y0);
    }

    #[test]
    fn test_sustained_bounded_spiking() {
        // Default regime is the published chaotic spiking-bursting mode: many
        // spikes, no clip-pegging, bounded trajectory.
        let mut state = CourageNekorkinMapNeuron::new();
        let mut spikes = 0;
        let mut max_abs = 0.0_f64;
        for _ in 0..20_000 {
            spikes += state.step(0.0).max(0);
            max_abs = max_abs.max(state.x.abs());
        }
        assert!(spikes > 1000, "expected sustained spiking, got {spikes}");
        assert!(
            max_abs < 10.0,
            "trajectory must stay bounded, got {max_abs}"
        );
    }
}
