// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li

//! Dependency-free safety oracle for the source McKean Heaviside system.

const STATE_BOUND: f64 = 1.0e6;

/// Complete state and configuration for the McKean/Tonnelier equations.
#[derive(Debug, Clone)]
pub struct McKeanNeuron {
    pub v: f64,
    pub w: f64,
    pub a: f64,
    pub lambda: f64,
    pub mu: f64,
    pub b: f64,
    pub dt: f64,
}
impl McKeanNeuron {
    /// Construct the normalized source profile with `H(0)=1`.
    pub fn new() -> Self {
        Self {
            v: 0.0,
            w: 0.0,
            a: 0.25,
            lambda: 1.0,
            mu: 1.0,
            b: 0.01,
            dt: 0.1,
        }
    }
    fn rhs(&self, v: f64, w: f64, current: f64) -> (f64, f64) {
        let h = if v >= self.a { 1.0 } else { 0.0 };
        (-self.lambda * v + self.mu * h - w + current, self.b * v)
    }
    fn candidate(&self, current: f64) -> (f64, f64) {
        let d = self.dt;
        let k1 = self.rhs(self.v, self.w, current);
        let k2 = self.rhs(self.v + d * k1.0 / 2.0, self.w + d * k1.1 / 2.0, current);
        let k3 = self.rhs(self.v + d * k2.0 / 2.0, self.w + d * k2.1 / 2.0, current);
        let k4 = self.rhs(self.v + d * k3.0, self.w + d * k3.1, current);
        (
            self.v + d * (k1.0 + 2.0 * k2.0 + 2.0 * k3.0 + k4.0) / 6.0,
            self.w + d * (k1.1 + 2.0 * k2.1 + 2.0 * k3.1 + k4.1) / 6.0,
        )
    }
    /// Advance atomically, returning `-1` for an invalid transition.
    pub fn step(&mut self, current: f64) -> i32 {
        if !validate_mckean(self) || !current.is_finite() {
            return -1;
        }
        let previous = self.v;
        let (v, w) = self.candidate(current);
        if !(v.is_finite() && w.is_finite() && v.abs() <= STATE_BOUND && w.abs() <= STATE_BOUND) {
            return -1;
        }
        self.v = v;
        self.w = w;
        i32::from(previous < self.a && v >= self.a)
    }
    /// Restore the source equilibrium state.
    pub fn reset(&mut self) {
        self.v = 0.0;
        self.w = 0.0;
    }
}
impl Default for McKeanNeuron {
    fn default() -> Self {
        Self::new()
    }
}
/// Validate source inequalities and the enrolled finite-state envelope.
pub fn validate_mckean(s: &McKeanNeuron) -> bool {
    [s.v, s.w, s.a, s.lambda, s.mu, s.b, s.dt]
        .into_iter()
        .all(f64::is_finite)
        && s.v.abs() <= STATE_BOUND
        && s.w.abs() <= STATE_BOUND
        && s.a > 0.0
        && s.lambda > 0.0
        && s.mu > s.lambda * s.a
        && s.b > 0.0
        && s.dt > 0.0
        && s.dt <= 1.0
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_step_and_atomic_failure() {
        let mut n = McKeanNeuron::new();
        assert_eq!(n.step(3.0), 1);
        let before = (n.v, n.w);
        assert_eq!(n.step(f64::NAN), -1);
        assert_eq!((n.v, n.w), before);
    }
}
