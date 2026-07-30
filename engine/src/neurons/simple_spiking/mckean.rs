// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li

//! Source-bound space-clamped McKean Heaviside system.

const STATE_BOUND: f64 = 1.0e6;

/// Complete state and configuration for the McKean/Tonnelier source equations.
#[derive(Clone, Debug)]
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
        let dt = self.dt;
        let k1 = self.rhs(self.v, self.w, current);
        let k2 = self.rhs(self.v + dt * k1.0 / 2.0, self.w + dt * k1.1 / 2.0, current);
        let k3 = self.rhs(self.v + dt * k2.0 / 2.0, self.w + dt * k2.1 / 2.0, current);
        let k4 = self.rhs(self.v + dt * k3.0, self.w + dt * k3.1, current);
        (
            self.v + dt * (k1.0 + 2.0 * k2.0 + 2.0 * k3.0 + k4.0) / 6.0,
            self.w + dt * (k1.1 + 2.0 * k2.1 + 2.0 * k3.1 + k4.1) / 6.0,
        )
    }
    /// Validate the source state, parameter inequalities, and RK4 envelope.
    pub fn valid(&self) -> bool {
        [
            self.v,
            self.w,
            self.a,
            self.lambda,
            self.mu,
            self.b,
            self.dt,
        ]
        .into_iter()
        .all(f64::is_finite)
            && self.v.abs() <= STATE_BOUND
            && self.w.abs() <= STATE_BOUND
            && self.a > 0.0
            && self.lambda > 0.0
            && self.mu > self.lambda * self.a
            && self.b > 0.0
            && self.dt > 0.0
            && self.dt <= 1.0
    }
    /// Advance atomically and report invalid transitions as an error.
    pub fn try_step(&mut self, current: f64) -> Result<i32, &'static str> {
        if !self.valid() || !current.is_finite() {
            return Err("invalid McKean state, configuration, or current");
        }
        let previous = self.v;
        let (v, w) = self.candidate(current);
        if !(v.is_finite() && w.is_finite() && v.abs() <= STATE_BOUND && w.abs() <= STATE_BOUND) {
            return Err("McKean RK4 candidate outside safety envelope");
        }
        let event = i32::from(previous < self.a && v >= self.a);
        self.v = v;
        self.w = w;
        Ok(event)
    }
    pub fn step(&mut self, current: f64) -> i32 {
        self.try_step(current).unwrap_or(-1)
    }
    /// Restore the source equilibrium state without changing configuration.
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_transition_is_atomic_and_uses_switching_event() {
        let mut n = McKeanNeuron::new();
        assert_eq!(n.step(3.0), 1);
        let before = (n.v, n.w);
        assert_eq!(n.step(f64::NAN), -1);
        assert_eq!((n.v, n.w), before);
    }
}
