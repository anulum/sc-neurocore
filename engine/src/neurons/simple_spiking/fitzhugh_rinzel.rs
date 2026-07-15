// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — FitzHugh-Rinzel Neuron Model

//! FitzHugh-Rinzel bursting neuron dynamics.

/// FitzHugh-Rinzel — 3D extension with slow bursting variable.
#[derive(Clone, Debug)]
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

    fn valid_numeric_contract(&self) -> bool {
        self.v.is_finite()
            && self.w.is_finite()
            && self.y.is_finite()
            && self.a.is_finite()
            && self.b.is_finite()
            && self.c.is_finite()
            && self.d.is_finite()
            && self.delta.is_finite()
            && self.mu.is_finite()
            && self.dt.is_finite()
            && self.v_threshold.is_finite()
            && self.b > 0.0
            && self.d > 0.0
            && self.delta > 0.0
            && self.mu > 0.0
            && self.dt > 0.0
    }

    fn derivatives(&self, v: f64, w: f64, y: f64, current: f64) -> Option<(f64, f64, f64)> {
        if !(v.is_finite() && w.is_finite() && y.is_finite() && current.is_finite()) {
            return None;
        }
        let dv = v - v.powi(3) / 3.0 - w + y + current;
        let dw = self.delta * (self.a + v - self.b * w);
        let dy = self.mu * (self.c - v - self.d * y);
        if dv.is_finite() && dw.is_finite() && dy.is_finite() {
            Some((dv, dw, dy))
        } else {
            None
        }
    }

    fn rk4_candidate(&self, current: f64) -> Option<(f64, f64, f64)> {
        let (v0, w0, y0, dt) = (self.v, self.w, self.y, self.dt);
        let (k1v, k1w, k1y) = self.derivatives(v0, w0, y0, current)?;
        let (k2v, k2w, k2y) = self.derivatives(
            v0 + 0.5 * dt * k1v,
            w0 + 0.5 * dt * k1w,
            y0 + 0.5 * dt * k1y,
            current,
        )?;
        let (k3v, k3w, k3y) = self.derivatives(
            v0 + 0.5 * dt * k2v,
            w0 + 0.5 * dt * k2w,
            y0 + 0.5 * dt * k2y,
            current,
        )?;
        let (k4v, k4w, k4y) =
            self.derivatives(v0 + dt * k3v, w0 + dt * k3w, y0 + dt * k3y, current)?;
        Some((
            v0 + dt * (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0,
            w0 + dt * (k1w + 2.0 * k2w + 2.0 * k3w + k4w) / 6.0,
            y0 + dt * (k1y + 2.0 * k2y + 2.0 * k3y + k4y) / 6.0,
        ))
    }

    pub fn step(&mut self, current: f64) -> i32 {
        if !self.valid_numeric_contract() || !current.is_finite() {
            return 0;
        }
        let v_prev = self.v;
        let Some((next_v, next_w, next_y)) = self.rk4_candidate(current) else {
            return 0;
        };
        if !(next_v.is_finite() && next_w.is_finite() && next_y.is_finite()) {
            return 0;
        }
        self.v = next_v;
        self.w = next_w;
        self.y = next_y;
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = -1.0;
        self.w = -0.5;
        self.y = 0.0;
    }

    /// Run `n_steps` RK4 updates under a constant input, returning the `v` trace
    /// and the upward-crossing spike count. Reuses `step` (RK4) so the trace is
    /// bit-identical to the per-step path and — because the right-hand side is
    /// exact arithmetic (`v.powi(3)` = `v*v*v`) — to the Python reference. The
    /// final state is left in `self.v` / `self.w` / `self.y`.
    pub fn simulate(&mut self, n_steps: usize, current: f64) -> (Vec<f64>, i64) {
        let mut trace = Vec::with_capacity(n_steps);
        let mut spikes: i64 = 0;
        for _ in 0..n_steps {
            let spiked = self.step(current);
            trace.push(self.v);
            if spiked == 1 {
                spikes += 1;
            }
        }
        (trace, spikes)
    }
}
impl Default for FitzHughRinzelNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_constructor_state() {
        let default = FitzHughRinzelNeuron::default();
        let constructed = FitzHughRinzelNeuron::new();
        assert_eq!(default.v, constructed.v);
    }

    #[test]
    fn simulate_matches_repeated_step() {
        let mut simulated = FitzHughRinzelNeuron::new();
        let mut repeated = FitzHughRinzelNeuron::new();
        let (trace, spikes) = simulated.simulate(2_000, 1.0);
        let mut expected_trace = Vec::with_capacity(2_000);
        let mut expected_spikes = 0_i64;
        for _ in 0..2_000 {
            if repeated.step(1.0) == 1 {
                expected_spikes += 1;
            }
            expected_trace.push(repeated.v);
        }
        assert_eq!(trace, expected_trace);
        assert_eq!(spikes, expected_spikes);
    }

    #[test]
    fn fhr_fires() {
        let mut n = FitzHughRinzelNeuron::new();
        let t: i32 = (0..2000).map(|_| n.step(1.0)).sum();
        assert!(t > 0);
    }

    #[test]
    fn fhr_reset_clears_state() {
        let mut n = FitzHughRinzelNeuron::new();
        for _ in 0..500 {
            n.step(1.0);
        }
        n.reset();
        assert!((n.v - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn fhr_bounded() {
        let mut n = FitzHughRinzelNeuron::new();
        for _ in 0..2000 {
            n.step(50.0);
        }
        assert!(n.v.is_finite());
    }

    #[test]
    fn fhr_invalid_input_preserves_state() {
        let mut n = FitzHughRinzelNeuron::new();
        let before = (n.v, n.w, n.y);
        assert_eq!(n.step(f64::NAN), 0);
        assert_eq!((n.v, n.w, n.y), before);
    }

    #[test]
    fn fhr_matches_rk4_candidate() {
        let mut n = FitzHughRinzelNeuron::new();
        n.v = -1.0;
        n.w = 0.2;
        n.y = 0.1;
        let expected = n.rk4_candidate(0.5).unwrap();
        assert_eq!(n.step(0.5), 0);
        assert!((n.v - expected.0).abs() < 1.0e-15);
        assert!((n.w - expected.1).abs() < 1.0e-15);
        assert!((n.y - expected.2).abs() < 1.0e-15);
    }

    #[test]
    fn fhr_overflow_candidate_preserves_state() {
        let mut n = FitzHughRinzelNeuron {
            v: 1.0e155,
            ..Default::default()
        };
        let before = (n.v, n.w, n.y);
        assert_eq!(n.step(0.5), 0);
        assert_eq!((n.v, n.w, n.y), before);
    }

    #[test]
    fn fhr_negative_no_crash() {
        let mut n = FitzHughRinzelNeuron::new();
        for _ in 0..500 {
            n.step(-5.0);
        }
        assert!(n.v.is_finite());
    }

    #[test]
    fn fhr_slow_y_variable() {
        let mut n = FitzHughRinzelNeuron::new();
        let y0 = n.y;
        for _ in 0..2000 {
            n.step(1.0);
        }
        assert!((n.y - y0).abs() > 1e-6, "slow y should evolve in 3D model");
    }
}
